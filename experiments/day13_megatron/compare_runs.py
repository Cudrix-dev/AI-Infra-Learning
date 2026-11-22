"""Day 13 - compare the layouts of one run id: losses, groups, P2P shapes, NCCL counts and the 1F1B order.
Standard library only (runs on Windows on a downloaded run).

    python experiments/day13_megatron/compare_runs.py outputs/day13_megatron/run_<RID>

Reads <run>/<tag>/summary.json (train_gpt.py) and <run>/<tag>/trace_analysis.json (analyze_trace.py) for every
tag present (tp1pp1, tp2pp1, tp1pp2, tp2pp2, tp2pp2_sp, and *_mine). Checks:
  1. every job's process groups equal the mapper's (train_gpt.py did the comparison inside the job);
  2. the losses of every step equal the one-GPU reference's within --rtol (same full weights, FP32, TF32 off;
     only the reduction order differs), and so does the step-0 gradient norm of every FULL parameter (TP shards
     reassembled) - the loss alone is a 4096-token mean that hides per-element rounding;
  3. the P2P buffer shape Megatron computed (get_tensor_shapes) is [s/TP if SP else s, b, h];
  4. per rank, the NCCL kernel counts in the profiled step equal the counts derived from the model (Section 3 of
     the GUIDE derives them; SP runs are reported, not checked);
  5. per rank, the F/B order equals non-interleaved 1F1B's (warmup = min(P - s - 1, M) forwards).
Exit 0 when every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

LAYERS, M = 6, 8


def one_f_one_b(P: int, s: int, m: int) -> list[str]:
    """Day 12's non-interleaved 1F1B order for stage s of P with m microbatches."""
    w = min(P - s - 1, m)
    order, f, b = [], 0, 0
    for _ in range(w):
        order.append(f"F{f}"); f += 1
    for _ in range(m - w):
        order.append(f"F{f}"); f += 1
        order.append(f"B{b}"); b += 1
    for _ in range(w):
        order.append(f"B{b}"); b += 1
    return order


def expected_counts(tp: int, pp: int, stage: int, sp: bool) -> Counter | None:
    """NCCL kernels per rank in one step, SP off. Keys (collective, phase, elems).
    Per microbatch: each TransformerLayer has 2 forward all-reduces (after linear_proj and linear_fc2, the
    row-parallel outputs: Day 11's g) and 2 backward all-reduces (input grads of linear_qkv and linear_fc1, the
    column-parallel inputs: Day 11's f). The first stage adds 1 forward all-reduce (vocab-parallel embedding);
    the last stage adds 1 backward all-reduce (output layer's input grad, column-parallel) and 3 forward
    all-reduces of b*s elements in vocab-parallel cross-entropy (max logit, target logit, sum of exp).
    P2P per stage of a 2-stage 1F1B: 1 + (M - 1) + 1 = M + 1 SendRecv kernels (Section 3)."""
    if sp:
        return None
    c = Counter()
    act = 256 * 2 * 512
    if tp > 1:
        per_stage_layers = LAYERS // pp
        first, last = stage == 0, stage == pp - 1
        c[("allreduce", "F", act)] = M * (2 * per_stage_layers + (1 if first else 0))
        c[("allreduce", "B", act)] = M * (2 * per_stage_layers + (1 if last else 0))
        if last:
            c[("allreduce", "F", 256 * 2)] = 3 * M
    if pp > 1:
        c[("p2p", "-", act)] = M + 1
    return c


def full_grad_norms(ranks: dict[int, dict]) -> dict[str, float]:
    """Per global parameter, the norm of the FULL step-0 gradient: TP shards' sums of squares added, replicated
    parameters (LayerNorms, row-parallel biases, position embedding) taken from TP rank 0. DP = 1, and every
    parameter lives on exactly one PP stage."""
    acc: dict[str, float] = {}
    for r in ranks.values():
        for name, g in (r.get("grad_sumsq_step0") or {}).items():
            if g["sumsq"] is None:
                continue
            if g["tp_sharded"]:
                acc[name] = acc.get(name, 0.0) + g["sumsq"]
            elif r["coords"]["tp"] == 0:
                acc[name] = g["sumsq"]
    return {k: v ** 0.5 for k, v in acc.items()}


def measured_counts(rank_analysis: dict) -> Counter:
    c = Counter()
    for k in rank_analysis["comms"]:
        if k["kernel"] == "ncclDevKernel_SendRecv":
            c[("p2p", "-", k["nelems"])] += 1
        else:
            ph = k["phase"][0] if k["phase"] != "outside F/B" else "-"
            c[(k["collective"], ph, k["nelems"])] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    tags = sorted(p.name for p in run.iterdir() if (p / "summary.json").is_file())
    S = {t: json.loads((run / t / "summary.json").read_text()) for t in tags}
    A = {t: json.loads((run / t / "trace_analysis.json").read_text()) for t in tags
         if (run / t / "trace_analysis.json").is_file()}
    R = {t: {int(p.stem[4:]): json.loads(p.read_text()) for p in (run / t).glob("rank*.json")} for t in tags}
    ok, report = True, {}
    ref_tag = next((t for t in tags if t.startswith("tp1pp1")), None)   # tp1pp1 or tp1pp1_mine
    ref = S.get(ref_tag)

    print(f"Run {run.name}: layouts {tags}\n")
    print("| layout | groups = mapper | losses (step 0..n) | max rel diff vs tp1pp1 | P2P shape | step ms unprofiled |")
    print("| --- | --- | --- | --- | --- | --- |")
    for t in tags:
        s = S[t]
        g_ok = bool(s["group_check"] and s["group_check"]["passed"]) and s["last_stage_tp_ranks_equal"]
        diff = None
        if ref and t != ref_tag:
            diff = max(abs(a - b) / abs(b) for a, b in zip(s["losses"], ref["losses"]))
        l_ok = diff is None or diff <= args.rtol
        shape_exp = [256 // (s["tp"] if s["sp"] else 1), 2, 512]
        sh_ok = s["p2p_tensor_shapes"] == [shape_exp]
        ms = [round(m, 1) for m, p in zip(s.get("step_ms", []), s.get("profiled", [])) if p is None][1:]
        ok &= g_ok and l_ok and sh_ok
        report[t] = {"groups_ok": g_ok, "loss_rel_diff": diff, "loss_ok": l_ok, "shape_ok": sh_ok,
                     "losses": s["losses"], "step_ms_unprofiled_after_first": ms}
        print(f"| {t} | {'ok' if g_ok else 'FAIL'} | {', '.join(f'{x:.7f}' for x in s['losses'])} | "
              f"{'-' if diff is None else f'{diff:.2e}'}{'' if l_ok else ' FAIL'} | "
              f"{s['p2p_tensor_shapes'][0]}{'' if sh_ok else ' FAIL'} | {ms} |")

    print("\nStep-0 gradient norm of every full parameter vs the one-GPU reference "
          "(the loss is a 4096-token mean and hides per-element rounding; these do not):")
    ref_norms = full_grad_norms(R[ref_tag]) if ref_tag else {}
    for t in tags:
        if t == ref_tag or not ref_norms:
            continue
        norms = full_grad_norms(R[t])
        same_keys = set(norms) == set(ref_norms)
        rel = {k: abs(norms[k] - ref_norms[k]) / ref_norms[k] for k in ref_norms if k in norms and ref_norms[k] > 0}
        worst = max(rel, key=rel.get) if rel else None
        exact = sum(norms[k] == ref_norms[k] for k in rel)
        g_ok = same_keys and worst is not None and rel[worst] <= args.rtol
        ok &= g_ok
        report[t]["grad_norms"] = {"params": len(norms), "same_params": same_keys, "bitwise_equal_norms": exact,
                                   "worst": worst, "worst_rel": rel.get(worst)}
        print(f"  {t}: {len(norms)} parameters (same set: {same_keys}), {exact} norms bitwise equal, worst rel diff "
              f"{rel[worst]:.2e} ({worst}) [{'ok' if g_ok else 'FAIL'}]")

    print("\nNCCL kernels per rank in the profiled step: (collective, phase, elements) -> count")
    for t in tags:
        if t not in A:
            continue
        s = S[t]
        for rank, ra in sorted(A[t]["ranks"].items(), key=lambda x: int(x[0])):
            stage = R[t][int(rank)]["coords"]["pp"]
            got = measured_counts(ra)
            exp = expected_counts(s["tp"], s["pp"], stage, s["sp"])
            order_exp = one_f_one_b(s["pp"], stage, M) if s["pp"] > 1 else [x for i in range(M) for x in (f"F{i}", f"B{i}")]
            o_ok = ra["order"] == order_exp
            c_ok = exp is None or got == exp
            ok &= o_ok and c_ok
            report.setdefault(t, {}).setdefault("ranks", {})[rank] = {
                "stage": stage, "counts": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sorted(got.items())},
                "counts_ok": None if exp is None else c_ok, "order": ra["order"], "order_ok": o_ok}
            counts = ", ".join(f"{k[0]}/{k[1]}/{k[2]}: {v}" for k, v in sorted(got.items()))
            print(f"  {t} rank {rank} (stage {stage}): {counts}  "
                  f"[{'derived counts ok' if c_ok and exp is not None else ('not checked (SP)' if exp is None else 'COUNTS FAIL')}]"
                  f"  order {' '.join(ra['order'])} [{'1F1B ok' if o_ok else 'ORDER FAIL'}]")
    if args.json:
        Path(args.json).write_text(json.dumps({"passed": ok, "layouts": report}, indent=1))
    print(f"\nCOMPARE {run.name}: {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
