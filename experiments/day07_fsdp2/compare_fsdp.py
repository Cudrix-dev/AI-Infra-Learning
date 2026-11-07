"""Day 07 - orchestrator: parity, collectives in the trace, memory timeline, timing, slow link. Plain python.

  --stage parity    tiny model, Day 04/06's setting: 6 optimizer steps of G = 32 (2 ranks x K 4 x m 4), the LAST window
                    short (G = 16). Reference: 1 process, one big batch per optimizer step, Day 01's AdamW.
                      A B C   Day 02's checks: step-0 gradient, params after 1 update, loss at step 0
                      D E     Day 04's checks: LR + Adam step counter, short final window
                      R       replicas: every NON-sharded parameter bitwise equal across ranks after every step
                              (all of them for ddp; FSDP shards every parameter, so R has nothing to check there)
                      S       AdamW state gathered from the shards matches the DDP + AdamW run
                    plus BITWISE against DDP + the same AdamW: required for the "boundary" group (one reduction of the
                    window's local sum), reported for "per_micro" (reduce-scatter every microbatch: predicted NOT
                    bitwise, Day 06's rs_every). Negative controls: each must be DETECTED by its named check.
  --stage count     medium, K = 4: one profiled run per variant -> collectives per optimizer step by kind, with the
                    FSDP group each one was launched for (analyze_fsdp_trace.py); K = 1 traces for the sequence and the
                    exposed-communication / idle-GPU numbers.
  --stage memory    medium, K = 1, m = 8: the block-boundary memory timeline of ddp and the three FSDP layouts, plus a
                    torch.cuda.memory snapshot per variant (rank 0) for pytorch.org/memory_viz.
  --stage timing    medium: ms per optimizer step, peak / between-microbatch memory, state per rank; K = 1 at m = 8 and
                    m = 32, K = 4 at m = 8 with sync / nosync; AdamW(foreach=True) everywhere, plus one per-tensor
                    AdamW pair (the DTensor dispatch cost).
  --stage slowlink  K = 1, m = 8 with NCCL_P2P_DISABLE=1: does per-block FSDP hide 1.5x DDP's bytes behind compute?
                    Plus two profiled runs for the exposed-communication numbers.
  --stage all       all of them, in that order.

  python experiments/day07_fsdp2/compare_fsdp.py --stage all --condition shared --run-id "a100_${RID}"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
sys.path.insert(0, str(HERE.parent / "day04_grad_accum"))
sys.path.insert(0, str(HERE))
from analyze_fsdp_trace import analyze_pair, print_sequence, summary  # noqa: E402
from check_parity import compare  # noqa: E402  (Day 02's A/B/C checks and tolerances)
from compare_accum import extra_checks, max_abs  # noqa: E402  (Day 04's D/E checks)
from launch_utils import make_env, run, smoke_test, torchrun_cmd  # noqa: E402

TRAIN = HERE / "train_fsdp.py"
OUT = HERE.parents[1] / "outputs" / "day07_fsdp2"
MiB = 2**20
STATE_RTOL = 1e-4
KINDS = ["AllGather", "ReduceScatter", "AllReduce", "Broadcast"]
M = 8  # rows per microbatch per rank (Days 03-06)

# name -> flags. K = 1 runs ignore --accum (the only microbatch is the last one).
V = {
    "ddp": ["--impl", "ddp"],
    "root_sync": ["--impl", "fsdp_root", "--accum", "sync"],
    "root_nosync": ["--impl", "fsdp_root", "--accum", "nosync"],
    "block_sync": ["--impl", "fsdp_block", "--accum", "sync"],
    "block_nosync": ["--impl", "fsdp_block", "--accum", "nosync"],
    "nrf_sync": ["--impl", "fsdp_block_nrf", "--accum", "sync"],
    "nrf_nosync": ["--impl", "fsdp_block_nrf", "--accum", "nosync"],
    "pf_sync": ["--impl", "fsdp_block_pf", "--accum", "sync"],
    "pf_nosync": ["--impl", "fsdp_block_pf", "--accum", "nosync"],
}
K1 = ["ddp", "root_sync", "block_sync", "nrf_sync", "pf_sync"]  # the K = 1 line-up (accum is irrelevant at K = 1)
P_SMALL = ["--bucket-mb", "0.25"]  # tiny model: several DDP buckets, as on Days 04 and 06
PARITY = [  # (name, flags, bitwise group, bitwise reference)
    ("ddp", [*V["ddp"], *P_SMALL], "reference", None),
    ("root_nosync", V["root_nosync"], "boundary", "ddp"),
    ("block_nosync", V["block_nosync"], "boundary", "ddp"),
    ("nrf_nosync", V["nrf_nosync"], "boundary", "ddp"),
    ("pf_nosync", V["pf_nosync"], "boundary", "ddp"),
    ("root_sync", V["root_sync"], "per_micro", "ddp"),
    ("block_sync", V["block_sync"], "per_micro", "ddp"),
    ("ddp_foreach", [*V["ddp"], *P_SMALL, "--foreach"], "reference", None),
    ("block_nosync_foreach", [*V["block_nosync"], "--foreach"], "boundary", "ddp_foreach"),
]
CONTROLS = [  # (bug, variant, check that must fail)
    ("no_init_bcast", "block_nosync", "C"),
    ("sync_never", "block_nosync", "A"),
    ("sum_grads", "block_nosync", "A"),
]


def load(tag: str) -> dict:
    return torch.load(OUT / tag / "result.pt", map_location="cpu", weights_only=True)


def launch(nproc: int, flags: list[str], env: dict) -> None:
    cmd = [sys.executable, str(TRAIN), *flags] if nproc == 1 else torchrun_cmd(nproc, TRAIN, flags)
    print("\n>>", " ".join(cmd), flush=True)
    if run(cmd, env)[0] != 0:
        raise SystemExit("run failed; not continuing")


# ---------------------------------------------------------------- R and S (Day 06's, R restricted to replicas)
def check_r(dp: dict) -> tuple[bool, str]:
    n = dp["meta"]["replicated_tensors"]
    if n == 0:
        return True, "no replicated parameters to compare (every parameter is a DTensor shard): vacuous"
    steps = dp["meta"]["in_sync_steps"]
    bad = [s for s, ok in steps if not ok]
    ok = not bad and dp["meta"]["in_sync_final"]
    return ok, (f"{n} replicated tensors bitwise equal across ranks after {len(steps)} checked steps: {ok}"
                + (f" (first divergence after step {bad[0]})" if bad else ""))


def state_diff(a: dict, b: dict) -> dict:
    out = {}
    for kind in ("exp_avg", "exp_avg_sq"):
        d = max((a[n][kind] - b[n][kind]).abs().max().item() for n in b)
        scale = max(b[n][kind].abs().max().item() for n in b)
        out[kind] = (d, scale)
    return out


def check_s(base: dict, dp: dict) -> tuple[bool, str]:
    if not dp["state_final"]:
        return False, "no optimizer state at all (AdamW never stepped a parameter)"
    sd = state_diff(dp["state_final"], base["state_final"])
    ok_t = all(d <= STATE_RTOL * scale for d, scale in sd.values())
    ok_step = dp["meta"]["adam_step"] == base["meta"]["adam_step"]
    text = "  ".join(f"{k} max|diff| {d:.3e} (max|ddp| {s:.3e})" for k, (d, s) in sd.items())
    return ok_t and ok_step, f"gathered AdamW state vs DDP + AdamW: {text}; step {dp['meta']['adam_step']}"


def bitwise(a: dict, b: dict) -> dict:
    res = {k: all(torch.equal(a[k][n], b[k][n]) for n in b[k]) for k in ("grads0", "params1", "grads_last", "final")}
    res["state"] = bool(a["state_final"]) and all(torch.equal(a["state_final"][n][k], b["state_final"][n][k])
                                                  for n in b["state_final"] for k in ("exp_avg", "exp_avg_sq"))
    return res


def all_checks(ref: dict, base: dict, dp: dict) -> dict:
    return {**compare(ref, dp), **extra_checks(ref, dp), "R": check_r(dp), "S": check_s(base, dp)}


def rank_line(r: dict) -> str:
    sb = r["state_bytes"]
    return (f"rank {r['rank']}: stores {r['local_numel']:>10,} of {r['full_numel']:,} elements ({r['dtensor_params']:2d} "
            f"DTensor) | exp_avg + exp_avg_sq {(sb.get('exp_avg', 0) + sb.get('exp_avg_sq', 0)) / MiB:7.2f} MiB"
            f" | delta first step {r['delta_first_step_mib']:7.2f} MiB")


# ---------------------------------------------------------------- stages
def stage_parity(args, env, rid: str) -> list[str]:
    common = ["--preset", "tiny", "--opt-steps", "6", "--global-batch", "32", "--last-global-batch", "16",
              "--condition", args.condition, "--check-every", "1"]
    tag = lambda n: f"par_{n}_{rid}"  # noqa: E731
    if not args.skip_run:
        launch(1, ["--impl", "single", "--micro-batch", "32", "--tag", tag("ref"), *common], env)  # K = 1
        for name, flags, _, _ in PARITY:
            launch(2, ["--micro-batch", "4", "--tag", tag(name), *common, *flags], env)
        for bug, var, _ in CONTROLS:
            launch(2, ["--micro-batch", "4", "--tag", tag(f"bug_{bug}"), *common, *V[var], "--bug", bug], env)
    ref = load(tag("ref"))
    results = {name: load(tag(name)) for name, _, _, _ in PARITY}
    base = results["ddp"]
    failed = []
    print("\n=== PARITY vs 1 process, one big batch per optimizer step (A B C: Day 02 | D E: Day 04 | R replicas | "
          "S optimizer state vs DDP + AdamW) ===")
    for name, _, _, _ in PARITY:
        dp = results[name]
        res = all_checks(ref, base, dp)
        m = dp["meta"]
        print(f"\n--- {name} (impl {m['impl']}, fsdp accum {m['fsdp_accum']}, K {m['accum']}, windows "
              f"{m['n_micro_per_window']}, groups {len(m['fsdp_groups'])}) ---")
        for k in "ABCDERS":
            print(f"{'PASS' if res[k][0] else 'FAIL'}  {k}  {res[k][1]}"
                  + ("  [S reference: this run]" if k == "S" and name == "ddp" else ""))
        for r in m["per_rank"]:
            print("      " + rank_line(r))
        if not all(res[k][0] for k in "ABCDERS"):
            failed.append(name)

    print("\n=== BITWISE vs DDP with the same AdamW (grads0, params1, grads_last, final params, gathered state) ===")
    for name, _, group, refname in PARITY:
        if refname is None:
            continue
        bw = bitwise(results[name], results[refname])
        same = all(bw.values())
        diff = max_abs(results[name], results[refname], "final")
        print(f"  [{group:9s}] {name:22s} == {refname:11s}: {same}  {bw}  (final params max|diff| {diff:.3e})")
        if group == "boundary" and not same:
            failed.append(f"{name} (not bitwise equal to {refname})")
    print("  boundary: required (same local window sum, one two-operand reduction at W = 2, elementwise AdamW on "
          "shards).\n  per_micro: predicted NOT bitwise (one reduction per microbatch, Day 06 rs_every), reported only.")

    print("\n=== NEGATIVE CONTROLS (lower-case pass/fail; the named check must FAIL) ===")
    for bug, var, must_fail in CONTROLS:
        dp = load(tag(f"bug_{bug}"))
        res = all_checks(ref, base, dp)
        print(f"\n--- bug_{bug} on {var} (expected detector: {must_fail}) ---")
        for k in "ABCDERS":
            print(f"{'pass' if res[k][0] else 'fail'}  {k}  {res[k][1]}")
        caught = not res[must_fail][0]
        print(f"      => {'DETECTED by check ' + must_fail if caught else 'MISSED by check ' + must_fail}")
        if not caught:
            failed.append(f"bug_{bug} (control missed)")
    print("\nPARITY: " + ("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}"))
    return failed


def medium_flags(name: str, k: int, m: int, args, steps: int, warmup: int, foreach: bool = True) -> list[str]:
    common = ["--preset", "medium", "--opt-steps", str(steps), "--warmup", str(warmup), "--condition", args.condition,
              "--check-every", "1000", "--log-every", "1000", "--no-tensors"] + (["--foreach"] if foreach else [])
    return [*V[name], "--global-batch", str(2 * m * k), "--micro-batch", str(m), *common]


COUNT_PLAN = [  # (K, m, variants): profiled medium runs
    (4, M, ["ddp", "root_sync", "root_nosync", "block_sync", "block_nosync", "nrf_sync", "nrf_nosync"]),
    (1, M, K1),
    (1, 32, ["ddp", "block_sync", "pf_sync"]),
]
SEQUENCES = [("root_sync", 1, M), ("block_sync", 1, M), ("pf_sync", 1, M), ("block_sync", 1, 32)]


def stage_count(args, env, rid: str) -> dict:
    out = {}
    for k, m, names in COUNT_PLAN:
        for name in names:
            tag = f"cnt_{name}_k{k}_m{m}_{rid}"
            if not args.skip_run:
                launch(2, ["--tag", tag, *medium_flags(name, k, m, args, steps=8, warmup=3), "--profile",
                           "--profile-steps", "2"], env)
            res = analyze_pair(OUT / tag / "trace_rank0.json")
            out[(name, k, m)] = {"summary": summary(res), "res": res}
    for k, m, names in COUNT_PLAN:
        print(f"\n=== COLLECTIVES per optimizer step: medium, K = {k}, m = {m}, rank-0 trace, 2 profiled steps ===")
        print(f"{'variant':12s} " + " ".join(f"{kk:>13s}" for kk in KINDS)
              + f" {'comm ms':>8s} {'min-rank':>8s} {'exposed':>8s} {'GPU busy':>9s} {'span':>7s} {'idle':>6s}")
        for name in names:
            sm = out[(name, k, m)]["summary"]
            ps = sm["per_step"]
            print(f"{name:12s} " + " ".join(f"{ps.get(kk, 0):13.1f}" for kk in KINDS)
                  + f" {sm['comm_ms']:8.2f} {sm.get('comm_min_rank_sum_ms', float('nan')):8.2f}"
                  f" {sm['exposed_comm_ms']:8.2f} {sm['gpu_busy_ms']:9.2f} {sm['span_ms']:7.2f} {sm['idle_ms']:6.2f}")
    print("\none ncclDevKernel_* per collective; a kernel belongs to the step its LAUNCH falls in. comm = union of rank 0's"
          " NCCL\nkernels (includes waiting for rank 1); min-rank = sum over collectives of min(duration on rank 0, "
          "rank 1);\nexposed = rank 0 comm with no other kernel running; idle = GPU span - union of all GPU work. "
          "Profiled\nruns: absolute ms are inflated (Day 03), use them to rank, not as benchmarks.")
    for name, k, m in SEQUENCES:
        print(f"\n=== SEQUENCE of collectives, {name}, K = {k}, m = {m} (which group, which phase, overlapped?) ===")
        print_sequence(out[(name, k, m)]["res"])
    js = {f"{n}_k{k}_m{m}": v["summary"] for (n, k, m), v in out.items()}
    js.update({f"{n}_k{k}_m{m}_sequence": out[(n, k, m)]["res"]["steps"][0]["sequence"] for n, k, m in SEQUENCES})
    (OUT / f"count_summary_{rid}.json").write_text(json.dumps(js, indent=2))
    return out


def print_timeline(name: str, r: dict) -> None:
    print(f"\n--- {name}, rank {r['rank']}: memory_allocated after each event / peak since the previous event (MiB) ---")
    base = r["timeline"][0][1] if r["timeline"] else 0.0
    for what, alloc, peak in r["timeline"]:
        print(f"  {what:58s} {alloc:8.1f}  (+{alloc - base:7.1f})   peak {peak:8.1f}")


def stage_memory(args, env, rid: str) -> None:
    names = K1
    for name in names:
        tag = f"mem_{name}_{rid}"
        if not args.skip_run:
            launch(2, ["--tag", tag, *medium_flags(name, 1, M, args, steps=8, warmup=3), "--memory-snapshot"], env)
    print(f"\n=== MEMORY TIMELINE: medium, K = 1, m = {M}, microbatch 0 of the last warmup step (state already "
          f"allocated) ===")
    rows = []
    for name in names:
        meta = json.loads((OUT / f"mem_{name}_{rid}" / "metrics.json").read_text())["meta"]
        r = meta["per_rank"][0]
        print_timeline(name, r)
        tl = r["timeline"]
        fwd = max(p for w, _, p in tl[: next(i for i, e in enumerate(tl) if e[0].startswith("loss")) + 1])
        bwd = max(p for w, _, p in tl[next(i for i, e in enumerate(tl) if e[0].startswith("loss")) + 1:
                                      next(i for i, e in enumerate(tl) if e[0].startswith("bwd_done")) + 1])
        opt = next(p for w, _, p in tl if w.startswith("opt_post"))
        rows.append({"variant": name, "start": tl[0][1], "fwd_peak": fwd, "bwd_peak": bwd, "opt_peak": opt,
                     "after_wrap": r["alloc_after_wrap_mib"], "model": r["alloc_model_mib"]})
    print(f"\n{'variant':11s} {'model':>7s} {'after wrap':>10s} {'start of step':>13s} {'fwd peak':>9s} "
          f"{'bwd peak':>9s} {'opt peak':>9s}")
    for x in rows:
        print(f"{x['variant']:11s} {x['model']:7.1f} {x['after_wrap']:10.1f} {x['start']:13.1f} {x['fwd_peak']:9.1f} "
              f"{x['bwd_peak']:9.1f} {x['opt_peak']:9.1f}")
    print("snapshots: outputs/day07_fsdp2/mem_<variant>_<rid>/memory_snapshot_rank0.pickle -> drag into "
          "https://pytorch.org/memory_viz")
    (OUT / f"memory_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def timing_table(title: str, plan: list[tuple[str, int, int, bool]], prefix: str, args, env, rid: str,
                 extra_env: dict | None = None) -> list[dict]:
    res = {p: [] for p in plan}
    for rep in range(args.repeats):
        for p in plan:
            name, k, m, fe = p
            tag = f"{prefix}_{name}_k{k}_m{m}{'' if fe else '_pt'}_r{rep}_{rid}"
            if not args.skip_run:
                launch(2, ["--tag", tag, *medium_flags(name, k, m, args, args.steps, args.warmup, foreach=fe)],
                       {**env, **(extra_env or {})})
            res[p].append(json.loads((OUT / tag / "metrics.json").read_text()))
    med = statistics.median
    rng = lambda v: f"{med(v):.2f} [{min(v):.2f}-{max(v):.2f}]"  # noqa: E731
    print(f"\n=== {title}: medium, {args.steps} optimizer steps ({args.warmup} warmup excluded), "
          f"repeats={args.repeats}, condition={args.condition}, run-id={rid} ===")
    print(f"{'K':>2s} {'m':>3s} {'optim':7s} {'variant':13s} {'ms / opt step [min-max]':>24s} {'fwd+bwd':>8s} "
          f"{'optim':>6s} {'tok/s':>8s} {'peak MiB':>8s} {'between':>8s} {'after wrap':>10s} {'state/rank':>10s}")
    rows = []
    for p in plan:
        name, k, m, fe = p
        t = [x["timing"] for x in res[p]]
        pr = res[p][0]["meta"]["per_rank"]
        core = [x["core_step_ms"] for x in t]
        ph = {q: med(x[q] for x in t) for q in ("fwd_bwd_ms", "optim_ms", "core_tokens_per_s")}
        state = max(sum(r["state_bytes"].get(kk, 0) for kk in ("exp_avg", "exp_avg_sq")) / MiB for r in pr)
        between = max(r.get("alloc_between_micro_mib", float("nan")) for r in pr)
        wrap = max(r["alloc_after_wrap_mib"] for r in pr)
        peak = med(x["peak_alloc_mib_max_rank"] for x in t)
        print(f"{k:2d} {m:3d} {'foreach' if fe else 'per-t':7s} {name:13s} {rng(core):>24s} {ph['fwd_bwd_ms']:8.2f} "
              f"{ph['optim_ms']:6.2f} {ph['core_tokens_per_s']:8.0f} {peak:8.0f} {between:8.1f} {wrap:10.1f} "
              f"{state:10.1f}")
        rows.append({"variant": name, "accum": k, "micro_batch": m, "foreach": fe, "core_ms": core, "phases_ms": ph,
                     "peak_mib": peak, "between_micro_mib": between, "after_wrap_mib": wrap,
                     "state_mib_per_rank": state, "per_rank": pr})
    print("ms / opt step: K x (fwd+bwd, all communication) + optimizer + scheduler; synchronized phase boundaries;\n"
          "slowest rank per step, median over steps; [min-max] over repeats. between = memory_allocated() after\n"
          "microbatch 0 (K >= 2), max over ranks; state/rank = exp_avg + exp_avg_sq stored on one rank.")
    return rows


def stage_timing(args, env, rid: str) -> None:
    plan = ([(n, 1, M, True) for n in K1] + [(n, 1, 32, True) for n in K1]
            + [(n, 4, M, True) for n in ["ddp", "root_nosync", "block_sync", "block_nosync", "nrf_sync", "pf_sync"]]
            + [("ddp", 1, M, False), ("block_sync", 1, M, False)])
    rows = timing_table("TIMING", plan, "time", args, env, rid)
    (OUT / f"timing_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def stage_slowlink(args, env, rid: str) -> None:
    slow = {"NCCL_P2P_DISABLE": "1"}
    plan = [(n, 1, M, True) for n in K1]
    rows = timing_table("SLOW LINK (NCCL_P2P_DISABLE=1)", plan, "slow", args, env, rid, extra_env=slow)
    prof = {}
    for name in ("ddp", "root_sync", "block_sync", "pf_sync"):
        tag = f"slowprof_{name}_k1_{rid}"
        if not args.skip_run:
            launch(2, ["--tag", tag, *medium_flags(name, 1, M, args, steps=8, warmup=3), "--profile",
                       "--profile-steps", "2"], {**env, **slow})
        prof[name] = analyze_pair(OUT / tag / "trace_rank0.json")
    print("\n=== SLOW LINK, profiled (K = 1, m = 8): how much communication stayed exposed? ===")
    for name, res in prof.items():
        sm = summary(res)
        print(f"{name:11s} per step {sm['per_step']} | comm {sm['comm_ms']:.2f} ms (min-rank sum "
              f"{sm.get('comm_min_rank_sum_ms', float('nan')):.2f}), exposed {sm['exposed_comm_ms']:.2f} | GPU busy "
              f"{sm['gpu_busy_ms']:.2f} / span {sm['span_ms']:.2f} ms")
    for name in ("block_sync", "pf_sync"):
        print(f"\n--- sequence, {name}, slow link ---")
        print_sequence(prof[name], max_rows=40)
    (OUT / f"slowlink_summary_{rid}.json").write_text(json.dumps(
        {"timing": rows, "profiled": {n: summary(r) for n, r in prof.items()}}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["parity", "count", "memory", "timing", "slowlink", "all"], default="all")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=25, help="optimizer steps per timing run")
    ap.add_argument("--warmup", type=int, default=5, help="optimizer steps excluded from timing")
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--skip-run", action="store_true", help="re-read results of an earlier --run-id")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    if args.skip_run and not args.run_id:
        raise SystemExit("--skip-run needs --run-id")
    rid = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    env = make_env(args.omp_threads)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"run id {rid}; results in {OUT}/")
    if not args.skip_run and not args.no_smoke:
        smoke_test(2, "cuda", args.omp_threads)
    if args.stage in ("parity", "all") and stage_parity(args, env, rid):
        print("parity failed: not timing a wrong implementation")
        return 1
    if args.stage in ("count", "all"):
        stage_count(args, env, rid)
    if args.stage in ("memory", "all"):
        stage_memory(args, env, rid)
    if args.stage in ("timing", "all"):
        stage_timing(args, env, rid)
    if args.stage in ("slowlink", "all"):
        stage_slowlink(args, env, rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
