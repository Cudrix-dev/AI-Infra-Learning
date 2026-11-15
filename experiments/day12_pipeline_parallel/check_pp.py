"""Day 12 - two-stage pipeline vs the single-GPU model: parity, per-rank event order, in-flight counts, P2P labels.

Launch with torchrun, one process per GPU (rank r runs stage r):

  # Step 1 check: one microbatch across the boundary (uses YOUR forward_step/backward_step, a provided F-then-B loop)
  torchrun --standalone --nproc_per_node=2 experiments/day12_pipeline_parallel/check_pp.py --impl mine \
      --test boundary --baseline outputs/day12_pp/tensors/baseline_float32_RID.pt
  # Step 2/3 checks and the full matrix (both schedules, several M), plus the three negative controls
  torchrun --standalone --nproc_per_node=2 experiments/day12_pipeline_parallel/check_pp.py --impl ref \
      --schedules gpipe,1f1b --microbatches 1,2,4,8 --controls --baseline ... --run-id RID

--impl ref | mine        --device cuda (NCCL, default) | cpu (gloo; same logic, no GPUs, no memory numbers)

Per (schedule, M) case the checker compares, BEFORE any optimizer step and with the same weights and data:
  y, loss            last stage (rank 1): the concatenated model output and the summed microbatch losses
  blkK.<param>       each stage's parameter gradients (blk0-1 on rank 0, blk2-3 on rank 1)
against two single-GPU references from baseline_1gpu.py: gradient accumulation with the SAME M (same arithmetic in
the same order; reported bitwise) and the FULL batch (M = 1; same mathematics, different rounding order).
Pass: max|pp - ref| / max|ref| <= 1e-5 (fp32) against BOTH, and
  order    each rank's F/B microbatch order equals pp_sim.op_order()           (the simulator's schedule)
  peak     each rank's peak in-flight count equals the simulator's peak
  labels   the k-th activation (gradient) rank 1 (rank 0) received has the label of the k-th one the other sent
Reported, not required: bitwise flags, GPipe vs 1F1B bitwise, the P2P call sequence vs the simulator's Megatron-style
program, GPU timestamps, and CUDA memory (max_memory_allocated - memory_allocated at the schedule start).

Exit codes: 0 all passed, 1 a check failed (or a control was not detected as predicted), 3 my_pp.py has TODOs left.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pp_controls  # noqa: E402
import pp_ref  # noqa: E402
import pp_sim  # noqa: E402
from pp_common import (OUT, P2P, REL_TOL, EventLog, Shapes, Stage, build_blocks, compare, environment,  # noqa: E402
                       microbatches, set_exact_math, stage_blocks, write_json, zero_grads)

RANK = 0
P = 2


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def run_boundary(st: Stage, M: int) -> None:
    """Provided for the Step 1 check: one microbatch, forward through both stages, then backward."""
    assert M == 1
    x = st.comm.recv_forward(0)
    out, inp = st.forward_step(0, x)
    st.comm.send_forward(out, 0)
    g = st.comm.recv_backward(0)
    dx = st.backward_step(0, inp, out, g)
    st.comm.send_backward(dx, 0)
    st.boundary_info = {   # what crossed the boundary, as seen by this rank
        "input_is_leaf": bool(getattr(inp, "is_leaf", False)), "input_requires_grad": bool(inp.requires_grad),
        "input_grad": None if st.is_first or dx is None else [list(dx.shape), str(dx.dtype)],
        "output": [list(out.shape), str(out.dtype)],
    }


NEEDED = {"boundary": ("forward_step", "backward_step"), "gpipe": ("forward_step", "backward_step", "run_gpipe"),
          "1f1b": ("forward_step", "backward_step", "run_1f1b")}


def todo_left(impl, schedules) -> list[str]:
    """Names (in the order you write them) whose body still raises NotImplementedError. Checked from the source
    before any communication, so an unfinished function never leaves the other rank waiting in a receive."""
    names = []
    for s in schedules:
        names += [n for n in NEEDED[s] if n not in names]
    return [n for n in names if "raise NotImplementedError" in inspect.getsource(getattr(impl, n))]


class Runner:
    def __init__(self, problem, shapes, device, dtype):
        self.problem, self.shapes, self.device, self.dtype = problem, shapes, device, dtype
        self.stage = RANK
        self.model = build_blocks(problem, stage_blocks(shapes, self.stage), shapes, device, dtype)
        self.x = problem["x"].to(device) if self.stage == 0 else None
        self.t = problem["target"].to(device) if self.stage == P - 1 else None

    def run(self, impl, schedule: str, M: int) -> dict:
        sh = self.shapes
        zero_grads(self.model)
        mb_shape = (sh.seq, sh.batch // M, sh.d_model)
        st = Stage(self.stage, P, self.model, None, None, impl=impl)
        if st.is_first:
            st.inputs = microbatches(self.x, M)
        if st.is_last:
            st.targets = microbatches(self.t, M)
            st.n_total = self.t.numel()
            st.y_out = torch.empty_like(self.t)
        cuda = self.device.type == "cuda"
        dist.barrier()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        st.log = EventLog(RANK, self.stage, self.device)          # t0 = right after the barrier
        st.comm = P2P(self.stage, P, mb_shape, self.dtype, self.device, st.log)
        fn = {"boundary": run_boundary, "gpipe": getattr(impl, "run_gpipe", None),
              "1f1b": getattr(impl, "run_1f1b", None)}[schedule]
        fn(st, M)
        peak_mib = None
        if cuda:
            torch.cuda.synchronize()
            peak_mib = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        events = st.log.finalize()
        out = {"grads": {k: v.detach().cpu().clone() for k, v in self.model.grads().items()},
               "events": events, "order": st.log.order(), "peak": st.log.peak, "comm_calls": st.log.comm_calls(),
               "chan": {f"{d}_{k}": st.log.channel(d, k) for d in ("send", "recv") for k in ("a", "g")},
               "mem_peak_mib": peak_mib, "boundary": getattr(st, "boundary_info", None)}
        if st.is_last:
            out["y"] = st.y_out.cpu()
            out["loss"] = torch.stack([l for _, l in sorted(st.losses, key=lambda p: p[0])]).sum().cpu() \
                if len(st.losses) == M else torch.tensor(float("nan"))
            # Sum in microbatch order, as the baseline does. (Sorting is a no-op unless a schedule reordered them.)
        return out


def check_case(local: dict, refs: dict, schedule: str, M: int, rel_tol: float) -> dict:
    """Rank-local checks: tensors vs the two baselines, order and peak vs the simulator."""
    sched_for_sim = "gpipe" if schedule == "boundary" else schedule
    same, full = refs[M], refs[1]
    tensors = []
    names = list(local["grads"])
    if "y" in local:
        names = ["y", "loss"] + names
    for n in names:
        got = local[n] if n in ("y", "loss") else local["grads"][n]
        a = same[n] if n in ("y", "loss") else same["grads"][n]
        b = full[n] if n in ("y", "loss") else full["grads"][n]
        ca, cb = compare(n, got, a, rel_tol), compare(n, got, b, rel_tol)
        tensors.append({"name": n, "rank": RANK, "vs_same_M": ca, "vs_full": cb, "ok": ca["ok"] and cb["ok"]})
    sim = pp_sim.simulate(sched_for_sim, P, M, [(1, 2)] * P)
    sim_stage = pp_sim.stage_metrics(sim, RANK)
    sim_comm = pp_sim.comm_sequence(pp_sim.program(sched_for_sim, P, M, RANK))
    return {"rank": RANK, "tensors": tensors,
            "order": local["order"], "sim_order": sim_stage["order"], "order_ok": local["order"] == sim_stage["order"],
            "peak": local["peak"], "sim_peak": sim_stage["peak_in_flight"],
            "peak_ok": local["peak"] == sim_stage["peak_in_flight"],
            "comm_calls": local["comm_calls"], "sim_comm_calls": sim_comm,
            "comm_matches_sim": local["comm_calls"] == sim_comm,
            "chan": local["chan"], "mem_peak_mib": local["mem_peak_mib"], "boundary": local["boundary"],
            "events": local["events"]}


def merge(per_rank: list[dict]) -> dict:
    r0, r1 = per_rank
    labels_a = r0["chan"]["send_a"] == r1["chan"]["recv_a"]
    labels_g = r1["chan"]["send_g"] == r0["chan"]["recv_g"]
    tensors = [t for r in per_rank for t in r["tensors"]]
    failed = sorted({t["name"] for t in tensors if not t["ok"]})
    if not all(r["order_ok"] for r in per_rank):
        failed.append("order")
    if not all(r["peak_ok"] for r in per_rank):
        failed.append("peak")
    if not (labels_a and labels_g):
        failed.append("labels")
    return {"ranks": per_rank, "tensors": tensors, "labels_ok": labels_a and labels_g,
            "labels": {"a_sent_r0": r0["chan"]["send_a"], "a_recv_r1": r1["chan"]["recv_a"],
                       "g_sent_r1": r1["chan"]["send_g"], "g_recv_r0": r0["chan"]["recv_g"]},
            "failed": failed, "passed": not failed}


def in_flight_trace(events) -> str:
    return " ".join(str(e["in_flight"]) for e in events if e["type"] == "compute")


def print_case(title: str, res: dict, shapes: Shapes, M: int, verbose: bool = True) -> None:
    say(f"\n=== {title}  (microbatch [{shapes.seq}, {shapes.batch // M}, {shapes.d_model}]) ===")
    for r in res["ranks"]:
        say(f"rank {r['rank']} / stage {r['rank']}  order   {r['order']:<34} simulator: "
            f"{'same' if r['order_ok'] else r['sim_order']}  -> {'ok' if r['order_ok'] else 'FAIL'}")
        say(f"{'':17}in flight after each op: {in_flight_trace(r['events'])}   peak {r['peak']} "
            f"(simulator {r['sim_peak']})  -> {'ok' if r['peak_ok'] else 'FAIL'}")
        say(f"{'':17}P2P calls: {' '.join(r['comm_calls']) or '(none)'}")
        if not r["comm_matches_sim"]:
            say(f"{'':17}  simulator's Megatron-style program: {' '.join(r['sim_comm_calls'])}  (info only)")
        if verbose:
            times = " | ".join(f"{e['op']}{e['mb']} {e['start_ms']:.2f}-{e['end_ms']:.2f}" for e in r["events"]
                               if e["type"] == "compute")
            say(f"{'':17}GPU ms since t0: {times}")
        if r["mem_peak_mib"] is not None:
            say(f"{'':17}CUDA peak during the schedule: {r['mem_peak_mib']:.1f} MiB")
        if r["boundary"]:
            b = r["boundary"]
            say(f"{'':17}boundary: stage input is_leaf={b['input_is_leaf']} requires_grad={b['input_requires_grad']}"
                f"; output {b['output']}; input grad sent back: {b['input_grad']}")
    lab = res["labels"]
    say(f"labels: activations sent by rank 0 {lab['a_sent_r0']} received by rank 1 {lab['a_recv_r1']}; gradients "
        f"sent by rank 1 {lab['g_sent_r1']} received by rank 0 {lab['g_recv_r0']}  -> "
        f"{'ok' if res['labels_ok'] else 'FAIL'}")
    say(f"{'tensor':<17} {'rank':>4} | {'vs accumulation same M: rel':>27} {'bitwise':>8} | "
        f"{'vs full batch: rel':>18} {'bitwise':>8} {'ratio':>8} | ok")
    for t in res["tensors"]:
        a, b = t["vs_same_M"], t["vs_full"]
        say(f"{t['name']:<17} {t['rank']:>4} | {a['rel']:>27.3e} {str(a['bitwise']):>8} | {b['rel']:>18.3e} "
            f"{str(b['bitwise']):>8} {b['ratio']:>8.4f} | {'ok' if t['ok'] else 'FAIL'}")


def main() -> None:
    global RANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--test", choices=["boundary", "schedules"], default="schedules")
    ap.add_argument("--schedules", default="gpipe,1f1b")
    ap.add_argument("--microbatches", default="4")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--controls", action="store_true", help="also run the three negative controls (built from ref)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--quiet-times", action="store_true", help="do not print GPU timestamps")
    args = ap.parse_args()

    set_exact_math()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    RANK = dist.get_rank()
    assert dist.get_world_size() == P, "this exercise uses exactly two ranks (two stages)"
    impl = pp_ref if args.impl == "ref" else importlib.import_module("my_pp")
    schedules = ["boundary"] if args.test == "boundary" else args.schedules.split(",")
    Ms = [1] if args.test == "boundary" else [int(v) for v in args.microbatches.split(",")]

    say(f"pipeline: {P} stages, backend {dist.get_backend()}, impl {args.impl}, schedules {schedules}, M {Ms}")
    todo = todo_left(impl, schedules)
    if todo:
        say(f"\nINCOMPLETE (TODOs left): {', '.join(todo)} in my_pp.py still raise NotImplementedError. "
            f"Fill them in (GUIDE Block 2), then rerun.")
        dist.destroy_process_group()
        sys.exit(3)
    if device.type == "cuda":
        names = [None] * P
        dist.all_gather_object(names, f"rank {RANK} -> stage {RANK} on cuda:{local} {torch.cuda.get_device_name(local)}")
        for n in names:
            say("  " + n)

    data = torch.load(args.baseline, map_location="cpu", weights_only=True)
    problem, refs = data["problem"], data["results"]
    shapes = Shapes(**data["shapes"])
    dtype = problem["x"].dtype
    rel_tol = REL_TOL[dtype]
    missing = [M for M in Ms if M not in refs]
    if missing:
        raise SystemExit(f"baseline {args.baseline} has no accumulation result for M={missing}")
    say(f"baseline {args.baseline} ({data['dtype']}, shapes {data['shapes']}), pass if rel <= {rel_tol:g}; stage 0 "
        f"owns blocks {list(stage_blocks(shapes, 0))}, stage 1 owns {list(stage_blocks(shapes, 1))}")

    runner = Runner(problem, shapes, device, dtype)
    # Warm-up exchange: creates the NCCL point-to-point communicator with both ranks participating.
    w_send = torch.full((4,), float(RANK), device=device, dtype=dtype)
    w_recv = torch.empty(4, device=device, dtype=dtype)
    peer = 1 - RANK
    for req in dist.batch_isend_irecv([dist.P2POp(dist.isend, w_send, peer), dist.P2POp(dist.irecv, w_recv, peer)]):
        req.wait()
    assert w_recv[0].item() == peer

    record = {"run_id": args.run_id, "impl": args.impl, "test": args.test, "schedules": schedules, "M": Ms,
              "baseline": str(args.baseline), "dtype": data["dtype"], "rel_tol": rel_tol, "shapes": data["shapes"],
              "environment": environment(), "cases": {}, "controls": {}}
    kept: dict = {}                       # (schedule, M) -> this rank's grads (and y), for GPipe vs 1F1B
    events_dir = OUT / f"events_{args.run_id}"
    code = 0
    try:
        for sched in schedules:
            for M in Ms:
                local_res = runner.run(impl, sched, M)
                kept[(sched, M)] = {**local_res["grads"], **({"y": local_res["y"]} if "y" in local_res else {})}
                mine = check_case(local_res, refs, sched, M, rel_tol)
                per_rank = [None] * P
                dist.all_gather_object(per_rank, mine)
                res = merge(per_rank)
                key = f"{sched}_M{M}"
                print_case(f"{args.impl} | {sched} | M={M}", res, shapes, M, verbose=not args.quiet_times)
                say(f"RESULT {args.impl} {sched} M={M}: {'PASSED' if res['passed'] else 'FAILED ' + str(res['failed'])}")
                if RANK == 0:
                    events_dir.mkdir(parents=True, exist_ok=True)
                    with (events_dir / f"{args.impl}_{key}.jsonl").open("w") as f:
                        for r in res["ranks"]:
                            for e in r["events"]:
                                f.write(json.dumps(e) + "\n")
                for r in res["ranks"]:
                    r.pop("events")
                record["cases"][key] = res
                code = code or (0 if res["passed"] else 1)
    except NotImplementedError as e:
        print(f"[rank {RANK}] NOT IMPLEMENTED: {e}", flush=True)
        os._exit(3)

    # GPipe vs 1F1B: same microbatches, same per-stage accumulation order -> expected identical numbers.
    if "gpipe" in schedules and "1f1b" in schedules:
        cross = {}
        for M in Ms:
            a, b = kept[("gpipe", M)], kept[("1f1b", M)]
            cross[M] = all(torch.equal(a[k], b[k]) for k in a)
        all_cross = [None] * P
        dist.all_gather_object(all_cross, cross)
        record["gpipe_vs_1f1b_bitwise"] = {M: all(c[M] for c in all_cross) for M in Ms}
        say("\nGPipe vs 1F1B, every tensor on both ranks bitwise equal: "
            + ", ".join(f"M={M}: {record['gpipe_vs_1f1b_bitwise'][M]}" for M in Ms))

    if args.test == "schedules":
        say(f"\n=== summary: effective batch {shapes.batch} sequences fixed, M varies ===")
        say(f"{'sched':<6} {'M':>2} {'mb':>13} | {'sim bubble':>10} {'sim peak':>9} | {'meas. order/peak':>16} | "
            f"{'worst rel vs full':>18} {'bitwise vs accum.':>18} | {'CUDA peak MiB r0, r1':>21}")
        for sched in schedules:
            for M in Ms:
                res = record["cases"][f"{sched}_M{M}"]
                sim = pp_sim.summary(pp_sim.simulate(sched, P, M, [(1, 2)] * P))
                worst = max(res["tensors"], key=lambda t: t["vs_full"]["rel"])
                bw = all(t["vs_same_M"]["bitwise"] for t in res["tensors"])
                mem = [r["mem_peak_mib"] for r in res["ranks"]]
                mem_s = "n/a" if mem[0] is None else f"{mem[0]:.1f}, {mem[1]:.1f}"
                ok = all(r["order_ok"] and r["peak_ok"] for r in res["ranks"])
                say(f"{sched:<6} {M:>2} {str([shapes.seq, shapes.batch // M, shapes.d_model]):>13} | "
                    f"{float(sim['bubble_fraction']):>10.3f} {str([p['peak_in_flight'] for p in sim['stages']]):>9} | "
                    f"{'match' if ok else 'MISMATCH':>16} | {worst['vs_full']['rel']:>10.2e} {worst['name']:>7} "
                    f"{str(bw):>18} | {mem_s:>21}")
        say("CUDA peak = torch.cuda.max_memory_allocated() - memory_allocated() at the schedule start, per rank; "
            "grads, inputs, targets and the output buffer were allocated before. It counts saved activations, "
            "received buffers and backward temporaries; NCCL's own buffers are not in it.")

    if args.controls:
        say(f"\n=== negative controls ({pp_controls.CONTROL_SCHEDULE}, M={pp_controls.CONTROL_M}) ===")
        all_ok = True
        for name, ctl in pp_controls.CONTROLS.items():
            local_res = runner.run(ctl, pp_controls.CONTROL_SCHEDULE, pp_controls.CONTROL_M)
            mine = check_case(local_res, refs, pp_controls.CONTROL_SCHEDULE, pp_controls.CONTROL_M, rel_tol)
            mine.pop("events")
            per_rank = [None] * P
            dist.all_gather_object(per_rank, mine)
            res = merge(per_rank)
            expected = pp_controls.EXPECTED[name]
            as_pred = set(res["failed"]) == expected
            all_ok &= as_pred
            ratios = {t["name"]: round(t["vs_full"]["ratio"], 4) for t in res["tensors"] if not t["ok"]}
            say(f"{name:<20} failed {res['failed']}")
            say(f"{'':20} ratios vs full batch {ratios}")
            if "order" in res["failed"] or "labels" in res["failed"]:
                say(f"{'':20} rank 0 order {res['ranks'][0]['order']} (simulator {res['ranks'][0]['sim_order']}); "
                    f"gradient labels sent by rank 1 {res['labels']['g_sent_r1']}, rank 0 expected "
                    f"{res['labels']['g_recv_r0']}")
            say(f"{'':20} -> {'DETECTED as predicted' if as_pred else 'NOT AS PREDICTED: expected ' + str(sorted(expected))}")
            record["controls"][name] = {"failed": res["failed"], "expected": sorted(expected), "as_predicted": as_pred,
                                        "ratios": ratios, "orders": [r["order"] for r in res["ranks"]],
                                        "labels": res["labels"]}
        say(f"CONTROLS: {'ALL DETECTED AS PREDICTED' if all_ok else 'SOME NOT AS PREDICTED'}")
        code = code or (0 if all_ok else 1)

    verdict = "PASSED" if code == 0 else "FAILED"
    label = "BOUNDARY" if args.test == "boundary" else "PARITY"
    say(f"\n{label} ({args.impl}, {data['dtype']}, schedules {','.join(schedules)}, M {','.join(map(str, Ms))}): "
        f"{verdict}")
    record["exit_code"] = code
    record["passed"] = code == 0
    if RANK == 0:
        tag = "boundary" if args.test == "boundary" else "parity"
        write_json(OUT / f"{tag}_{args.impl}_{args.run_id}.json", record)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(code)


if __name__ == "__main__":
    main()
