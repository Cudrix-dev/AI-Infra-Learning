"""Day 13 follow-up - find what stalls one pipeline stage for tens of milliseconds, and test fixes.

The Day 13 timing traces showed 40-50 ms gaps inside one forward on both TP ranks of a stage, with no communication
running; the other stage then waited on its SendRecv for the same time (notes Section 6). This probe runs many
steps of the same Megatron job WITHOUT the profiler and records, on every rank, host-side timestamps of:

  - each step (CUDA-synchronized at both ends, as in train_gpt.py);
  - every forward_step_func call ("F k"), every backward_step call ("B k") and every P2P call of the schedule
    (send_forward, send_forward_recv_backward, ...), entry and exit;
  - every Python garbage collection (gc.callbacks: generation, start, stop, objects collected).

All timestamps are time.perf_counter_ns(), i.e. CLOCK_MONOTONIC, which is the same clock in every process on one
host, so ranks can be compared directly.

Options toggle the suspected causes and the candidate fixes (GUIDE Appendix B):

  --profile-steps 2:stack,3:timing   wrap these steps in torch.profiler as train_gpt.py does (to reproduce)
  --gc auto|manual                   manual = gc.disable() + gc.freeze() after setup, gc.collect() at a fixed point
  --gc-interval N                    with --gc manual: collect every N steps (0 = never during the run)
  --pin-numa                         bind each rank to the CPUs of its GPU's NUMA node

Usage (four GPUs, from the project root):
  export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1
  torchrun --standalone --nproc_per_node=4 experiments/day13_megatron/stall_probe.py --steps 300 --run-id base
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import gzip
import json
import os
import shutil
import statistics
import sys
import time
import weakref
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_gpt as tg  # noqa: E402  (model, init and data of the Day 13 job, unchanged)

NS = 1e-6  # ns -> ms


def parse_args():
    ap = argparse.ArgumentParser(description="Day 13: find and fix host stalls in the TP x PP job")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=5, help="steps excluded from the statistics")
    ap.add_argument("--profile-steps", default="", help='e.g. "2:stack,3:timing" (empty = no profiler)')
    ap.add_argument("--gc", choices=["auto", "manual"], default="auto")
    ap.add_argument("--gc-interval", type=int, default=0, help="--gc manual: gc.collect() every N steps (0 = never)")
    ap.add_argument("--pin-numa", action="store_true")
    ap.add_argument("--release-profiler", choices=["later", "collect"], default="later",
                    help="after exporting a profiled step: 'later' = drop the reference and let the GC free the "
                    "profiler whenever it next runs (what train_gpt.py does); 'collect' = gc.collect() right away, "
                    "outside the timed step")
    ap.add_argument("--stall-ms", type=float, default=10.0, help="a host call longer than this counts as a stall")
    ap.add_argument("--pid-dir", default="", help="write pid_rank<r>.txt here (for host_sampler.py)")
    ap.add_argument("--allow-ptrace", action="store_true",
                    help="prctl(PR_SET_PTRACER_ANY) so stall_catcher.py's gdb can attach (no CAP_SYS_PTRACE here)")
    ap.add_argument("--gc-debug-steps", default="", help='e.g. "2,3,4": before these steps, rank 0 prints the size '
                    'of each GC generation and the largest containers in generation 0 (diagnosis only)')
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="outputs/day13_stall")
    a = ap.parse_args()
    a.sp, a.lr, a.seed, a.impl = False, 1e-4, 1234, "ref"   # what train_gpt's helpers expect
    return a


# ------------------------------------------------------------------------------------------------- recording
class Recorder:
    """Host-side events of this rank: (kind, name, start_ns, end_ns, extra)."""

    def __init__(self):
        self.events = []
        self.step = -1
        self._gc_start = None

    def span(self, kind, name, t0, t1, **extra):
        self.events.append({"step": self.step, "kind": kind, "name": name, "t0": t0, "t1": t1, **extra})

    def gc_callback(self, phase, info):
        # Called by CPython around every collection; keep it cheap (no allocation-heavy work here).
        now = time.perf_counter_ns()
        if phase == "start":
            self._gc_start = now
        elif self._gc_start is not None:
            self.span("gc", f"gen{info['generation']}", self._gc_start, now, collected=info["collected"])
            self._gc_start = None


REC = Recorder()


def timed(kind, name_fn, fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter_ns()
        try:
            return fn(*a, **kw)
        finally:
            REC.span(kind, name_fn(), t0, time.perf_counter_ns())
    return wrapper


def instrument():
    """Wrap Megatron's backward_step and the P2P calls the 1F1B schedule makes. Both are looked up at call time
    (module global / class attribute), so replacing them is enough; numerics are unchanged."""
    from megatron.core.pipeline_parallel import p2p_communication, schedules

    count = {"B": 0}
    original_bwd = schedules.backward_step

    def backward_step(*a, **kw):
        k = count["B"]
        count["B"] += 1
        t0 = time.perf_counter_ns()
        try:
            return original_bwd(*a, **kw)
        finally:
            REC.span("B", f"B{k}", t0, time.perf_counter_ns())

    schedules.backward_step = backward_step
    cls = p2p_communication.P2PCommunicator
    for name in ("recv_forward", "recv_backward", "send_forward", "send_backward",
                 "send_forward_recv_backward", "send_backward_recv_forward"):
        original = getattr(cls, name)
        setattr(cls, name, timed("p2p", (lambda n=name: n), original))
    return count


def traced_forward(count):
    def forward_step(data_iterator, model):
        k = count["F"]
        count["F"] += 1
        t0 = time.perf_counter_ns()
        try:
            return tg.ref_forward_step_func(data_iterator, model)
        finally:
            REC.span("F", f"F{k}", t0, time.perf_counter_ns())
    return forward_step


def pin_to_gpu_numa(local_rank: int) -> str:
    """Restrict this process (and the threads it creates later) to the CPUs local to its GPU, read from sysfs via
    the GPU's PCI address. Call before NCCL and autograd start their threads: new threads inherit the mask."""
    p = torch.cuda.get_device_properties(local_rank)
    bus = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    cpulist = Path(f"/sys/bus/pci/devices/{bus}/local_cpulist").read_text().strip()
    cpus = set()
    for part in cpulist.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    os.sched_setaffinity(0, cpus)
    return cpulist


def gc_snapshot(step: int):
    """Diagnosis: how many objects each generation holds now, and which gen-0 containers are largest. A gen-0
    collection traverses every object in gen 0, and each container's traverse visits all of its items."""
    sizes = [len(gc.get_objects(generation=g)) for g in range(3)]
    young = gc.get_objects(generation=0)
    big = []
    for o in young:
        try:
            n = len(o)
        except TypeError:
            continue
        if n > 1000:
            big.append((n, type(o).__name__, repr(o)[:90] if not isinstance(o, (list, dict, tuple, set)) else
                        f"first item: {type(next(iter(o))).__name__}" if n else ""))
    big.sort(reverse=True)
    print(f"[gc-debug] before step {step}: gc.get_count()={gc.get_count()} objects per generation {sizes}; "
          f"frozen {gc.get_freeze_count()}; largest gen-0 containers: {big[:5]}", flush=True)
    del young, big


# ------------------------------------------------------------------------------------------------- analysis
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def analyze(per_rank: list[dict], args) -> dict:
    """Merge all ranks: step-time distribution, host stalls (F/B calls far longer than their median), and whether a
    garbage collection on the same rank overlaps each stall."""
    world = len(per_rank)
    steps = [s for s in per_rank[0]["steps"] if s["step"] >= args.warmup and not s["profiled"]]
    ms = [max(r["steps"][s["step"]]["ms"] for r in per_rank) for s in steps]   # the step ends with the slowest rank
    med = statistics.median(ms)
    out = {"n_steps": len(ms), "step_ms": {"median": med, "p90": pct(ms, 0.9), "p99": pct(ms, 0.99),
                                           "max": max(ms), "mean": statistics.fmean(ms)},
           "slow_steps": sum(m > med + args.stall_ms for m in ms), "ranks": []}
    for r in per_rank:
        ev = [e for e in r["events"] if e["step"] >= args.warmup and e["step"] not in r["profiled_steps"]]
        gcs = [e for e in ev if e["kind"] == "gc"]
        calls = [e for e in ev if e["kind"] in ("F", "B")]
        by_kind = {}
        for e in calls:
            by_kind.setdefault(e["kind"], []).append((e["t1"] - e["t0"]) * NS)
        medians = {k: statistics.median(v) for k, v in by_kind.items()}
        stalls = []
        for e in calls:
            d = (e["t1"] - e["t0"]) * NS
            if d > medians[e["kind"]] + args.stall_ms:
                overlap = [g for g in gcs if g["t0"] < e["t1"] and g["t1"] > e["t0"]]
                stalls.append({"step": e["step"], "call": e["name"], "ms": round(d, 2),
                               "gc_overlap": [f"{g['name']} {(g['t1'] - g['t0']) * NS:.1f} ms" for g in overlap]})
        gc_summary = {}
        for g in gcs:
            gc_summary.setdefault(g["name"], []).append((g["t1"] - g["t0"]) * NS)
        out["ranks"].append({
            "rank": r["rank"], "cpus": r["cpus"],
            "call_median_ms": {k: round(v, 3) for k, v in medians.items()},
            "stalls": len(stalls), "stalls_with_gc": sum(bool(s["gc_overlap"]) for s in stalls),
            "stall_examples": stalls[:6],
            "gc": {k: {"count": len(v), "median_ms": round(statistics.median(v), 3), "max_ms": round(max(v), 2),
                       "total_ms": round(sum(v), 1)} for k, v in sorted(gc_summary.items())}})
    out["world"] = world
    # Profiled steps, reported separately: per rank the longest host F/B call and the collections in that step.
    out["profiled"] = []
    for step in per_rank[0]["profiled_steps"]:
        entry = {"step": step, "ms": max(r["steps"][step]["ms"] for r in per_rank), "ranks": []}
        for r in per_rank:
            ev = [e for e in r["events"] if e["step"] == step]
            calls = sorted((e for e in ev if e["kind"] in ("F", "B")), key=lambda e: e["t1"] - e["t0"])
            gcs = [e for e in ev if e["kind"] == "gc"]
            top = calls[-1]
            entry["ranks"].append({
                "rank": r["rank"], "longest_call": top["name"], "longest_ms": round((top["t1"] - top["t0"]) * NS, 2),
                "gc_in_step": [f"{g['name']} {(g['t1'] - g['t0']) * NS:.1f} ms ({g['collected']} freed)" for g in gcs],
                "gc_overlap_longest": any(g["t0"] < top["t1"] and g["t1"] > top["t0"] for g in gcs)})
        out["profiled"].append(entry)
    return out


def report(res: dict, args) -> str:
    s = res["step_ms"]
    lines = [f"STALL PROBE {args.run_id}: tp{args.tp}pp{args.pp}, gc={args.gc}"
             f"{f' (collect every {args.gc_interval} steps)' if args.gc == 'manual' else ''}, "
             f"pin_numa={args.pin_numa}, profiler steps={args.profile_steps or 'none'}",
             f"  {res['n_steps']} measured steps (warmup {args.warmup} and profiled steps excluded): step ms "
             f"median {s['median']:.1f}, p90 {s['p90']:.1f}, p99 {s['p99']:.1f}, max {s['max']:.1f}, "
             f"mean {s['mean']:.1f}; steps > median + {args.stall_ms:g} ms: {res['slow_steps']}"]
    for r in res["ranks"]:
        gcs = ", ".join(f"{k}: {v['count']}x median {v['median_ms']} ms max {v['max_ms']} ms total {v['total_ms']} ms"
                        for k, v in r["gc"].items()) or "none"
        lines.append(f"  rank {r['rank']} (cpus {r['cpus']}): host F/B call median {r['call_median_ms']}; "
                     f"stalls (call > median + {args.stall_ms:g} ms) {r['stalls']}, of which overlap a GC "
                     f"{r['stalls_with_gc']}; GC {gcs}")
        for ex in r["stall_examples"]:
            lines.append(f"      step {ex['step']} {ex['call']} {ex['ms']} ms, GC overlap: {ex['gc_overlap'] or 'none'}")
    for rank, rel in res.get("prof_release", {}).items():
        if rel:
            lines.append(f"  rank {rank} profiler release ({res['args']['release_profiler']}): {rel}")
    for p in res["profiled"]:
        lines.append(f"  profiled step {p['step']} ({res['args']['profile_steps']}): {p['ms']:.1f} ms")
        for r in p["ranks"]:
            lines.append(f"      rank {r['rank']}: longest host call {r['longest_call']} {r['longest_ms']} ms "
                         f"(GC overlaps it: {r['gc_overlap_longest']}); GC in step: {r['gc_in_step'] or 'none'}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    profiled = {}
    for item in filter(None, args.profile_steps.split(",")):
        step, kind = item.split(":")
        profiled[int(step)] = kind
    local_rank = int(os.environ["LOCAL_RANK"])
    cpus = pin_to_gpu_numa(local_rank) if args.pin_numa else "all"

    ps = tg.initialize_distributed(args.tp, args.pp)
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    rank, world = dist.get_rank(), dist.get_world_size()
    if args.pid_dir:
        Path(args.pid_dir).mkdir(parents=True, exist_ok=True)
        if args.allow_ptrace:
            import ctypes
            ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)   # PR_SET_PTRACER, PR_SET_PTRACER_ANY
        (Path(args.pid_dir) / f"pid_rank{rank}.txt").write_text(str(os.getpid()))
    out_dir = Path(args.out) / f"run_{args.run_id}"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    model, _ = tg.model_provider(args, ps)
    tg.canonical_init(model, args.tp, ps.get_tensor_model_parallel_rank())
    count = instrument()
    count["F"] = 0
    step_fn = traced_forward(count)
    fb_func = get_forward_backward_func()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch = tg.make_global_batch(args)
    is_last = ps.is_pipeline_last_stage()

    gc.callbacks.append(REC.gc_callback)
    if args.gc == "manual":
        # Megatron's --manual-gc (megatron/training/training.py:3586-3593): disable threshold-triggered collection,
        # collect once now; we also freeze the setup objects so later collections need not traverse them.
        gc.disable()
        gc.collect()
        gc.freeze()

    prof_release = []
    debug_steps = {int(s) for s in filter(None, args.gc_debug_steps.split(","))}
    steps = []
    for step in range(args.steps):
        REC.step = step
        if rank == 0 and step in debug_steps:
            gc_snapshot(step)
        count.update(F=0, B=0)
        optimizer.zero_grad(set_to_none=True)
        kind = profiled.get(step)
        prof = (profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=kind == "stack",
                        with_stack=kind == "stack") if kind else contextlib.nullcontext())
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter_ns()
        with prof:
            losses = fb_func(forward_step_func=step_fn, data_iterator=iter(batch), model=model,
                             num_microbatches=args.microbatches, seq_length=tg.SEQ,
                             micro_batch_size=args.micro_batch_size, decoder_seq_length=tg.SEQ, forward_only=False)
            optimizer.step()
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        REC.span("step", f"step{step}", t0, t1)
        if kind:   # the same export train_gpt.py does (it allocates many Python objects)
            raw = out_dir / f"{kind}_step{step}_rank{rank}.json"
            prof.export_chrome_trace(str(raw))
            with raw.open("rb") as src, gzip.open(str(raw) + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            raw.unlink()
            # torch.profiler.profile keeps bound methods of itself in self.action_map (torch/profiler/profiler.py
            # __init__), a reference cycle: dropping our reference does not free it, only the cyclic GC can.
            ref = weakref.ref(prof)
            del prof
            info = {"step": step, "alive_after_del": ref() is not None}
            if args.release_profiler == "collect":
                c0 = time.perf_counter_ns()
                info["freed"] = gc.collect()
                info["collect_ms"] = round((time.perf_counter_ns() - c0) * NS, 2)
                info["alive_after_collect"] = ref() is not None
            prof_release.append(info)
        if args.gc == "manual" and args.gc_interval and (step + 1) % args.gc_interval == 0:
            # Same step on every rank, outside the schedule: all ranks pause together, nobody waits on a peer
            # that is collecting. Timed by the gc callback like any other collection.
            gc.collect()
        loss = float(torch.stack([d["lm loss"] for d in losses]).mean()) if is_last and losses else None
        ms_alloc = torch.cuda.memory_stats()
        steps.append({"step": step, "ms": (t1 - t0) * NS, "loss": loss, "profiled": kind,
                      # caching-allocator counters: a new cudaMalloc / cudaFree in steady state would show here
                      "alloc": {k: ms_alloc.get(k, 0) for k in ("num_device_alloc", "num_device_free",
                                                                 "num_alloc_retries", "num_sync_all_streams")}})

    gc.callbacks.remove(REC.gc_callback)
    record = {"rank": rank, "cpus": cpus, "steps": steps, "events": REC.events, "prof_release": prof_release,
              "profiled_steps": sorted(profiled), "args": vars(args)}
    (out_dir / f"rank{rank}.json").write_text(json.dumps(record))
    gathered = [None] * world
    dist.all_gather_object(gathered, record)
    if rank == 0:
        res = analyze(gathered, args)
        res["args"] = vars(args)
        res["prof_release"] = {}
        losses = next(g for g in gathered if g["steps"][0]["loss"] is not None)["steps"]
        res["loss_first_last"] = [losses[0]["loss"], losses[-1]["loss"]]
        res["prof_release"] = {g["rank"]: g["prof_release"] for g in gathered}
        (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
        text = report(res, args)
        (out_dir / "report.txt").write_text(text + "\n")
        print(text, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
