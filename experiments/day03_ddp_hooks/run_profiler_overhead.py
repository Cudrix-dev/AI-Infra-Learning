"""Does the profiler itself create the rank offset seen in profiled Graph + bucket-view steps?

Same configuration as the bucket-view study (medium, global batch 16, optimizer Graph, --grad-as-view; MiniDDP with
rebuilt 25 MiB buckets and a 1 MiB first bucket, PyTorch DDP defaults), run three ways:
  none      no profiler
  cuda      torch.profiler with ProfilerActivity.CUDA only (CUPTI kernels and runtime calls, no CPU-op recording)
  cpu_cuda  the CPU+CUDA profiler used for the published traces
All three use the SAME metric: each rank's perf_counter core-start/core-end boundaries (core end follows a device
synchronize), compared across ranks on the same step indices: the steps the profiler records in the profiled runs.
perf_counter is CLOCK_MONOTONIC, shared by both processes on this host.

  python experiments/day03_ddp_hooks/run_profiler_overhead.py --run-id NAME
Results: outputs/day03_profiler_overhead/NAME/ (summary.json + per-run folders; no tensors saved).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEPS, WARMUP, PROFILE_STEPS = 18, 5, 6
FIRST = WARMUP + 1  # train_ddp records steps [warmup+1, warmup+1+profile_steps) when profiling
WINDOW = list(range(FIRST, FIRST + PROFILE_STEPS))
CONFIGS = {"none": [], "cuda": ["--profile", "--profile-activities", "cuda"],
           "cpu_cuda": ["--profile", "--profile-activities", "cpu,cuda"]}


def run(out, tag, impl, extra, env):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           str(HERE / "train_ddp.py"), "--impl", impl, "--optimizer-mode", "graph", "--grad-as-view",
           "--preset", "medium", "--steps", str(STEPS), "--warmup", str(WARMUP), "--global-batch", "16",
           "--profile-steps", str(PROFILE_STEPS), "--output-root", str(out), "--tag", tag, "--condition", "unknown",
           "--check-every", str(STEPS), "--log-every", str(STEPS), "--no-tensors", *extra]
    if impl == "mini_ddp":
        cmd += ["--rebuild", "--bucket-mb", "25", "--first-bucket-mb", "1"]
    print(f"\nSTART {tag}", flush=True)
    with open(out / f"{tag}.log", "w") as log:
        if subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode:
            raise SystemExit(f"{tag} failed: see {out / (tag + '.log')}")
    b0, b1 = (json.loads((out / tag / f"boundaries_rank{r}.json").read_text()) for r in (0, 1))
    timing = json.loads((out / tag / "metrics.json").read_text())["timing"]
    rows = []
    for u, v in zip(b0, b1):
        if u["step"] in WINDOW:
            rows.append({"step": u["step"], "start_ms": (v["core_start"] - u["core_start"]) * 1e3,
                         "end_ms": (v["core_end"] - u["core_end"]) * 1e3,
                         "core_ms_rank0": (u["core_end"] - u["core_start"]) * 1e3,
                         "core_ms_rank1": (v["core_end"] - v["core_start"]) * 1e3})
    return {"tag": tag, "core_step_ms_all_timed": timing["core_step_ms"], "window": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    out = ROOT / "outputs" / "day03_profiler_overhead" / a.run_id
    if out.exists():
        raise SystemExit(f"{out} exists: use a new --run-id")
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    for k in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_ALGO"):
        env.pop(k, None)
    names = list(CONFIGS)
    runs = []
    for rep in range(a.repeats):
        for impl in (["mini_ddp", "torch_ddp"] if rep % 2 == 0 else ["torch_ddp", "mini_ddp"]):
            for cfg in names[rep % 3:] + names[:rep % 3]:  # rotate order against drift
                r = run(out, f"r{rep}_{impl}_{cfg}", impl, CONFIGS[cfg], env)
                runs.append({"repeat": rep, "impl": impl, "config": cfg, **r})
    summary = []
    print(f"\nsteps {WINDOW} of each run; |rank1 - rank0| over {a.repeats} repeats x {PROFILE_STEPS} steps")
    print(f"{'impl':10s} {'config':9s} {'start median':>12s} {'start max':>10s} {'end median':>11s} {'end max':>8s} "
          f"{'core ms (window, max rank) median':>34s}")
    for impl in ("mini_ddp", "torch_ddp"):
        for cfg in names:
            w = [x for r in runs if r["impl"] == impl and r["config"] == cfg for x in r["window"]]
            st, en = [abs(x["start_ms"]) for x in w], [abs(x["end_ms"]) for x in w]
            core = [max(x["core_ms_rank0"], x["core_ms_rank1"]) for x in w]
            row = {"impl": impl, "config": cfg, "n_steps": len(w),
                   "abs_start_offset_ms": {"median": statistics.median(st), "max": max(st)},
                   "abs_end_offset_ms": {"median": statistics.median(en), "max": max(en)},
                   "core_ms_window_slowest_rank_median": statistics.median(core)}
            summary.append(row)
            print(f"{impl:10s} {cfg:9s} {row['abs_start_offset_ms']['median']:12.3f} {max(st):10.3f} "
                  f"{row['abs_end_offset_ms']['median']:11.3f} {max(en):8.3f} {row['core_ms_window_slowest_rank_median']:34.2f}")
    (out / "summary.json").write_text(json.dumps({"window_steps": WINDOW, "summary": summary, "runs": runs}, indent=2))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
