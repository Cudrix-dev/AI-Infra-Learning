"""Day 02 - scaling table: 1, 2, 4, 8 GPUs, weak OR strong, for one sync variant, with repeats.

weak   scaling: per-GPU batch fixed, global batch grows with GPU count  -> ideal: tokens/s grows N x
strong scaling: global batch fixed, per-GPU batch shrinks               -> ideal: step time shrinks N x
Report them separately. Each row is a separate launch of ddp_manual.py.

Throughput here is the CORE scope (forward + backward + sync + optimizer, slowest rank per step, median over timed
steps; see ddp_manual.py). The baseline for speedup/efficiency is the FIRST --nprocs entry, measured in the SAME
repeat, on the first visible GPU. On a host with unequal GPUs or links that baseline is host- and mode-specific:
never compare efficiencies across hosts as if they shared one baseline.

  python experiments/day02_data_parallel/run_scaling.py --mode weak   --preset medium --per-gpu-batch 16 --nprocs 1 2
  python experiments/day02_data_parallel/run_scaling.py --mode strong --preset medium --global-batch 32 --nprocs 1 2 4 8 --sync flat --repeats 3 --condition shared

Which GPUs are used: the first N visible ones. Set CUDA_VISIBLE_DEVICES first (gpu_probe.py --want 8 helps on a shared host).
With --repeats R the whole table is measured R times (outer loop). The summary gives the median efficiency and the
min-max range over repeats. A min-max range of 3 repeats is a spread, not a confidence interval. On a shared host
(--condition shared) treat every number as provisional.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch_utils import DDP, OUT, make_env, run, smoke_test, torchrun_cmd  # noqa: E402


def fmt_range(vals: list[float], spec: str) -> str:
    if len(vals) == 1:
        return format(vals[0], spec)
    return f"{format(statistics.median(vals), spec)} [{format(min(vals), spec)}-{format(max(vals), spec)}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weak", "strong"], required=True)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--nprocs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--per-gpu-batch", type=int, default=16)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sync", choices=["per_param", "flat"], default="flat")
    ap.add_argument("--allreduce", choices=["builtin", "ring"], default="builtin")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for every run, single-process included")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    ap.add_argument("--run-id", default=None, help="suffix for all tags (default: timestamp)")
    ap.add_argument("--label", default="")
    ap.add_argument("--no-smoke", action="store_true", help="skip the hello_dist.py smoke test (not recommended)")
    args = ap.parse_args()

    if any(n < 1 for n in args.nprocs):
        raise SystemExit("--nprocs entries must be >= 1")
    if len(set(args.nprocs)) != len(args.nprocs):
        raise SystemExit("--nprocs entries must be distinct")
    if min(args.steps, args.repeats, args.omp_threads, args.per_gpu_batch, args.global_batch) < 1:
        raise SystemExit("--steps, --repeats, --omp-threads and the batch sizes must be >= 1")
    if args.mode == "strong":
        bad = [n for n in args.nprocs if args.global_batch % n]
        if bad:
            raise SystemExit(f"strong scaling: --global-batch {args.global_batch} is not divisible by {bad}")
    rid = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    batch_flag = (["--per-gpu-batch", str(args.per_gpu_batch)] if args.mode == "weak"
                  else ["--global-batch", str(args.global_batch)])
    env = make_env(args.omp_threads)

    if not args.no_smoke:
        for n in sorted(x for x in args.nprocs if x > 1):
            smoke_test(n, args.device, args.omp_threads)

    per_repeat: list[list[dict]] = []
    meta0 = None
    for rep in range(args.repeats):
        rows = []
        for n in args.nprocs:
            tag = f"scale_{args.mode}_{args.sync}_{args.allreduce}{args.label}_n{n}_r{rep}_{rid}"
            base = ["--preset", args.preset, "--steps", str(args.steps), "--device", args.device, "--tag", tag,
                    "--sync", args.sync, "--allreduce", args.allreduce, "--check-every", "1000",
                    "--condition", args.condition, *batch_flag]
            cmd = [sys.executable, str(DDP), *base] if n == 1 else torchrun_cmd(n, DDP, base)
            print(f"\n>> repeat {rep + 1}/{args.repeats}, n={n}:", " ".join(cmd), flush=True)
            if run(cmd, env)[0] != 0:
                raise SystemExit(f"run failed (n={n}, repeat {rep + 1}); not continuing")
            m = json.loads((OUT / tag / "metrics.json").read_text())
            if m["meta"]["world"] != n:
                raise SystemExit(f"{tag}: recorded world {m['meta']['world']} != requested {n}")
            rows.append(m["timing"])
            meta0 = meta0 or m["meta"]
        per_repeat.append(rows)

    print(f"\n{args.mode} scaling, preset={args.preset}, sync={args.sync}, allreduce={args.allreduce}, "
          f"steps={args.steps} (warmup excluded, see metrics.json), repeats={args.repeats}, condition={args.condition}, "
          f"OMP_NUM_THREADS={args.omp_threads}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, run-id={rid}")
    if args.condition != "exclusive":
        print("NOTE: not an exclusive host -> numbers are PROVISIONAL (other jobs may share GPUs, links or CPU).")
    head = (f"{'GPUs':>4s} {'global B':>8s} {'fwd+bwd ms':>11s} {'sync ms':>9s} {'sync/core':>9s} {'core ms':>9s} "
            f"{'tokens/s (core)':>16s} {'speedup':>10s} {'efficiency':>22s} {'peak MiB r0':>11s}")
    print(head)
    for i, n in enumerate(args.nprocs):
        med = lambda key: statistics.median(rep[i][key] for rep in per_repeat)  # noqa: E731
        speedups, effs = [], []
        for rep in per_repeat:  # the baseline is the first row of the SAME repeat
            s = rep[i]["core_tokens_per_s"] / rep[0]["core_tokens_per_s"]
            speedups.append(s)
            effs.append(s / (n / args.nprocs[0]))
        peaks = [rep[i].get("peak_alloc_mib_rank0") for rep in per_repeat]
        peak = "-" if any(p is None for p in peaks) else f"{statistics.median(peaks):.0f}"
        print(f"{n:4d} {per_repeat[0][i]['global_batch']:8d} {med('fwd_bwd_ms'):11.2f} {med('comm_ms'):9.2f} "
              f"{100 * med('comm_over_core'):8.1f}% {med('core_step_ms'):9.2f} {med('core_tokens_per_s'):16,.0f} "
              f"{fmt_range(speedups, '.2f'):>10s} {fmt_range(effs, '.2f'):>22s} {peak:>11s}")
    print("columns: medians over repeats of per-run medians (slowest rank per step). speedup/efficiency: median [min-max] over repeats.")
    print("'sync ms' is the gradient-synchronization phase; the first row (1 process) has none. sync/core = sync ms / core step ms.")
    print("efficiency = speedup / (N / N_first). For strong scaling the ideal is step time / N; for weak, tokens/s x N.")
    print("This is the CORE scope; ddp_manual.py also records the 'loop' scope (includes data generation and logging) in metrics.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
