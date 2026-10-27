"""Day 02 - all-reduce bandwidth vs message size: where is the latency-bound / bandwidth-bound crossover?

For each message size it times NCCL's all_reduce (and optionally our Python ring) and reports

  time        MAX over ranks of each rank's median time per call (a max of medians; this is NOT the
              per-step-max-then-median definition used in ddp_manual.py, and it is a different quantity)
  algbw       S / time                      (what the application sees)
  busbw       algbw * 2(W-1)/W              (comparable to the link speed, independent of W; the metric used by nccl-tests)

Read the table like this: small messages -> time stays roughly flat (latency-bound: 2(W-1) steps of alpha);
large messages -> busbw plateaus at what the interconnect can carry (bandwidth-bound).
Your gradient payload (~105 MiB for the medium preset) sits in the plateau; your 30 tiny tensors sit far left.

From the repo root:
  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/bench_allreduce.py --tag bench_2gpu
  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/bench_allreduce.py --ring --tag bench_2gpu_ring
Experiments (set before the command; each is a hypothesis to test, not a known result):
  NCCL_DEBUG=INFO          which transport did NCCL pick (P2P / SHM / NET) and what ring did it build?
  NCCL_P2P_DISABLE=1       disable the P2P transport (NCCL then uses another one, e.g. shared host memory). On one observed
                           host this was a needed workaround; do not assume it for other hosts, and use it only for that run.
  NCCL_ALGO=Ring | Tree    force the algorithm (matters with 4-8 GPUs)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ring_allreduce import ring_all_reduce  # noqa: E402

OUT = HERE.parents[1] / "outputs" / "day02_data_parallel"


def timed(fn, device, iters: int, warmup: int) -> float:
    """Median seconds per call, synchronizing around every call so the number is the real completion time."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index]) if device.type == "cuda" else dist.barrier()  # start together
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-mib", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ring", action="store_true", help="also time our Python ring_all_reduce")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown",
                    help="record whether the host was exclusive; shared-host numbers are provisional")
    args = ap.parse_args()
    if args.iters < 1 or args.warmup < 0 or args.max_mib < 1:
        raise SystemExit("--iters and --max-mib must be >= 1, --warmup >= 0")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()

    sizes = []
    b = 1024  # 1 KiB
    while b <= args.max_mib * 2**20:
        sizes.append(b)
        b *= 4
    if args.max_mib * 2**20 not in sizes:
        sizes.append(args.max_mib * 2**20)

    impls = [("builtin", lambda t: dist.all_reduce(t))]
    if args.ring:
        impls.append(("ring", ring_all_reduce))

    results = []
    if rank == 0:
        print(f"world={world} backend={backend} device={device} condition={args.condition} NCCL_ALGO={os.environ.get('NCCL_ALGO')} "
              f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"{'impl':8s} {'size':>10s} {'time':>10s} {'algbw GB/s':>11s} {'busbw GB/s':>11s}")
    for name, fn in impls:
        for nbytes in sizes:
            t = torch.ones(nbytes // 4, dtype=torch.float32, device=device)
            sec = timed(lambda: fn(t), device, args.iters, args.warmup)
            m = torch.tensor([sec], device=device)
            dist.all_reduce(m, op=dist.ReduceOp.MAX)  # slowest rank wins
            sec = m.item()
            algbw = nbytes / sec / 1e9
            busbw = algbw * 2 * (world - 1) / world
            results.append({"impl": name, "bytes": nbytes, "seconds": sec, "algbw_GBps": algbw, "busbw_GBps": busbw})
            if rank == 0:
                label = f"{nbytes / 2**20:.0f} MiB" if nbytes >= 2**20 else f"{nbytes / 1024:.0f} KiB"
                unit = f"{sec * 1e3:.3f} ms" if sec >= 1e-3 else f"{sec * 1e6:.1f} us"
                print(f"{name:8s} {label:>10s} {unit:>10s} {algbw:11.2f} {busbw:11.2f}", flush=True)
            del t

    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.tag}.json").write_text(json.dumps({"world": world, "backend": backend, "results": results,
                                                          "time_definition": "max over ranks of per-rank median seconds per call",
                                                          "condition": args.condition, "torch": torch.__version__,
                                                          "iters": args.iters, "warmup": args.warmup,
                                                          "env": {k: v for k, v in os.environ.items()
                                                                  if k.startswith(("NCCL_", "TORCH_NCCL_", "OMP_", "CUDA_VISIBLE"))}}, indent=2))
        print(f"wrote {OUT / (args.tag + '.json')}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
