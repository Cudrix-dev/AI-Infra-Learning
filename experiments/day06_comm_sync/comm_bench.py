"""Day 06 follow-up - communication only: the three collectives of flat_ar / zero1 on preallocated FP32 buffers.

No model, no autograd, no AdamW. One flat buffer of the medium model's size (27,443,200 FP32 elements = 104.6875 MiB)
and the SAME in-place layouts as FlatZeroAdamW:

  all_reduce      all_reduce(buf)                         every element: rank0 + rank1
  reduce_scatter  reduce_scatter_single(buf[lo:hi], buf)  in place: output = this rank's slice of the input
  all_gather      all_gather_single(buf, buf[lo:hi])      in place: input = this rank's slice of the output
  rs_ag_pair      reduce_scatter then all_gather, back to back. A COMMUNICATION-ONLY pair: no division, no optimizer
                  between them, so it is not a ZeRO-1 optimizer step

Per iteration and operation: reset the inputs (outside the timed range), synchronize, Gloo CPU barrier (arrival
alignment), then time the operation with host perf_counter around (dispatch -> return -> synchronize) and with CUDA
events recorded on the current stream before the call and after it returns. The event interval includes the NCCL
stream's dependency on the current stream and any residual peer waiting; it is not "pure transfer time". After
timing, the output is validated exactly (inputs are small integers, so the FP32 sums are exact). Operation order is
rotated every iteration.

Writes bench_rank{r}.jsonl (one row per iteration and operation) and meta_rank{r}.json.

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day06_comm_sync/comm_bench.py \
      --out outputs/day06_comm_sync/try/comm/p2p --iters 20 --warmup 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

reduce_scatter = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
all_gather = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
OPS = ["all_reduce", "reduce_scatter", "all_gather", "rs_ag_pair"]
ENV_KEYS = ("NCCL_", "TORCH_NCCL_", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--numel", type=int, default=27_443_200)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rank, world, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if args.numel % world:
        raise SystemExit("numel must be divisible by the world size")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
    ctrl = dist.new_group(backend="gloo")
    S = args.numel // world
    lo, hi = rank * S, (rank + 1) * S
    buf = torch.empty(args.numel, dtype=torch.float32, device=device)
    expect_sum = float(sum(r + 1 for r in range(world)))  # rank r contributes r + 1 everywhere

    def reset(op: str) -> None:
        if op == "all_gather":
            buf.fill_(-1.0)  # stale values that must be overwritten
            buf[lo:hi].fill_(rank + 1)
        else:
            buf.fill_(rank + 1)

    def run(op: str) -> None:
        if op == "all_reduce":
            dist.all_reduce(buf)
        elif op == "reduce_scatter":
            reduce_scatter(buf[lo:hi], buf)
        elif op == "all_gather":
            all_gather(buf, buf[lo:hi])
        else:
            reduce_scatter(buf[lo:hi], buf)
            all_gather(buf, buf[lo:hi])

    def valid(op: str) -> bool:
        if op == "reduce_scatter":
            return bool((buf[lo:hi] == expect_sum).all())
        if op == "all_gather":
            return all(bool((buf[r * S : (r + 1) * S] == r + 1).all()) for r in range(world))
        return bool((buf == expect_sum).all())  # all_reduce, and rs_ag_pair (every slice reduced, then gathered)

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rows = []
    for it in range(args.warmup + args.iters):
        order = OPS[it % len(OPS):] + OPS[: it % len(OPS)]
        for op in order:
            reset(op)
            torch.cuda.synchronize(device)
            b0 = time.perf_counter_ns()
            dist.barrier(group=ctrl)
            b1 = time.perf_counter_ns()
            d0 = time.perf_counter_ns()
            e0.record()
            run(op)
            d1 = time.perf_counter_ns()
            e1.record()
            torch.cuda.synchronize(device)
            d2 = time.perf_counter_ns()
            ok = valid(op)
            rows.append({"iter": it, "warmup": it < args.warmup, "op": op, "rank": rank, "bar_in": b0, "bar_out": b1,
                         "disp0": d0, "disp1": d1, "done": d2, "event_ms": e0.elapsed_time(e1), "valid": ok})
            if not ok:
                raise RuntimeError(f"rank {rank}: {op} produced wrong values at iteration {it}")
    with (out / f"bench_rank{rank}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / f"meta_rank{rank}.json").write_text(json.dumps({
        "rank": rank, "args": vars(args), "numel": args.numel, "mib": args.numel * 4 / 2**20,
        "affinity": sorted(os.sched_getaffinity(0)),
        "env": {k: v for k, v in os.environ.items() if k.startswith(ENV_KEYS)},
        "torch": torch.__version__, "nccl_runtime": ".".join(map(str, torch.cuda.nccl.version())),
        "all_valid": all(r["valid"] for r in rows)}, indent=2))
    dist.barrier(group=ctrl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
