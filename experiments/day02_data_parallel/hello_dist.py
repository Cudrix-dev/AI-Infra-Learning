"""Day 02 - smallest possible torchrun test. Run this FIRST, before any real code, on every machine.

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/hello_dist.py --device cuda
  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/hello_dist.py --device cpu

Expected with 2 ranks: each rank prints 3.0 after all_reduce (1 + 2), and the process exits 0.
It also probes ReduceOp.AVG on this backend and torch version. Observed 2026-10-21 on torch 2.14.1: AVG worked on
BOTH gloo and NCCL (1.5 on 2 ranks). That is version-dependent; this script shows what YOUR install does.
Add NCCL_DEBUG=INFO to see which transport NCCL chose (P2P, SHM, ...). Seeing "P2P" in that log does NOT prove that
P2P traffic works: on one observed host the P2P channels were created and the first all-reduce then timed out.
If the first all_reduce hangs, the timeout below turns it into an error after --timeout seconds and the exit code is
non-zero; do not go on to long runs until this passes.
"""

import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--timeout", type=int, default=60, help="seconds before a stuck collective raises")
    args = ap.parse_args()
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")

    env = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                                          "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "OMP_NUM_THREADS")}
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(f"LOCAL_RANK {local_rank} but only {torch.cuda.device_count()} visible GPUs")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=args.timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "-"
    print(f"[rank {rank}] pid={os.getpid()} device={device} ({gpu}) backend={dist.get_backend()} env={env}", flush=True)

    t = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(t)  # default op is SUM, result written in place on every rank
    expected = world * (world + 1) / 2
    ok = t.item() == expected
    print(f"[rank {rank}] after all_reduce: {t.item()} (expected {expected}) {'ok' if ok else 'MISMATCH'}", flush=True)

    # Why sync_grads still does SUM and then divides: explicit normalisation that does not depend on which
    # backend/version supports ReduceOp.AVG. Does this install accept AVG?
    u = torch.tensor([float(rank + 1)], device=device)
    try:
        dist.all_reduce(u, op=dist.ReduceOp.AVG)
        print(f"[rank {rank}] ReduceOp.AVG worked on {backend}: {u.item()} (expected {(world + 1) / 2})", flush=True)
    except Exception as e:  # noqa: BLE001 - we want to see whatever it is
        print(f"[rank {rank}] ReduceOp.AVG on {backend} -> {type(e).__name__}: {str(e)[:150]}", flush=True)

    if device.type == "cuda" and rank == 0:
        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}", flush=True)

    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
