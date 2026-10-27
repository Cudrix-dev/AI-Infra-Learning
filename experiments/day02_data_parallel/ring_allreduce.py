"""Day 02 (stretch) - ring all-reduce built from point-to-point send/recv only.

Idea: split the tensor into W chunks (W = world size) and pass chunks around a ring
(rank r sends to r+1 and receives from r-1) in two phases.

  Phase 1, reduce-scatter (W-1 steps): rank r sends chunk (r - s) and receives chunk (r - s - 1),
      ADDING it to its own copy.  After W-1 steps, rank r holds the FULLY SUMMED chunk (r + 1) % W.
  Phase 2, all-gather (W-1 steps): rank r sends chunk (r + 1 - s) and receives chunk (r - s),
      OVERWRITING its copy.  After W-1 steps every rank holds every fully summed chunk.

With W equal chunks each rank SENDS 2 * (W-1)/W * S bytes in total (S = tensor size; received volume is the same
number again, so sent+received is twice that), almost independent of W. With uneven chunks an individual rank's sent
bytes can differ from this formula; the average over ranks still matches it.

Works with NCCL (CUDA tensors) and gloo (CPU tensors).  Self-test, from the repo root:
  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/ring_allreduce.py --device cuda
  python -m torch.distributed.run --standalone --nproc_per_node 4 experiments/day02_data_parallel/ring_allreduce.py --device cuda
  python -m torch.distributed.run --standalone --nproc_per_node 3 experiments/day02_data_parallel/ring_allreduce.py --device cpu
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def ring_all_reduce(t: torch.Tensor) -> int:
    """In-place SUM all-reduce over the default group using only point-to-point operations.

    Every rank must call this with the same shape and dtype (the "same collectives, same order,
    same shapes on every rank" rule).  Returns the number of bytes THIS rank sent.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    if world == 1:
        return 0
    assert t.is_contiguous(), "ring_all_reduce needs a contiguous tensor"

    flat = t.view(-1)  # 1-D view of the same memory
    # `world` chunks whose sizes differ by at most 1. tensor_split returns VIEWS, so writing into
    # chunks[i] writes into `t`.  If numel < world, some chunks are empty.
    chunks = torch.tensor_split(flat, world)
    right, left = (rank + 1) % world, (rank - 1) % world
    sent_bytes = 0

    def exchange(send_chunk: torch.Tensor, recv_chunk: torch.Tensor) -> None:
        """Send to the right neighbour AND receive from the left neighbour in ONE batch.

        Why a batch: with NCCL, a send only completes when the peer has posted the matching receive.
        If every rank issued send() and then recv() separately, rank r would wait for r+1 to start
        receiving while r+1 waits for r+2 ... a deadlock around the ring.  batch_isend_irecv groups
        both operations (ncclGroupStart/End underneath) so they make progress together.
        (With gloo it simply issues isend + irecv, which is equally safe.)
        Empty chunks are skipped; sender and receiver agree on sizes, so they skip together.
        """
        nonlocal sent_bytes
        ops = []
        if send_chunk.numel() > 0:
            ops.append(dist.P2POp(dist.isend, send_chunk, right))
            sent_bytes += send_chunk.numel() * send_chunk.element_size()
        if recv_chunk.numel() > 0:
            ops.append(dist.P2POp(dist.irecv, recv_chunk, left))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()  # NCCL: makes the current CUDA stream wait for the transfer (does not block the CPU)

    # ---- Phase 1: reduce-scatter. Send my partial sum of chunk (rank - s), add what arrives.
    for s in range(world - 1):
        send_idx = (rank - s) % world
        recv_idx = (rank - s - 1) % world
        # Receive into a scratch buffer: we must not overwrite chunks[recv_idx] before adding to it.
        # (chunks[send_idx] is untouched while its send is in flight, so the send buffer is safe.)
        scratch = torch.empty_like(chunks[recv_idx])
        exchange(chunks[send_idx], scratch)
        chunks[recv_idx].add_(scratch)

    # Now rank r owns the fully reduced chunk (r + 1) % world.

    # ---- Phase 2: all-gather. Pass the finished chunks around; receive straight into place.
    for s in range(world - 1):
        send_idx = (rank + 1 - s) % world
        recv_idx = (rank - s) % world
        exchange(chunks[send_idx], chunks[recv_idx])

    return sent_bytes


# ----------------------------------------------------------------------------- self-test
def _selftest(device_kind: str) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_kind == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(f"LOCAL_RANK {local_rank} but only {torch.cuda.device_count()} visible GPUs "
                             "(check CUDA_VISIBLE_DEVICES / --nproc_per_node)")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    all_ok = True
    if rank == 0:
        print(f"ring_all_reduce self-test: world size {world}, backend {backend}, device {device_kind}")
        print(f"{'numel':>10s} {'int64 exact':>12s} {'f32 max|diff|':>14s} {'avg sent B':>11s} {'2(W-1)/W*S':>11s}")

    for n in (1, world, 7, 1000, 100_003, 4_000_000):  # n=1 < world exercises the empty-chunk path
        # (a) Exact test with integers: addition is associative, so the ring must match bit for bit.
        a = ((torch.arange(n, dtype=torch.int64) + 1) * (rank + 1)).to(device)
        want = a.clone()
        dist.all_reduce(want)  # reference: the backend's own all_reduce, SUM
        sent = ring_all_reduce(a)
        exact = torch.equal(a, want)
        st = torch.tensor([float(sent)], dtype=torch.float64, device=device)
        dist.all_reduce(st)  # average sent volume over ranks (uneven chunks make single ranks differ)
        avg_sent = st.item() / world

        # (b) float32: the SUMMATION ORDER differs from the backend's, so expect tiny rounding differences.
        g = torch.Generator().manual_seed(1234 + rank)
        f = torch.randn(n, generator=g).to(device)
        want_f = f.clone()
        dist.all_reduce(want_f)
        ring_all_reduce(f)
        diff = (f - want_f).abs().max().item()
        close = torch.allclose(f, want_f, rtol=1e-5, atol=1e-5)

        # (c) all ranks must hold identical values after the ring (not just "close to the backend's answer").
        ref = f.clone()
        dist.broadcast(ref, src=0)
        same_bits = torch.equal(f, ref)

        ok = torch.tensor([int(exact and close and same_bits)], device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)  # all ranks agree on the verdict before anyone prints
        all_ok &= bool(ok.item())
        if rank == 0:
            theory = 2 * (world - 1) / world * n * 8  # int64 = 8 bytes per element
            print(f"{n:10d} {str(exact):>12s} {diff:14.3e} {avg_sent:11.0f} {theory:11.0f}  {'ok' if ok.item() else 'FAIL'}")

    if rank == 0:
        print("RING SELF-TEST PASSED" if all_ok else "RING SELF-TEST FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    _selftest(ap.parse_args().device)
