"""Day 02 - paper-and-pencil ring all-reduce, simulated in pure Python (no torch needed).

Each rank starts with W chunks, and chunk c on rank r contains only rank r's data.
We track HOW MANY TIMES each rank's data has been added into every chunk (a count per rank), using the same index
formulas as ring_allreduce.py.  A plain set of ranks would hide a bug that adds someone's data twice; counts do not.
The final assertion requires every count to be exactly 1.  Note this still tracks coverage, not values: the real
numeric self-test (ring_allreduce.py under torchrun) stays necessary.

  python experiments/day02_data_parallel/simulate_ring.py          # W = 4
  python experiments/day02_data_parallel/simulate_ring.py 3        # W = 3
"""

from __future__ import annotations

import sys


def fmt(counts: list[int]) -> str:
    return "{" + ",".join(str(r) if c == 1 else f"{r}x{c}" for r, c in enumerate(counts) if c) + "}"


def show(state, title: str) -> None:
    W = len(state)
    width = 3 * W + 3
    print(f"\n{title}")
    print("        " + "".join(f"{'chunk ' + str(c):<{width}s}" for c in range(W)))
    for r in range(W):
        print(f"rank {r}  " + "".join(f"{fmt(state[r][c]):<{width}s}" for c in range(W)))


def main(W: int) -> None:
    if W < 1:
        raise SystemExit("world size must be >= 1")
    # state[r][c][k] = how many times rank k's data is already summed into rank r's copy of chunk c
    state = [[[1 if k == r else 0 for k in range(W)] for _ in range(W)] for r in range(W)]
    show(state, "start: every rank holds only its own data in every chunk")

    print(f"\n=== Phase 1: reduce-scatter, W-1 = {W - 1} steps (s = 0, 1, ...) ===")
    print("  rank r sends chunk (r-s) to rank r+1 (the row below; the last rank sends to rank 0)")
    print("  rank r receives chunk (r-s-1) from rank r-1 (the row above; rank 0 receives from the last rank) and ADDS it")
    for s in range(W - 1):
        sends = {r: (r - s) % W for r in range(W)}  # chunk index rank r sends to rank r+1
        payload = {r: list(state[r][sends[r]]) for r in range(W)}  # snapshot: all sends happen "at once"
        for r in range(W):
            prev = (r - 1) % W  # the rank that sends to r
            recv_idx = (r - s - 1) % W
            assert sends[prev] == recv_idx, "sender and receiver must agree on the chunk index"
            state[r][recv_idx] = [a + b for a, b in zip(state[r][recv_idx], payload[prev])]
        moves = ", ".join(f"r{r}->r{(r + 1) % W}: chunk {sends[r]}" for r in range(W))
        show(state, f"after reduce-scatter step {s}   ({moves})")

    full = [1] * W
    owner = {c: [r for r in range(W) if state[r][c] == full] for c in range(W)}
    print("\nfully reduced chunk owners:", owner)
    for r in range(W):
        assert owner[(r + 1) % W] == [r], "rank r must own chunk (r+1) % W, with every contribution exactly once"

    print(f"\n=== Phase 2: all-gather, W-1 = {W - 1} steps (s = 0, 1, ...) ===")
    print("  rank r sends chunk (r+1-s) to rank r+1 (the row below; the last rank sends to rank 0)")
    print("  rank r receives chunk (r-s) from rank r-1 (the row above; rank 0 receives from the last rank) and OVERWRITES its copy")
    for s in range(W - 1):
        sends = {r: (r + 1 - s) % W for r in range(W)}
        payload = {r: list(state[r][sends[r]]) for r in range(W)}
        for r in range(W):
            prev = (r - 1) % W
            recv_idx = (r - s) % W
            assert sends[prev] == recv_idx
            state[r][recv_idx] = payload[prev]
        moves = ", ".join(f"r{r}->r{(r + 1) % W}: chunk {sends[r]}" for r in range(W))
        show(state, f"after all-gather step {s}   ({moves})")

    assert all(state[r][c] == full for r in range(W) for c in range(W)), "some chunk misses or double-counts a rank"
    steps = 2 * (W - 1)
    print(f"\nDONE. Every rank holds every chunk with each of the {W} ranks' data counted exactly once.")
    print(f"{steps} communication steps; each rank SENT {steps} chunks = {steps}/{W} = {steps / W:.2f} x the tensor size S")
    print("  (theory for equal chunks: 2(W-1)/W * S sent per rank, approaching 2*S; received volume is the same again)")
    print(f"naive 'everyone sends to rank 0, then rank 0 broadcasts': rank 0 receives {W - 1} x S and sends {W - 1} x S")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
