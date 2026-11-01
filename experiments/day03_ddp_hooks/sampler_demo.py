"""Day 03 side exercise - what DistributedSampler hands each rank (CPU only, no process group needed).

Day 02 and train_ddp.py shard a batch by CONTIGUOUS rows: rank r takes rows [r*b, (r+1)*b). A real data loader with
DistributedSampler instead gives rank r the indices r, r+W, r+2W, ... of a (shuffled, padded) index list. Both are
valid data parallelism; they put different samples on each rank, so a parity test must use the same rule in the
reference. Things to see below:
  1. interleaving                       rank r gets every W-th index
  2. padding when N % W != 0            drop_last=False repeats the first indices so every rank gets the same count
                                        (equal shard sizes = "the mean of local means is the global mean", Day 02 §1.3)
  3. drop_last=True                     the tail is dropped instead
  4. set_epoch                          same seed + epoch on every rank -> one shared permutation, then interleaved;
                                        forgetting set_epoch replays the same order every epoch

  python experiments/day03_ddp_hooks/sampler_demo.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DistributedSampler


class _Range(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def show(n: int, world: int, shuffle: bool, drop_last: bool, epoch: int = 0) -> None:
    ds = _Range(n)
    parts = []
    for r in range(world):
        s = DistributedSampler(ds, num_replicas=world, rank=r, shuffle=shuffle, seed=0, drop_last=drop_last)
        s.set_epoch(epoch)
        parts.append(list(s))
    flat = sorted(i for p in parts for i in p)
    dup = len(flat) - len(set(flat))
    miss = n - len(set(flat))
    print(f"N={n:2d} W={world} shuffle={shuffle!s:5s} drop_last={drop_last!s:5s} epoch={epoch}: "
          + " | ".join(f"rank{r} {p}" for r, p in enumerate(parts))
          + f"   (duplicated {dup}, never seen {miss})")


if __name__ == "__main__":
    show(8, 2, shuffle=False, drop_last=False)
    show(10, 3, shuffle=False, drop_last=False)
    show(10, 3, shuffle=False, drop_last=True)
    show(10, 2, shuffle=True, drop_last=False, epoch=0)
    show(10, 2, shuffle=True, drop_last=False, epoch=0)
    show(10, 2, shuffle=True, drop_last=False, epoch=1)
