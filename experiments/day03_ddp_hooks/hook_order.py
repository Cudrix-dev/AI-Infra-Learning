"""Day 03 - when does each gradient become ready, and what does that do to buckets? (one process, no NCCL)

Part 1  Tensor.register_hook vs register_post_accumulate_grad_hook on a parameter used TWICE: which one sees the
        complete gradient, and when is p.grad written?
Part 2  For the Day 01 model: registration order vs the order in which backward finishes each parameter's gradient
        (recorded with post-accumulate hooks, exactly what MiniDDP and DDP's rebuild record).
Part 3  Bucket layouts for a cap: built in reverse-registration order (DDP's guess before the first backward) vs in
        the recorded ready order (after the rebuild). For each bucket: when its LAST gradient arrives, and how long
        in-order launching makes it wait for an earlier bucket.

  python experiments/day03_ddp_hooks/hook_order.py                       # tiny, cap 0.25 MiB, CPU is fine
  python experiments/day03_ddp_hooks/hook_order.py --preset medium --bucket-mb 25 --first-bucket-mb 1 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day01_single_gpu"))
sys.path.insert(0, str(HERE))
from train import PRESETS, Config, attention_ctx, build, get_batch, setup_reproducibility  # noqa: E402

from mini_ddp import MiB, assign_buckets  # noqa: E402


def part1() -> None:
    print("== Part 1: tensor hook vs post-accumulate hook (w is used twice: y = w*x1 + w*x2) ==")
    w = torch.nn.Parameter(torch.tensor(1.0))
    w.register_hook(lambda g: print(f"  register_hook                  : incoming grad {g.item():.1f}, "
                                    f"w.grad = {None if w.grad is None else w.grad.item()}"))
    w.register_post_accumulate_grad_hook(lambda p: print(f"  register_post_accumulate_grad_hook: w.grad = {p.grad.item():.1f}"))
    for it in range(2):  # two backwards WITHOUT zero_grad = gradient accumulation (Day 4)
        print(f"  backward #{it + 1}")
        (w * 2.0 + w * 3.0).backward()
    print("  -> autograd sums the two uses (2 + 3) before the parameter's AccumulateGrad node, so both hooks fire once\n"
          "     per backward. Only the post-accumulate hook sees p.grad AFTER accumulation (5, then 10): that is the\n"
          "     moment the gradient is final and may be communicated.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="tiny")
    ap.add_argument("--bucket-mb", type=float, default=0.25)
    ap.add_argument("--first-bucket-mb", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    part1()

    cfg = Config(**PRESETS[args.preset])
    cfg.batch_size = args.batch
    setup_reproducibility(cfg)
    device = torch.device(args.device)
    model, _, _ = build(cfg, device)
    named = list(model.named_parameters())
    sizes = [p.numel() * p.element_size() for _, p in named]

    order: list[int] = []
    for i, (_, p) in enumerate(named):
        p.register_post_accumulate_grad_hook(lambda _p, i=i: order.append(i))
    x, y = get_batch(cfg, 0, device)
    with attention_ctx(cfg):
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size), y.reshape(-1))
    loss.backward()
    pos = {i: k for k, i in enumerate(order)}  # param index -> position in the ready sequence

    print(f"== Part 2: {args.preset}: {len(named)} parameters, {sum(sizes) / MiB:.2f} MiB of fp32 gradients ==")
    print(f"{'reg#':>4s} {'ready#':>6s} {'MiB':>8s}  name")
    for i, (n, _) in enumerate(named):
        print(f"{i:4d} {pos[i]:6d} {sizes[i] / MiB:8.3f}  {n}")
    rev = list(reversed(range(len(named))))
    mism = sum(a != b for a, b in zip(rev, order))
    print(f"ready order == reverse registration order? {order == rev}  ({mism}/{len(named)} positions differ)\n")

    caps = [int(args.bucket_mb * MiB)]
    if args.first_bucket_mb is not None:
        caps = [int(args.first_bucket_mb * MiB), caps[0]]
    for title, ord_ in (("reverse registration (initial guess)", rev), ("recorded ready order (after rebuild)", order)):
        buckets = assign_buckets(sizes, ord_, caps)
        print(f"== Part 3: buckets from {title}, caps {[c / MiB for c in caps]} MiB: {len(buckets)} buckets ==")
        print(f"{'bkt':>3s} {'MiB':>8s} {'#p':>3s} {'first ready':>11s} {'last ready':>10s} {'launch at':>9s}  members")
        launch_prev, total_wait = -1, 0
        for k, b in enumerate(buckets):
            first, last = min(pos[i] for i in b), max(pos[i] for i in b)
            launch = max(last, launch_prev)  # in-order rule: not before every earlier bucket was launched
            total_wait += launch - last
            launch_prev = launch
            names = ", ".join(named[i][0] for i in b[:3]) + (" ..." if len(b) > 3 else "")
            print(f"{k:3d} {sum(sizes[i] for i in b) / MiB:8.3f} {len(b):3d} {first:11d} {last:10d} {launch:9d}  {names}")
        print(f"   a bucket can be sent only after its LAST grad ('last ready') and after bucket k-1 ('launch at').\n"
              f"   sum over buckets of (launch - last) = {total_wait} hook-steps of extra in-order waiting\n")


if __name__ == "__main__":
    main()
