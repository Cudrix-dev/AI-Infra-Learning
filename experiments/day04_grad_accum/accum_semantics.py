"""Day 04 - gradient accumulation semantics in ONE process (no NCCL): bookkeeping, loss scaling, optimizer sensitivity.

Part 1  batch-size bookkeeping: micro m, accumulation K, world W -> per-GPU batch, global batch, optimizer steps per
        epoch (with and without dropping the short final window). Pure arithmetic.
Part 2  the gradient of K microbatches with loss * (rows_j / rows_window) equals the big-batch gradient; what each
        scaling mistake does to it (ratio of gradient norms, max |diff|).
Part 3  why some of those mistakes are INVISIBLE in training: run 10 optimizer steps with SGD, AdamW and AdamW + grad
        clipping, once with correct gradients, once with every gradient c times too large (persistent bug, e.g. no
        loss scaling), once with only the last step's gradient halved (one-off bug, e.g. 1/K in a short final window).

  python experiments/day04_grad_accum/accum_semantics.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day01_single_gpu"))
from train import PRESETS, Config, attention_ctx, build, get_batch, setup_reproducibility  # noqa: E402


# ----------------------------------------------------------------------------- Part 1
def bookkeeping(n_rows: int, m: int, k: int, w: int) -> dict:
    """One epoch of n_rows samples, DistributedSampler-style: every rank gets ceil(n/W) rows (padding repeats rows)."""
    rows_per_rank = math.ceil(n_rows / w)
    micro_per_rank = math.ceil(rows_per_rank / m)  # the last microbatch may be short
    return {"m": m, "K": k, "W": w, "per_gpu_batch": k * m, "global_batch": w * k * m,
            "micro_per_epoch": micro_per_rank,
            "opt_steps_keep_last": math.ceil(micro_per_rank / k),  # short final window stepped (scaled correctly!)
            "opt_steps_drop_last": micro_per_rank // k,  # short final window discarded
            "last_window_micro": micro_per_rank % k or k}


def part1() -> None:
    print("=== Part 1: bookkeeping  G = W x K x m;  optimizer steps = windows, not microbatches ===")
    print(f"{'N rows':>7s} {'m':>3s} {'K':>3s} {'W':>2s} {'per-GPU':>7s} {'global':>6s} {'micro/epoch/rank':>16s} "
          f"{'steps keep':>10s} {'steps drop':>10s} {'last window':>11s}")
    for n, m, k, w in [(10_000, 8, 1, 2), (10_000, 8, 4, 2), (10_000, 8, 8, 2), (10_000, 16, 4, 1), (1_000, 8, 4, 2)]:
        b = bookkeeping(n, m, k, w)
        print(f"{n:7d} {m:3d} {k:3d} {w:2d} {b['per_gpu_batch']:7d} {b['global_batch']:6d} {b['micro_per_epoch']:16d} "
              f"{b['opt_steps_keep_last']:10d} {b['opt_steps_drop_last']:10d} {b['last_window_micro']:>8d}/{k}")
    print("The LR schedule (warmup, decay) and Adam's bias correction count OPTIMIZER steps. Halving m and doubling K\n"
          "keeps G and the number of optimizer steps; it changes memory and speed, not the optimization problem.\n")


# ----------------------------------------------------------------------------- Part 2
def grads_of(model, cfg, chunks: list[tuple[torch.Tensor, torch.Tensor, float]], zero_each: bool = False) -> torch.Tensor:
    """Backward each (x, y, scale) chunk WITHOUT zeroing in between (= accumulation); return all grads flattened."""
    model.zero_grad(set_to_none=True)
    for x, y, scale in chunks:
        if zero_each:
            model.zero_grad(set_to_none=True)
        with attention_ctx(cfg):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        (loss * scale).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def part2(cfg: Config, device: torch.device) -> None:
    print("=== Part 2: accumulated gradient vs one big batch (same rows, same parameters) ===")
    model, _, _ = build(cfg, device)
    x, y = get_batch(replace(cfg, batch_size=32), 0, device)

    def micro(rows: int, m: int, scale):  # split the first `rows` rows into microbatches; scale(rows_j, n_micro)
        xs, ys = x[:rows].split(m), y[:rows].split(m)
        return [(a, b, scale(a.size(0), len(xs))) for a, b in zip(xs, ys)]

    def report(label: str, g: torch.Tensor, ref: torch.Tensor) -> None:
        print(f"  {label:58s} |g|/|g_ref| = {(g.norm() / ref.norm()).item():6.3f}   "
              f"max|diff| = {(g - ref).abs().max().item():.3e}")

    big32 = grads_of(model, cfg, [(x, y, 1.0)])
    report("K=4 x m=8, loss * 1/4                (correct)", grads_of(model, cfg, micro(32, 8, lambda r, n: 1 / 4)), big32)
    report("K=4 x m=8, loss * 1                  (no_loss_scale)", grads_of(model, cfg, micro(32, 8, lambda r, n: 1.0)), big32)
    report("K=4 x m=8, zero_grad every micro     (zero_every_micro)",
           grads_of(model, cfg, micro(32, 8, lambda r, n: 1 / 4), zero_each=True), big32)
    big16 = grads_of(model, cfg, [(x[:16], y[:16], 1.0)])
    report("short window: 2 x m=8, loss * 1/2    (correct)", grads_of(model, cfg, micro(16, 8, lambda r, n: 1 / 2)), big16)
    report("short window: 2 x m=8, loss * 1/K=1/4 (fixed_k_scale)",
           grads_of(model, cfg, micro(16, 8, lambda r, n: 1 / 4)), big16)
    big20 = grads_of(model, cfg, [(x[:20], y[:20], 1.0)])
    report("uneven: 20 rows -> 8,8,4, loss * rows_j/20 (correct)",
           grads_of(model, cfg, micro(20, 8, lambda r, n: r / 20)), big20)
    report("uneven: 20 rows -> 8,8,4, loss * 1/3  (mean of means)",
           grads_of(model, cfg, micro(20, 8, lambda r, n: 1 / n)), big20)
    print("  The correct rows show only float summation-order noise; every bug changes the gradient itself.\n"
          "  With masked tokens (LLM training) the weight is (valid tokens in micro) / (valid tokens in window).\n")


# ----------------------------------------------------------------------------- Part 3
def trajectory(cfg: Config, device: torch.device, opt: str, scale_at, steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Train `steps` optimizer steps from the same init on the same batches; scale_at(s) multiplies the loss of step s.
    Returns (initial params, final params), flattened."""
    setup_reproducibility(cfg)
    model, _, _ = build(cfg, device)
    p0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    if opt == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=0.05)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg.weight_decay, foreach=False)
    for s in range(steps):
        x, y = get_batch(replace(cfg, batch_size=16), s, device)
        optim.zero_grad(set_to_none=True)
        with attention_ctx(cfg):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) * scale_at(s)
        loss.backward()
        if opt == "adamw_clip":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
    return p0, torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def part3(cfg: Config, device: torch.device, steps: int = 10) -> None:
    print(f"=== Part 3: does the optimizer notice a wrongly scaled gradient? ({steps} steps, same init and batches) ===")
    print("  metric: |theta_bug - theta_ok| / |theta_ok - theta_0|  (0 = invisible, 1 = as large as all of training)")
    print(f"  {'optimizer':12s} {'every grad x4 (persistent)':>28s} {'last grad x0.5 (one-off)':>26s}")
    for opt in ("sgd", "adamw", "adamw_clip"):
        p0, ok = trajectory(cfg, device, opt, lambda s: 1.0, steps)
        _, pers = trajectory(cfg, device, opt, lambda s: 4.0, steps)
        _, once = trajectory(cfg, device, opt, lambda s: 0.5 if s == steps - 1 else 1.0, steps)
        move = (ok - p0).norm()
        print(f"  {opt:12s} {((pers - ok).norm() / move).item():28.3e} {((once - ok).norm() / move).item():26.3e}")
    print("  AdamW divides m by sqrt(v): a CONSTANT factor on every gradient cancels (up to eps), so a missing 1/K is\n"
          "  nearly invisible in the loss curve. SGD moves c times further. Clipping changes the effective threshold.\n"
          "  A ONE-OFF factor does not cancel, because m and v remember the correctly scaled past. Checking the\n"
          "  gradient directly (check A / E in compare_accum.py) catches both; checking the loss curve may not.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preset", choices=PRESETS, default="tiny")
    args = ap.parse_args()
    device = torch.device(args.device)
    cfg = Config(**PRESETS[args.preset])
    setup_reproducibility(cfg)
    print(f"torch {torch.__version__}, device {device}, preset {args.preset}\n")
    part1()
    part2(cfg, device)
    part3(cfg, device)


if __name__ == "__main__":
    main()
