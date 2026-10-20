"""Day 01 - checkpoint/resume parity check.

Runs train.py three times (four with --with-bad-resume) and compares them:

  full     : steps 0..N-1 in one process                          (reference)
  part     : steps 0..K-1, saves ckpt_stepK.pt, exits             (simulated crash)
  resumed  : new process, loads ckpt_stepK.pt, runs steps K..N-1
  bad      : like resumed, but loads model weights only           (negative control)

Checks
  A. run-to-run reproducibility : full vs part,    losses of steps [0, K)
  B. resume correctness          : full vs resumed, losses of steps [K, N)
  C. final state                 : full/final.pt vs resumed/final.pt (model, Adam moments, step, scheduler)

Usage (from the repo root). Unknown flags are forwarded to train.py:
  python experiments/day01_single_gpu/check_resume.py
  python experiments/day01_single_gpu/check_resume.py --with-bad-resume
  python experiments/day01_single_gpu/check_resume.py --no-deterministic --atol 1e-5
  python experiments/day01_single_gpu/check_resume.py --preset medium --steps 100 --stop-at 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train.py"
OUT = HERE.parents[1] / "outputs" / "day01_single_gpu"


def run(*args: str) -> None:
    cmd = [sys.executable, str(TRAIN), *args]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def losses(tag: str) -> dict[int, float]:
    m = json.loads((OUT / tag / "metrics.json").read_text())
    return {int(k): v for k, v in m["losses"].items()}


def max_loss_diff(a: dict[int, float], b: dict[int, float], steps) -> float:
    return max(abs(a[s] - b[s]) for s in steps)


def tree_max_diff(a, b, path: str = "") -> tuple[float, str]:
    """Largest |a - b| over two nested dict/list/tensor/number trees, plus where it occurred."""
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return float("inf"), f"{path}: keys differ {set(a) ^ set(b)}"
        return max((tree_max_diff(a[k], b[k], f"{path}.{k}") for k in a), default=(0.0, path), key=lambda r: r[0])
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf"), f"{path}: length differs"
        return max((tree_max_diff(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))),
                   default=(0.0, path), key=lambda r: r[0])
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return float("inf"), f"{path}: shape/dtype differs"
        if not a.is_floating_point():
            return (0.0 if torch.equal(a, b) else float("inf")), path
        return (a.double() - b.double()).abs().max().item() if a.numel() else 0.0, path
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)), path
    return (0.0 if a == b else float("inf")), path


def load(tag: str, name: str) -> dict:
    return torch.load(OUT / tag / name, map_location="cpu", weights_only=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--stop-at", type=int, default=100)
    p.add_argument("--atol", type=float, default=0.0, help="0.0 = demand bit-exact (deterministic mode)")
    p.add_argument("--with-bad-resume", action="store_true")
    args, forward = p.parse_known_args()
    N, K = args.steps, args.stop_at
    assert 0 < K < N, "need 0 < stop-at < steps"
    common = ["--steps", str(N), *forward]

    run("--tag", "full", *common)
    run("--tag", "part", "--stop-at", str(K), *common)
    ckpt = str(OUT / "part" / f"ckpt_step{K}.pt")
    run("--tag", "resumed", "--resume", ckpt, *common)
    if args.with_bad_resume:
        run("--tag", "bad", "--resume", ckpt, "--bad-resume", *common)

    full, part, resumed = losses("full"), losses("part"), losses("resumed")
    a = max_loss_diff(full, part, range(0, K))
    b = max_loss_diff(full, resumed, range(K, N))
    fa, fb = load("full", "final.pt"), load("resumed", "final.pt")
    c_model, where_m = tree_max_diff(fa["model"], fb["model"], "model")
    c_opt, where_o = tree_max_diff(fa["optimizer"], fb["optimizer"], "optimizer")
    c_sched, where_s = tree_max_diff(fa["scheduler"], fb["scheduler"], "scheduler")

    rows = [
        ("A  reproducibility: losses [0,K)  full vs part", a, ""),
        ("B  resume: losses [K,N)  full vs resumed", b, ""),
        ("C1 final model params", c_model, where_m),
        ("C2 final optimizer state (exp_avg, exp_avg_sq, step, lr)", c_opt, where_o),
        ("C3 final scheduler state", c_sched, where_s),
    ]
    if args.with_bad_resume:
        bad = losses("bad")
        rows.append(("X  NEGATIVE CONTROL: losses [K,N) full vs bad", max_loss_diff(full, bad, range(K, N)), "should be LARGE"))

    print(f"\nN={N} K={K} atol={args.atol} forwarded={forward}")
    ok = True
    for name, diff, where in rows:
        passed = diff <= args.atol
        if not name.startswith("X"):
            ok &= passed
        tag = "" if name.startswith("X") else ("PASS" if passed else "FAIL")
        print(f"{tag:4s}  {name:58s} max|diff| = {diff:.3e}  {where}")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
