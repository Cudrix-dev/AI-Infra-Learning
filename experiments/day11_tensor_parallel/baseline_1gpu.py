"""Day 11 - the single-GPU reference: the dense MLP, one forward + backward, everything saved for the parity check.

    python experiments/day11_tensor_parallel/baseline_1gpu.py --run-id RID [--dtype float32|float64] [--blocks 1|2]

Writes outputs/day11_tp/tensors/baseline_<dtype>_b<blocks>_<RID>.pt (inputs, parameters, y, dx, every gradient; not
archived, it is reproducible from the seed) and outputs/day11_tp/baseline_<dtype>_b<blocks>_<RID>.json (shapes,
norms, SHA-256 of the tensors, environment). Uses ONE GPU: run it with CUDA_VISIBLE_DEVICES=0.

The model is plain nn.Linear layers loaded with the same tensors make_problem() returns, so the baseline is
"PyTorch's own MLP", not a re-implementation of the TP math.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp_common import (ACT, DTYPES, OUT, TENSORS, Shapes, environment, flatten_problem, loss_fn,  # noqa: E402
                       make_problem, set_exact_math, sha256_tensors, shapes_to_dict, write_json)


class DenseBlock(nn.Module):
    def __init__(self, d: int, ffn: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, ffn)   # weight [ffn, d]
        self.fc2 = nn.Linear(ffn, d)   # weight [d, ffn]

    def forward(self, x):
        return self.fc2(ACT(self.fc1(x)))


def run_dense(problem: dict, shapes: Shapes, device, dtype) -> dict:
    """Forward + backward of the dense model. Returns CPU tensors: y, loss, dx and per-block grads."""
    model = nn.Sequential(*[DenseBlock(shapes.d_model, shapes.ffn) for _ in range(shapes.blocks)]).to(device, dtype)
    with torch.no_grad():
        for blk, p in zip(model, problem["blocks"]):
            blk.fc1.weight.copy_(p["W1"]); blk.fc1.bias.copy_(p["b1"])
            blk.fc2.weight.copy_(p["W2"]); blk.fc2.bias.copy_(p["b2"])
    x = problem["x"].to(device).requires_grad_(True)
    y = model(x)
    loss = loss_fn(y, problem["target"].to(device))
    loss.backward()
    grads = [{"W1": b.fc1.weight.grad, "b1": b.fc1.bias.grad, "W2": b.fc2.weight.grad, "b2": b.fc2.bias.grad}
             for b in model]
    return {"y": y.detach().cpu(), "loss": loss.detach().cpu(), "dx": x.grad.cpu(),
            "grads": [{k: v.cpu() for k, v in g.items()} for g in grads]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32")
    ap.add_argument("--blocks", type=int, choices=[1, 2], default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    set_exact_math()
    dtype = DTYPES[args.dtype]
    shapes = Shapes(blocks=args.blocks)
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.device_count() >= 1
        torch.cuda.set_device(0)

    problem = make_problem(shapes, dtype)
    res = run_dense(problem, shapes, device, dtype)
    res2 = run_dense(problem, shapes, device, dtype)   # determinism of the baseline itself
    repeat_bitwise = all(torch.equal(res[k], res2[k]) for k in ("y", "dx")) and all(
        torch.equal(a[n], b[n]) for a, b in zip(res["grads"], res2["grads"]) for n in a)

    tag = f"{args.dtype}_b{args.blocks}_{args.run_id}"
    TENSORS.mkdir(parents=True, exist_ok=True)
    path = TENSORS / f"baseline_{tag}.pt"
    if path.exists():
        raise SystemExit(f"{path} exists; use a new --run-id")
    torch.save({"problem": problem, "result": res, "shapes": shapes_to_dict(shapes), "dtype": args.dtype}, path)

    print(f"Dense single-GPU MLP, {args.dtype}, {args.blocks} block(s), device {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"x {list(problem['x'].shape)}  ->  y {list(res['y'].shape)}   loss = {res['loss'].item():.10g}")
    print(f"{'tensor':<10} {'shape':<14} {'max|.|':>12} {'L2 norm':>12}")
    rows = [("y", res["y"]), ("dx", res["dx"])]
    for k, g in enumerate(res["grads"]):
        rows += [(f"blk{k}.{n}", t) for n, t in g.items()]
    for name, t in rows:
        print(f"{name:<10} {str(list(t.shape)):<14} {t.abs().max().item():>12.4e} {t.double().norm().item():>12.4e}")
    print(f"repeat run bitwise identical: {repeat_bitwise}")
    print(f"saved {path}")

    write_json(OUT / f"baseline_{tag}.json", {
        "run_id": args.run_id, "dtype": args.dtype, "shapes": shapes_to_dict(shapes), "loss": res["loss"].item(),
        "repeat_bitwise": repeat_bitwise, "tensor_file": str(path.relative_to(OUT.parents[1])),
        "problem_sha256": sha256_tensors(flatten_problem(problem)),
        "norms": {name: {"shape": list(t.shape), "max_abs": t.abs().max().item(), "l2": t.double().norm().item()}
                  for name, t in rows},
        "environment": environment(),
    })
    print("BASELINE: SAVED")


if __name__ == "__main__":
    main()
