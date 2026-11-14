"""Day 11 - shared pieces: problem shapes, the dense (single-GPU) MLP, the loss, comparisons, a collective counter.

Everything that must be IDENTICAL between the single-GPU baseline and the tensor-parallel run lives here:
  * make_problem(): input x, target, and every block's (W1, b1, W2, b2), drawn on the CPU from one seed in float64
    and then cast, so the values do not depend on the device or on how many processes exist;
  * dense_forward(): the plain MLP, y = W2 . gelu(W1 . x + b1) + b2 per block (PyTorch's F.linear layout);
  * loss_fn(): mean squared error against the target.

Layout follows Megatron: activations are [seq, batch, hidden]. F.linear only touches the LAST dimension, so the leading
dimensions are just "tokens".

PyTorch weight layout reminder: nn.Linear(in, out).weight has shape [out_features, in_features] and computes
y = x @ W.T + b. "Column-parallel" and "row-parallel" are named after the math form Y = X A with A = W.T [in, out]:
  * column-parallel splits A's columns = W's ROWS (dim 0 of the PyTorch weight, the output features);
  * row-parallel splits A's rows = W's COLUMNS (dim 1 of the PyTorch weight, the input features).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / "outputs" / "day11_tp"          # JSON results (archived by the runner)
TENSORS = OUT / "tensors"                       # baseline .pt files (NOT archived: ~10-40 MB each, reproducible)

DTYPES = {"float32": torch.float32, "float64": torch.float64}
# Pass criterion for every compared tensor: max|tp - dense| / max|dense| <= REL_TOL. Relative to the tensor's own
# scale, because the MSE gradients are small (the mean divides by seq*batch*d_model = 262,144).
REL_TOL = {torch.float32: 1e-5, torch.float64: 1e-12}
ACT = F.gelu  # exact (erf) GELU; any ELEMENTWISE function works with a sharded input, see GUIDE 1.2


@dataclass(frozen=True)
class Shapes:
    seq: int = 256
    batch: int = 2
    d_model: int = 512
    ffn: int = 2048     # 4 x d_model, as in Day 01's medium model; split over TP ranks
    blocks: int = 1     # 1 = one MLP (the core check); 2 = two MLPs stacked, to show errors propagating upstream


def set_exact_math() -> None:
    """FP32 matmuls in full precision (no TF32), so FP32 differences are rounding-order only."""
    torch.set_float32_matmul_precision("highest")


def make_problem(shapes: Shapes, dtype: torch.dtype, seed: int = 1234) -> dict:
    """Input, target and parameters for `shapes.blocks` MLP blocks, on the CPU."""
    g = torch.Generator().manual_seed(seed)
    s, b, d, f = shapes.seq, shapes.batch, shapes.d_model, shapes.ffn

    def randn(*size, scale=1.0):
        return (torch.randn(*size, generator=g, dtype=torch.float64) * scale).to(dtype)

    blocks = []
    for _ in range(shapes.blocks):
        blocks.append({
            "W1": randn(f, d, scale=d ** -0.5),   # [out=ffn, in=d]   -> column-parallel: split dim 0
            "b1": randn(f, scale=0.1),            # [ffn]             -> split like W1's rows
            "W2": randn(d, f, scale=f ** -0.5),   # [out=d, in=ffn]   -> row-parallel: split dim 1
            "b2": randn(d, scale=0.1),            # [d]               -> replicated, added once AFTER the reduction
        })
    return {"x": randn(s, b, d), "target": randn(s, b, d), "blocks": blocks}


def dense_forward(x: torch.Tensor, blocks: list[dict]) -> torch.Tensor:
    for p in blocks:
        x = F.linear(ACT(F.linear(x, p["W1"], p["b1"])), p["W2"], p["b2"])
    return x


def loss_fn(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The SAME loss on every TP rank: y is replicated, so each rank computes the full loss and backpropagates 1.0.
    No division by the TP size anywhere (contrast with DDP, where ranks hold DIFFERENT data and the mean is taken
    over ranks)."""
    return ((y - target) ** 2).mean()


# Which dimension of each parameter tensor-parallelism splits (None = replicated on every TP rank).
SPLIT_DIM = {"W1": 0, "b1": 0, "W2": 1, "b2": None}


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, rel_tol: float) -> dict:
    """max abs error, error relative to max|ref|, bitwise flag, and the projection ratio <got,ref>/<ref,ref>.

    The ratio tells a SCALE bug from a random one: 2.000 means "doubled", 0.500 "halved", anything else "wrong".
    """
    got64, ref64 = got.detach().double().cpu(), ref.detach().double().cpu()
    if got64.shape != ref64.shape:
        return {"name": name, "ok": False, "shape": list(got64.shape), "ref_shape": list(ref64.shape),
                "max_abs": float("inf"), "rel": float("inf"), "bitwise": False, "ratio": float("nan")}
    diff = (got64 - ref64).abs().max().item()
    scale = ref64.abs().max().item()
    rel = diff / scale if scale > 0 else diff
    ratio = (got64 * ref64).sum().item() / (ref64 * ref64).sum().item()
    return {"name": name, "ok": bool(rel <= rel_tol), "shape": list(got64.shape), "max_abs": diff, "rel": rel,
            "bitwise": bool(torch.equal(got.detach().cpu(), ref.detach().cpu())), "ratio": ratio}


def sha256_tensors(tensors: dict) -> str:
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def flatten_problem(problem: dict) -> dict:
    out = {"x": problem["x"], "target": problem["target"]}
    for k, p in enumerate(problem["blocks"]):
        out.update({f"b{k}.{n}": t for n, t in p.items()})
    return out


class CollectiveCounter:
    """Counts torch.distributed collectives per phase ("forward" / "backward") by wrapping the module functions.

    It wraps the ATTRIBUTES of torch.distributed, so code that calls `dist.all_reduce(...)` or
    `torch.distributed.all_reduce(...)` at call time is counted (yours, the reference, Megatron Core). Code that did
    `from torch.distributed import all_reduce` before the wrap would not be. Calls made while `phase` is None (the
    checker's own gathers) are not counted.
    """

    NAMES = ("all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter_tensor", "broadcast")

    def __init__(self) -> None:
        self.phase: str | None = None
        self.events: list[dict] = []
        self._orig = {}

    def __enter__(self):
        for name in self.NAMES:
            orig = getattr(dist, name, None)
            if orig is None:
                continue
            self._orig[name] = orig

            def wrapped(*args, fn_name=name, fn=orig, **kwargs):
                if self.phase is not None:
                    t = args[0] if args else kwargs.get("tensor")
                    if isinstance(t, (list, tuple)):
                        t = args[1] if len(args) > 1 else kwargs.get("tensor")
                    op = kwargs.get("op", args[1] if fn_name == "all_reduce" and len(args) > 1 else None)
                    self.events.append({
                        "phase": self.phase, "op": fn_name, "reduce_op": str(op) if op is not None else "SUM(default)",
                        "shape": list(t.shape) if isinstance(t, torch.Tensor) else None,
                        "bytes": t.numel() * t.element_size() if isinstance(t, torch.Tensor) else None,
                        "async": bool(kwargs.get("async_op", False)),
                    })
                return fn(*args, **kwargs)

            setattr(dist, name, wrapped)
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(dist, name, orig)

    def take(self) -> list[dict]:
        ev, self.events = self.events, []
        return ev


def environment() -> dict:
    info = {
        "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "platform": platform.platform(), "hostname": platform.node(),
        "container_id": os.environ.get("CONTAINER_ID", "?"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        try:
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:  # noqa: BLE001
            pass
    return info


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def shapes_to_dict(shapes: Shapes) -> dict:
    return asdict(shapes)


def die_not_implemented(err: NotImplementedError) -> None:
    print(f"\nNOT IMPLEMENTED YET: {err}\nFill in the TODOs in my_tp.py (see GUIDE Block 2).", flush=True)
    sys.exit(3)
