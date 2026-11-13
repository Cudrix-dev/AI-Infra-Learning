"""Day 10 - small helpers shared by the check, inspect and benchmark scripts (no kernels here)."""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT_DIR = PROJECT / "outputs" / "day10_softmax"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # so `import softmax_ref` works from the project root

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Pass/fail tolerances against torch.softmax in the SAME dtype. Both sides compute in fp32 and round once, so the
# fp16/bf16 outputs may differ by about one ulp of the output dtype where the two fp32 values straddle a rounding
# boundary: fp16 ulp = 2^-10 relative (~9.8e-4), bf16 ulp = 2^-7 (~7.8e-3). The rtol values allow about one ulp (fp16)
# and two ulps (bf16, torch.testing's default). fp32: Triton's exp is the fast hardware exp2 path and torch uses
# expf, so a few fp32 ulps of difference (~1e-6 relative) are expected; rtol 1e-5 leaves a margin.
# atol covers outputs near zero (most entries of a long row), where a relative test is meaningless.
TOLERANCES = {
    torch.float32: dict(rtol=1e-5, atol=1e-6),
    torch.float16: dict(rtol=1e-3, atol=1e-5),
    torch.bfloat16: dict(rtol=1.6e-2, atol=1e-5),
}


def load_impl(name: str):
    """'ref' -> softmax_ref, 'mine' -> my_softmax. Both expose softmax(), rmsnorm(), compiled_softmax_kernel()."""
    return importlib.import_module({"ref": "softmax_ref", "mine": "my_softmax"}[name])


def is_not_implemented(err: BaseException) -> bool:
    """True if `err` comes from the untouched starter (its static_assert or NotImplementedError)."""
    text = f"{type(err).__name__}: {err}"
    return isinstance(err, NotImplementedError) or "not implemented yet" in text


def env_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        import triton
        triton_version = triton.__version__
    except Exception as e:  # noqa: BLE001
        triton_version = f"unavailable ({e})"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.max.sm,clocks.max.mem",
                          "--format=csv,noheader"], capture_output=True, text=True)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "triton": triton_version,
        "gpu": p.name,
        "sm_count": p.multi_processor_count,
        "capability": f"sm_{p.major}{p.minor}",
        "total_mem_gib": round(p.total_memory / 2**30, 2),
        "l2_cache_mib": round(getattr(p, "L2_cache_size", 0) / 2**20, 2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "nvidia_smi": smi.stdout.strip(),
        "hostname": platform.node(),
        "container_id": os.environ.get("CONTAINER_ID", "?"),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; use a new --run-id")
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {path.relative_to(PROJECT).as_posix()}")
