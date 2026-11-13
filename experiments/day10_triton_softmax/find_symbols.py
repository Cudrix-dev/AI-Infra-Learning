"""Day 10 - file:line of the Block 3 reading list in YOUR installed triton and torch (+ C++ files to open on GitHub).

    python experiments/day10_triton_softmax/find_symbols.py

A text search for `def name` / `class name`: a navigation aid, not proof of a call chain (same caveat as Day 02).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import triton

TORCH = Path(torch.__file__).resolve().parent
TRITON = Path(triton.__file__).resolve().parent

READING_LIST = [
    (TRITON, "language/core.py", ["program_id", "arange", "load", "store", "reduce", "constexpr", "static_assert"]),
    (TRITON, "language/standard.py", ["max", "sum", "softmax"]),
    (TRITON, "language/math.py", ["exp", "exp2"]),
    (TRITON, "runtime/jit.py", ["JITFunction", "run", "_do_compile", "specialize_impl"]),
    (TRITON, "compiler/compiler.py", ["compile", "CompiledKernel", "_init_handles"]),
    (TORCH, "_decomp/decompositions.py", ["_softmax"]),
    (TORCH, "_inductor/runtime/triton_heuristics.py", ["persistent_reduction", "reduction", "_persistent_reduction_configs"]),
    (TORCH, "_inductor/cudagraph_trees.py", ["cudagraphify_impl", "CUDAGraphTreeManager", "CUDAGraphNode"]),
    (TORCH, "compiler/__init__.py", ["cudagraph_mark_step_begin"]),
]
GITHUB = [
    "aten/src/ATen/native/cuda/SoftMax.cu: host_softmax, dispatch_softmax_forward, cunn_SoftMaxForward "
    "(github.com/pytorch/pytorch, tag v2.14.1)",
    "aten/src/ATen/native/cuda/PersistentSoftmax.cuh: softmax_warp_forward (rows up to 1024 elements: one warp per row)",
]


def locate(root: Path, rel: str, symbol: str) -> str:
    path = root / rel
    if not path.exists():
        return "(file not found)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    defn = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    shown = f"{root.name}/{rel}"
    for i, line in enumerate(lines, 1):
        if defn.search(line):
            return f"{shown}:{i}"
    for i, line in enumerate(lines, 1):
        if re.search(rf"\b{re.escape(symbol)}\b", line):
            return f"(no def; first mention {shown}:{i})"
    return "(not found)"


def main() -> None:
    print(f"torch {torch.__version__} at {TORCH}\ntriton {triton.__version__} at {TRITON}\n")
    print("| symbol | location |\n| --- | --- |")
    for root, rel, symbols in READING_LIST:
        for s in symbols:
            print(f"| `{s}` | {locate(root, rel, s)} |")
    print("\nC++ (not in the wheel):")
    for g in GITHUB:
        print(f"- {g}")


if __name__ == "__main__":
    main()
