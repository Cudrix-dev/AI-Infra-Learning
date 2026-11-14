"""Day 11 - file:line of the Block 3 reading list in YOUR Megatron-LM checkout (and in your torch install).

    python experiments/day11_tensor_parallel/find_symbols.py /workspace/Megatron-LM

A text search for `def name` / `class name` (first match): a navigation aid, not proof of a call chain. The GUIDE's
line numbers are for core_v0.19.2 (commit 4b4acac9); run this if your checkout is a different version.
No torch import is needed for the Megatron part, so it also runs on Windows against a local clone.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

READING_LIST = [
    ("megatron/core/tensor_parallel/mappings.py", [
        "_reduce", "_CopyToModelParallelRegion", "_ReduceFromModelParallelRegion", "_ScatterToModelParallelRegion",
        "_GatherFromModelParallelRegion", "copy_to_tensor_model_parallel_region",
        "reduce_from_tensor_model_parallel_region", "_ReduceScatterToSequenceParallelRegion"]),
    ("megatron/core/tensor_parallel/layers.py", [
        "param_is_not_tensor_parallel_duplicate", "_initialize_affine_weight_cpu",
        "LinearWithGradAccumulationAndAsyncCommunication", "linear_with_grad_accumulation_and_async_allreduce",
        "ColumnParallelLinear", "RowParallelLinear"]),
    ("megatron/core/transformer/mlp.py", ["MLP"]),
    ("megatron/core/parallel_state.py", ["initialize_model_parallel", "get_tensor_model_parallel_group"]),
    ("megatron/core/optimizer/clip_grads.py", ["get_grad_norm_fp32"]),
]
# Lines inside a function body worth jumping to (regex, first match after the def)
INSIDE = [
    ("megatron/core/tensor_parallel/layers.py", "LinearWithGradAccumulationAndAsyncCommunication",
     r"if ctx\.allreduce_dgrad:", "backward: async dx all-reduce launched"),
    ("megatron/core/tensor_parallel/layers.py", "ColumnParallelLinear", r"self\.allreduce_dgrad = \(",
     "allreduce_dgrad decided"),
    ("megatron/core/tensor_parallel/layers.py", "ColumnParallelLinear",
     r"input_parallel = copy_to_tensor_model_parallel_region", "the f path (only when allreduce_dgrad is False)"),
    ("megatron/core/tensor_parallel/layers.py", "RowParallelLinear",
     r"output_ = reduce_from_tensor_model_parallel_region", "the g call"),
    ("megatron/core/tensor_parallel/layers.py", "RowParallelLinear", r"output = \(output_ \+ self\.bias\)",
     "bias added after the reduction"),
]


def find_def(lines, symbol):
    defn = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    for i, line in enumerate(lines, 1):
        if defn.search(line):
            return i
    return None


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/Megatron-LM")
    if not (root / "megatron/core").is_dir():
        raise SystemExit(f"{root}/megatron/core not found. Usage: find_symbols.py PATH_TO_MEGATRON_LM")
    try:
        desc = subprocess.run(["git", "-C", str(root), "describe", "--tags"], capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        desc = head = "?"
    print(f"Megatron-LM at {root}: {desc or '?'} {head or ''}\n")
    print("| symbol | location |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        for s in symbols:
            i = find_def(lines, s)
            print(f"| `{s}` | {rel}:{i if i else '(not found)'} |")
    print()
    for rel, owner, pattern, what in INSIDE:
        lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        start = find_def(lines, owner) or 1
        hit = next((i for i, line in enumerate(lines[start - 1:], start) if re.search(pattern, line)), None)
        print(f"| {owner}: {what} | {rel}:{hit if hit else '(not found)'} |")
    try:
        import torch
        import torch.autograd.function as fn
        import torch.distributed.distributed_c10d as c10d
        print(f"\ntorch {torch.__version__}:")
        for mod, name in ((fn, "Function"), (fn, "FunctionCtx"), (c10d, "all_reduce")):
            path = Path(mod.__file__)
            i = find_def(path.read_text(encoding="utf-8").splitlines(), name)
            print(f"| `{name}` | torch/{path.relative_to(Path(torch.__file__).parent).as_posix()}:{i} |")
    except ImportError:
        print("\n(torch not importable here; skipped the torch symbols)")


if __name__ == "__main__":
    main()
