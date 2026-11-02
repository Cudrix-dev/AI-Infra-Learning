"""Day 04 - file:line of today's reading list (DDP.no_sync and the flags it touches), in YOUR installed torch.

Same caveat as Days 02/03: a text search, a navigation aid, not proof of a call chain. Besides the def lines it lists
EVERY line that reads or writes the two sync flags, because today's question is "who looks at this flag, and when?".
The C++ Reducer is not in the wheel: read it on GitHub at the tag that matches your torch (printed below).

  python experiments/day04_grad_accum/find_symbols.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from find_symbols import ROOT, locate  # noqa: E402

DDP_FILE = "nn/parallel/distributed.py"
READING_LIST = [  # (file relative to the torch package, [symbols]) in READING ORDER
    (DDP_FILE, ["no_sync", "_pre_forward", "_post_forward", "_check_sync_bufs_pre_fwd", "will_sync_module_buffers",
                "_sync_buffers", "_DDPSink"]),
    ("_tensor.py", ["register_post_accumulate_grad_hook"]),
]
FLAGS = ["require_backward_grad_sync", "require_forward_param_sync"]

CPP = [  # (file, what to find) on GitHub
    ("torch/csrc/distributed/c10d/reducer.cpp",
     "Reducer::prepare_for_backward (expect_autograd_hooks_ = true; reset_bucket_counting), Reducer::autograd_hook "
     "(the early `if (!expect_autograd_hooks_) return;` and the comment above it, BEFORE grad_ready_order_indices_ "
     "and push_rebuilt_params), Reducer::finalize_backward (expect_autograd_hooks_ = false), "
     "Reducer::prepare_for_forward, should_rebuild_buckets in reducer.hpp"),
    ("torch/csrc/autograd/functions/accumulate_grad.h",
     "AccumulateGrad::accumulateGrad: steals the incoming grad if .grad is None, otherwise adds into .grad (in place "
     "when it can). This is the line that makes accumulation 'just work' across backwards"),
]


def main() -> None:
    ver = str(torch.__version__).split("+")[0]
    print(f"torch {torch.__version__} at {ROOT}\n")
    print("| Symbol | Where in your install |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        for s in symbols:
            print(f"| `{s}` ({rel.rsplit('/', 1)[-1]}) | {locate(ROOT / rel, s)} |")
    path = ROOT / DDP_FILE
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for flag in FLAGS:
        print(f"\nEvery line of {path.relative_to(ROOT.parent).as_posix()} that mentions `{flag}`:")
        for i, line in enumerate(lines, 1):
            if re.search(rf"\b{flag}\b", line):
                print(f"  {i:5d}  {line.strip()}")
    print(f"\nC++ side, at the release tag (e.g. https://github.com/pytorch/pytorch/blob/v{ver}/<file>):")
    for f, what in CPP:
        print(f"  - {f}: {what}")


if __name__ == "__main__":
    main()
