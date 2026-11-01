"""Day 03 - file:line of every Python symbol on today's DDP reading list, in YOUR installed torch.

Same caveat as Day 02: a text search for "def/class name", a navigation aid, not proof of a call chain.
The C++ Reducer is not in the wheel: read it on GitHub at the tag that matches your torch (links printed below).

  python experiments/day03_ddp_hooks/find_symbols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from find_symbols import ROOT, locate  # noqa: E402

READING_LIST = [  # (file relative to the torch package, [symbols]) in READING ORDER
    ("_tensor.py", ["register_post_accumulate_grad_hook", "register_hook"]),
    ("nn/parallel/distributed.py", [
        # __init__ and forward are not listed: the first "def __init__" / "def forward" in this file belong to other
        # classes (_DDPSink, the join hook). Search inside class DistributedDataParallel for those two.
        "_BucketCapacityConfig", "compute_bucket_size_limits", "DistributedDataParallel",
        "_ddp_init_helper", "_pre_forward", "_lazy_init", "_post_forward", "_sync_buffers", "no_sync",
        "register_comm_hook", "_DDPSink"]),
    ("distributed/utils.py", ["_sync_module_states", "_verify_param_shape_across_processes"]),
    ("distributed/algorithms/ddp_comm_hooks/default_hooks.py", ["allreduce_hook", "_allreduce_fut", "fp16_compress_hook"]),
]

CPP = [  # (file, symbols) on GitHub
    ("torch/csrc/distributed/c10d/reducer.hpp", "kDefaultFirstBucketBytes, kDefaultBucketBytesCap, struct Bucket (gradients, "
                                                "bucket_views_in/out, pending, future_work)"),
    ("torch/csrc/distributed/c10d/reducer.cpp", "Reducer::Reducer (add_post_hook on each grad_accumulator), autograd_hook, "
                                                "mark_variable_ready, mark_variable_ready_dense, mark_bucket_ready, "
                                                "all_reduce_bucket, run_comm_hook, finalize_backward, finalize_bucket_dense, "
                                                "copy_bucket_to_grad, rebuild_buckets, sync_bucket_indices, "
                                                "initialize_buckets, initialize_bucket_views, compute_bucket_assignment_by_size"),
    ("torch/csrc/distributed/c10d/default_comm_hooks.cpp", "AllReduceCommHook / _AllReduceBySumCommHook::runHook"),
    ("torch/csrc/autograd/functions/accumulate_grad.h", "AccumulateGrad::accumulateGrad (steals or adds into .grad; the "
                                                        "post hooks run after it)"),
]


def main() -> None:
    ver = str(torch.__version__).split("+")[0]
    print(f"torch {torch.__version__} at {ROOT}\n")
    print("| Symbol | Where in your install |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        for s in symbols:
            print(f"| `{s}` ({rel.rsplit('/', 1)[-1]}) | {locate(ROOT / rel, s)} |")
    print(f"\nC++ side, at the release tag (e.g. https://github.com/pytorch/pytorch/blob/v{ver}/<file>; if your exact "
          "patch tag does not exist, use the nearest one and say so in your notes):")
    for f, syms in CPP:
        print(f"  - {f}: {syms}")


if __name__ == "__main__":
    main()
