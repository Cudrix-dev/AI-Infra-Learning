"""Day 06 - file:line of today's reading list (ZeroRedundancyOptimizer, AdamW's per-tensor loop, the two collectives,
the DDP + ZeRO overlap hook, and the FSDP2 accumulation switch for Day 7), in YOUR installed torch.

Same caveat as Days 02-04: a text search for def/class lines, a navigation aid, not proof of a call chain.
A symbol printed as "not found" may have moved in your release; search the file by name.

  python experiments/day06_zero1/find_symbols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from find_symbols import ROOT, locate  # noqa: E402

READING_LIST = [  # (file relative to the torch package, [symbols]) in READING ORDER (guide Block 3)
    ("distributed/optim/zero_redundancy_optimizer.py",
     ["ZeroRedundancyOptimizer", "_partition_parameters", "_init_local_optimizer", "step", "_local_step",
      "_sync_params", "_broadcast_params_from_rank", "_build_param_buckets", "consolidate_state_dict",
      "state_dict"]),
    ("optim/adam.py", ["_single_tensor_adam", "_init_group"]),
    ("optim/adamw.py", ["AdamW"]),
    ("distributed/distributed_c10d.py", ["reduce_scatter_single", "all_gather_single", "reduce_scatter_tensor",
                                         "all_gather_into_tensor"]),
    ("distributed/algorithms/ddp_comm_hooks/ddp_zero_hook.py", ["hook_with_zero_step", "_perform_local_step",
                                                                "_broadcast_bucket"]),
    ("distributed/fsdp/_fully_shard/_fully_shard.py", ["set_requires_gradient_sync", "set_reshard_after_backward"]),
]

DEEPSPEED = [  # not in the torch wheel: read on GitHub (deepspeedai/DeepSpeed), Day 8 preview only
    ("deepspeed/runtime/zero/stage_1_and_2.py",
     "DeepSpeedZeroOptimizer: how the flat fp32 partitions are built (single_partition_of_fp32_groups), where the "
     "gradients are reduced for stage 1 vs stage 2 (search reduce_gradients, reduce_ipg_grads, "
     "partition_gradients), and step(): local update, then all-gather of the updated partitions"),
    ("deepspeed/runtime/engine.py",
     "is_gradient_accumulation_boundary(), and where backward()/step() consult it"),
]


def main() -> None:
    print(f"torch {torch.__version__} at {ROOT}\n")
    print("| Symbol | Where in your install |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        path = ROOT / rel
        for s in symbols:
            where = locate(path, s) if path.exists() else f"(file {rel} not in this release)"
            print(f"| `{s}` ({rel.rsplit('/', 1)[-1]}) | {where} |")
    print("\nDeepSpeed (not installed here; read on GitHub, Day 8 preview):")
    for f, what in DEEPSPEED:
        print(f"  - {f}: {what}")


if __name__ == "__main__":
    main()
