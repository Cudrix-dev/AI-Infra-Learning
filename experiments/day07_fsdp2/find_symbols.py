"""Day 07 - file:line of today's FSDP2 reading list (guide Block 3), in YOUR installed torch, in reading order.

Same caveat as Days 02-06: a text search for def/class lines, a navigation aid, not proof of a call chain. Several FSDP2
classes define methods with the same name (FSDPModule.unshard, FSDPParamGroup.unshard); the search is per FILE, so the
file column tells you which one you got. A symbol printed as "not found" may have moved in your release.

  python experiments/day07_fsdp2/find_symbols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from find_symbols import ROOT, locate  # noqa: E402

FS = "distributed/fsdp/_fully_shard/"
READING_LIST = [  # (file relative to the torch package, [symbols]) in READING ORDER
    (FS + "_fully_shard.py", ["fully_shard", "FSDPModule", "set_requires_gradient_sync", "set_reshard_after_backward",
                              "set_modules_to_forward_prefetch", "set_gradient_divide_factor"]),
    (FS + "_fsdp_init.py", ["_get_managed_states", "_init_default_mesh"]),
    (FS + "_fsdp_param.py", ["FSDPParam", "_init_sharded_param", "init_unsharded_param", "to_sharded",
                             "to_unsharded", "free_storage", "alloc_storage"]),
    (FS + "_fsdp_state.py", ["FSDPState", "_root_pre_forward", "_lazy_init", "_pre_forward", "_post_forward",
                             "_pre_backward", "_root_post_backward_final_callback"]),
    (FS + "_fsdp_param_group.py", ["FSDPParamGroup", "unshard", "wait_for_unshard", "reshard", "pre_forward",
                                   "post_forward", "pre_backward", "post_backward", "_backward_prefetch",
                                   "_prefetch_unshard", "RegisterPostBackwardFunction"]),
    (FS + "_fsdp_collectives.py", ["foreach_all_gather", "foreach_all_gather_copy_out", "foreach_reduce",
                                   "foreach_reduce_scatter_copy_in", "_get_gradient_divide_factors"]),
    ("distributed/tensor/_api.py", ["DTensor", "full_tensor", "to_local"]),
    ("optim/adam.py", ["_single_tensor_adam", "_multi_tensor_adam"]),
]


def main() -> None:
    print(f"torch {torch.__version__} at {ROOT}\n")
    print("| Symbol | Where in your install |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        path = ROOT / rel
        for s in symbols:
            where = locate(path, s) if path.exists() else f"(file {rel} not in this release)"
            print(f"| `{s}` ({rel.rsplit('/', 1)[-1]}) | {where} |")


if __name__ == "__main__":
    main()
