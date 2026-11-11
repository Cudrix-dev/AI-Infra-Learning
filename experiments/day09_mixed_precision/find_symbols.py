"""Day 09 - file:line of today's reading list (guide Block 3), in YOUR installed torch, in reading order.

Same caveat as Days 02-08: a text search for def/class lines, a navigation aid, not proof of a call chain. Two stops are
C++ and not in the wheel; they are printed at the end with the GitHub path to open (use the tag of your torch version).

  python experiments/day09_mixed_precision/find_symbols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from find_symbols import ROOT, locate  # noqa: E402

READING_LIST = [  # (file relative to the torch package, [symbols], required?) in READING ORDER
    ("amp/autocast_mode.py", ["autocast", "__enter__", "__exit__"], True),
    ("amp/grad_scaler.py", ["OptState", "GradScaler", "scale", "_unscale_grads_", "unscale_", "_maybe_opt_step",
                            "step", "update"], True),
    ("nn/utils/clip_grad.py", ["clip_grad_norm_", "_get_total_norm", "_clip_grads_with_norm_"], True),
    ("utils/checkpoint.py", ["checkpoint", "_checkpoint_without_reentrant_generator_impl", "_CheckpointFrame",
                             "_checkpoint_hook", "_recomputation_hook", "_get_autocast_kwargs"], True),
    ("optim/adam.py", ["_init_group", "_single_tensor_adam"], False),
    ("utils/checkpoint.py", ["CheckpointFunction", "create_selective_checkpoint_contexts", "CheckpointPolicy"], False),
    ("nn/functional.py", ["multi_head_attention_forward", "_canonical_mask"], False),
    ("distributed/algorithms/ddp_comm_hooks/default_hooks.py", ["_compress_hook", "bf16_compress_hook"], False),
    ("distributed/fsdp/_fully_shard/_fsdp_api.py", ["MixedPrecisionPolicy"], False),
]
CPP = [  # not in the installed package: open on GitHub at your release tag (v2.14.1 here)
    ("aten/src/ATen/autocast_mode.h", "the CUDA op lists as macros: AT_FORALL_LOWER_PRECISION_FP (addmm, linear, "
                                      "bmm, scaled_dot_product_attention, ...), AT_FORALL_FP32 (layer_norm, nll_loss, "
                                      "...), AT_FORALL_FP32_SET_OPT_DTYPE (softmax, log_softmax, ...)", True),
    ("aten/src/ATen/autocast_mode.cpp", "TORCH_LIBRARY_IMPL(aten, AutocastCUDA, m), where those lists are registered; "
                                        "cached_cast: which tensors are cast once per region and cached (fp32 leaves "
                                        "that require grad, not views)", True),
    ("aten/src/ATen/native/transformers/attention.cpp", "_scaled_dot_product_attention_math: the upcast of half / "
                                                        "bf16 q, k, v to float", False),
]


def main() -> None:
    print(f"torch {torch.__version__} at {ROOT}\n")
    print("| Stop | Symbol | Where in your install |\n| --- | --- | --- |")
    for rel, symbols, req in READING_LIST:
        path = ROOT / rel
        for s in symbols:
            where = locate(path, s) if path.exists() else f"(file {rel} not in this release)"
            print(f"| {'required' if req else 'optional'} | `{s}` ({rel.rsplit('/', 1)[-1]}) | {where} |")
    print("\nC++ (GitHub, pytorch/pytorch at your tag):")
    for path, what, req in CPP:
        print(f"  [{'required' if req else 'optional'}] {path}: {what}")


if __name__ == "__main__":
    main()
