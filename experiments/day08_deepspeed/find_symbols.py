"""Day 08 - file:line of today's DeepSpeed reading list (guide Block 3), in YOUR installed deepspeed, in reading order.

Same caveat as Days 02-07: a text search for def/class lines, a navigation aid, not proof of a call chain. Several
DeepSpeed classes define methods with the same name (DeepSpeedZeroOptimizer.step and DeepSpeedZeroOptimizer_Stage3.step,
reduce_ready_partitions_and_remove_grads in both); the search is per FILE, so the file column tells you which one you got.
DeepSpeed refactors often: a symbol printed as "not found" may have moved in your release (grep -rn).

  python experiments/day08_deepspeed/find_symbols.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from find_symbols import ROOT, locate  # noqa: E402  (ROOT = site-packages/torch; locate prints paths from site-packages)

import deepspeed  # noqa: E402

DS = Path(deepspeed.__file__).resolve().parent
if DS.parent != ROOT.parent:
    raise SystemExit(f"deepspeed ({DS}) and torch ({ROOT}) are in different site-packages; edit find_symbols.py")

RT = "runtime/"
Z = "runtime/zero/"
READING_LIST = [  # (file relative to the deepspeed package, [symbols]) in READING ORDER
    ("__init__.py", ["initialize"]),
    (RT + "engine.py", ["DeepSpeedEngine", "_configure_distributed_model", "_broadcast_model", "_configure_optimizer",
                        "_configure_zero_optimizer", "forward", "backward", "_backward_epilogue",
                        "allreduce_gradients", "buffered_allreduce_fallback", "is_gradient_accumulation_boundary",
                        "set_gradient_accumulation_boundary", "step", "_take_model_step", "no_sync"]),
    (Z + "config.py", ["DeepSpeedZeroConfig", "reduce_bucket_size", "use_multi_rank_bucket_allreduce",
                       "overlap_comm_valid", "prefetch_bucket_size", "param_persistence_threshold",
                       "max_reuse_distance"]),
    (Z + "stage_1_and_2.py", ["DeepSpeedZeroOptimizer", "create_gradient_handling_hooks", "process_gradients",
                              "setup_buckets", "reduce_gradients", "reduce_ready_partitions_and_remove_grads",
                              "reduce_independent_p_g_buckets_and_remove_grads", "reduce_ipg_grads",
                              "average_tensor", "allreduce_and_scatter", "copy_grads_in_partition", "step",
                              "_optimizer_step", "_update_model_bit16_weights"]),
    (RT + "utils.py", ["all_gather_dp_groups"]),
    (Z + "parameter_offload.py", ["DeepSpeedZeRoOffload", "mark_persistent_parameters", "setup_zero_stage3_hooks",
                                  "_register_deepspeed_module", "pre_sub_module_forward_function",
                                  "post_sub_module_forward_function", "pre_sub_module_backward_function"]),
    (Z + "partitioned_param_coordinator.py", ["PartitionedParameterCoordinator", "fetch_sub_module",
                                              "_fetch_sub_module_impl", "release_sub_module", "__all_gather_params_",
                                              "__params_to_release"]),
    (Z + "partition_parameters.py", ["Init", "_convert_to_deepspeed_param", "all_gather_coalesced", "partition",
                                     "_partition_param"]),
    (Z + "stage3.py", ["DeepSpeedZeroOptimizer_Stage3", "reduce_ready_partitions_and_remove_grads",
                       "__add_grad_to_ipg_bucket", "__reduce_and_partition_ipg_grads", "__avg_scatter_grads",
                       "partition_grads", "step", "_post_step"]),
    (RT + "comm/coalesced_collectives.py", ["reduce_scatter_coalesced"]),
    ("utils/tensor_fragment.py", ["safe_get_full_grad", "safe_get_full_fp32_param", "safe_get_full_optimizer_state"]),
]


def main() -> None:
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"deepspeed {deepspeed.__version__} at {DS}\n")
    print("| Symbol | Where in your install |\n| --- | --- |")
    for rel, symbols in READING_LIST:
        path = DS / rel
        for s in symbols:
            if needle and needle not in s:
                continue
            where = locate(path, s) if path.exists() else f"(file {rel} not in this release)"
            print(f"| `{s}` ({rel.rsplit('/', 1)[-1]}) | {where} |")


if __name__ == "__main__":
    main()
