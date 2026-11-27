"""Day 14 - file:line of the Block 3 reading list (MoE) in YOUR Megatron-LM checkout (no torch needed).

    python experiments/day14_moe/find_symbols.py /workspace/Megatron-LM

A text search: `def`/`class` lines, then the first line matching a pattern after a given def. A navigation aid,
not proof of a call chain. The GUIDE's line numbers come from this script on core_v0.19.2 (commit 4b4acac9).
Exit code 1 if any symbol or pattern is not found (another Megatron version: re-read the GUIDE's stops by name).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LAYER = "megatron/core/transformer/moe/moe_layer.py"
ROUTER = "megatron/core/transformer/moe/router.py"
UTILS = "megatron/core/transformer/moe/moe_utils.py"
DISP = "megatron/core/transformer/moe/token_dispatcher.py"
EXPERTS = "megatron/core/transformer/moe/experts.py"
SHARED = "megatron/core/transformer/moe/shared_experts.py"
MAP = "megatron/core/tensor_parallel/mappings.py"
PS = "megatron/core/parallel_state.py"
DDP = "megatron/core/distributed/distributed_data_parallel.py"

DEFS = [
    (LAYER, ["MoELayer"]),
    (ROUTER, ["TopKRouter", "gating", "_apply_aux_loss", "_apply_global_aux_loss", "attach_and_log_load_balancing_loss",
              "apply_z_loss", "_apply_expert_bias", "routing", "forward"]),
    (UTILS, ["switch_load_balancing_loss_func", "z_loss_func", "sinkhorn", "get_capacity", "MoEAuxLossAutoScaler",
             "permute", "unpermute", "sort_chunks_by_idxs", "topk_routing_with_score_function",
             "apply_router_token_dropping", "get_updated_expert_bias"]),
    (DISP, ["MoEAllGatherTokenDispatcher", "MoEAlltoAllTokenDispatcher", "preprocess", "dispatch_preprocess",
            "token_dispatch", "dispatch_postprocess", "token_combine", "_maybe_dtoh_and_synchronize",
            "MoEFlexTokenDispatcher"]),
    (EXPERTS, ["TEGroupedMLP", "SequentialMLP"]),
    (SHARED, ["SharedExpertMLP"]),
    (MAP, ["_AllToAll", "all_to_all"]),
    (PS, ["initialize_model_parallel"]),
]
# (file, owner def, regex, what) -> first matching line after the owner's def
INSIDE = [
    # Stop 1: one MoE layer, router, top-k
    (LAYER, "MoELayer", r"    def forward\(", "MoELayer.forward"),
    (LAYER, "MoELayer", r"    def route\(", "route"),
    (LAYER, "MoELayer", r"    def routed_experts_compute\(", "routed_experts_compute"),
    (UTILS, "topk_routing_with_score_function", r"if use_pre_softmax:", "softmax BEFORE top-k (pre_softmax=True)"),
    (UTILS, "topk_routing_with_score_function", r"probs = torch\.softmax\(scores, dim=-1", "default: softmax AFTER top-k"),
    (UTILS, "topk_routing_with_score_function", r"probs = scores / \(scores\.sum", "sigmoid: renormalize only if topk > 1"),
    # Stop 2: aux loss, scope, AutoScaler, z-loss
    (UTILS, "switch_load_balancing_loss_func", r"aggregated_probs_per_expert = probs\.sum", "sum_i P_i * f_i"),
    (UTILS, "MoEAuxLossAutoScaler", r"    def backward", "AutoScaler.backward"),
    (ROUTER, "_apply_aux_loss", r"reduce_group=self\.tp_cp_group", "aux_loss: counts reduced over tp_cp only"),
    (ROUTER, "_apply_global_aux_loss", r"reduce_group=self\.tp_dp_cp_group", "global_aux_loss: also over DP"),
    (ROUTER, "attach_and_log_load_balancing_loss", r"activation = MoEAuxLossAutoScaler\.apply\(", "aux attached to probs"),
    (ROUTER, "apply_z_loss", r"z_loss = z_loss_func\(", "z-loss on the logits"),
    # Stop 3: capacity and dropping
    (UTILS, "get_capacity", r"capacity = math\.ceil", "C = ceil(T / E * cf)"),
    (UTILS, "apply_router_token_dropping", r"expert_capacity = get_capacity\(", "capacity from the LOCAL tokens"),
    (UTILS, "apply_router_token_dropping", r"capacity_indices = torch\.topk\(routing_probs", "drop policy 'probs'"),
    # Stop 4: counts, splits, host sync
    (DISP, "preprocess", r"self\.num_out_tokens = routing_map\.size\(0\) \* self\.config\.moe_router_topk",
     "dropless: static output size"),
    (DISP, "preprocess", r"self\.input_splits = num_local_tokens_per_expert\.reshape", "input_splits (per dst rank)"),
    (DISP, "preprocess", r"gather_from_sequence_parallel_region\(", "all-gather of every rank's counts"),
    (DISP, "preprocess", r"self\.output_splits = num_global_tokens_per_rank", "output_splits (per src rank)"),
    (DISP, "preprocess", r'_maybe_update_cuda_sync_point\("before_ep_alltoall"\)', "sync point before the a2a"),
    (DISP, "_maybe_dtoh_and_synchronize", r"if point == self\.cuda_sync_point", "where the host waits"),
    # Stop 5: permute, a2a, sort_chunks, combine
    (DISP, "dispatch_preprocess", r"= permute\(", "permutation 1 (by expert)"),
    (DISP, "token_dispatch", r"global_input_tokens = all_to_all\(", "dispatch a2a (tokens)"),
    (DISP, "token_dispatch", r"global_probs = all_to_all\(", "dispatch a2a (probs travel too)"),
    (DISP, "dispatch_postprocess", r"sort_chunks_by_idxs\(", "(src, expert) -> (expert, src)"),
    (DISP, "token_combine", r"= all_to_all\(", "combine a2a: splits swapped"),
    (DISP, "token_combine", r"output = unpermute\(", "unpermute (combine_postprocess)"),
    (MAP, "_AllToAll", r"    def backward", "_AllToAll.backward"),
    (UTILS, "unpermute", r"output_tokens\.index_add_\(", "index_add (deterministic mode)"),
    (UTILS, "sort_chunks_by_idxs", r"\.tolist\(\)", "split sizes to the host"),
    # Stop 6: expert gradients and groups
    (DDP, "DistributedDataParallel", r"expert_gradient_scaling_factor = self\.expt_dp_group\.size\(\) / self\.dp_cp_group",
     "average_in_collective: edp / dp"),
    (DDP, "DistributedDataParallel", r"expert_gradient_scaling_factor = 1\.0 / data_parallel_world_size",
     "default: 1 / dp"),
    (DDP, "DistributedDataParallel", r"is_expert_parallel = not getattr\(param, 'allreduce', True\)",
     "expert params: allreduce=False"),
    (PS, "initialize_model_parallel", r"expert_data_parallel_size = world_size //", "EDP = world / (etp*ep*pp)"),
    (PS, "initialize_model_parallel", r"expert_decoder_rank_generator = RankGenerator\(", "separate expert generator"),
    (PS, "initialize_model_parallel", r"for ranks in expert_decoder_rank_generator\.get_ranks\('ep'\)", "EP groups"),
    (PS, "initialize_model_parallel", r"for ranks in expert_decoder_rank_generator\.get_ranks\(\"dp\"\)", "EDP groups"),
    # Stop 7 (optional): expert compute, shared experts, bias, sinkhorn
    (EXPERTS, "TEGroupedMLP", r"tokens_per_expert: list\[int\] = tokens_per_expert\.tolist\(\)", "grouped GEMM splits"),
    (EXPERTS, "SequentialMLP", r"tokens_per_expert = tokens_per_expert\.tolist\(\)", "loop over local experts"),
    (UTILS, "get_updated_expert_bias", r"torch\.sign\(offset\)", "aux-loss-free bias update"),
    (ROUTER, "sinkhorn_load_balancing", r"sinkhorn\(", "sinkhorn (moe_router_load_balancing_type='sinkhorn')"),
    (ROUTER, "routing", r"topk_routing_with_score_function\(", "TopKRouter.routing -> top-k"),
    (ROUTER, "routing", r"apply_router_token_dropping\(", "TopKRouter.routing -> capacity"),
]


def start_of(lines, rel, symbol) -> int:
    """Search methods inside the class that matters: the ALL-TO-ALL dispatcher (the all-gather one comes first)
    and TopKRouter (the base Router defines its own routing/forward; gating stays the base's)."""
    owner = {DISP: "MoEAlltoAllTokenDispatcher", ROUTER: "TopKRouter"}.get(rel)
    if owner is None or symbol == owner or symbol == "gating" or not symbol[0].islower() and rel == DISP:
        return 1
    return find_def(lines, owner) or 1


def find_def(lines, symbol, start=1):
    defn = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    for i, line in enumerate(lines[start - 1:], start):
        if defn.search(line):
            return i
    return None


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/Megatron-LM")
    if not (root / "megatron/core/transformer/moe").is_dir():
        raise SystemExit(f"{root}/megatron/core/transformer/moe not found. Usage: find_symbols.py PATH_TO_MEGATRON_LM")
    try:
        desc = subprocess.run(["git", "-C", str(root), "describe", "--tags"], capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        desc = head = "?"
    print(f"Megatron-LM at {root}: {desc or '?'} {head or ''}\n")
    print("| symbol | location |\n| --- | --- |")
    cache = {}
    missing = 0
    for rel, symbols in DEFS:
        lines = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8", errors="replace").splitlines())
        for s in symbols:
            i = find_def(lines, s, start_of(lines, rel, s))
            missing += i is None
            print(f"| `{s}` | {rel.removeprefix('megatron/core/')}:{i if i else '(not found)'} |")
    print("\n| inside | location |\n| --- | --- |")
    for rel, owner, pattern, what in INSIDE:
        lines = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8", errors="replace").splitlines())
        start = find_def(lines, owner, start_of(lines, rel, owner)) or 1
        hit = next((i for i, line in enumerate(lines[start - 1:], start) if re.search(pattern, line)), None)
        missing += hit is None
        print(f"| {owner}: {what} | {rel.removeprefix('megatron/core/')}:{hit if hit else '(not found)'} |")
    print(f"\n{missing} not found")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
