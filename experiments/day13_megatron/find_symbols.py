"""Day 13 - file:line of the Block 3 reading list in YOUR Megatron-LM checkout (no torch needed).

    python experiments/day13_megatron/find_symbols.py /workspace/Megatron-LM

A text search: `def`/`class` lines, then the first line matching a pattern after a given def. A navigation aid,
not proof of a call chain. The GUIDE's line numbers come from this script on core_v0.19.2 (commit 4b4acac9).
The profiler's Python frames name a function by the line of its FIRST DECORATOR when it has one (e.g. layers.py
backward is 581 in a trace, `def backward` is 583): Python's co_firstlineno.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PS = "megatron/core/parallel_state.py"
LAY = "megatron/core/tensor_parallel/layers.py"
MAP = "megatron/core/tensor_parallel/mappings.py"
CE = "megatron/core/tensor_parallel/cross_entropy.py"
SCHED = "megatron/core/pipeline_parallel/schedules.py"
P2P = "megatron/core/pipeline_parallel/p2p_communication.py"
GPT = "megatron/core/models/gpt/gpt_model.py"
LM = "megatron/core/models/common/language_module/language_module.py"
BLOCK = "megatron/core/transformer/transformer_block.py"
FIN = "megatron/core/distributed/finalize_model_grads.py"
SOFT = "megatron/core/fusions/fused_softmax.py"

DEFS = [
    (PS, ["generate_masked_orthogonal_rank_groups", "RankGenerator", "default_embedding_ranks",
          "default_position_embedding_ranks", "initialize_model_parallel", "get_model_parallel_group",
          "get_tensor_model_parallel_group", "get_pipeline_model_parallel_group", "get_data_parallel_group",
          "get_embedding_group", "get_position_embedding_group"]),
    (LAY, ["set_tensor_model_parallel_attributes", "VocabParallelEmbedding",
           "LinearWithGradAccumulationAndAsyncCommunication", "ColumnParallelLinear", "RowParallelLinear"]),
    (MAP, ["_reduce", "_gather_along_first_dim", "_reduce_scatter_along_first_dim",
           "copy_to_tensor_model_parallel_region", "reduce_from_tensor_model_parallel_region",
           "gather_from_sequence_parallel_region", "reduce_scatter_to_sequence_parallel_region"]),
    (CE, ["VocabParallelCrossEntropy", "_VocabParallelCrossEntropy", "vocab_parallel_cross_entropy"]),
    (GPT, ["GPTModel"]),
    (LM, ["compute_language_model_loss", "setup_embeddings_and_output_layer"]),
    (BLOCK, ["set_input_tensor"]),
    (SCHED, ["get_forward_backward_func", "forward_step_calc_loss", "forward_step", "backward_step",
             "forward_backward_no_pipelining", "get_tensor_shapes", "forward_backward_pipelining_without_interleaving"]),
    (P2P, ["_batched_p2p_ops", "P2PCommunicator", "_communicate", "send_forward_recv_backward",
           "send_backward_recv_forward"]),
    (FIN, ["_get_shared_word_embedding_weight", "_allreduce_word_embedding_grads",
           "_allreduce_non_tensor_model_parallel_grads", "finalize_model_grads"]),
]
# (file, owner def, regex, what) -> first matching line after the owner's def
INSIDE = [
    (PS, "initialize_model_parallel", r'order: str = "tp-cp-ep-dp-pp"', "default order, TP fastest"),
    (PS, "initialize_model_parallel", r"decoder_rank_generator = RankGenerator\(", "the rank generator"),
    (PS, "initialize_model_parallel", r"for ranks in decoder_rank_generator\.get_ranks\('tp-gtp_remat-pp'\)", "model-parallel groups (TP x PP)"),
    (PS, "initialize_model_parallel", r"for ranks in decoder_rank_generator\.get_ranks\('tp'\)", "TP groups"),
    (PS, "initialize_model_parallel", r"for ranks in decoder_rank_generator\.get_ranks\('pp'\)", "PP groups"),
    (PS, "initialize_model_parallel", r"embedding_ranks = get_embedding_ranks\(ranks", "embedding group per PP group"),
    (PS, "RankGenerator", r"def get_ranks", "get_ranks(token)"),
    (LAY, "VocabParallelEmbedding", r"output = reduce_from_tensor_model_parallel_region\(",
     "embedding: all-reduce of the partial lookups (SP off)"),
    (LAY, "VocabParallelEmbedding", r"output = reduce_scatter_to_sequence_parallel_region\(",
     "embedding: reduce-scatter instead (SP on)"),
    (LAY, "LinearWithGradAccumulationAndAsyncCommunication", r"handle = torch\.distributed\.all_reduce\(",
     "backward: async all-reduce of the input gradient (Day 11's f)"),
    (LAY, "LinearWithGradAccumulationAndAsyncCommunication", r"handle = _reduce_scatter_along_first_dim\(|"
     r"handle = dist_reduce_scatter_func\(", "backward with SP: reduce-scatter of the input gradient"),
    (LAY, "LinearWithGradAccumulationAndAsyncCommunication", r"CUDA_DEVICE_MAX_CONNECTIONS=1",
     "why CUDA_DEVICE_MAX_CONNECTIONS=1"),
    (LAY, "ColumnParallelLinear", r"self\.allreduce_dgrad = \(", "allreduce_dgrad when TP > 1 and SP off"),
    (LAY, "ColumnParallelLinear", r"input_parallel = copy_to_tensor_model_parallel_region\(", "forward: f (identity)"),
    (LAY, "RowParallelLinear", r"output_ = reduce_scatter_to_sequence_parallel_region\(", "forward with SP: reduce-scatter"),
    (LAY, "RowParallelLinear", r"output_ = reduce_from_tensor_model_parallel_region\(", "forward: g (all-reduce)"),
    (MAP, "_reduce", r"all_reduce\(", "the all-reduce call"),
    (CE, "VocabParallelCrossEntropy", r"all_reduce\(", "first all-reduce (max logit)"),
    (SCHED, "forward_step", r"set_input_tensor\(input_tensor\)", "received activation handed to the model"),
    (SCHED, "forward_step_calc_loss", r"output_tensor /= num_microbatches", "loss divided by num_microbatches"),
    (SCHED, "backward_step", r"torch\.autograd\.backward\(output_tensor\[0\]", "backward with grad_tensors"),
    (SCHED, "get_tensor_shapes", r"effective_seq_length = effective_seq_length // tp_group\.size\(\)",
     "SP divides the P2P sequence length by TP"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"num_warmup_microbatches = p2p_communicator\.total",
     "warmup = stages - stage - 1"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"send_forward_recv_backward\(",
     "steady: send activation + receive gradient"),
    (SCHED, "forward_backward_pipelining_without_interleaving", r"send_backward_recv_forward\(",
     "steady: send gradient + receive next activation"),
    (P2P, "_batched_p2p_ops", r"batch_isend_irecv\(ops\)", "one batch for send + recv"),
    (GPT, "GPTModel", r"    def forward", "GPTModel.forward"),
    (GPT, "GPTModel", r"self\.compute_language_model_loss\(", "labels given: per-token loss"),
    (LM, "compute_language_model_loss", r"vocab_parallel_cross_entropy\(|tensor_parallel\.vocab_parallel_cross_entropy",
     "vocab-parallel cross-entropy"),
    (LM, "setup_embeddings_and_output_layer", r"torch\.distributed\.all_reduce\(weight\.data",
     "tied embeddings only: copy stage 0's weight to the last stage"),
    (FIN, "_get_shared_word_embedding_weight", r"if model_module\.share_embeddings_and_output_weights",
     "untied: nothing to reduce across the embedding group"),
    (FIN, "_allreduce_non_tensor_model_parallel_grads", r"config\.sequence_parallel and getattr\(param, \"sequence_parallel\"",
     "SP: LayerNorm / row-bias grads all-reduced over TP"),
    (SOFT, "FusedScaleMaskSoftmax", r"mask is None and sq > 1", "causal mask built when attention_mask is None"),
]


def find_def(lines, symbol, start=1, toplevel=False):
    indent = "" if toplevel else r"\s*"
    defn = re.compile(rf"^{indent}(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b")
    for i, line in enumerate(lines[start - 1:], start):
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
    cache = {}
    missing = 0
    for rel, symbols in DEFS:
        lines = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8", errors="replace").splitlines())
        for s in symbols:
            i = find_def(lines, s, toplevel=rel == SCHED)
            missing += i is None
            print(f"| `{s}` | {rel.removeprefix('megatron/core/')}:{i if i else '(not found)'} |")
    print("\n| inside | location |\n| --- | --- |")
    for rel, owner, pattern, what in INSIDE:
        lines = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8", errors="replace").splitlines())
        start = find_def(lines, owner, toplevel=rel == SCHED) or 1
        hit = next((i for i, line in enumerate(lines[start - 1:], start) if re.search(pattern, line)), None)
        missing += hit is None
        print(f"| {owner}: {what} | {rel.removeprefix('megatron/core/')}:{hit if hit else '(not found)'} |")
    print(f"\n{missing} not found")


if __name__ == "__main__":
    main()
