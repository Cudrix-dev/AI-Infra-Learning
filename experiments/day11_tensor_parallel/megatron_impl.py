"""Day 11 (optional) - the same MLP built from Megatron Core's ColumnParallelLinear and RowParallelLinear.

Used by `check_tp_parity.py --impl megatron`: same baseline tensors, same shards (tp_ref's shard functions), same
checks and the same collective counter. Needs a Megatron-LM checkout on PYTHONPATH (no pip install, no Transformer
Engine, no apex):

    git clone --depth 1 --branch core_v0.19.2 https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM
    export PYTHONPATH=/workspace/Megatron-LM CUDA_DEVICE_MAX_CONNECTIONS=1

Mapping to your code (GUIDE Block 3):
  ColumnParallelLinear(gather_output=False)          = your ColumnParallelLinear; f is NOT a separate Function here:
                                                       with TP > 1 Megatron passes allreduce_dgrad=True and launches
                                                       the dx all-reduce inside the linear's own backward (async).
  RowParallelLinear(input_is_parallel=True,
                    skip_bias_add=False)              = your RowParallelLinear; g = reduce_from_tensor_model_parallel_region.
"""

from __future__ import annotations

import torch

import tp_ref
from tp_common import ACT


def setup(tp: int):
    """Megatron's process groups for TP = tp (DP = PP = 1). Returns the TP group."""
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, create_gloo_process_groups=False)
    return parallel_state.get_tensor_model_parallel_group()


class MegatronBlock(tp_ref.TPMLPBlock):
    def forward(self, x):
        h, _ = self.fc1(x)          # Megatron layers return (output, bias-or-None)
        out, _ = self.fc2(ACT(h))
        return out


def build(problem: dict, rank: int, tp: int, group, device) -> tp_ref.TPMLP:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    dtype = problem["x"].dtype
    config = ModelParallelConfig(tensor_model_parallel_size=tp, params_dtype=dtype, perform_initialization=False,
                                 use_cpu_initialization=False, gradient_accumulation_fusion=False,
                                 sequence_parallel=False)
    no_init = lambda w: None  # noqa: E731  (weights are copied in below)
    blocks = []
    for p in problem["blocks"]:
        ffn, d = p["W1"].shape
        fc1 = ColumnParallelLinear(d, ffn, config=config, init_method=no_init, bias=True, gather_output=False,
                                   skip_bias_add=False, tp_group=group)
        fc2 = RowParallelLinear(ffn, d, config=config, init_method=no_init, bias=True, input_is_parallel=True,
                                skip_bias_add=False, tp_group=group)
        W1_r, b1_r = tp_ref.shard_column_parallel(p["W1"], p["b1"], rank, tp)
        W2_r = tp_ref.shard_row_parallel(p["W2"], rank, tp)
        with torch.no_grad():
            for param, val in ((fc1.weight, W1_r), (fc1.bias, b1_r), (fc2.weight, W2_r), (fc2.bias, p["b2"])):
                assert param.shape == val.shape, (param.shape, val.shape)
                param.copy_(val.to(device))
        blocks.append(MegatronBlock(fc1, fc2))
    return tp_ref.TPMLP(blocks).to(device)
