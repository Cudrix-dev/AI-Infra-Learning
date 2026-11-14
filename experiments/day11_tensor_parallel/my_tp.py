"""Day 11 - YOUR tensor-parallel MLP. Fill in the TODOs in the order below (GUIDE Block 2).

Test after each step (all from the project root, on the dual-A100 host):

  Step 1  shard_column_parallel, shard_row_parallel
  Step 2  CopyToTP (f), ReduceFromTP (g)
          -> torchrun --standalone --nproc_per_node=2 experiments/day11_tensor_parallel/check_tp_parity.py --impl mine --test unit
             (unit tests print one line per function; a function still raising NotImplementedError exits with code 3)
  Step 3  ColumnParallelLinear.forward, RowParallelLinear.forward
          -> python experiments/day11_tensor_parallel/baseline_1gpu.py --run-id try1            (once; GPU 0)
          -> torchrun --standalone --nproc_per_node=2 experiments/day11_tensor_parallel/check_tp_parity.py \
                 --impl mine --test parity --baseline outputs/day11_tp/tensors/baseline_float32_b1_try1.pt

The model wrapper (TPMLPBlock, TPMLP, build_tp_model) is shared with the reference in tp_ref.py; you only write the
pieces below. Keep the names and signatures: the checker imports them by name.

Rules the checker relies on:
  * call collectives as `dist.all_reduce(...)` (module attribute), so the collective counter sees them;
  * pass `group=` to every collective;
  * do not change the input tensor of a Function in place (clone first).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------------
# Step 1: sharding. PyTorch weights are [out_features, in_features].
# ---------------------------------------------------------------------------------------------------------------


def shard_column_parallel(W: torch.Tensor, b: torch.Tensor | None, rank: int, tp: int):
    """Return (W_r, b_r): this rank's slice of a COLUMN-parallel linear layer.

    W: [out, in] full weight, b: [out] full bias (or None). Expected shapes: W_r [out/tp, in], b_r [out/tp].
    """
    # TODO(1a): which dimension of W holds the OUTPUT features? Split that one into tp equal contiguous chunks and
    #           keep chunk `rank`. Split the bias the same way (it has one entry per output feature).
    # TODO(1b): return tensors that own their storage (.clone()), not views of the full matrix.
    # TODO(1c): assert that the split dimension is divisible by tp.
    assert W.shape[0] % tp == 0  # compute number or rows per rank
    rows_per_rank = W.shape[0] // tp
    start = rank * rows_per_rank
    stop = start + rows_per_rank
    weight_shard = W[start:stop, :].clone()
    bias_shard = b[start:stop].clone() if b is not None else None
    return weight_shard, bias_shard


def shard_row_parallel(W: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    """Return W_r: this rank's slice of a ROW-parallel linear layer. Expected shape: [out, in/tp].

    The bias of a row-parallel layer is NOT sharded (why? GUIDE 1.2), so there is no bias argument.
    """
    # TODO(1d): split the INPUT-feature dimension of W into tp contiguous chunks; keep chunk `rank`.
    # TODO(1e): a slice along dim 1 is not contiguous; return a contiguous copy.
    assert W.shape[1] % tp == 0  # compute number of columns per rank
    cols_per_rank = W.shape[1] // tp
    start = rank * cols_per_rank
    stop = start + cols_per_rank
    return W[:, start:stop].clone(memory_format=torch.contiguous_format)


# ---------------------------------------------------------------------------------------------------------------
# Step 2: the two conjugate communication operators.
# ---------------------------------------------------------------------------------------------------------------


class CopyToTP(torch.autograd.Function):
    """f: placed in front of the column-parallel layer. Its input x is the same full tensor on every rank."""

    @staticmethod
    def forward(ctx, x, group):
        # TODO(2a): what must the forward do to an input every rank already holds in full?
        #           Remember `group` on ctx for the backward.
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # TODO(2b): grad_output on rank r is only the contribution to dx through rank r's slice of W1.
        #           Turn it into the full dx on every rank. Which reduce op: SUM or AVG? (GUIDE 2.3)
        #           Work on a clone; return one gradient per forward input (x, group) -> (dx, None).
        dx = grad_output.clone()
        dist.all_reduce(dx, op=dist.ReduceOp.SUM, group=ctx.group)
        return dx, None


class ReduceFromTP(torch.autograd.Function):
    """g: placed after the row-parallel matmul. Its input on rank r is a partial sum z_r of the output."""

    @staticmethod
    def forward(ctx, x, group):
        # TODO(2c): produce z = sum over ranks of z_r, identical on every rank. Do not modify x in place.
        result = x.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # TODO(2d): dz is already identical on every rank (same loss, same y). What is d(sum_r z_r)/d z_r?
        #           Does this need any communication? (GUIDE 2.2, prediction P5)
        return grad_output, None


# ---------------------------------------------------------------------------------------------------------------
# Step 3: the two layers. __init__ is given; write forward.
# ---------------------------------------------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """weight: [out/tp, in] (this rank's rows), bias: [out/tp]. Input replicated [.., in] -> output sharded [.., out/tp]."""

    def __init__(
        self, weight_shard: torch.Tensor, bias_shard: torch.Tensor | None, group
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias_shard) if bias_shard is not None else None
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(3a): pass x through CopyToTP (with self.group), then a local F.linear with this rank's weight and bias.
        #           No forward communication here.
        x_for_tp = CopyToTP.apply(x, self.group)
        return F.linear(x_for_tp, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """weight: [out, in/tp] (this rank's columns), bias: [out] (full). Input sharded [.., in/tp] -> output replicated [.., out]."""

    def __init__(
        self, weight_shard: torch.Tensor, bias_full: torch.Tensor | None, group
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias_full) if bias_full is not None else None
        self.group = group

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        # TODO(3b): local F.linear WITHOUT the bias -> a partial sum of the output.
        # TODO(3c): ReduceFromTP over self.group -> the full output on every rank.
        # TODO(3d): add the bias exactly once. Where? (GUIDE 1.2, prediction P3)
        partial = F.linear(x_shard, self.weight, bias=None)
        result = ReduceFromTP.apply(partial, self.group)
        if self.bias is not None:
            result += self.bias
        return result
