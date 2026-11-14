"""Day 11 - REFERENCE tensor-parallel MLP: column-parallel Linear -> GELU -> row-parallel Linear.

Read this AFTER your my_tp.py passes (or when you are stuck): it is the answer key for Block 2.

The two communication operators (Megatron's names in brackets; the Megatron-LM paper calls them f and g):

    CopyToTP     (f, _CopyToModelParallelRegion)      forward: identity           backward: all-reduce SUM
    ReduceFromTP (g, _ReduceFromModelParallelRegion)  forward: all-reduce SUM     backward: identity

They are conjugates: one is the other's transpose. Where the forward copies (replicates) a tensor, the backward
must add up the per-rank contributions to that tensor's gradient; where the forward sums partial results, the
backward hands the (already replicated) gradient to every rank unchanged.

One block on rank r of a TP group of size P (ffn_r = ffn / P), activations [seq, batch, .]:

    x  [s,b,d]       replicated
    x' = f(x)        replicated (same tensor; f only matters in backward)
    h_r = x' W1_r^T + b1_r     [s,b,ffn_r]   sharded   (W1_r = W1[r*ffn_r:(r+1)*ffn_r, :], rows of the weight)
    a_r = gelu(h_r)            [s,b,ffn_r]   sharded   (elementwise: no communication)
    z_r = a_r W2_r^T           [s,b,d]       PARTIAL SUM (W2_r = W2[:, r*ffn_r:(r+1)*ffn_r], columns of the weight)
    z  = g(z_r) = sum_r z_r    [s,b,d]       replicated
    y  = z + b2                [s,b,d]       replicated (b2 added once, after the sum)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tp_common import ACT


# ---------------------------------------------------------------------------------------------------------------
# Sharding: which slice of the full weight each rank owns
# ---------------------------------------------------------------------------------------------------------------

def shard_column_parallel(W: torch.Tensor, b: torch.Tensor | None, rank: int, tp: int):
    """Column-parallel: split the OUTPUT features, i.e. dim 0 of the PyTorch [out, in] weight, and the bias with it."""
    out_features = W.shape[0]
    assert out_features % tp == 0, f"out_features={out_features} not divisible by tp={tp}"
    n = out_features // tp
    rows = slice(rank * n, (rank + 1) * n)
    # .clone(): the shard must own its storage (a view would keep the full matrix alive and share it between layers)
    return W[rows].clone(), (b[rows].clone() if b is not None else None)


def shard_row_parallel(W: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    """Row-parallel: split the INPUT features, i.e. dim 1 of the PyTorch [out, in] weight. The bias is NOT split."""
    in_features = W.shape[1]
    assert in_features % tp == 0, f"in_features={in_features} not divisible by tp={tp}"
    n = in_features // tp
    return W[:, rank * n:(rank + 1) * n].contiguous()  # a column slice is strided: contiguous() copies it


# ---------------------------------------------------------------------------------------------------------------
# The two conjugate communication operators
# ---------------------------------------------------------------------------------------------------------------

def _all_reduce_sum(t: torch.Tensor, group) -> torch.Tensor:
    """Out-of-place SUM all-reduce. Megatron reduces in place to save memory (mappings.py `_reduce`); here a clone
    keeps autograd's tensors untouched, which is easier to reason about while learning."""
    out = t.contiguous().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


class CopyToTP(torch.autograd.Function):
    """f: identity forward, all-reduce SUM backward.

    Forward: every rank already holds the full x. Backward: rank r's dx is only the contribution through ITS shard
    of W1 (dx_r = dh_r @ W1_r), so the true dx = sum_r dx_r needs an all-reduce. SUM, not mean: the shards are
    different terms of one sum, not copies of one estimate.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)  # a new tensor object; the identity is what matters

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce_sum(grad_output, ctx.group), None


class ReduceFromTP(torch.autograd.Function):
    """g: all-reduce SUM forward, identity backward.

    Forward: z = sum_r z_r. Backward: dz_r = dz for every r (the derivative of a sum w.r.t. each term is 1), and dz
    is already replicated because every rank computed the same loss from the same y. Reducing it again would
    multiply every upstream gradient by the TP size.
    """

    @staticmethod
    def forward(ctx, x, group):
        return _all_reduce_sum(x, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ---------------------------------------------------------------------------------------------------------------
# The two layers
# ---------------------------------------------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Holds W[rows of this rank] and b[same rows]. Input replicated, output sharded along the last dim."""

    def __init__(self, weight_shard: torch.Tensor, bias_shard: torch.Tensor | None, group) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias_shard) if bias_shard is not None else None
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = CopyToTP.apply(x, self.group)              # f: marks where backward must sum dx over ranks
        return F.linear(x, self.weight, self.bias)     # [.., in] @ [out_r, in]^T -> [.., out_r]  (no communication)


class RowParallelLinear(nn.Module):
    """Holds W[:, columns of this rank] and the FULL bias. Input sharded along the last dim, output replicated."""

    def __init__(self, weight_shard: torch.Tensor, bias_full: torch.Tensor | None, group) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias_full) if bias_full is not None else None
        self.group = group

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x_shard, self.weight)       # [.., in_r] @ [out, in_r]^T -> [.., out], a PARTIAL sum
        out = ReduceFromTP.apply(partial, self.group)  # g: sum the partials -> replicated
        if self.bias is not None:
            out = out + self.bias                      # once, after the sum (adding before would count it P times)
        return out


# ---------------------------------------------------------------------------------------------------------------
# The model, built from any implementation module (ref, yours, a negative control)
# ---------------------------------------------------------------------------------------------------------------

class TPMLPBlock(nn.Module):
    def __init__(self, fc1: nn.Module, fc2: nn.Module) -> None:
        super().__init__()
        self.fc1, self.fc2 = fc1, fc2

    def forward(self, x):
        return self.fc2(ACT(self.fc1(x)))  # the activation runs on the shard: elementwise, so no communication


class TPMLP(nn.Module):
    def __init__(self, blocks: list[TPMLPBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def grads(self) -> list[dict]:
        """Per block: the LOCAL gradient of each parameter (a shard for W1, b1, W2; the full tensor for b2)."""
        out = []
        for blk in self.blocks:
            out.append({"W1": blk.fc1.weight.grad, "b1": blk.fc1.bias.grad,
                        "W2": blk.fc2.weight.grad, "b2": blk.fc2.bias.grad})
        return out


def build_tp_model(impl, problem: dict, rank: int, tp: int, group, device) -> TPMLP:
    """Shard the full parameters with `impl`'s shard functions and wrap them in `impl`'s layer classes.

    `impl` is any module with shard_column_parallel, shard_row_parallel, ColumnParallelLinear, RowParallelLinear.
    """
    blocks = []
    for p in problem["blocks"]:
        W1_r, b1_r = impl.shard_column_parallel(p["W1"], p["b1"], rank, tp)
        W2_r = impl.shard_row_parallel(p["W2"], rank, tp)
        fc1 = impl.ColumnParallelLinear(W1_r.to(device), b1_r.to(device), group)
        fc2 = impl.RowParallelLinear(W2_r.to(device), p["b2"].clone().to(device), group)
        blocks.append(TPMLPBlock(fc1, fc2))
    return TPMLP(blocks)
