"""Day 11 - negative controls: the reference with ONE line changed each. The parity check must catch every one.

Each control is a small module-like namespace with the four names build_tp_model() needs. EXPECTED lists, per
number of blocks, the checks that MUST fail; every other check must still pass. A control whose failures do not
match exactly is reported as "not as predicted", which means either the harness or the explanation is wrong.

Check names: y (forward output), dx (input gradient), replicated (y, dx and every b2 gradient bitwise identical on
all ranks), and per block k: blk{k}.W1, blk{k}.b1, blk{k}.W2, blk{k}.b2 (gathered gradients vs the single-GPU baseline).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

import tp_ref
from tp_ref import ReduceFromTP, _all_reduce_sum


# 1. no_dx_allreduce: f's backward forgets the all-reduce. Each rank keeps only its own partial dx.
class CopyToTP_NoReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None                          # <- the bug: partial dx_r returned as if it were dx


class ColumnNoReduce(tp_ref.ColumnParallelLinear):
    def forward(self, x):
        return F.linear(CopyToTP_NoReduce.apply(x, self.group), self.weight, self.bias)


# 2. g_bwd_allreduce: "all-reduce in both directions". g's backward reduces a gradient that is already replicated.
class ReduceFromTP_Twice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _all_reduce_sum(x, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce_sum(grad_output, ctx.group), None   # <- the bug: sums P identical copies -> P x dz


class RowTwice(tp_ref.RowParallelLinear):
    def forward(self, x_shard):
        return ReduceFromTP_Twice.apply(F.linear(x_shard, self.weight), self.group) + self.bias


# 3. avg_dx: the DDP habit. f's backward averages instead of summing.
class CopyToTP_Avg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        tp = dist.get_world_size(ctx.group)
        return _all_reduce_sum(grad_output, ctx.group) / tp, None   # <- the bug: mean over ranks -> dx / P


class ColumnAvg(tp_ref.ColumnParallelLinear):
    def forward(self, x):
        return F.linear(CopyToTP_Avg.apply(x, self.group), self.weight, self.bias)


# 4. bias_before_reduce: the row-parallel bias is added to every rank's partial sum, i.e. P times in total.
class RowBiasBefore(tp_ref.RowParallelLinear):
    def forward(self, x_shard):
        return ReduceFromTP.apply(F.linear(x_shard, self.weight, self.bias), self.group)   # <- bias inside the sum


def _impl(column=tp_ref.ColumnParallelLinear, row=tp_ref.RowParallelLinear):
    return SimpleNamespace(shard_column_parallel=tp_ref.shard_column_parallel,
                           shard_row_parallel=tp_ref.shard_row_parallel,
                           ColumnParallelLinear=column, RowParallelLinear=row)


CONTROLS = {
    "no_dx_allreduce": _impl(column=ColumnNoReduce),
    "g_bwd_allreduce": _impl(row=RowTwice),
    "avg_dx": _impl(column=ColumnAvg),
    "bias_before_reduce": _impl(row=RowBiasBefore),
}

_ALL1 = ["blk0.W1", "blk0.b1", "blk0.W2", "blk0.b2"]
_ALL2 = _ALL1 + ["blk1.W1", "blk1.b1", "blk1.W2", "blk1.b2"]

# Checks that MUST fail (everything else must pass), keyed by number of blocks. Derivations: GUIDE 2.4 and 3.3.
EXPECTED = {
    "no_dx_allreduce": {
        1: ["dx", "replicated"],
        2: ["dx", "replicated", "blk0.W1", "blk0.b1", "blk0.W2", "blk0.b2"],   # block 1's partial dx is block 0's dy
    },
    "g_bwd_allreduce": {
        1: ["dx", "blk0.W1", "blk0.b1", "blk0.W2"],                       # x2; blk0.b2 is outside g, so it stays right
        2: ["dx", "blk0.W1", "blk0.b1", "blk0.W2", "blk0.b2", "blk1.W1", "blk1.b1", "blk1.W2"],   # x4 in block 0, x2 in block 1
    },
    "avg_dx": {
        1: ["dx"],                                                  # x1/2
        2: ["dx"] + _ALL1,                                          # block 0 x1/2, dx x1/4
    },
    "bias_before_reduce": {
        1: ["y", "dx"] + _ALL1,                                     # forward off by (P-1) b2; MSE passes it back
        2: ["y", "dx"] + _ALL2,
    },
}

# Collectives per phase that each control should issue for ONE block (all-reduce counts): (forward, backward)
EXPECTED_COUNTS_PER_BLOCK = {
    "ref": (1, 1), "mine": (1, 1), "megatron": (1, 1),
    "no_dx_allreduce": (1, 0), "g_bwd_allreduce": (1, 2), "avg_dx": (1, 1), "bias_before_reduce": (1, 1),
}
