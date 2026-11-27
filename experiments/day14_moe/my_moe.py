"""Day 14 - YOUR top-1 MoE pieces. Fill in the TODOs in order (GUIDE Block 2). Keep the names and signatures.

Test after each stage, from the project root (CPU works for every stage; 4 local gloo processes stand in for GPUs):

  Stage 1  TODO 1 top1_route, TODO 2 permute + unpermute
           -> python experiments/day14_moe/check_local.py --impl mine
  Stage 2  TODO 3 exchange_counts, TODO 4 AllToAllV
           -> python experiments/day14_moe/check_ep.py --impl mine --cpu --nproc 4 --cases ragged --tests dispatch
  Stage 3  TODO 5 finish_grads, then the full parity check
           -> python experiments/day14_moe/baseline_1gpu.py --cpu --run-id try1
           -> python experiments/day14_moe/check_ep.py --impl mine --cpu --nproc 4 \
                  --baseline outputs/day14_moe/tensors/baseline_small_try1.pt
           On the four-GPU host: the same two commands without --cpu --nproc 4, under torchrun (GUIDE Block 4).

While any function below still contains `raise NotImplementedError`, check_ep.py prints `INCOMPLETE (TODOs left)`
and exits with code 3 before it creates a process group; check_local.py tests what is finished and exits 3 at the end.

Rules the checker relies on:
  * call collectives as `dist.all_to_all_single(...)` / `dist.all_reduce(...)` (module attributes: counted);
  * pass `group=` to every collective;
  * do not modify input tensors in place (autograd may have saved them).
The reference is moe_ref.py. Read it after your version runs, or when stuck.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------------------------- TODO 1 (10 min)
def top1_route(logits: torch.Tensor, forced: torch.Tensor | None = None):
    """logits [T, E] -> (idx [T] long, weight [T], probs [T, E]).

    probs:  softmax over ALL E experts (dim -1).
    idx:    the argmax of probs, unless `forced` is given (a [T] tensor of expert ids: use it as is).
    weight: the combine weight of the chosen expert. GUIDE 1.1 / P1: which of these keeps a gradient path from
            the task loss to the router, and why must the weight NOT be renormalized to 1 for top-1?
    """
    # TODO(1a): probs = softmax of logits over the expert dimension.
    # TODO(1b): idx = argmax, or forced (move it to logits.device, dtype long).
    # TODO(1c): weight = probs at (token, idx) -> shape [T]. Hint: gather along dim 1 with idx.unsqueeze(1).
    probs = torch.softmax(logits, dim=-1)
    if forced is None:
        weight, idx = probs.max(dim=-1)
    else:
        idx = forced.to(logits.device, torch.long)
        weight = probs.gather(dim=1, index=idx.unsqueeze(1)).squeeze(1)
    return idx, weight, probs


# ---------------------------------------------------------------------------------------------- TODO 2 (25 min)
def permute(idx: torch.Tensor, n_experts: int):
    """idx [T] (expert of each token) -> (order [T] long, counts [E] long).

    order: x[order] groups the tokens by expert: all of expert 0 first (in original token order), then expert 1, ...
    counts: tokens per expert, including zeros for experts that got none.
    GUIDE 1.3 / P3. Why must the sort be STABLE for the layout to be reproducible?
    """
    # TODO(2a): order = a stable argsort of idx.
    # TODO(2b): counts = a bincount of idx with minlength=n_experts (so idle experts appear as 0).
    order = torch.argsort(
        idx, stable=True
    )  # [2, 0, 3, 0, 2, 0, 3, 3] -> [1, 3, 5, 0, 4, 2, 6, 7]
    counts = torch.bincount(idx, minlength=n_experts)
    return order, counts


def unpermute(
    y_sorted: torch.Tensor, order: torch.Tensor, n_tokens: int
) -> torch.Tensor:
    """The inverse of x_sorted = x[order]: return y [n_tokens, ...] with y[order[k]] = y_sorted[k].

    len(order) may be smaller than n_tokens (dropped tokens): their rows must be exactly zero.
    Do NOT write y_sorted[order] (P3: on which inputs would that wrong inverse still pass?).
    """
    # TODO(2c): start from zeros of shape (n_tokens, *y_sorted.shape[1:]) and scatter y_sorted into rows `order`.
    #           Use an OUT-OF-PLACE differentiable op (index_add / index_copy / index_put on a fresh zeros tensor).
    y = y_sorted.new_zeros((n_tokens, *y_sorted.shape[1:]))
    """
    equal to:
    for k in range(y_sorted.shape[0]):
        y[order[k]] = y_sorted[k]
    """
    return y.index_copy_(0, order, y_sorted)


# ---------------------------------------------------------------------------------------------- TODO 3 (15 min)
def exchange_counts(send_counts: torch.Tensor, group) -> torch.Tensor:
    """send_counts [EP, E_local] (int64): row j = how many tokens I send to each local expert of EP rank j.
    Return recv_counts [EP, E_local]: row j = how many tokens EP rank j sends to each of MY local experts.

    One all_to_all with EQUAL splits: every peer gets exactly one row of E_local integers. Do not call .tolist()
    here; the layer does one device -> host copy for all sizes afterwards (GUIDE 2.2, P4).
    """
    # TODO(3): allocate the receive tensor (same shape/dtype/device) and call dist.all_to_all_single(..., group=group).
    recv_counts = send_counts.new_empty(send_counts.shape)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return recv_counts


# ---------------------------------------------------------------------------------------------- TODO 4 (25 min)
class AllToAllV(torch.autograd.Function):
    """Uneven all-to-all of rows, differentiable.

    forward(x, out_splits, in_splits, group): x has sum(in_splits) rows; send in_splits[j] rows to EP rank j and
    receive out_splits[j] rows from rank j. Return a tensor with sum(out_splits) rows (same trailing dims as x).
    backward(grad_out): which rows must go back to whom? GUIDE 2.2 / P4: write the split lists of the backward call.
    out_splits / in_splits are Python lists of ints (the host needs them to enqueue the call).
    """

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        # TODO(4a): save what backward needs on ctx.
        # TODO(4b): out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:])); then
        #           dist.all_to_all_single(out, x.contiguous(), output_split_sizes=..., input_split_sizes=...,
        #           group=group); return out.
        ctx.out_splits = out_splits
        ctx.in_splits = in_splits
        ctx.group = group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(
            out,
            x.contiguous(),
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=group,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # TODO(4c): allocate the input-gradient tensor (how many rows?) and run the reverse exchange.
        # TODO(4d): return one value per forward input: (grad_x, None, None, None).
        grad_x = grad_out.new_empty((sum(ctx.in_splits),) + tuple(grad_out.shape[1:]))
        dist.all_to_all_single(
            grad_x,
            grad_out.contiguous(),
            output_split_sizes=ctx.in_splits,
            input_split_sizes=ctx.out_splits,
            group=ctx.group,
        )
        return (
            grad_x,
            None,
            None,
            None,
        )  # return four values because forward had four inputs (x, out_splits, in_splits, group), but only x has a gradient


# ---------------------------------------------------------------------------------------------- TODO 5 (20 min)
def finish_grads(model: torch.nn.Module, groups) -> None:
    """Called after loss_r.backward() on every rank (loss_r = mean over THIS rank's tokens). Make every gradient
    equal to the gradient of the GLOBAL mean loss L = (1/DP) * sum_r loss_r (GUIDE 3.2, P6).

    groups (moe_common.Groups): .world == .dp (dense DP size), .dp_group, .edp, .edp_group.
    Expert parameters have `p.expert_parallel = True`; every other parameter is replicated (dense).
      * dense params:  what must happen across the dense DP group?
      * expert params: the backward all_to_all already summed the contributions of every source rank into the
                       owner's .grad. What factor is still missing? Over which group (if any) do you reduce?
    Do nothing when groups is None or groups.world == 1 (the one-GPU baseline).
    """
    # TODO(5a): return early for the baseline.
    # TODO(5b): dense grads -> all_reduce (SUM) over groups.dp_group, then divide by groups.dp.
    #           (Per parameter is fine; one flat bucket is what Day 03 built.)
    # TODO(5c): expert grads -> the missing factor, and the EDP reduction when groups.edp > 1.
    if groups is None or groups.world == 1:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        if getattr(p, "expert_parallel", False):
            if groups.edp > 1:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=groups.edp_group)
        else:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=groups.dp_group)
        p.grad.div_(groups.dp)
