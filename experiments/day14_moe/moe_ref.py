"""Day 14 - reference implementation. my_moe.py has the same five names for YOU WRITE; everything else here is shared.

The five pieces (GUIDE Block 2, in order):
  TODO 1  top1_route       router logits -> (expert index, combine weight, full softmax)
  TODO 2  permute          stable argsort by expert -> (order, tokens per expert)
          unpermute        the inverse: scatter expert outputs back to token order (index_add)
  TODO 3  exchange_counts  all_to_all of the [EP, E_local] count matrix: what every peer will send me
  TODO 4  AllToAllV        autograd Function: uneven all_to_all_single forward, the reverse exchange backward
  TODO 5  finish_grads     dense grads averaged over dense DP; expert grads scaled by 1/DP (and summed over EDP)

Shared (not YOU WRITE): sort_chunks (by source rank -> by local expert), run_experts (Megatron's SequentialMLP
loop), the split-metadata validator, aux / z losses and the aux-loss autoscaler, the capacity mask.

Rules the checker relies on: call collectives as `dist.all_to_all_single(...)` / `dist.all_reduce(...)` (module
attributes, so the collective counter sees them) and pass `group=` to every collective.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ TODO 1: gating
def top1_route(logits: torch.Tensor, forced: torch.Tensor | None = None):
    """logits [T, E] -> expert index [T] (long), combine weight [T], probs [T, E].

    The combine weight is the selected entry of the FULL softmax over all E experts. It is the only path from the
    task loss back to the router: the index itself is an argmax (piecewise constant, no gradient). Renormalizing a
    single top-1 weight to 1 (softmax over the selected logit alone, Megatron's default when pre-softmax is off)
    makes the weight a constant and cuts that path (GUIDE 1.1, Block 3 Stop 2).
    """
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1) if forced is None else forced.to(device=logits.device, dtype=torch.long)
    weight = probs.gather(1, idx.unsqueeze(1)).squeeze(1)
    return idx, weight, probs


# ------------------------------------------------------------------------------------------------ TODO 2: permutation
def permute(idx: torch.Tensor, n_experts: int):
    """idx [T] -> order [T], counts [E].

    order[k] = the token that goes to position k when tokens are grouped by expert (expert 0's tokens first, each
    group in original token order: a STABLE sort, so the layout does not depend on the sort implementation).
    x_sorted = x[order]. counts[e] = number of tokens routed to expert e (bincount, may contain zeros).
    """
    order = torch.argsort(idx, stable=True)
    counts = torch.bincount(idx, minlength=n_experts)
    return order, counts


def unpermute(y_sorted: torch.Tensor, order: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Inverse of x[order]: y[order[k]] = y_sorted[k]. Rows that no k points to (dropped tokens) stay zero.

    index_add_ is the scatter; its backward is a gather, grad_y_sorted = grad_y[order]. For top-1 every token appears
    at most once in order, so no two rows are added into the same destination (with top-2 they are: GUIDE 1.6).
    """
    out = y_sorted.new_zeros((n_tokens,) + tuple(y_sorted.shape[1:]))
    return out.index_add(0, order, y_sorted)


# ------------------------------------------------------------------------------------------------ TODO 3: counts
def exchange_counts(send_counts: torch.Tensor, group) -> torch.Tensor:
    """send_counts [EP, E_local]: row j = tokens I will send to each local expert of EP rank j.
    Returns recv_counts [EP, E_local]: row j = tokens EP rank j will send to each of MY local experts.

    Equal splits (E_local integers per peer), so the shapes are known in advance. This is the exchange that makes
    the payload shapes known: only after it can each rank allocate its receive buffer.
    """
    send = send_counts.contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv


# ------------------------------------------------------------------------------------------------ TODO 4: all_to_all
class AllToAllV(torch.autograd.Function):
    """Uneven all-to-all of rows. forward: send in_splits[j] rows to rank j, receive out_splits[j] rows from rank j.

    Backward is the same exchange in reverse: the gradient of the rows I received goes back to whoever sent them,
    so the split lists swap roles (Megatron's _AllToAll, tensor_parallel/mappings.py:424, backward :470).
    out_splits / in_splits are Python lists of ints: NCCL needs the sizes on the host when the call is enqueued.
    """

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), output_split_sizes=out_splits, input_split_sizes=in_splits,
                               group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = grad_out.new_empty((sum(ctx.in_splits),) + tuple(grad_out.shape[1:]))
        dist.all_to_all_single(grad_in, grad_out.contiguous(), output_split_sizes=ctx.in_splits,
                               input_split_sizes=ctx.out_splits, group=ctx.group)
        return grad_in, None, None, None


# ------------------------------------------------------------------------------------------------ TODO 5: gradients
def finish_grads(model: torch.nn.Module, groups) -> None:
    """After loss_r.backward() on every rank, where loss_r is the mean over rank r's OWN tokens (Day 02's rule).

    Global loss L = (1/DP) sum_r loss_r (equal token counts per rank).
      * Replicated (dense + router) params: rank r holds d loss_r / dp -> AVERAGE over the dense DP group.
      * Expert params: the backward all_to_all already delivered every source rank's d loss_r / d y rows to the
        expert's owner, so its grad is sum_r d loss_r / d W_e: multiply by 1/DP (NOT 1/EDP), then SUM over the
        EDP replicas (nothing to sum when EDP = 1). Megatron: expert_gradient_scaling_factor = 1/dp_size
        (distributed/distributed_data_parallel.py:213).
    Expert params carry `p.expert_parallel = True` (Megatron marks them with `allreduce = False`).
    """
    if groups is None or groups.world == 1:
        return
    dense = [p for p in model.parameters() if p.grad is not None and not getattr(p, "expert_parallel", False)]
    flat = torch.cat([p.grad.reshape(-1) for p in dense])          # one bucket (Day 03), one all-reduce
    dist.all_reduce(flat, group=groups.dp_group)
    flat /= groups.dp
    off = 0
    for p in dense:
        n = p.grad.numel()
        p.grad.copy_(flat[off:off + n].view_as(p.grad))
        off += n
    for p in model.parameters():
        if getattr(p, "expert_parallel", False) and p.grad is not None:
            p.grad /= groups.dp
            if groups.edp > 1:
                dist.all_reduce(p.grad, group=groups.edp_group)


# ================================================================================================= shared helpers
def sort_chunks(x: torch.Tensor, sizes: list[int], order: list[int]) -> torch.Tensor:
    """Split x into len(sizes) row chunks and concatenate them in `order` (Megatron's sort_chunks_by_idxs, :572).

    Used when a rank owns E_local > 1 experts: the all_to_all delivers chunks ordered (source rank, local expert);
    the experts want (local expert, source rank). torch.split/cat are differentiable, so backward is the inverse.
    """
    chunks = torch.split(x, sizes, dim=0)
    return torch.cat([chunks[i] for i in order], dim=0)


def chunk_orders(ep: int, e_loc: int) -> tuple[list[int], list[int]]:
    """Chunk index src * e_loc + le. by_expert: (src, le) -> (le, src); back: the inverse order."""
    by_expert = [src * e_loc + le for le in range(e_loc) for src in range(ep)]
    back = [le * ep + src for src in range(ep) for le in range(e_loc)]
    return by_expert, back


def run_experts(x_sorted: torch.Tensor, counts: list[int], experts) -> torch.Tensor:
    """Megatron's SequentialMLP: split by tokens-per-expert, run each expert on its chunk, concatenate.

    Every expert runs, even on an empty chunk, so its parameters get a ZERO gradient instead of None (an expert that
    received no token in this step must still take part in the gradient all-reduce / optimizer step).
    One GEMM pair per local expert: with many small experts the launches add up (GUIDE 4.2, grouped GEMM).
    """
    chunks = torch.split(x_sorted, counts, dim=0)
    return torch.cat([expert(c) for expert, c in zip(experts, chunks)], dim=0)


class SplitMetadataError(RuntimeError):
    pass


def split_problems(all_in: list, all_out: list, all_rows: list) -> list[str]:
    """Consistency of every rank's split lists, judged from all of them (pure Python, no communication).

    rank r sends in[r][j] rows to j; rank j must expect exactly that many from r: in[r][j] == out[j][r].
    Each in[r] must also sum to the rows rank r actually has, and every list must have one entry per EP rank.
    """
    ep = len(all_in)
    bad = []
    for r in range(ep):
        if len(all_in[r]) != ep or len(all_out[r]) != ep:
            bad.append(f"rank {r}: split lists have lengths {len(all_in[r])}/{len(all_out[r])}, expected {ep}")
            continue
        if sum(all_in[r]) != all_rows[r]:
            bad.append(f"rank {r}: input splits sum to {sum(all_in[r])} but it holds {all_rows[r]} rows")
    if not bad:
        for r in range(ep):
            for j in range(ep):
                if all_in[r][j] != all_out[j][r]:
                    bad.append(f"rank {r} sends {all_in[r][j]} rows to rank {j}, which expects {all_out[j][r]}")
    return bad


def validate_splits_collective(in_splits: list[int], out_splits: list[int], rows: int, group) -> None:
    """Gather every rank's split lists (a few ints) and check them ALL on EVERY rank before the payload exchange.

    Every rank evaluates the same predicate on the same data, so either all ranks continue or all raise at the
    same point: a deterministic verdict instead of a hung or corrupted NCCL all_to_all (GUIDE 2.4).
    """
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, (list(in_splits), list(out_splits), int(rows)), group=group)
    bad = split_problems([g[0] for g in gathered], [g[1] for g in gathered], [g[2] for g in gathered])
    if bad:
        raise SplitMetadataError("; ".join(bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))


def aux_loss(probs: torch.Tensor, counts: torch.Tensor, n_tokens: int, n_experts: int, topk: int = 1):
    """Switch load-balancing loss without alpha: E * sum_i f_i * P_i (Megatron's switch_load_balancing_loss_func).

    f_i = counts_i / (T * k): fraction of routed assignments that went to expert i. Integer counts: no gradient.
    P_i = mean router probability of expert i over the LOCAL tokens: the differentiable part.
    Pass GLOBAL counts and the GLOBAL token count to get the one-GPU value as the mean over ranks (GUIDE 2.3).
    Balanced (f = P = 1/E): 1. Collapsed onto one expert (f = P = one-hot): E.
    """
    f = counts.to(probs.dtype) / (n_tokens * topk)
    return n_experts * (f * probs.mean(dim=0)).sum()


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (ST-MoE): mean over tokens of logsumexp(logits)^2. Keeps logits small; Megatron z_loss_func."""
    return torch.logsumexp(logits, dim=-1).square().mean()


class AuxLossAutoScaler(torch.autograd.Function):
    """Identity on `output`; in backward, hands `aux` a gradient of `scale` (Megatron's MoEAuxLossAutoScaler, :284).

    Attaching the aux loss to the combine weights means the layer's OUTPUT and the logged task loss are unchanged,
    yet loss.backward() on the task loss also backpropagates d(aux)/d(router). scale = 1 here: one microbatch, and
    each rank's aux term is averaged over dense DP together with the task-loss gradient.
    """

    scale = 1.0

    @staticmethod
    def forward(ctx, output, aux):
        ctx.save_for_backward(aux)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (aux,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux) * AuxLossAutoScaler.scale


def capacity_keep_mask(idx: torch.Tensor, n_experts: int, cap: int, policy: str = "position",
                       weight: torch.Tensor | None = None) -> torch.Tensor:
    """Which tokens fit: each expert keeps at most `cap` of the tokens routed to it.

    position: the first `cap` tokens in token order (needs the STABLE sort; Megatron drop_policy="position").
    probs:    the `cap` tokens with the largest combine weight (Megatron's default drop_policy="probs").
    """
    keep = torch.zeros_like(idx, dtype=torch.bool)
    for e in range(n_experts):
        members = (idx == e).nonzero(as_tuple=True)[0]          # ascending token order
        if policy == "probs":
            members = members[torch.argsort(weight[members], descending=True, stable=True)]
        keep[members[:cap]] = True
    return keep


def renormalized_top1(logits: torch.Tensor, forced: torch.Tensor | None = None):
    """The variant to AVOID for top-1: softmax over the selected logit only -> weight == 1, no router gradient.

    This is what Megatron computes with moe_router_pre_softmax=False and topk=1 (moe_utils.py:710, softmax branch).
    """
    probs = torch.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1) if forced is None else forced.to(device=logits.device, dtype=torch.long)
    top = logits.gather(1, idx.unsqueeze(1))
    weight = torch.softmax(top, dim=-1).squeeze(1)              # == 1.0 exactly, d weight / d logits == 0
    return idx, weight, probs


def naive_moe(x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, experts, n_experts: int) -> torch.Tensor:
    """Oracle without any permutation: for each expert, a boolean mask, then write back with the same mask."""
    out = torch.zeros_like(x)
    for e in range(n_experts):
        m = idx == e
        out = out.index_put((m.nonzero(as_tuple=True)[0],), experts[e](x[m]))
    return out * weight.unsqueeze(1)


def gelu_mlp(x, w1, b1, w2, b2):
    return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)
