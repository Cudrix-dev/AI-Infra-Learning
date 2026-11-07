"""Day 07 - the part you write: put Day 01's model under FSDP2 (`fully_shard`), whole model vs per block.

FSDP2 is ZeRO-3 organised by MODULE. Every `fully_shard(m)` call makes ONE parameter group out of the parameters of
`m` that no earlier call has claimed. Each group is the unit of communication:

    forward  pre-hook of m  : all-gather the group's shards -> unsharded params (ONE all_gather_into_tensor)
    forward  post-hook of m : free the unsharded params            (reshard_after_forward=True)
    backward pre-hook of m  : all-gather again                     (only if they were freed after forward)
    backward post of m      : reduce-scatter the group's gradients (ONE reduce_scatter_tensor), free them

So WHERE you call fully_shard decides how big each all-gather is, how much is unsharded at once (peak memory), and how
much communication can hide behind the neighbouring block's compute. Three layouts today:

    root        fully_shard(model) only: ONE group = all 78 tensors. Gathered once at the start of forward, kept
                unsharded through backward (the root never reshards after forward), reduce-scattered once at the end
                of backward. Shards the optimizer state and the BETWEEN-step memory, but the peak holds P of params
                and P of gradients unsharded, and nothing overlaps.
    block       fully_shard(block) for each of the n_layers blocks, THEN fully_shard(model) for what is left (token and
                position embeddings, final norm, head): n_layers + 1 groups. Bottom-up order matters: a call claims
                only parameters not claimed before, so calling the root first would put everything in one group.
    block_nrf   the same with reshard_after_forward=False on the blocks: no re-gather in backward (one all-gather per
                block per microbatch instead of two) at the price of keeping every block unsharded from its forward to
                its backward: ZeRO-2-like during the step.

Gradient accumulation (Day 04 / Day 06 sec. 1.4): FSDP2's default reduce-scatters EVERY microbatch and adds the
averaged shard into the sharded .grad (Day 06's zero1_rs_every row). `set_requires_gradient_sync(False)` on the
non-final microbatches keeps the full UNSHARDED gradient instead and reduce-scatters once, at the last microbatch
(Day 06's zero1 row; the FSDP1 name was no_sync).

Things FSDP2 does NOT do that DDP did:
    * it does not broadcast rank 0's initial parameters. fully_shard keeps chunk r of RANK r's OWN tensor. If ranks
      initialised differently, the "full" model is a patchwork of rank 0's chunk 0 and rank 1's chunk 1. No crash, no
      divergence check fires (every element lives on one rank only), just a different model. --bug no_init_bcast.
    * there are no replicas: every element of a parameter, gradient and AdamW moment lives on exactly one rank.

Parts marked YOU WRITE: rewrite them yourself (guide Block 2), then diff against this file.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

LAYOUTS = ("root", "block", "block_nrf", "block_pf")


# ============================================================================ YOU WRITE (1): the initial broadcast
@torch.no_grad()
def broadcast_module_state(model: nn.Module, src: int = 0) -> None:
    """Make every rank's parameters AND buffers equal to rank `src`'s BEFORE fully_shard (DDP's constructor does this
    for you; FSDP2 does not). One broadcast per tensor is fine here: it runs once.

    Why buffers too: this model's only buffer (the causal mask) is derived from the config and therefore already
    equal, but a real model can carry state in buffers (e.g. BatchNorm running stats, RoPE caches)."""
    for t in [*model.parameters(), *model.buffers()]:
        dist.broadcast(t.data, src=src)


# ============================================================================ YOU WRITE (2): the wrapping policy
def apply_fully_shard(model: nn.Module, layout: str) -> list[str]:
    """Apply FSDP2 IN PLACE according to `layout`. Returns the FSDP groups in forward order, for logging.

    Rules to encode:
      * bottom-up: blocks first, the root last (the root call claims only what the blocks left over);
      * the root (model) is ALWAYS wrapped, even for the per-block layout: otherwise the embeddings / norm / head would
        not be managed at all (not sharded, gradients never reduced, and each block would become its own FSDP root);
      * reshard_after_forward: leave it at None (the default: True for non-root groups, False for the root) except
        for block_nrf, where the blocks get False.
    fully_shard uses a 1-D DeviceMesh over all ranks when `mesh` is not given (_init_default_mesh).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    groups = []
    if layout != "root":
        raf = False if layout == "block_nrf" else None
        for i, blk in enumerate(model.blocks):
            fully_shard(blk, reshard_after_forward=raf)
            groups.append(f"blocks.{i}")
    fully_shard(model)  # root: auto reshard_after_forward -> False (it is needed again right at backward start)
    if layout == "block_pf":
        set_forward_prefetch(model)
    groups.insert(0, "root (every parameter)" if layout == "root" else "root (embeddings, final norm, head)")
    return groups


# ============================================================================ given: explicit forward prefetch
def set_forward_prefetch(model: nn.Module) -> None:
    """block_pf: issue the NEXT group's all-gather right after this group's own unshard, before this group's compute
    is launched (FSDPState._pre_forward -> _prefetch_unshard). Default ("implicit") forward prefetching only issues
    block i+1's all-gather when the CPU reaches block i+1's pre-forward; if the CPU is the bottleneck, the GPU has
    already finished block i by then and the all-gather runs exposed. Backward keeps the default (one group ahead,
    FSDPParamGroup._backward_prefetch)."""
    blocks = list(model.blocks)
    model.set_modules_to_forward_prefetch([blocks[0]])  # root's pre-forward also launches blocks.0's all-gather
    for cur, nxt in zip(blocks, blocks[1:]):
        cur.set_modules_to_forward_prefetch([nxt])


# ============================================================================ YOU WRITE (3): the accumulation switch
def set_microbatch_sync(model: nn.Module, last: bool, accum: str, bug: str = "none") -> None:
    """Call BEFORE each microbatch's forward.

      accum == "sync"   : every microbatch reduce-scatters (FSDP2's default; nothing to do but be explicit)
      accum == "nosync" : only the LAST microbatch reduce-scatters; the others accumulate the unsharded gradient

    `set_requires_gradient_sync(flag)` recurses into every FSDP submodule by default (recurse=True) and sets
    reduce_grads on each FSDPParamGroup; post_backward then skips foreach_reduce and keeps the unsharded .grad.
    Contrast with DDP's no_sync(): a context manager read in FORWARD. Here it is a sticky flag: if you never set it
    back to True, nothing is ever reduced (--bug sync_never).
    """
    assert isinstance(model, FSDPModule)
    if bug == "sync_never":
        model.set_requires_gradient_sync(False)  # BUG: the flag is never re-enabled for the last microbatch
        return
    model.set_requires_gradient_sync(accum == "sync" or last)


# ============================================================================ negative control helper (given)
def sum_instead_of_average(model: nn.Module) -> None:
    """--bug sum_grads: divide factor 1 -> FSDP2 uses ReduceOp.PREMUL_SUM(1/1): the reduced gradient is the SUM over
    ranks, W x too large. Adam is nearly scale-invariant, so watch which checks notice (guide 1.7)."""
    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.set_gradient_divide_factor(1.0)


# ============================================================================ evidence helpers (given)
def to_full(t: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """A full, plain CPU copy of a (possibly DTensor-sharded) tensor. COLLECTIVE for a DTensor: all ranks must call.

    A missing gradient (None) becomes zeros of `like`'s full shape; a run with gradients that were never reduced
    (sync_never) then shows a zero gradient in check A instead of crashing the evidence code."""
    if t is None:
        shape = like.shape  # DTensor.shape is the GLOBAL shape
        return torch.zeros(shape, dtype=like.dtype)
    if isinstance(t, DTensor):
        t = t.full_tensor()  # all-gather of the shards (+ un-padding), outside the timed region
    return t.detach().cpu().clone()


def shard_report(params: list[torch.Tensor]) -> dict:
    """What THIS rank stores for its parameters: local elements, padding, how many are DTensors (FSDP-managed)."""
    local = padded = full = n_dt = 0
    for p in params:
        full += p.numel()  # global numel for a DTensor
        if isinstance(p, DTensor):
            n_dt += 1
            lt = p.to_local()
            local += lt.numel()
            # FSDP2 pads dim 0 so every rank's shard has chunk 0's size (_init_sharded_param): the padded storage
            # is what all-gather moves and what memory_allocated counts
            padded += lt.untyped_storage().nbytes() // lt.element_size()
        else:
            local += p.numel()
            padded += p.numel()
    return {"full_numel": full, "local_numel": local, "local_padded_numel": padded, "dtensor_params": n_dt,
            "plain_params": len(params) - n_dt}


def optimizer_state_report(optim: torch.optim.Optimizer) -> dict:
    """Bytes of AdamW state stored on THIS rank (local shard of each DTensor moment), by kind."""
    by_key, devices = {}, {}
    for st in optim.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                lv = v.to_local() if isinstance(v, DTensor) else v
                by_key[k] = by_key.get(k, 0) + lv.numel() * lv.element_size()
                devices[k] = str(lv.device)
    return {"state_bytes": by_key, "state_devices": devices}
