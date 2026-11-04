"""Day 06 - ZeRO-1 from scratch: AdamW state sharded across data-parallel ranks.

ZeRO stage 1 (Rajbhandari et al., 2020) removes the REDUNDANT copies of the optimizer state that plain data
parallelism keeps on every rank. Each rank still holds the full parameters and the full gradient during backward,
but AdamW's two moment tensors (exp_avg, exp_avg_sq) exist only for this rank's 1/W slice. One optimizer step:

    backward(s)                 every rank: full local gradient g_r in a flat buffer (P bytes)
    reduce_scatter(flat grad)   rank r receives sum_r' g_r' for ITS slice only; divide by W  -> averaged grad shard
    AdamW on the slice          only S = padded/W elements are updated; state is allocated for S elements only
    all_gather(flat param)      every rank receives every other rank's updated slice -> full parameters again

reduce_scatter + all_gather move exactly the bytes of one ring all_reduce (an all_reduce IS a reduce-scatter
followed by an all-gather), so ZeRO-1 adds no communication VOLUME over DDP. It adds a second collective after the
optimizer, which here is not overlapped with anything.

The same class runs three ways, so the comparisons are matched (same flat layout, same single-tensor AdamW, same
gradient views, one change at a time):

    shard=False                     "flat_ar"  UNSHARDED control: all_reduce(flat grad), AdamW over the whole flat
                                               parameter. Same code path as ZeRO-1 minus the sharding.
    shard=True                      "zero1"    accumulate the full gradient locally for K microbatches (Day 04's
                                               no_sync analogue), ONE reduce-scatter at the window boundary
    shard=True, rs_every=True       "zero1_rs_every"  reduce-scatter after EVERY microbatch into a P/W shard
                                               accumulator and free the full gradient in between (the accumulation
                                               pattern of ZeRO-2 / FSDP, run on top of ZeRO-1)

Why a FLAT buffer with slices that cut through tensors? AdamW is ELEMENTWISE: every element's update depends only on
its own param, grad, exp_avg, exp_avg_sq and the shared scalars (lr, betas, eps, wd, step). So any partition of the
elements gives the same numbers, and a flat buffer gives W exactly equal shards (balanced memory, equal-size
collectives). Optimizers that need a whole tensor (Adafactor's row/col factors, LAMB's per-tensor trust ratio,
Shampoo, Muon's orthogonalization) or a global norm (gradient clipping) cannot be sliced like this without extra
communication. torch's ZeroRedundancyOptimizer partitions WHOLE tensors instead (Block 3).

Padding: reduce-scatter / all-gather of one flat tensor need equal shards, so the flat buffers are padded with zeros
to a multiple of W. A zero pad has zero gradient, zero moments and zero update; it never reaches a real parameter.

Not handled (real implementations do): mixed precision (bf16 params + fp32 master shard), several dtypes/devices,
parameter groups with different hyper-parameters, gradient clipping (needs an all_reduce of the shard norms),
overlapping the reduce-scatter with backward (buckets, Day 03), sharded checkpoints and resharding on restart.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

MiB = 2**20
BUGS = ("none", "no_reduce_scatter", "no_allgather", "rs_overwrite")

# torch 2.14 deprecates reduce_scatter_tensor / all_gather_into_tensor in favour of *_single (same semantics: one
# flat input, one flat output, equal shards). Fall back to the old names on older releases.
reduce_scatter = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
all_gather = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


class FlatZeroAdamW:
    """AdamW whose state lives only on the owning rank's slice of one flat parameter buffer.

    Training-loop contract (train_zero1.py):
        zero_grad()          once per window (IN PLACE for the boundary variants: p.grad are views of flat_grad)
        backward()           K times; rs_every: reduce_microbatch() after each one
        reduce_grads()       once per window, after the last backward (a no-op for rs_every)
        step_local()         AdamW on this rank's slice (the whole buffer for shard=False)
        gather_params()      all_gather the updated slices (a no-op for shard=False)
    """

    def __init__(self, params, *, lr: float, weight_decay: float, betas=(0.9, 0.999), eps: float = 1e-8,
                 shard: bool = True, rs_every: bool = False, bug: str = "none"):
        if bug not in BUGS:
            raise ValueError(f"unknown bug {bug!r}")
        if rs_every and not shard:
            raise ValueError("rs_every only makes sense with shard=True")
        self.world, self.rank = dist.get_world_size(), dist.get_rank()
        self.params = [p for p in params if p.requires_grad]
        if len({(p.dtype, p.device) for p in self.params}) != 1:
            raise ValueError("one dtype and one device per model (real ZeRO keys its flat buffers by both)")
        self.shard, self.rs_every, self.bug = shard, rs_every, bug
        dtype, device = self.params[0].dtype, self.params[0].device
        self.numels = [p.numel() for p in self.params]
        self.offsets = [sum(self.numels[:i]) for i in range(len(self.params))]
        self.numel = sum(self.numels)
        # Pad to a multiple of W even for shard=False, so both variants have the same buffer sizes.
        self.padded = math.ceil(self.numel / self.world) * self.world
        self.shard_numel = self.padded // self.world
        if shard:
            self.lo, self.hi = self.rank * self.shard_numel, (self.rank + 1) * self.shard_numel
        else:
            self.lo, self.hi = 0, self.padded

        # ======== YOU WRITE (1): one flat parameter buffer; every model parameter becomes a VIEW of it ========
        # After this, the all-gather can write straight into the parameters (no copy-back), and the optimizer's slice
        # below aliases the same memory. The old per-tensor parameter storage is freed by the `p.data =` assignment.
        self.flat_param = torch.zeros(self.padded, dtype=dtype, device=device)
        with torch.no_grad():
            for p, off, n in zip(self.params, self.offsets, self.numels):
                self.flat_param[off : off + n].copy_(p.reshape(-1))
                p.data = self.flat_param[off : off + n].view_as(p)
        # ONE broadcast of rank 0's values replaces Day 02's per-tensor broadcast_params (ranks start differently on
        # purpose: train_zero1.py seeds each rank with seed + rank).
        dist.broadcast(self.flat_param, src=0)

        # ======== YOU WRITE (2): where gradients live between microbatches ========
        if not rs_every:
            # Boundary variants: p.grad is a view of ONE persistent flat gradient buffer (Day 03's grad_as_view).
            # AccumulateGrad adds every microbatch into it in place, so after K backwards flat_grad holds this rank's
            # window gradient sum_j w_j g_j, already flat and ready for ONE reduce-scatter. Cost: P bytes for the
            # whole run, also between microbatches. zero_grad must therefore zero IN PLACE (never set_to_none).
            self.flat_grad = torch.zeros_like(self.flat_param)
            for p, v in zip(self.params, self.views(self.flat_grad)):
                p.grad = v
            self.grad_acc = None
            shard_grad = self.flat_grad[self.lo : self.hi]  # reduce_grads() writes the averaged slice here (in place)
        else:
            # rs_every: no persistent full gradient. Each microbatch's p.grad is reduce-scattered and freed; only the
            # averaged SHARD survives between microbatches (P/W bytes), summed over the window in grad_acc.
            self.flat_grad = None
            self.grad_acc = torch.zeros(self.shard_numel, dtype=dtype, device=device)
            shard_grad = self.grad_acc

        # ======== YOU WRITE (3): the optimizer sees ONLY this rank's slice ========
        # nn.Parameter(view) shares storage with the view (it detaches, it does not copy), so AdamW's in-place update
        # of opt_param IS an update of flat_param[lo:hi], i.e. of this rank's part of the model parameters.
        # AdamW allocates exp_avg / exp_avg_sq lazily, at the first step(), with opt_param's shape: S elements, not P.
        self.opt_param = torch.nn.Parameter(self.flat_param[self.lo : self.hi])
        if self.opt_param.data_ptr() != self.flat_param[self.lo :].data_ptr():
            raise RuntimeError("opt_param must alias flat_param[lo:hi]")
        self.opt_param.grad = shard_grad  # a view too: the reduce-scatter result is the optimizer's gradient
        # Same AdamW as Day 01 (foreach=False, fused=False): the per-tensor reference loop, here over ONE tensor.
        self.optim = torch.optim.AdamW([self.opt_param], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                       foreach=False, fused=False)
        self.collectives = {"reduce_scatter": 0, "all_reduce": 0, "all_gather": 0}

    # ------------------------------------------------------------------ helpers
    def views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        """Per-parameter views of a full (padded) flat buffer, in parameter order."""
        return [flat[off : off + n].view_as(p) for p, off, n in zip(self.params, self.offsets, self.numels)]

    def zero_grad(self) -> None:
        """Once per window."""
        if self.rs_every:
            self.grad_acc.zero_()
            for p in self.params:
                p.grad = None
        else:
            self.flat_grad.zero_()  # IN PLACE: set_to_none would cut p.grad loose from flat_grad (control
            #                          grad_set_to_none in train_zero1.py)

    # ======== YOU WRITE (4): the gradient side of one window ========
    def reduce_microbatch(self) -> None:
        """rs_every only, after EVERY backward: reduce-scatter this microbatch's gradient, keep only the shard."""
        if not self.rs_every:
            return
        pad = self.padded - self.numel
        tmp = torch.cat([p.grad.reshape(-1) for p in self.params]
                        + ([self.params[0].new_zeros(pad)] if pad else []))  # transient P bytes
        for p in self.params:
            p.grad = None  # the full gradient is gone: between microbatches only grad_acc (P/W) remains
        out = tmp[self.lo : self.hi]
        if self.bug != "no_reduce_scatter":
            reduce_scatter(out, tmp, op=dist.ReduceOp.SUM)  # in place: out is tmp's own slice
            self.collectives["reduce_scatter"] += 1
        out.div_(self.world)
        if self.bug == "rs_overwrite":
            self.grad_acc.copy_(out)  # BUG: keeps only the latest microbatch's shard
        else:
            self.grad_acc.add_(out)  # window sum of averaged shards: sum_j fl(g_j^0 + g_j^1)/2 (other rounding than
            #                          one reduce-scatter of the locally accumulated sum, see the guide)

    def reduce_grads(self) -> None:
        """Once per window, after the last backward: average this rank's gradient slice over all ranks."""
        if self.rs_every:
            return  # already reduced, microbatch by microbatch, into grad_acc (= opt_param.grad)
        g = self.flat_grad[self.lo : self.hi]
        if self.shard:
            if self.bug != "no_reduce_scatter":
                # IN PLACE: NCCL's reduce-scatter is in place when recvbuff == sendbuff + rank * recvcount, i.e. the
                # output is this rank's own slice of the input. No extra P/W buffer for the shard gradient.
                reduce_scatter(g, self.flat_grad, op=dist.ReduceOp.SUM)
                self.collectives["reduce_scatter"] += 1
            # (no_reduce_scatter: BUG, this rank steps its slice with its own LOCAL gradient)
        else:
            dist.all_reduce(self.flat_grad, op=dist.ReduceOp.SUM)  # every rank needs every element: all_reduce
            self.collectives["all_reduce"] += 1
        g.div_(self.world)  # SUM then divide, like Day 02; for W = 2 the factor 0.5 is exact

    # ======== YOU WRITE (5): the parameter side ========
    def step_local(self) -> None:
        self.optim.step()  # updates opt_param == flat_param[lo:hi] in place; the other slices are now stale

    def gather_params(self) -> None:
        if not self.shard or self.bug == "no_allgather":
            return  # shard=False: every rank updated everything. no_allgather: BUG, other slices stay stale
        # IN PLACE all-gather: the input is this rank's slice of the output (sendbuff == recvbuff + rank * count).
        # Because the model parameters are views of flat_param, this single call refreshes all of them.
        all_gather(self.flat_param, self.flat_param[self.lo : self.hi])
        self.collectives["all_gather"] += 1

    # ------------------------------------------------------------------ evidence (outside the timed region)
    def _full(self, part: torch.Tensor) -> torch.Tensor:
        """The full padded buffer from every rank's slice (an extra all_gather, for EVIDENCE only)."""
        if not self.shard:
            return part.detach().clone()
        out = torch.empty(self.padded, dtype=part.dtype, device=part.device)
        all_gather(out, part.detach().contiguous())
        return out

    def full_grad(self) -> list[torch.Tensor]:
        """The averaged gradient the optimizer used, per parameter (all ranks must call)."""
        return [v.cpu().clone() for v in self.views(self._full(self.opt_param.grad))]

    def full_state(self) -> dict:
        """AdamW state gathered from all slices, per parameter, as torch.optim.AdamW would hold it (all ranks call).

        This is what a checkpoint of a ZeRO-1 run has to do (or save per-rank shards and reshard on load).
        """
        st = self.optim.state[self.opt_param]
        out = {k: [v.cpu().clone() for v in self.views(self._full(st[k]))] for k in ("exp_avg", "exp_avg_sq")}
        out["step"] = float(st["step"])
        return out

    def state_report(self) -> dict:
        """What THIS rank owns and stores (the 'state per rank' of today's done-when)."""
        st = self.optim.state.get(self.opt_param, {})
        tensors = {k: v for k, v in st.items() if torch.is_tensor(v)}
        return {
            "owned_range": [self.lo, self.hi],
            "owned_numel": self.hi - self.lo,
            "owned_real_numel": max(0, min(self.hi, self.numel) - self.lo),  # without the zero pad
            "pad_numel": self.padded - self.numel,
            "state_bytes": {k: v.numel() * v.element_size() for k, v in tensors.items()},
            "state_devices": {k: str(v.device) for k, v in tensors.items()},
            "persistent_grad_bytes": (self.flat_grad.numel() if self.flat_grad is not None
                                      else self.grad_acc.numel()) * self.flat_param.element_size(),
            "flat_param_bytes": self.flat_param.numel() * self.flat_param.element_size(),
        }
