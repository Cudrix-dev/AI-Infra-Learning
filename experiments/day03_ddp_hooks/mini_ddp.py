"""Day 03 - MiniDDP: bucketed, overlapped gradient all-reduce driven by autograd hooks.

What torch's DistributedDataParallel does for a dense, single-GPU-per-process model, in Python, so every moving part
is visible:

  construction    broadcast rank 0's parameters AND buffers; assign parameters to buckets; allocate one flat
                  gradient buffer per bucket; register a post-accumulate-grad hook on every parameter
  forward()       reset the per-iteration counters (grads still pending per bucket, next bucket to launch)
  backward        autograd calls hook(p) right after p.grad is final for this backward ->
                  copy p.grad * (1/world) into its slot of the bucket buffer -> when the bucket's last grad has
                  arrived, launch an ASYNC all_reduce for it, strictly IN BUCKET ORDER
                  => early buckets travel over NVLink while autograd is still computing later gradients
  finish_grad_sync()  wait for every bucket's Work (stream dependency, not a CPU block), copy the averaged buckets
                  back into p.grad (skipped with grad_as_view: p.grad then IS a view of the bucket)

Real DDP does the same in C++ (torch/csrc/distributed/c10d/reducer.cpp, read it in Block 3):
  Reducer::autograd_hook -> mark_variable_ready -> mark_variable_ready_dense (copy + scale) -> mark_bucket_ready
  (in-order launch) -> all_reduce_bucket; finalize_backward (wait + copy back) is queued with
  Engine::queue_callback so the user never calls it. Here finish_grad_sync() is called explicitly by the training
  loop, to keep the control flow visible.

Not handled (DDP does): unused parameters (find_unused_parameters), no_sync() / gradient accumulation (Day 4),
several devices per process, mixed dtypes in one model, sparse gradients, comm hooks, static_graph, join.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.distributed as dist

MiB = 2**20


# ----------------------------------------------------------------------------- bucket assignment (pure function)
def assign_buckets(sizes_bytes: list[int], order: list[int], caps_bytes: list[int]) -> list[list[int]]:
    """Greedily pack parameter indices, taken in `order`, into buckets.

    caps_bytes[k] is the byte limit of bucket k; the last entry repeats for all later buckets (DDP passes
    [first_bucket_cap, bucket_cap]). Same rule as DDP's compute_bucket_assignment_by_size: a bucket is CLOSED as soon
    as its size reaches its limit, so a bucket can overshoot the limit by its last tensor, and a tensor larger than
    the cap gets a bucket of its own. A cap of 0 therefore gives one tensor per bucket (the per-parameter baseline).

    `order` should be the order in which gradients become ready: bucket 0 must be the first to fill up, because
    buckets are launched in index order (see MiniDDP._launch_in_order).
    """
    buckets: list[list[int]] = []
    cur: list[int] = []
    cur_bytes = 0
    for i in order:
        cur.append(i)
        cur_bytes += sizes_bytes[i]
        limit = caps_bytes[min(len(buckets), len(caps_bytes) - 1)]
        if cur_bytes >= limit:
            buckets.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        buckets.append(cur)
    return buckets


class _Bucket:
    """One flat gradient buffer plus a view per parameter. The buffer lives for the whole run (like DDP's)."""

    def __init__(self, params: list[torch.Tensor], indices: list[int]):
        self.indices = indices  # positions in MiniDDP.params
        self.params = params
        numel = sum(p.numel() for p in params)
        # zeros, not empty: a slot that is never written (see bug early_launch) must read as 0, not as garbage
        self.flat = torch.zeros(numel, dtype=params[0].dtype, device=params[0].device)
        self.views, off = [], 0
        for p in params:
            # contiguous view with p's shape: our params are contiguous, so this matches p.grad's layout
            # (DDP uses as_strided with p's strides to honour the "gradient layout contract" in general)
            self.views.append(self.flat[off : off + p.numel()].view_as(p))
            off += p.numel()
        self.pending = len(params)  # grads still to arrive in this backward
        self.ready = False  # all grads copied in: may be launched once every earlier bucket is launched
        self.launched = False  # its all_reduce has been issued in this backward
        self.work: dist.Work | None = None  # handle of the async all_reduce

    @property
    def nbytes(self) -> int:
        return self.flat.numel() * self.flat.element_size()


# ----------------------------------------------------------------------------- the wrapper
class MiniDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_cap_mb: float = 25.0, first_bucket_mb: float | None = None,
                 grad_as_view: bool = False, rebuild: bool = False, bug: str = "none", comm: bool = True):
        super().__init__()
        self.module = module
        self.world = dist.get_world_size()
        self.params = [p for p in module.parameters() if p.requires_grad]
        if len({(p.dtype, p.device) for p in self.params}) != 1:
            raise ValueError("MiniDDP supports one dtype and one device per model (DDP keys buckets by both)")
        if bug not in ("none", "early_launch", "no_wait"):
            raise ValueError(f"unknown bug {bug!r}")
        # comm=False: hooks, copies and bookkeeping run but no all_reduce is issued. A TIMING ABLATION ONLY (ranks
        # diverge): it measures what the hooks cost by themselves, so overhead = hooks + communication can be split.
        self.grad_as_view, self.bug, self.comm = grad_as_view, bug, comm
        cap = int(bucket_cap_mb * MiB)
        self.caps = [int(first_bucket_mb * MiB), cap] if first_bucket_mb is not None else [cap]
        self.sizes = [p.numel() * p.element_size() for p in self.params]

        self._sync_module_states()

        # ======== YOU WRITE: bucket construction + hook registration (Block 2, Stage 2) ========
        # Before the first backward we do not know the order in which grads become ready. Like DDP's initial guess,
        # assume "reverse of registration order" (the last layer's grads come first in backward).
        self._build(list(reversed(range(len(self.params)))))
        self.initial_layout = self.layout()

        # Record the ACTUAL ready order during the first backward. With rebuild=True the buckets are rebuilt from it
        # before the next forward (DDP: push_rebuilt_params in autograd_hook, rebuild_buckets in _pre_forward).
        self.ready_order: list[int] = []
        self._recording = True
        self._rebuild_requested, self._rebuild_now = rebuild, False

        # register_post_accumulate_grad_hook(hook): autograd calls hook(param) AFTER param.grad has been accumulated,
        # i.e. when this parameter's gradient for this backward is complete. (Tensor.register_hook instead sees the
        # incoming gradient BEFORE it is added to .grad.) DDP attaches the C++ equivalent: a post-hook on the
        # parameter's AccumulateGrad node.
        self._hooks = [p.register_post_accumulate_grad_hook(partial(self._on_grad_ready, i))
                       for i, p in enumerate(self.params)]
        self._expect_hooks = False

    # ------------------------------------------------------------------ construction helpers
    @torch.no_grad()
    def _sync_module_states(self) -> None:
        """Rank 0's parameters AND buffers to everyone (DDP: _sync_module_states -> _broadcast_coalesced).

        Day 02 broadcast parameters only. Buffers (BatchNorm statistics, here the causal mask) are model state too.
        """
        for t in list(self.module.parameters()) + list(self.module.buffers()):
            dist.broadcast(t.data, src=0)

    def _build(self, order: list[int]) -> None:
        groups = assign_buckets(self.sizes, order, self.caps)
        self.buckets = [_Bucket([self.params[i] for i in g], g) for g in groups]
        self.locate = {i: (b, s) for b, g in enumerate(groups) for s, i in enumerate(g)}  # param -> (bucket, slot)
        if self.grad_as_view:  # after a rebuild, existing grads move into the new buffers (DDP: initialize_bucket_views)
            for b in self.buckets:
                for p, v in zip(b.params, b.views):
                    if p.grad is not None:
                        v.copy_(p.grad)
                        p.grad = v

    def _rebuild(self) -> None:
        """Rebuild buckets in the recorded gradient-ready order. All ranks MUST end up with the same layout,
        otherwise bucket k on rank 0 and bucket k on rank 1 have different sizes and the all_reduce hangs or mixes
        data. DDP broadcasts rank 0's assignment (Reducer::sync_bucket_indices); we broadcast rank 0's order."""
        if sorted(self.ready_order) != list(range(len(self.params))):  # local check; every param must fire once
            raise RuntimeError("ready order does not cover every parameter exactly once (unused parameter?)")
        order = torch.tensor(self.ready_order, dtype=torch.int64, device=self.params[0].device)
        dist.broadcast(order, src=0)
        self._build(order.tolist())

    # ------------------------------------------------------------------ per iteration
    def forward(self, *args, **kwargs):
        if self._rebuild_now:  # before forward, while activation memory is still low (same reason as DDP)
            self._rebuild()
            self._rebuild_now = False
        # ======== YOU WRITE: reset the per-iteration state ========
        for b in self.buckets:
            b.pending, b.ready, b.launched, b.work = len(b.params), False, False, None
        self._next_bucket = 0  # index of the next bucket to launch; buckets go out strictly in order
        self._expect_hooks = True
        return self.module(*args, **kwargs)

    def _on_grad_ready(self, i: int, p: torch.Tensor) -> None:
        """Autograd hook: p.grad is complete for this backward. Runs on the autograd engine's thread, in the middle
        of backward, with grad mode off. Keep it cheap: every microsecond here is CPU time inside backward."""
        if not self._expect_hooks:  # a backward that did not go through our forward(): nothing to do
            return
        if self._recording:
            self.ready_order.append(i)
        # ======== YOU WRITE: copy into the bucket, count down, launch (Block 2, Stage 2) ========
        b, s = self.locate[i]
        bucket = self.buckets[b]
        if self.bug == "early_launch" and bucket.launched:
            return  # BUG: this bucket was already sent; the late gradient is silently dropped (its slot stays 0)
        view = bucket.views[s]
        if p.grad.data_ptr() != view.data_ptr():
            # Scale while copying: one pass over the data instead of copy + div (DDP: at::mul_out(bucket_view, grad,
            # 1/div_factor)). Pre-dividing also keeps fp16 sums in range. For world=2 the factor 0.5 is exact, so
            # pre-scaling gives the same bits as Day 02's SUM-then-divide.
            torch.mul(p.grad, 1.0 / self.world, out=view)
            if self.grad_as_view:
                p.grad = view  # from now on AccumulateGrad writes straight into the bucket: no copy, one buffer less
        else:  # p.grad already IS the bucket slot (grad_as_view, grads zeroed in place, not set to None)
            view.mul_(1.0 / self.world)
        bucket.pending -= 1
        if self.bug == "early_launch":  # BUG: "the last slot is filled" is not "all slots are filled"
            bucket.ready = bucket.ready or s == len(bucket.params) - 1
        else:
            bucket.ready = bucket.pending == 0
        if bucket.ready:
            self._launch_in_order()

    def _launch_in_order(self) -> None:
        """Launch every consecutive ready bucket starting at _next_bucket (DDP: Reducer::mark_bucket_ready).

        Why in order? Rule 1 from Day 02: every rank must issue the same collectives in the same order. Hooks fire in
        the same order on every rank for this model, but nothing guarantees that in general (different graphs,
        thread timing); fixing the launch order to the bucket index makes the order a property of the layout,
        which all ranks share. The price: a bucket that is ready early waits for slower buckets before it.
        """
        while self._next_bucket < len(self.buckets) and self.buckets[self._next_bucket].ready:
            bucket = self.buckets[self._next_bucket]
            # async_op=True: the collective is enqueued on NCCL's internal stream, ordered after the work already on
            # the current stream (our copy into the bucket), and this call returns immediately with a Work handle.
            if self.comm:
                bucket.work = dist.all_reduce(bucket.flat, op=dist.ReduceOp.SUM, async_op=True)
            bucket.launched = True
            self._next_bucket += 1

    def finish_grad_sync(self) -> None:
        """Call after loss.backward(), before optimizer.step(). (DDP: finalize_backward, queued automatically.)"""
        # ======== YOU WRITE: wait + copy back (Block 2, Stage 2) ========
        if self._next_bucket != len(self.buckets):
            # Local check. If it fails on one rank only, the others are left waiting in a collective: DDP's
            # equivalent is the "Expected to have finished reduction in the prior iteration" error.
            raise RuntimeError(f"only {self._next_bucket}/{len(self.buckets)} buckets were launched: "
                               "a parameter's hook did not fire (unused parameter?)")
        for bucket in self.buckets:
            if self.bug != "no_wait" and bucket.work is not None:
                # NCCL Work.wait(): makes the CURRENT stream wait for the all_reduce's completion event; it does not
                # block the CPU. Without it the copy below (current stream) can read the bucket before NCCL wrote it.
                bucket.work.wait()
            if not self.grad_as_view:
                for p, v in zip(bucket.params, bucket.views):
                    p.grad.copy_(v)  # one kernel per parameter (DDP: copy_bucket_to_grad; batched_grad_copy fuses)
        self._expect_hooks = False
        if self._recording:
            self._recording = False
            self._rebuild_now = self._rebuild_requested

    # ------------------------------------------------------------------ introspection
    def layout(self) -> list[dict]:
        """Bucket sizes and members, for logs and notes."""
        return [{"bucket": k, "mib": b.nbytes / MiB, "n_params": len(b.params), "param_indices": b.indices}
                for k, b in enumerate(self.buckets)]
