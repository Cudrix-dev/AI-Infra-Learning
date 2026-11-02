"""Day 04 - MiniDDP + no_sync(): gradient accumulation without communicating every microbatch.

Day 03's MiniDDP arms its hooks in forward() and launches one all_reduce per bucket during every backward. For
gradient accumulation we want K backwards per optimizer step but ONE gradient sync: the first K-1 backwards should
only accumulate into p.grad locally (AccumulateGrad already adds into an existing .grad), and the last backward should
communicate the SUM of all K local contributions.

DistributedDataParallel does it with one flag (torch/nn/parallel/distributed.py, read it in Block 3):

    @contextmanager
    def no_sync(self):
        old = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try: yield
        finally: self.require_backward_grad_sync = old

  _pre_forward:  if grad enabled and require_backward_grad_sync: reducer.prepare_for_forward()
  _post_forward: if grad enabled and require_backward_grad_sync: reducer.prepare_for_backward(...)   <- arms the hooks
                 else: require_forward_param_sync = False                       <- next forward skips the buffer sync
  C++ Reducer::autograd_hook: if (!expect_autograd_hooks_) return;              <- hooks still FIRE, then return

The decision is taken at FORWARD time, not at backward time. That is why the docs warn that the forward pass must be
inside no_sync(): a forward outside it has already armed the hooks, and the backward will communicate even if it runs
inside the context (the `bwd_only_nosync` control in train_accum.py shows exactly that).

This file adds the same mechanism to MiniDDP. Nothing else changes: the bucket that is all-reduced on the last
microbatch simply contains p.grad = g_1 + ... + g_K (already weighted by the loss scaling of the training loop).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day03_ddp_hooks"))
from mini_ddp import MiniDDP  # noqa: E402


class MiniDDPAccum(MiniDDP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_backward_grad_sync = True  # same name as DDP's flag
        # Evidence counters (cheap ints; read by train_accum.py). DDP has no Python-visible equivalent: for DDP the
        # collectives are counted in the profiler trace (count_collectives.py).
        self.hook_calls_armed = 0  # hooks that copied a gradient into a bucket
        self.hook_calls_skipped = 0  # hooks that fired under no_sync and returned at once
        self.collectives = 0  # bucket all_reduces issued

    # ======== YOU WRITE: the context manager (Block 2, Stage 2) ========
    @contextmanager
    def no_sync(self):
        """Inside this context, forward() does not arm the hooks: the following backward accumulates locally."""
        old = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            # restore, do not set True: nested no_sync() blocks must not re-enable sync when the inner one exits
            self.require_backward_grad_sync = old

    # ======== YOU WRITE: decide at forward time whether the coming backward communicates ========
    def forward(self, *args, **kwargs):
        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            # Day 03 path: (rebuild if due,) reset pending counts / next bucket, arm the hooks, run the module
            return super().forward(*args, **kwargs)
        # no_sync (or no_grad) forward: disarm. The hooks will still be called by autograd for every parameter, and
        # return immediately (MiniDDP._on_grad_ready checks _expect_hooks first, like Reducer::autograd_hook checks
        # expect_autograd_hooks_). AccumulateGrad keeps adding this microbatch's gradient into p.grad.
        self._expect_hooks = False
        return self.module(*args, **kwargs)

    def _on_grad_ready(self, i: int, p: torch.Tensor) -> None:
        if self._expect_hooks:
            self.hook_calls_armed += 1
        else:
            self.hook_calls_skipped += 1
        super()._on_grad_ready(i, p)

    def finish_grad_sync(self) -> None:
        """Call after EVERY backward. A no-op if the forward ran under no_sync (nothing was launched)."""
        if not self._expect_hooks:
            return
        if self.comm:
            self.collectives += len(self.buckets)
        super().finish_grad_sync()  # wait + copy back + (first time) schedule the rebuild
