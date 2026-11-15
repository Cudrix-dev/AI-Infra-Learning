"""Day 12 - REFERENCE two-stage pipeline: autograd across the process boundary, GPipe, and non-interleaved 1F1B.

Same names and signatures as my_pp.py. Read it after your version passes, or when stuck.

Vocabulary used in the comments:
  output      what this stage's forward returns: the activation sent to the next stage, or (last stage) the loss
  input       what this stage's forward consumed: the data microbatch (first stage) or the received activation
  in flight   microbatch m on this stage between the start of F(m) and the end of B(m); its saved tensors live here
"""

from __future__ import annotations

from collections import deque

import torch

from pp_common import Stage, mb_loss


# ---------------------------------------------------------------------------------------------------------------
# Step 1: one stage's forward and backward for ONE microbatch (Megatron: forward_step / backward_step)
# ---------------------------------------------------------------------------------------------------------------

def forward_step(st: Stage, mb: int, received):
    """Run this stage's blocks on microbatch mb. Returns (output, input_tensor) for the schedule to keep."""
    if st.is_first:
        x = st.inputs[mb]                  # data: needs no gradient
    else:
        # The received buffer is plain data. Making it a LEAF that requires grad turns it into the start of this
        # rank's autograd graph: backward will stop here and leave d loss / d input in x.grad (GUIDE §2.2).
        x = received.requires_grad_()
    y = st.module(x)
    if st.is_last:
        st.save_output(mb, y)
        loss = mb_loss(y, st.targets[mb], st.n_total)    # this microbatch's share of the full-batch mean
        st.save_loss(mb, loss)
        return loss, x                     # the last stage's "output" is a scalar: backward starts from 1.0
    return y, x                            # y still has its grad_fn; only y's DATA is sent


def backward_step(st: Stage, mb: int, input_tensor, output, grad_output):
    """Backward of this stage for microbatch mb. Returns d loss / d input (None on the first stage)."""
    if st.is_last:
        torch.autograd.backward(output)                          # output is the scalar loss
    else:
        # The graph ends at `output`, which is not a scalar: supply its upstream gradient explicitly.
        torch.autograd.backward(output, grad_tensors=grad_output)
    if st.is_first:
        return None                                              # nothing to send further back
    return input_tensor.grad                                     # the leaf's gradient: goes to the previous stage


# ---------------------------------------------------------------------------------------------------------------
# Step 2: GPipe. All forwards, then all backwards (ascending microbatch order on every stage).
# ---------------------------------------------------------------------------------------------------------------

def run_gpipe(st: Stage, M: int) -> None:
    saved = deque()                                              # (mb, input, output) per microbatch in flight
    for m in range(M):
        x = st.comm.recv_forward(m)                              # None on the first stage
        out, inp = st.forward_step(m, x)
        st.comm.send_forward(out, m)                             # no-op on the last stage
        saved.append((m, inp, out))
    for m in range(M):
        mb, inp, out = saved.popleft()                           # oldest first: matches the arrival order of grads
        g = st.comm.recv_backward(mb)                            # None on the last stage
        dx = st.backward_step(mb, inp, out, g)
        st.comm.send_backward(dx, mb)                            # no-op on the first stage


# ---------------------------------------------------------------------------------------------------------------
# Step 3: non-interleaved 1F1B (Megatron: forward_backward_pipelining_without_interleaving)
# ---------------------------------------------------------------------------------------------------------------

def run_1f1b(st: Stage, M: int) -> None:
    warmup = min(st.P - st.stage - 1, M)          # stage 0 of 2: 1; the last stage: 0
    remaining = M - warmup
    saved = deque()

    # Warmup: forwards only, to fill the pipeline below this stage.
    for i in range(warmup):
        x = st.comm.recv_forward(i)
        out, inp = st.forward_step(i, x)
        st.comm.send_forward(out, i)
        saved.append((i, inp, out))

    # The first steady-state forward needs its input before the loop.
    x = st.comm.recv_forward(warmup) if remaining > 0 else None

    # Steady state: one forward (the newest microbatch), then one backward (the OLDEST microbatch in flight).
    for i in range(remaining):
        f_mb = warmup + i
        out, inp = st.forward_step(f_mb, x)
        saved.append((f_mb, inp, out))
        b_mb = saved[0][0]
        # Send the new activation and receive the oldest microbatch's gradient in ONE call: the next stage is about
        # to do the mirror image (send that gradient, receive this activation). Posting both at once is what keeps
        # two blocking ranks from each waiting for the other to receive first (GUIDE §2.4).
        g = st.comm.send_forward_recv_backward(out, f_mb, b_mb)
        mb, inp_b, out_b = saved.popleft()
        dx = st.backward_step(mb, inp_b, out_b, g)
        if i == remaining - 1:
            st.comm.send_backward(dx, mb)
            x = None
        else:
            x = st.comm.send_backward_recv_forward(dx, mb, f_mb + 1)

    # Cooldown: the backwards still owed for the warmup forwards.
    for _ in range(warmup):
        mb, inp_b, out_b = saved.popleft()
        g = st.comm.recv_backward(mb)
        dx = st.backward_step(mb, inp_b, out_b, g)
        st.comm.send_backward(dx, mb)
    assert not saved, "every forward must have had its backward"
