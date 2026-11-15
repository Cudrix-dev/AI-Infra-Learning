"""Day 12 - negative controls: the reference with ONE change each. The parity check must catch every one.

All three run the 1F1B schedule with M = 4. EXPECTED lists the checks that MUST fail; every other check must pass.
A control whose failures differ is reported as "NOT AS PREDICTED" (the harness or the explanation is wrong).

Check names: y (model output), loss, blk{k}.fc{1,2}.{weight,bias} (16 parameter gradients; blk0-1 live on stage 0,
blk2-3 on stage 1), order (each rank's F/B order equals the simulator's), peak (each rank's peak in-flight count equals
the simulator's), labels (the k-th activation/gradient received carries the label of the k-th one sent).
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

import pp_ref
from pp_common import Stage


# 1. detached_input: stage 1 forgets to make the received activation require grad. Its own gradients are fine;
#    input.grad is None, so there is nothing to send back. (Real code would crash sending None; this control sends
#    zeros instead, so the effect on stage 0 is visible as a number.)
def _fwd_detached(st: Stage, mb, received):
    if st.is_first:
        return pp_ref.forward_step(st, mb, received)
    x = received                                          # <- the bug: no requires_grad_()
    y = st.module(x)
    if st.is_last:
        st.save_output(mb, y)
        loss = pp_ref.mb_loss(y, st.targets[mb], st.n_total)
        st.save_loss(mb, loss)
        return loss, x
    return y, x


def _bwd_detached(st: Stage, mb, input_tensor, output, grad_output):
    dx = pp_ref.backward_step(st, mb, input_tensor, output, grad_output)
    if not st.is_first and dx is None:
        dx = torch.zeros_like(input_tensor)
    return dx


# 2. lifo_backward: the 1F1B loop takes the NEWEST saved microbatch for backward instead of the oldest. The shapes
#    match, so nothing crashes: stage 0 pairs the gradient of microbatch m with the saved graph of another one.
def _run_1f1b_lifo(st: Stage, M: int) -> None:
    warmup = min(st.P - st.stage - 1, M)
    remaining = M - warmup
    saved = deque()
    for i in range(warmup):
        x = st.comm.recv_forward(i)
        out, inp = st.forward_step(i, x)
        st.comm.send_forward(out, i)
        saved.append((i, inp, out))
    x = st.comm.recv_forward(warmup) if remaining > 0 else None
    for i in range(remaining):
        f_mb = warmup + i
        out, inp = st.forward_step(f_mb, x)
        saved.append((f_mb, inp, out))
        b_mb = saved[-1][0]                               # <- the bug: newest instead of oldest
        g = st.comm.send_forward_recv_backward(out, f_mb, b_mb)
        mb, inp_b, out_b = saved.pop()                    # <- the bug: pop() instead of popleft()
        dx = st.backward_step(mb, inp_b, out_b, g)
        if i == remaining - 1:
            st.comm.send_backward(dx, mb)
            x = None
        else:
            x = st.comm.send_backward_recv_forward(dx, mb, f_mb + 1)
    for _ in range(warmup):
        mb, inp_b, out_b = saved.pop()
        g = st.comm.recv_backward(mb)
        dx = st.backward_step(mb, inp_b, out_b, g)
        st.comm.send_backward(dx, mb)


# 3. mean_per_microbatch: each microbatch's loss is its own mean (no 1/M weight), as in a non-accumulating loop.
def _fwd_mean(st: Stage, mb, received):
    if not st.is_last:
        return pp_ref.forward_step(st, mb, received)
    x = received.requires_grad_() if not st.is_first else st.inputs[mb]
    y = st.module(x)
    st.save_output(mb, y)
    loss = ((y - st.targets[mb]) ** 2).mean()             # <- the bug: mean over the microbatch, not sum / N
    st.save_loss(mb, loss)
    return loss, x


CONTROLS = {
    "detached_input": SimpleNamespace(forward_step=_fwd_detached, backward_step=_bwd_detached,
                                      run_gpipe=pp_ref.run_gpipe, run_1f1b=pp_ref.run_1f1b),
    "lifo_backward": SimpleNamespace(forward_step=pp_ref.forward_step, backward_step=pp_ref.backward_step,
                                     run_gpipe=pp_ref.run_gpipe, run_1f1b=_run_1f1b_lifo),
    "mean_per_microbatch": SimpleNamespace(forward_step=_fwd_mean, backward_step=pp_ref.backward_step,
                                           run_gpipe=pp_ref.run_gpipe, run_1f1b=pp_ref.run_1f1b),
}

STAGE0 = [f"blk{k}.{n}" for k in (0, 1) for n in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")]
STAGE1 = [f"blk{k}.{n}" for k in (2, 3) for n in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")]

EXPECTED = {   # for schedule 1f1b, M = 4, two stages
    "detached_input": set(STAGE0),                         # ratio 0: stage 0 received zero gradients
    # Stage 0's B order and gradient labels are permuted, and every stage-0 gradient that depends on a saved
    # activation is wrong. blk1.fc2.bias is NOT: it is the sum over all tokens of the gradient arriving at stage 0's
    # output, sum_m g_m, and a permutation of the pairs does not change that sum (found in the pre-test).
    "lifo_backward": (set(STAGE0) - {"blk1.fc2.bias"}) | {"order", "labels"},
    "mean_per_microbatch": set(STAGE0) | set(STAGE1) | {"loss"},   # everything scaled by M = 4; y unaffected
}
CONTROL_SCHEDULE, CONTROL_M = "1f1b", 4
