"""Day 12 - YOUR two-stage pipeline. Fill in the TODOs in the order below (GUIDE Block 2).

Test after each step (from the project root on the dual-A100 host; add `--device cpu` to run on CPU with gloo):

  Step 1  forward_step, backward_step   (autograd across the process boundary, one microbatch)
          -> torchrun --standalone --nproc_per_node=2 experiments/day12_pipeline_parallel/check_pp.py \
                 --impl mine --test boundary --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt
  Step 2  run_gpipe
          -> ... check_pp.py --impl mine --schedules gpipe --microbatches 4 --baseline <same file>
  Step 3  run_1f1b
          -> ... check_pp.py --impl mine --schedules 1f1b --microbatches 4 --baseline <same file>
          then all of them: --schedules gpipe,1f1b --microbatches 1,2,4,8

Each check prints what it compared and exits 0 (passed), 1 (a check failed) or 3 (a TODO still raises).

What you are given (pp_common.py, read its docstrings):
  st.stage, st.P, st.is_first, st.is_last     which stage this rank runs (stage == rank here)
  st.module                                   this stage's blocks (an nn.Module); call st.module(x)
  st.inputs[m]   (first stage)                microbatch m of the data, [seq, batch/M, d]
  st.targets[m]  (last stage)                 microbatch m of the target
  st.n_total                                  elements in the FULL-batch output; mb_loss() divides by it
  st.save_output(m, y), st.save_loss(m, l)    last stage: record y and the loss for the parity check
  st.comm.recv_forward(m), .send_forward(t, m), .recv_backward(m), .send_backward(t, m),
  st.comm.send_forward_recv_backward(t, send_mb, recv_mb), .send_backward_recv_forward(t, send_mb, recv_mb)
                                              the P2P helpers (no-ops where there is no neighbour)
  st.forward_step(m, received), st.backward_step(m, input, output, grad_output)
                                              LOGGED wrappers around YOUR two functions below: schedules call these,
                                              so the event log sees every op and its microbatch id

Rules the checker relies on:
  * schedules call st.forward_step / st.backward_step (not forward_step directly), so every op is logged;
  * pass the true microbatch id to every call: the labels are how the checker matches the two ranks' logs;
  * keep the names and signatures: the checker imports them by name.
"""

from __future__ import annotations

from collections import deque  # noqa: F401  (you will want it for the saved microbatches)

import torch  # noqa: F401

from pp_common import Stage, mb_loss  # noqa: F401


# ---------------------------------------------------------------------------------------------------------------
# Step 1: one stage, one microbatch. GUIDE §2.
# ---------------------------------------------------------------------------------------------------------------


def forward_step(st: Stage, mb: int, received):
    """Run this stage's blocks on microbatch mb. Return (output, input_tensor).

    first stage : input is st.inputs[mb] (data; `received` is None)
    other stages: input is `received`, a plain buffer that arrived over P2P
    last stage  : output is this microbatch's loss (mb_loss), after st.save_output and st.save_loss
    other stages: output is the stage's activation (it will be sent; keep the tensor itself for backward)
    """
    # TODO(1a): choose the input. On a non-first stage, what must you do to `received` so that, after backward,
    #           the gradient d loss / d received exists on THIS rank? (Hint: it must become a leaf of this rank's
    #           graph. What attribute does autograd check on a leaf?)
    # TODO(1b): run st.module on it.
    # TODO(1c): last stage: st.save_output(mb, y); loss = mb_loss(y, st.targets[mb], st.n_total);
    #           st.save_loss(mb, loss); return the loss as the output. Other stages: return the activation.
    if st.is_first:
        x = st.inputs[mb]
    else:
        x = received.requires_grad_()  # Track gradients on the received leaf.
    y = st.module(x)
    if st.is_last:
        st.save_output(mb, y)
        loss = mb_loss(y, st.targets[mb], st.n_total)
        st.save_loss(mb, loss)
        return loss, x
    return y, x


def backward_step(st: Stage, mb: int, input_tensor, output, grad_output):
    """Backward of this stage for microbatch mb. Return d loss / d input_tensor (None on the first stage).

    output       what forward_step returned for this microbatch (the loss, or the activation)
    grad_output  None on the last stage; otherwise the received d loss / d output, same shape as output
    """
    # TODO(1d): last stage: start backward from the scalar loss. Other stages: `output` is not a scalar, so
    #           torch.autograd.backward needs the upstream gradient explicitly (which argument?).
    # TODO(1e): return the gradient that the previous stage needs (where did autograd put it?), or None on the
    #           first stage.
    if st.is_last:
        output.backward()  # the first returned value in forward_step
    else:
        torch.autograd.backward(output, grad_tensors=grad_output)
    if st.is_first:
        return None
    return input_tensor.grad


# ---------------------------------------------------------------------------------------------------------------
# Step 2: GPipe. GUIDE §3.1.
# ---------------------------------------------------------------------------------------------------------------


def run_gpipe(st: Stage, M: int) -> None:
    """All M forwards, then all M backwards in ASCENDING microbatch order (the order the simulator assumes)."""
    # TODO(2a): forward loop over m = 0..M-1: recv_forward(m) -> st.forward_step(m, x) -> send_forward(out, m).
    #           Keep (m, input, output) for every microbatch: what would go wrong if you kept only the output?
    # TODO(2b): backward loop: take the OLDEST saved microbatch, recv_backward(its id) -> st.backward_step(...)
    #           -> send_backward(dx, its id). Why must both stages walk the microbatches in the same order?
    queue = deque()
    # forward loop
    for m in range(M):
        received = st.comm.recv_forward(m)
        output, input_tensor = st.forward_step(m, received)
        st.comm.send_forward(output, m)
        queue.append((m, input_tensor, output))
    # backward loop
    while queue:
        m, input_tensor, output = queue.popleft()
        grad_output = st.comm.recv_backward(m)
        input_grad = st.backward_step(m, input_tensor, output, grad_output)
        st.comm.send_backward(input_grad, m)


# ---------------------------------------------------------------------------------------------------------------
# Step 3: non-interleaved 1F1B. GUIDE §3.2 and Block 3 Stop 2.
# ---------------------------------------------------------------------------------------------------------------


def run_1f1b(st: Stage, M: int) -> None:
    """Warmup forwards, then steady (F, B) pairs, then cooldown backwards: Megatron's non-interleaved schedule."""
    # TODO(3a): warmup = min(P - stage - 1, M). Run that many forwards exactly as in GPipe's forward loop.
    # TODO(3b): if any microbatches remain, receive the input of the first steady-state forward BEFORE the loop.
    # TODO(3c): steady loop, i = 0 .. M - warmup - 1:
    #             forward microbatch warmup + i and save it;
    #             ONE call that sends that activation and receives the gradient of the OLDEST saved microbatch;
    #             backward of the oldest saved microbatch;
    #             ONE call that sends its input gradient and receives the next forward's input,
    #             except in the last iteration, which only sends.
    #           Use send_forward_recv_backward / send_backward_recv_forward. Why not send_forward followed by
    #           recv_backward? (GUIDE §2.4; try it in the simulator with --comm-style naive --send rendezvous.)
    # TODO(3d): cooldown: the warmup microbatches still need their backwards (recv_backward, backward, send_backward).
    queue = deque()
    warmup = min(st.P - st.stage - 1, M)
    for m in range(warmup):
        received = st.comm.recv_forward(m)
        output, input_tensor = st.forward_step(m, received)
        st.comm.send_forward(output, m)
        queue.append((m, input_tensor, output))
    if warmup < M:
        received = st.comm.recv_forward(
            warmup
        )  # receive the input of the first steady-state forward
    for m in range(warmup, M):
        output, input_tensor = st.forward_step(m, received)
        queue.append((m, input_tensor, output))
        oldest_m, oldest_input_tensor, oldest_output = queue.popleft()
        grad_output = st.comm.send_forward_recv_backward(output, m, oldest_m)
        input_grad = st.backward_step(
            oldest_m, oldest_input_tensor, oldest_output, grad_output
        )
        if m < M - 1:
            received = st.comm.send_backward_recv_forward(input_grad, oldest_m, m + 1)
        else:
            st.comm.send_backward(input_grad, oldest_m)

    # cooldown: the warmup microbatches still need their backwards (recv_backward, backward, send_backward).
    while queue:
        m, input_tensor, output = queue.popleft()
        grad_output = st.comm.recv_backward(m)
        input_grad = st.backward_step(m, input_tensor, output, grad_output)
        st.comm.send_backward(input_grad, m)
