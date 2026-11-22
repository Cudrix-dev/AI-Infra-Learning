"""Day 13 - YOUR two functions of a Megatron Core training step (used with `train_gpt.py --impl mine`).

Megatron's schedule (get_forward_backward_func) owns the loop over microbatches, the P2P transfers and every
backward. What it needs from you is exactly two callables (GUIDE Block 1, Section 2; Block 2, Stage 2):

  forward_step_func(data_iterator, model) -> (output_tensor, loss_func)
      Called once per microbatch on EVERY stage, in the schedule's order.
  loss_func(output_tensor) -> (loss, {"lm loss": value_to_log})
      Called by Megatron only on the LAST stage, with what the model returned.

The reference versions are `ref_forward_step_func` / `ref_loss_func` in train_gpt.py. Read them only after yours
runs, or when stuck. While a TODO remains, train_gpt.py prints `INCOMPLETE (TODOs left)` and exits with code 3
before it creates any process group.
"""

from __future__ import annotations

import torch


def loss_func(output_tensor: torch.Tensor):
    """YOU WRITE (TODO 3, 2-3 lines).

    output_tensor: the per-token cross-entropy GPTModel returns on the last stage when it is given labels,
    shape [micro_batch_size, seq] (FP32).
    Return (loss, {"lm loss": logged}) where
      - loss is the scalar Megatron backpropagates: the mean over this microbatch's tokens. Do NOT divide by the
        number of microbatches yourself - check schedules.py forward_step_calc_loss to see who does.
      - logged is the same value for printing. Careful: Megatron modifies `loss` IN PLACE afterwards
        (`output_tensor /= num_microbatches`). What does that do to a tensor that shares its storage?
    """
    loss = output_tensor.mean()
    return loss, {"lm loss": loss.detach().clone()}


def forward_step_func(data_iterator, model):
    """YOU WRITE (TODO 4, ~5 lines).

    Take the next microbatch (a dict with "tokens", "labels", "position_ids", CPU tensors [b, s]) from
    data_iterator, move it to the current CUDA device, run the model and return (output_tensor, loss_func).

      model(input_ids, position_ids, attention_mask, labels=labels)

    - attention_mask: pass None. The local spec's attention is causal and builds the mask itself.
    - On a stage that is not the first, the model ignores input_ids and uses the activation the schedule received
      and handed over with model.set_input_tensor(...). Should a non-first stage still call next(data_iterator)?
      Each rank has its own iterator over the same microbatches; the last stage needs microbatch k's labels on
      its k-th call. (GUIDE Block 2, Stage 2 answers this.)
    - On a stage that is not the last, output_tensor is the hidden state [s, b, h] the schedule will send on;
      loss_func is then never called.
    """
    microbatch = next(data_iterator)
    device = torch.cuda.current_device()
    tokens = microbatch["tokens"].to(device)
    labels = microbatch["labels"].to(device)
    position_ids = microbatch["position_ids"].to(device)

    output_tensor = model(tokens, position_ids, attention_mask=None, labels=labels)
    return output_tensor, loss_func
