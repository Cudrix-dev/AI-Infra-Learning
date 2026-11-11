"""Day 09 - mixed precision and activation checkpointing: the two pieces you write, and the small helpers around them.

Two parts are YOURS today (guide Block 2). Both are short; what matters is the ORDER of the calls.

    YOU WRITE (1)  window_step: ONE optimizer step over K microbatches in fp32, bf16 or fp16, with Day 04's loss
                   weights, gradient clipping and, for fp16, GradScaler. The only correct order is

                       for each microbatch:  autocast forward + loss -> scaler.scale(loss * w_j).backward()
                       once per window:      scaler.unscale_ -> clip_grad_norm_ -> scaler.step -> scaler.update

                   plus one negative control, bug="clip_before_unscale", that swaps the middle two calls.
    YOU WRITE (2)  model_forward: Day 01's TinyTransformer.forward, with the option to checkpoint every block.

my_amp.py has the same two signatures with empty bodies. Write them there, run the checks with `--impl mine`
(fp16_order.py and probe_amp.py --part ckpt), then diff against this file.

Everything else here is given: the autocast context, the GradScaler factory, the SDPA backend switch, and AmpNet (a
two-line nn.Module so DDP can wrap "Day 01's model + this forward").
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

PRECISIONS = ("fp32", "bf16", "fp16")  # the three the harness trains with
PROBE_ONLY = ("bf16_pure",)  # probe_amp.py only: the whole model cast to bf16, AdamW on bf16 params, NO fp32 master
LOW_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


# ============================================================================ given helpers
def autocast_ctx(precision: str, device_type: str = "cuda"):
    """torch.autocast for bf16 / fp16; nothing for fp32 and bf16_pure.

    autocast does NOT change any tensor you own. Inside the region, each op on autocast's lower-precision list
    (matmul-like: mm, addmm, bmm, linear, ...) casts its floating inputs to `dtype` before running; ops on the fp32 list
    (layer_norm, ...) cast to float32; everything else runs in whatever dtype its inputs already have. Parameters stay
    fp32, so the gradients autograd accumulates into p.grad are fp32 too. Run the probe to see the exact op list.
    """
    if precision in LOW_DTYPE:
        return torch.autocast(device_type, dtype=LOW_DTYPE[precision])
    return nullcontext()


def make_scaler(precision: str, enabled: bool = True, scaler_cls=None):
    """GradScaler for fp16 only (bf16 has fp32's exponent range: nothing underflows that fp32 would keep).

    Defaults: init_scale 2**16, growth_factor 2, backoff_factor 0.5, growth_interval 2000 optimizer steps.
    enabled=False builds a disabled scaler (scale() returns its input, unscale_ does nothing, get_scale() is 1.0):
    fp16_order.py --underflow uses it to show what happens WITHOUT loss scaling.
    """
    if precision != "fp16":
        return None
    cls = scaler_cls or torch.amp.GradScaler
    return cls("cuda", enabled=enabled)


def sdpa_context(kind: str):
    """'math' = the plain matmul + softmax backend, as on Days 01-08 (deterministic, and the attention matrix is
    materialized). 'auto' = let PyTorch pick (flash / memory-efficient attention for bf16/fp16): optional rows only."""
    if kind == "math":
        return sdpa_kernel(SDPBackend.MATH)
    if kind == "auto":
        return nullcontext()
    raise ValueError(kind)


class AmpNet(nn.Module):
    """Day 01's model with `forward_fn(model, tokens, ckpt)` as its forward. Parameters are the model's (names get a
    'model.' prefix). It exists so that DDP can wrap the checkpointed forward: DDP must be the module you CALL."""

    def __init__(self, model: nn.Module, ckpt: bool, forward_fn):
        super().__init__()
        self.model, self.ckpt, self.forward_fn = model, ckpt, forward_fn

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_fn(self.model, tokens, self.ckpt)


def _noop(event: str) -> None:
    return None


def load_impl(name: str):
    """'ref' = this file, 'mine' = my_amp.py (your version of the two YOU WRITE functions)."""
    return importlib.import_module("amp_setup" if name == "ref" else "my_amp")


# ============================================================================ YOU WRITE (2): the forward
def model_forward(model, tokens: torch.Tensor, ckpt: bool) -> torch.Tensor:
    """Day 01's TinyTransformer.forward (train.py), with every block optionally under activation checkpointing.

    Without checkpointing, autograd keeps every tensor each op saved for backward (Day 01's memory probe: LN inputs,
    QKV, the attention probabilities [B, H, T, T], the FFN hidden [B, T, 4D], ...) until backward reaches that op.

    checkpoint(block, h, mask, use_reentrant=False) runs the block forward WITHOUT keeping those tensors: a
    saved-tensor hook drops each one and remembers only how to get it back. What it does keep is the block's INPUTS
    (h and the mask) for the recompute. When backward needs the first dropped tensor of a block, it re-runs that
    block's forward (same autocast state, same RNG state) to rebuild the tensors, then continues. Cost: one more
    forward per block per step. Saving: all of a block's internal activations, minus one block's worth that is alive
    during its own recompute.

    use_reentrant=False is the recommended implementation: it works with torch.autograd.grad, with inputs that do not
    require grad, and with DDP. (The reentrant one is the old autograd.Function version; read CheckpointFunction.)

    Why YOU write it: you should be able to point at the one line that decides what is saved vs recomputed. The
    embedding, the final LayerNorm, the head and the loss are NOT checkpointed here; the probe shows what that leaves.
    """
    length = tokens.size(1)
    pos = torch.arange(length, device=tokens.device)
    h = model.token_emb(tokens) + model.pos_emb(pos)  # [B, T, D]
    mask = model.causal_mask[:length, :length]
    for block in model.blocks:
        if ckpt:
            # positional args: TransformerEncoderLayer.forward(src, src_mask=None, ...)
            h = checkpoint(block, h, mask, use_reentrant=False)
        else:
            h = block(h, src_mask=mask)
    return model.head(model.norm(h))  # [B, T, V]


# ============================================================================ YOU WRITE (1): one optimizer step
def window_step(net, optimizer, scaler, micro, *, precision: str, max_norm: float, no_sync=nullcontext,
                bug: str = "none", on=_noop) -> dict:
    """ONE optimizer step over the microbatches of one window (Day 04's contract), in `precision`.

    micro     list of (x, y, w): w = rows_j / rows_in_window, so sum_j w_j * mean_loss_j = the window's mean loss
    scaler    a GradScaler for fp16 (from make_scaler), else None
    no_sync   DDP's no_sync for microbatches that must NOT communicate (all but the last); nullcontext on one GPU.
              It must wrap the FORWARD too (Day 04): DDP decides in forward whether the coming backward reduces.
    bug       "none" or "clip_before_unscale" (the negative control)
    on(event) the harness's callback; call it at exactly these points, nothing else depends on it:
                "fwd"      after each microbatch's forward + loss (logits already deleted)
                "bwd"      after each microbatch's backward
                "pre_clip" just before clip_grad_norm_ (the gradients clipping is about to see)
                "pre_step" just before the optimizer step (the gradients AdamW is about to use)
                "opt"      after the optimizer step and scaler.update()

    Returns {"loss": window loss (fp32 tensor, correctly weighted), "norm": the total norm clip_grad_norm_ returned,
             "scale": the loss scale used for this window (1.0 without a scaler), "skipped": did GradScaler skip?}

    The order, and why each step sits where it does:
      1. zero_grad ONCE per window: accumulation means p.grad += across microbatches (Day 04).
      2. forward + loss INSIDE autocast; backward OUTSIDE (autograd replays each op's backward in the dtype its forward
         ran in, so it needs no autocast region of its own).
      3. scaler.scale(loss * w_j).backward(): fp16 gradients of small values would underflow to 0 (fp16's smallest
         subnormal is 6e-8). Multiplying the loss by S (2**16) multiplies every gradient by S, so they survive the fp16
         activation-gradient path. p.grad (fp32, the params are fp32) accumulates S * sum_j w_j g_j.
         The scale must stay CONSTANT over the window, or the microbatches would be summed with different factors:
         that is why scaler.update() is called once, after the step, never inside the loop.
      4. scaler.unscale_(optimizer): p.grad /= S in place, and records found_inf if any gradient is inf/nan. Once per
         window (GradScaler raises if you call it twice before update()).
      5. clip_grad_norm_(params, max_norm): computes ||p.grad|| over all params and rescales if above max_norm. It knows
         nothing about S: it must see the TRUE gradient, so it comes after unscale_ (the bug: before).
      6. scaler.step(optimizer): skips optimizer.step() if found_inf (the scale was too big), else steps. It does not
         unscale again, because unscale_ already ran for this optimizer.
      7. scaler.update(): halve S after a skipped step, or double it after growth_interval good steps.
    """
    optimizer.zero_grad(set_to_none=True)
    window_loss = torch.zeros((), device=micro[0][0].device)
    last = len(micro) - 1
    for j, (x, y, w) in enumerate(micro):
        with no_sync() if j < last else nullcontext():
            with autocast_ctx(precision):
                logits = net(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            del logits
            on("fwd")
            weighted = loss * w
            (scaler.scale(weighted) if scaler is not None else weighted).backward()
        on("bwd")
        window_loss += loss.detach().float() * w
    scale = scaler.get_scale() if scaler is not None else 1.0
    if scaler is not None and bug != "clip_before_unscale":
        scaler.unscale_(optimizer)
    on("pre_clip")
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    if scaler is not None and bug == "clip_before_unscale":
        scaler.unscale_(optimizer)  # BUG: clip saw S * g, so it scaled the gradient down by about 1 / (S * ||g||)
    on("pre_step")
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    on("opt")
    skipped = scaler is not None and scaler.get_scale() < scale  # update() halves S after an inf/nan window
    return {"loss": window_loss, "norm": norm.detach(), "scale": float(scale), "skipped": bool(skipped)}
