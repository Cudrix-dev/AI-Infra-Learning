"""Day 09 - YOUR version of the two YOU WRITE functions (guide Block 2). Same signatures as amp_setup.py.

Run every check against your code with --impl mine:

  python experiments/day09_mixed_precision/fp16_order.py --impl mine            # YOU WRITE (1): order + control
  python experiments/day09_mixed_precision/probe_amp.py --part ckpt --impl mine # YOU WRITE (2): checkpointing

Then compare with the reference:  diff experiments/day09_mixed_precision/my_amp.py experiments/day09_mixed_precision/amp_setup.py
Read the docstrings in amp_setup.py first: they state the contract (what `on` must be called with, and when).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch  # noqa: F401  (you will need it)
import torch.nn.functional as F  # noqa: F401
from torch.utils.checkpoint import checkpoint  # noqa: F401

from amp_setup import _noop, autocast_ctx  # noqa: F401  (given helpers you may use)


# ============================================================================ YOU WRITE (2): the forward
def model_forward(model, tokens: torch.Tensor, ckpt: bool) -> torch.Tensor:
    """Day 01's TinyTransformer.forward, with every block under checkpoint(..., use_reentrant=False) when ckpt.

    Steps: positions -> token_emb + pos_emb -> the causal mask slice -> each block (checkpointed or not) -> norm -> head.
    Checkpoint the BLOCKS only. Ask yourself: which tensors does checkpoint() keep for each block, and when does the
    block's forward run a second time?
    """
    raise NotImplementedError("YOU WRITE (2): model_forward")


# ============================================================================ YOU WRITE (1): one optimizer step
def window_step(net, optimizer, scaler, micro, *, precision: str, max_norm: float, no_sync=nullcontext,
                bug: str = "none", on=_noop) -> dict:
    """ONE optimizer step over K microbatches (see amp_setup.window_step for the full contract).

    Order to implement (scaler is None for fp32 / bf16):
      zero_grad once
      for each (x, y, w):  [no_sync unless last] autocast forward + cross_entropy; del logits; on("fwd");
                           backward of scaler.scale(loss * w) (or loss * w); on("bwd"); accumulate the logged loss
      scale = current loss scale
      unscale_ (fp16, unless bug == "clip_before_unscale"); on("pre_clip")
      norm = clip_grad_norm_(net.parameters(), max_norm)
      unscale_ HERE instead when bug == "clip_before_unscale"; on("pre_step")
      scaler.step + scaler.update (fp16) or optimizer.step(); on("opt")
      return {"loss": ..., "norm": ..., "scale": ..., "skipped": ...}

    Before writing, answer: why must the scale stay constant over the K microbatches, and what does clip_grad_norm_
    compute if it runs before unscale_?
    """
    raise NotImplementedError("YOU WRITE (1): window_step")
