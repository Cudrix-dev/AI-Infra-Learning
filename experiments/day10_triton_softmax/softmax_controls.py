"""Day 10 - negative controls: three deliberately broken variants of the reference kernel.

check_softmax.py --controls runs them to prove that the checks can fail. Each bug passes the "easy" cases and is
caught only by the case designed for it, which is why those cases exist:

  no_max       skip the row-max subtraction       passes random rows; overflows (inf/NaN) on the stability case
  zero_fill    masked loads use other=0.0          passes power-of-two N; wrong whenever N < BLOCK_SIZE
  no_store_mask  store without the mask            right values in the row; overwrites the sentinel padding after it

The flags are constexpr, so each variant is a separately compiled kernel and the reference stays untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from softmax_ref import launch_config, prepare_rows

CONTROLS = ("no_max", "zero_fill", "no_store_mask")


@triton.jit
def _control_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr,
                    SUBTRACT_MAX: tl.constexpr, FILL_NEG_INF: tl.constexpr, MASK_STORE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    ptrs = in_ptr + row.to(tl.int64) * in_row_stride + cols
    if FILL_NEG_INF:
        x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    else:
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)  # BUG (zero_fill)
    if SUBTRACT_MAX:
        x = x - tl.max(x, axis=0)
    num = tl.exp(x)  # BUG (no_max) when SUBTRACT_MAX is False: exp(1e4) = inf
    y = num / tl.sum(num, axis=0)
    out = out_ptr + row.to(tl.int64) * out_row_stride + cols
    if MASK_STORE:
        tl.store(out, y.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(out, y.to(out_ptr.dtype.element_ty))  # BUG (no_store_mask): writes BLOCK_SIZE elements


def softmax_control(x: torch.Tensor, bug: str, out: torch.Tensor | None = None) -> torch.Tensor:
    """Broken softmax. For no_store_mask, ONLY call it with an `out` view whose padding after each row is at least
    BLOCK_SIZE - N elements wide, or the last row writes past the allocation (check_softmax.py sizes it that way)."""
    if bug not in CONTROLS:
        raise ValueError(bug)
    x2, out2 = prepare_rows(x, out)
    cfg = launch_config(x2.shape[1])
    if bug == "no_store_mask":
        assert out is not None and out.stride(0) - x2.shape[1] >= cfg.block_size - x2.shape[1], "unsafe padding"
    _control_kernel[(x2.shape[0],)](
        out2, x2, x2.stride(0), out2.stride(0), x2.shape[1], BLOCK_SIZE=cfg.block_size, num_warps=cfg.num_warps,
        SUBTRACT_MAX=bug != "no_max", FILL_NEG_INF=bug != "zero_fill", MASK_STORE=bug != "no_store_mask")
    return out2.view(x.shape)
