"""Day 10 - YOUR row-wise fused softmax (and, optionally, your RMSNorm).

Fill in the kernel body below (YOU WRITE), then run the same checks and benchmark as the reference:

    python experiments/day10_triton_softmax/check_softmax.py --impl mine     # must print SOFTMAX CHECK: PASSED
    python experiments/day10_triton_softmax/bench_softmax.py --impl mine --stage quick

The wrapper `softmax()` is given and identical in behaviour to the reference: it validates the input with the shared
`prepare_rows` (so both implementations accept and reject the same inputs), picks BLOCK_SIZE and num_warps with the
shared `launch_config`, and launches ONE PROGRAM PER ROW. You only write what one program does with one row.

Do not open softmax_ref.py's kernel until your own version passes or you are stuck for more than 20 minutes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from softmax_ref import launch_config, prepare_rows


@triton.jit
def my_softmax_kernel(
    out_ptr,  # *T output; element (r, c) lives at out_ptr + r * out_row_stride + c
    in_ptr,  # *T input;  element (r, c) lives at in_ptr  + r * in_row_stride  + c
    in_row_stride,  # in ELEMENTS, not bytes (Triton pointer arithmetic is typed, like C)
    out_row_stride,  # may differ from in_row_stride (the check passes a padded output view)
    n_cols,  # the real row length N (runtime value)
    BLOCK_SIZE: tl.constexpr,  # power of two >= n_cols (compile-time constant)
):
    # ---------------------------------------- YOU WRITE (about 12 lines) ----------------------------------------
    # 1. Which row is mine?            tl.program_id(axis=0). The grid is (n_rows,), so program i owns row i.
    #    Convert it to int64 before multiplying by a stride: row * stride can exceed 2^31 on large tensors.
    row = tl.program_id(0).to(tl.int64)
    # 2. Column indices and the mask.  tl.arange(0, BLOCK_SIZE); lanes >= n_cols are padding.
    cols = tl.arange(0, BLOCK_SIZE)
    # 3. Masked load.                  tl.load(ptr, mask=..., other=...). Choose `other` so that padding lanes change
    #                                  neither the row max nor the sum of exponentials. (Which value is that? Why is
    #                                  0.0 wrong? The check has a non-power-of-two case that catches it.)
    valid = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=valid, other=float("-inf"))
    # 4. Compute in fp32.              .to(tl.float32) right after the load; fp16/bf16 inputs must not be summed in
    #                                  their own precision.
    x = x.to(tl.float32)
    # 5. Stable softmax.               m = tl.max(x, axis=0); e = tl.exp(x - m); s = tl.sum(e, axis=0); y = e / s.
    #                                  The stability case (inputs around 1e4) fails if you skip the subtraction.
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    # 6. Masked store, cast once.      tl.store(ptr, y.to(out_ptr.dtype.element_ty), mask=...). Without the mask the
    #                                  padding lanes are written past the row; the check's sentinel padding catches it.
    tl.store(
        out_ptr + row * out_row_stride + cols,
        y.to(out_ptr.dtype.element_ty),
        mask=valid,
    )

    # --------------------------------------------------------------------------------------------------------------


def softmax(
    x: torch.Tensor, out: torch.Tensor | None = None, num_warps: int | None = None
) -> torch.Tensor:
    """Same contract as softmax_ref.softmax (read its module docstring)."""
    x2, out2 = prepare_rows(x, out)
    n_rows, n_cols = x2.shape
    if n_rows == 0 or n_cols == 0:
        return out2.view(x.shape)
    cfg = launch_config(n_cols, num_warps)
    my_softmax_kernel[(n_rows,)](
        out2,
        x2,
        x2.stride(0),
        out2.stride(0),
        n_cols,
        BLOCK_SIZE=cfg.block_size,
        num_warps=cfg.num_warps,
    )
    return out2 if x.dim() == 2 else out2.view(x.shape)


KERNEL = my_softmax_kernel  # launched directly (no wrapper) by bench_softmax.py's `triton_raw` row


def compiled_softmax_kernel(x: torch.Tensor, num_warps: int | None = None):
    """Launch once and return Triton's CompiledKernel (used by inspect_kernels.py --impl mine)."""
    x2, out2 = prepare_rows(x)
    cfg = launch_config(x2.shape[1], num_warps)
    return my_softmax_kernel[(x2.shape[0],)](
        out2,
        x2,
        x2.stride(0),
        out2.stride(0),
        x2.shape[1],
        BLOCK_SIZE=cfg.block_size,
        num_warps=cfg.num_warps,
    )


# ---------------------------------------- STRETCH (optional): RMSNorm ----------------------------------------
# y = x / sqrt(mean(x^2) + eps) * weight, weight shape [N]. Same one-program-per-row structure. Think about:
# which padding value is neutral for a SUM of squares, whether the mean divides by n_cols or BLOCK_SIZE, and how
# many bytes per row the kernel must move compared with softmax (the weight vector is shared by all rows).
# Check it with:  python experiments/day10_triton_softmax/check_softmax.py --impl mine --op rmsnorm
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError(
        "my rmsnorm is not implemented yet (STRETCH in my_softmax.py)"
    )
