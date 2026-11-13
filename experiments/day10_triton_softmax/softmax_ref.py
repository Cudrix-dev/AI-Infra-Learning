"""Day 10 - reference row-wise fused softmax in Triton (and the RMSNorm stretch reference).

This is the REFERENCE. Write your own kernel in `my_softmax.py` and run every script with `--impl mine`.

Contract of `softmax(x)` (the same for `my_softmax.softmax`):

    y = softmax(x, dim=-1)      one independent softmax per row of the LAST dimension

Supported inputs (anything else raises ValueError before a kernel is launched):
  * device:  CUDA
  * dtypes:  float32, float16, bfloat16. Internally every row is computed in float32 and rounded to the input dtype
             only once, when it is stored. torch's CUDA softmax also accumulates half/bf16 in float32, so the two
             should agree to within about one unit in the last place (ulp) of the output dtype.
  * layout:  the last dimension must be contiguous (stride 1). A 2-D input may have ANY row stride (for example the
             column slice `base[:, :N]` of a wider tensor). An N-D input, e.g. attention scores [B, H, T, T], must be
             viewable as [rows, N] (true for any contiguous tensor); its rows are the flattened leading dimensions.
  * row length 1 <= N <= MAX_COLS = 65536. One program holds a whole row in registers (see `launch_config`), so very
             long rows would need a looped ("online") softmax, which is not what this kernel does.
  * `out=` (optional, 2-D inputs only): a CUDA tensor of the same shape and dtype with a contiguous last dimension
             and any row stride. The check script uses it to prove that masked stores never touch padding.

Semantics that follow from the math, matching torch:
  * a row that contains -inf entries gives exactly 0 at those positions (attention's causal mask);
  * a row that is entirely -inf gives NaN everywhere (max = -inf, and -inf - (-inf) = NaN), as torch.softmax does;
  * a row with +inf or NaN gives NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

MAX_COLS = 65536
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


# ----------------------------------------------------------------------------------------------------------------------
# The kernel: ONE PROGRAM INSTANCE PER ROW.
#
# Launch grid = (n_rows,). Program `pid` reads row `pid`, so program instances and rows map 1:1.
# Inside a program, `tl.arange(0, BLOCK_SIZE)` is a vector of BLOCK_SIZE column indices. Triton spreads that vector
# over the program's 32 * num_warps threads: each thread owns BLOCK_SIZE / (32 * num_warps) elements (at least 1;
# if the block is smaller than the thread count, elements are replicated across threads).
# BLOCK_SIZE is a compile-time constant (tl.constexpr), so each distinct BLOCK_SIZE (and num_warps) is a separate
# compiled kernel. It must be a power of two and >= n_cols, so the lanes n_cols..BLOCK_SIZE-1 are padding: `mask`
# switches them off for both the load and the store.
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def softmax_kernel(
    out_ptr,  # *T: output, element (r, c) at out_ptr + r * out_row_stride + c
    in_ptr,  # *T: input,  element (r, c) at in_ptr  + r * in_row_stride  + c
    in_row_stride,  # elements (not bytes) between the starts of two consecutive input rows
    out_row_stride,  # the same for the output; may differ from the input's
    n_cols,  # runtime row length N; NOT a constexpr, so all N with the same BLOCK_SIZE share one compiled kernel
    BLOCK_SIZE: tl.constexpr,  # next power of two >= n_cols
):
    row = tl.program_id(axis=0)
    # int64 before multiplying: row * stride overflows int32 once a tensor has more than 2^31 elements (8 GiB of fp32).
    row_in = in_ptr + row.to(tl.int64) * in_row_stride
    row_out = out_ptr + row.to(tl.int64) * out_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    # Masked load: lanes >= n_cols read nothing from memory and take the value `other`. -inf is the identity of max
    # and exp(-inf - m) = 0 adds nothing to the sum, so padding lanes cannot change either reduction.
    # (other=0.0 would be a bug: 0 can become the row max, and every padding lane would add exp(0 - m) to the sum.)
    x = tl.load(row_in + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)  # fp16/bf16 rows are computed in fp32; a no-op for fp32 inputs

    # Stable softmax: softmax(x) = softmax(x - m) for any constant m. With m = max(x) every exponent is <= 0, so
    # exp() cannot overflow and the largest term is exactly exp(0) = 1, so the sum is >= 1 (no division by ~0).
    row_max = tl.max(x, axis=0)  # reduction across the whole block: across threads and warps (via shared memory)
    z = x - row_max
    num = tl.exp(z)
    den = tl.sum(num, axis=0)
    y = num / den

    # Round once to the output dtype and write back only the real columns. Without the mask, the padding lanes
    # (value 0 here) would be written past the end of the row, into the next row or past the allocation.
    tl.store(row_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@dataclass(frozen=True)
class LaunchConfig:
    block_size: int  # BLOCK_SIZE constexpr: elements per program (power of two >= N)
    num_warps: int  # warps per program; threads = 32 * num_warps

    @property
    def elems_per_thread(self) -> float:
        return self.block_size / (32 * self.num_warps)


def launch_config(n_cols: int, num_warps: int | None = None) -> LaunchConfig:
    """BLOCK_SIZE and num_warps for a row of n_cols elements.

    Heuristic (the one used by the classic Triton softmax tutorial): more warps for longer rows, so the elements per
    thread (= registers per thread) grow more slowly. 4 warps up to 1024, 8 up to 2048, 16 beyond: at N = 65536 that
    is still 128 fp32 values per thread, which exceeds what fits in registers. `inspect_kernels.py` shows the spills.
    `num_warps` overrides the heuristic (bench_softmax.py --stage warps sweeps it).
    """
    block = triton.next_power_of_2(n_cols)
    if num_warps is None:
        num_warps = 4
        if block >= 2048:
            num_warps = 8
        if block >= 4096:
            num_warps = 16
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"num_warps must be a power of two in [1, 32], got {num_warps}")
    return LaunchConfig(block, num_warps)


def prepare_rows(x: torch.Tensor, out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the contract above and return (x2, out2): 2-D [rows, N] views that the kernel indexes directly.

    Shared by the reference and by my_softmax.py, so both implementations accept and reject exactly the same inputs.
    """
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("softmax: x must be a CUDA tensor")
    if x.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"softmax: dtype {x.dtype} not supported (float32, float16, bfloat16)")
    if x.dim() < 1:
        raise ValueError("softmax: x must have at least one dimension")
    n = x.shape[-1]
    if n > MAX_COLS:
        raise ValueError(f"softmax: row length {n} > MAX_COLS={MAX_COLS}; one program holds a whole row in registers")
    if n > 1 and x.stride(-1) != 1:
        raise ValueError("softmax: the last dimension must be contiguous (stride 1); call .contiguous() first")
    if x.dim() == 2:
        x2 = x  # keeps an arbitrary row stride
    else:
        try:
            x2 = x.view(-1, n)  # never copies: fails instead if the leading dims cannot be flattened
        except RuntimeError as e:
            raise ValueError(f"softmax: leading dimensions cannot be flattened into rows without a copy ({e})") from e
    if out is None:
        out2 = torch.empty(x2.shape, dtype=x.dtype, device=x.device)
    else:
        if x.dim() != 2:
            raise ValueError("softmax: out= is only supported for 2-D inputs")
        if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
            raise ValueError("softmax: out must match x in shape, dtype and device")
        if n > 1 and out.stride(-1) != 1:
            raise ValueError("softmax: out's last dimension must be contiguous")
        out2 = out
    return x2, out2


def softmax(x: torch.Tensor, out: torch.Tensor | None = None, num_warps: int | None = None) -> torch.Tensor:
    """Row-wise softmax over the last dimension. See the module docstring for the contract."""
    x2, out2 = prepare_rows(x, out)
    n_rows, n_cols = x2.shape
    if n_rows == 0 or n_cols == 0:
        return out2.view(x.shape)
    cfg = launch_config(n_cols, num_warps)
    grid = (n_rows,)  # one program instance per row
    softmax_kernel[grid](
        out2, x2, x2.stride(0), out2.stride(0), n_cols,
        BLOCK_SIZE=cfg.block_size, num_warps=cfg.num_warps,
    )
    # Host-side cost matters for small inputs: every line of this wrapper runs on the CPU per call (~6 us for
    # prepare_rows in the pre-test), so skip the .view() when there is nothing to reshape.
    return out2 if x.dim() == 2 else out2.view(x.shape)


KERNEL = softmax_kernel  # the uniform name bench_softmax.py's `triton_raw` row launches directly (no wrapper)


def compiled_softmax_kernel(x: torch.Tensor, num_warps: int | None = None):
    """Launch once on x and return Triton's CompiledKernel (registers, spills, shared memory, PTX). Used by
    inspect_kernels.py; not needed for normal use."""
    x2, out2 = prepare_rows(x)
    cfg = launch_config(x2.shape[1], num_warps)
    return softmax_kernel[(x2.shape[0],)](
        out2, x2, x2.stride(0), out2.stride(0), x2.shape[1], BLOCK_SIZE=cfg.block_size, num_warps=cfg.num_warps)


# ----------------------------------------------------------------------------------------------------------------------
# STRETCH (optional): RMSNorm, the same one-program-per-row pattern with a different reduction.
#     y = x / sqrt(mean(x^2) + eps) * weight          weight: [N]
# Differences from softmax worth noticing: the padding value must be 0 (the identity of SUM, not of MAX), the mean
# divides by n_cols (not BLOCK_SIZE), and there is a second input (weight) shared by every row, which is read
# once per program (from L2 after the first programs touch it).
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def rmsnorm_kernel(out_ptr, in_ptr, w_ptr, in_row_stride, out_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_ptr + row.to(tl.int64) * in_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(mean_sq + eps) * w
    tl.store(out_ptr + row.to(tl.int64) * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension; same input contract as softmax, weight is a contiguous [N] tensor of x's
    dtype. Compare with torch.nn.functional.rms_norm(x, (N,), weight, eps)."""
    x2, out2 = prepare_rows(x)
    if weight.shape != (x.shape[-1],) or weight.dtype != x.dtype or not weight.is_contiguous():
        raise ValueError("rmsnorm: weight must be a contiguous [N] tensor with x's dtype")
    n_rows, n_cols = x2.shape
    if n_rows == 0 or n_cols == 0:
        return out2.view(x.shape)
    cfg = launch_config(n_cols)
    rmsnorm_kernel[(n_rows,)](out2, x2, weight, x2.stride(0), out2.stride(0), n_cols, eps,
                              BLOCK_SIZE=cfg.block_size, num_warps=cfg.num_warps)
    return out2.view(x.shape)
