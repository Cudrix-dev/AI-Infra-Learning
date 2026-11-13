"""Day 10 - correctness checks for the row-wise softmax (and the RMSNorm stretch), against torch.

    python experiments/day10_triton_softmax/check_softmax.py --impl ref --controls     # reference + negative controls
    python experiments/day10_triton_softmax/check_softmax.py --impl mine               # your kernel, same checks
    python experiments/day10_triton_softmax/check_softmax.py --impl mine --op rmsnorm  # stretch

What is checked (each case compares with torch.softmax in the SAME dtype, tolerances in day10_common.TOLERANCES):
  sweep       33 row lengths from 1 to 65536, most of them NOT powers of two (padding lanes exist), x 3 dtypes,
              x 4 layouts:
                contiguous   [M, N] contiguous, M = 37 (5 for N > 8192)
                single_row   [1, N]: one program in the whole grid
                row_strided  input base[:, :N] of a [M, N + 13] tensor, AND output written into the same kind of view
                             of a buffer pre-filled with a sentinel (123.0): the 13 padding columns after every row
                             must still hold the sentinel afterwards (proves the store is masked)
                nd_rows      [2, 3, 7, N] contiguous (attention-scores-like); rows are the flattened leading dims
  stability   inputs near 1e4 (3e4 in fp16) and with a spread of 1e3: exp() would overflow without the max
              subtraction. causal_mask: -inf above the diagonal, as in attention. all_neg_inf: a fully masked row
              must give NaN, like torch.
  contract    unsupported inputs raise ValueError (float64, CPU, non-contiguous last dim, N > MAX_COLS); empty works.
Also reported, not pass/fail: the error of BOTH implementations against a float64 softmax, and max |row sum - 1|.

The run ends with `SOFTMAX CHECK: PASSED` (or `RMSNORM CHECK: PASSED`) and, with --controls, `CONTROLS: ALL DETECTED`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from day10_common import DTYPES, OUT_DIR, TOLERANCES, env_info, is_not_implemented, load_impl, write_json  # noqa: E402

NS = [1, 2, 3, 5, 16, 17, 64, 100, 127, 128, 129, 255, 256, 257, 500, 781, 1000, 1023, 1024, 1025,
      2047, 2048, 3000, 4096, 5000, 8191, 8192, 10007, 16384, 20000, 32768, 50000, 65536]
NS_QUICK = [1, 3, 17, 100, 128, 129, 781, 1024, 1025, 3000, 8192, 10007, 65536]
PAD = 13  # padding columns after each row in the row_strided layout (odd on purpose: row stride is not 16-aligned)
SENTINEL = 123.0  # exactly representable in fp32/fp16/bf16, and never a softmax output (outputs are in [0, 1])


def make_x(shape, dtype, gen, scale=1.0, offset=0.0):
    return (torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32) * scale + offset).to(dtype)


def close(y, ref, dtype) -> bool:
    return torch.allclose(y.float(), ref.float(), equal_nan=True, **TOLERANCES[dtype])


def max_abs(a, b) -> float:
    d = (a.double() - b.double()).abs()
    d = d[~torch.isnan(d)]
    return float(d.max()) if d.numel() else 0.0


class Recorder:
    def __init__(self):
        self.rows, self.fail = [], []

    def add(self, **row):
        self.rows.append(row)
        if not row["ok"]:
            self.fail.append(row)
            print(f"  FAIL {row}")


def run_layout(impl, rec, op, dtype_name, n, layout, gen):
    dtype = DTYPES[dtype_name]
    sentinel_ok = True
    if layout == "contiguous":
        x = make_x((37 if n <= 8192 else 5, n), dtype, gen)
        y = call(impl, op, x, gen)
    elif layout == "single_row":
        x = make_x((1, n), dtype, gen)
        y = call(impl, op, x, gen)
    elif layout == "nd_rows":
        x = make_x((2, 3, 7, n), dtype, gen)
        y = call(impl, op, x, gen)
    elif layout == "row_strided":
        m = 19
        x = make_x((m, n + PAD), dtype, gen)[:, :n]
        out_base = torch.full((m, n + PAD), SENTINEL, dtype=dtype, device="cuda")
        out = out_base[:, :n]
        y = call(impl, op, x, gen, out=out)
        sentinel_ok = bool((out_base[:, n:] == SENTINEL).all())
        assert y.data_ptr() == out.data_ptr()
    else:
        raise ValueError(layout)
    ref = reference(op, x, gen)
    ok_vals = close(y, ref, dtype)
    row = dict(op=op, dtype=dtype_name, n=n, layout=layout, shape=list(x.shape), ok=ok_vals and sentinel_ok,
               values_ok=ok_vals, sentinel_ok=sentinel_ok, max_abs_vs_torch=max_abs(y, ref))
    if op == "softmax":
        ref64 = torch.softmax(x.double(), dim=-1)
        row.update(max_abs_vs_fp64=max_abs(y, ref64), torch_max_abs_vs_fp64=max_abs(ref, ref64),
                   max_rowsum_err=float((y.double().sum(-1) - 1).abs().max()))
    rec.add(**row)


def call(impl, op, x, gen, out=None):
    if op == "softmax":
        return impl.softmax(x, out=out) if out is not None else impl.softmax(x)
    if out is not None:  # rmsnorm has no out=; emulate by copying, which still checks values (not the store mask)
        out.copy_(impl.rmsnorm(x, weight_for(x)))
        return out
    return impl.rmsnorm(x, weight_for(x))


_WEIGHTS = {}


def weight_for(x):
    key = (x.shape[-1], x.dtype)
    if key not in _WEIGHTS:
        g = torch.Generator(device="cuda").manual_seed(1234 + x.shape[-1])
        _WEIGHTS[key] = (1 + 0.1 * torch.randn(x.shape[-1], generator=g, device="cuda")).to(x.dtype)
    return _WEIGHTS[key]


def reference(op, x, gen):
    if op == "softmax":
        return torch.softmax(x, dim=-1)
    # fp32 RMSNorm, rounded once to the input dtype (the same precision policy as the kernel)
    w = weight_for(x)
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), w.float(), 1e-6).to(x.dtype)


def stability_cases(impl, rec, dtype_name, gen):
    """Inputs where exp(x) overflows unless the row max is subtracted first, plus attention-style -inf masks."""
    dtype = DTYPES[dtype_name]
    offset = 3e4 if dtype == torch.float16 else 1e4  # fp16's largest finite value is 65504
    cases = {
        "large_offset": make_x((16, 1000), dtype, gen, offset=offset),
        "large_spread": make_x((16, 1000), dtype, gen, scale=1e3),
    }
    t = 129  # causal mask like attention scores [T, T]: entry (i, j) = -inf for j > i; row 0 has one finite entry
    causal = make_x((t, t), dtype, gen)
    causal.masked_fill_(torch.ones(t, t, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
    cases["causal_mask"] = causal
    allinf = make_x((4, 300), dtype, gen)
    allinf[2] = float("-inf")  # a fully masked row: torch gives NaN for the whole row, and so must we
    cases["all_neg_inf_row"] = allinf
    for name, x in cases.items():
        y = impl.softmax(x)
        ref = torch.softmax(x, dim=-1)
        ok = close(y, ref, dtype)
        finite_rows = torch.isfinite(ref).all(-1)
        extra = {}
        if name == "all_neg_inf_row":
            ok = ok and bool(torch.isnan(y[2]).all()) and bool(torch.isfinite(y[[0, 1, 3]]).all())
        if name == "causal_mask":
            ok = ok and bool((y.float().masked_select(torch.isinf(x)) == 0).all())  # masked entries exactly 0
        if name.startswith("large"):
            naive = torch.exp(x.float())  # what you get without subtracting the max
            extra["naive_exp_nonfinite_frac"] = float((~torch.isfinite(naive / naive.sum(-1, keepdim=True))).float()
                                                      .mean())
        rec.add(op="softmax", dtype=dtype_name, n=x.shape[-1], layout=f"stability:{name}", shape=list(x.shape),
                ok=ok, values_ok=ok, sentinel_ok=True, max_abs_vs_torch=max_abs(y, ref),
                max_rowsum_err=float((y.double()[finite_rows].sum(-1) - 1).abs().max()), **extra)


def contract_cases(impl, op) -> list[str]:
    """Unsupported inputs must raise ValueError (not crash, not silently compute garbage)."""
    problems = []
    w = torch.ones(8, device="cuda")

    def f(x):
        return impl.softmax(x) if op == "softmax" else impl.rmsnorm(x, w.to(x.dtype) if x.is_cuda else w)

    bad = {
        "float64": torch.randn(4, 8, device="cuda", dtype=torch.float64),
        "cpu": torch.randn(4, 8),
        "last_dim_not_contiguous": torch.randn(8, 4, device="cuda").t(),
        "too_long_row": torch.randn(1, 65537, device="cuda"),
    }
    for name, x in bad.items():
        try:
            f(x)
            problems.append(f"{name}: no error")
        except ValueError:
            pass
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name}: {type(e).__name__} instead of ValueError")
    if op == "softmax":
        empty = impl.softmax(torch.empty(0, 8, device="cuda"))
        if empty.shape != (0, 8):
            problems.append("empty input: wrong shape")
    return problems


def run_controls(gen) -> bool:
    """Run the three broken kernels on an easy case and on the case designed to catch each. A control is DETECTED
    if the easy case passes (the bug hides) and the targeted case fails (the check catches it)."""
    from softmax_controls import softmax_control
    from softmax_ref import launch_config

    def strided_out(m, n, dtype):
        block = launch_config(n).block_size
        base = torch.full((m, block + 8), SENTINEL, dtype=dtype, device="cuda")  # wide enough for unmasked stores
        return base, base[:, :n]

    print("\n== negative controls (broken kernels) ==")
    all_detected = True
    dt = torch.float32
    plan = {
        "no_max": ("random rows N=1000", lambda: make_x((16, 1000), dt, gen),
                   "rows near 1e4 (stability)", lambda: make_x((16, 1000), dt, gen, offset=1e4)),
        "zero_fill": ("N=1024 (no padding lanes)", lambda: make_x((16, 1024), dt, gen),
                      "N=1000 (24 padding lanes)", lambda: make_x((16, 1000), dt, gen)),
        "no_store_mask": ("N=1024 (no padding lanes)", lambda: make_x((16, 1024), dt, gen),
                          "N=1000 (24 padding lanes)", lambda: make_x((16, 1000), dt, gen)),
    }
    for bug, (easy_name, easy, hard_name, hard) in plan.items():
        verdicts = []
        for x in (easy(), hard()):
            m, n = x.shape
            base, out = strided_out(m, n, dt)
            y = softmax_control(x, bug, out=out)
            vals = close(y, torch.softmax(x, -1), dt)
            sent = bool((base[:, n:] == SENTINEL).all())
            verdicts.append((vals, sent))
        easy_pass = all(verdicts[0])
        hard_pass = all(verdicts[1])
        detected = easy_pass and not hard_pass
        all_detected &= detected
        how = "values" if not verdicts[1][0] else "sentinel padding"
        print(f"  {bug:<14} {easy_name}: {'pass' if easy_pass else 'FAIL'} | {hard_name}: "
              f"{'pass' if hard_pass else 'FAIL'}  => {'DETECTED by ' + how if detected else 'NOT DETECTED'}")
    print("CONTROLS: ALL DETECTED" if all_detected else "CONTROLS: SOME NOT DETECTED")
    return all_detected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--op", choices=["softmax", "rmsnorm"], default="softmax")
    ap.add_argument("--controls", action="store_true", help="also run the three broken kernels (needs --impl ref)")
    ap.add_argument("--quick", action="store_true", help=f"{len(NS_QUICK)} row lengths instead of {len(NS)}")
    ap.add_argument("--run-id", default=None, help="write outputs/day10_softmax/check_<op>_<impl>_<run-id>.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    impl = load_impl(args.impl)
    gen = torch.Generator(device="cuda").manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    env = env_info()
    print(f"impl={args.impl} op={args.op} | torch {env['torch']} triton {env['triton']} | {env['gpu']}")

    rec = Recorder()
    t0 = time.perf_counter()
    try:
        for dtype_name in DTYPES:
            for n in NS_QUICK if args.quick else NS:
                for layout in ("contiguous", "single_row", "row_strided", "nd_rows"):
                    if layout == "nd_rows" and n > 4096:
                        continue
                    run_layout(impl, rec, args.op, dtype_name, n, layout, gen)
            if args.op == "softmax":
                stability_cases(impl, rec, dtype_name, gen)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        if is_not_implemented(e):
            print(f"\n{args.op} for impl={args.impl} is not implemented yet: fill in the YOU WRITE part of "
                  f"my_softmax.py.\n({type(e).__name__})")
            return 3
        raise
    contract_problems = contract_cases(impl, args.op)
    elapsed = time.perf_counter() - t0

    print(f"\n== summary by dtype ({len(rec.rows)} cases, {elapsed:.1f} s including JIT compiles) ==")
    print(f"{'dtype':<5} {'cases':>5} {'fail':>4}  {'max|y-torch|':>12}  {'max|y-fp64|':>11}  {'torch:|.-fp64|':>14}"
          f"  {'max|rowsum-1|':>13}  {'rtol/atol':>14}")
    for d in DTYPES:
        rows = [r for r in rec.rows if r["dtype"] == d]
        if not rows:
            continue
        mx = lambda k: max((r.get(k, 0.0) for r in rows), default=0.0)  # noqa: E731
        tol = TOLERANCES[DTYPES[d]]
        print(f"{d:<5} {len(rows):>5} {sum(not r['ok'] for r in rows):>4}  {mx('max_abs_vs_torch'):>12.3e}  "
              f"{mx('max_abs_vs_fp64'):>11.3e}  {mx('torch_max_abs_vs_fp64'):>14.3e}  {mx('max_rowsum_err'):>13.3e}  "
              f"{tol['rtol']:.1e}/{tol['atol']:.0e}")
    if args.op == "softmax":
        print("\nstability cases (naive = exp without the max subtraction, computed in fp32 by torch):")
        for r in rec.rows:
            if r["layout"].startswith("stability"):
                naive = (f"  naive non-finite: {r['naive_exp_nonfinite_frac']:.0%}"
                         if "naive_exp_nonfinite_frac" in r else "")
                print(f"  {r['dtype']:<5} {r['layout'][10:]:<16} {'pass' if r['ok'] else 'FAIL'}"
                      f"  max|y-torch| {r['max_abs_vs_torch']:.2e}{naive}")
        sent_cases = [r for r in rec.rows if r["layout"] == "row_strided"]
        print(f"\nmasked store: sentinel padding intact in {sum(r['sentinel_ok'] for r in sent_cases)}/"
              f"{len(sent_cases)} row_strided cases")
    print("contract (ValueError for unsupported inputs): " + ("ok" if not contract_problems else
                                                             "PROBLEMS: " + "; ".join(contract_problems)))

    passed = not rec.fail and not contract_problems
    controls_ok = None
    if args.controls:
        if args.impl != "ref" or args.op != "softmax":
            print("--controls ignored: it only applies to --impl ref --op softmax")
        else:
            controls_ok = run_controls(gen)
    label = "SOFTMAX" if args.op == "softmax" else "RMSNORM"
    print(f"\n{label} CHECK: {'PASSED' if passed else 'FAILED'} (impl={args.impl}, {len(rec.rows)} cases, "
          f"{len(rec.fail)} failed)")
    if args.run_id:
        write_json(OUT_DIR / f"check_{args.op}_{args.impl}_{args.run_id}.json",
                   dict(env=env, impl=args.impl, op=args.op, passed=passed, controls_detected=controls_ok,
                        contract_problems=contract_problems, tolerances={str(k): v for k, v in TOLERANCES.items()},
                        cases=rec.rows))
    return 0 if passed and controls_ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
