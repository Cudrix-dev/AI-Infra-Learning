"""Day 10 - benchmark: Triton softmax vs eager torch.softmax vs an unfused composition vs torch.compile, on one GPU.

    python experiments/day10_triton_softmax/bench_softmax.py --impl ref --stage core --run-id RID
    python experiments/day10_triton_softmax/bench_softmax.py --impl mine --stage quick
    python experiments/day10_triton_softmax/bench_softmax.py --impl ref --stage launch --run-id RID   # optional

Implementations, all called on the SAME input tensor for a given shape (matched inputs):
  triton      softmax_ref.softmax or my_softmax.softmax (one program per row, one read + one write per element)
  torch       eager torch.softmax(x, -1): ATen's own fused CUDA kernel (also one read + one write)
  naive       eager composition amax -> sub -> exp -> sum -> div: five kernels, three full-size intermediates
  compile     torch.compile(torch.softmax, dynamic=False): Inductor generates its own Triton kernel
  compile_ro  torch.compile(..., mode="reduce-overhead"): the same, replayed through CUDA Graphs (launch stage only)
  compile_ro_static  the same with the input marked static (no input copy before each replay; launch stage only)
  triton_raw  the Triton kernel launched directly, without the Python wrapper (launch stage only)
  copy       out.copy_(x): no math, the same bytes; an ACHIEVED-bandwidth reference for this size, not a softmax

Timing, per implementation and shape (all on one GPU; CUDA_VISIBLE_DEVICES selects it):
  first_ms    wall time of the first call, synchronized. Includes Triton JIT / Dynamo + Inductor compilation when the
              kernel is new in this process (Triton: a new BLOCK_SIZE/num_warps/dtype/alignment specialization).
              The runner points TRITON_CACHE_DIR and TORCHINDUCTOR_CACHE_DIR at fresh directories, so these are cold.
  warm_ms     the slowest of the next 10 synchronized calls (cudagraph-tree recording happens here for compile_ro).
  sync_us     steady state, ONE call then synchronize, median of 30: launch + kernel + synchronization latency.
  loop_us     steady state, `iters` back-to-back calls between two CUDA events, divided by iters, median of 5
              repeats. The CPU enqueues ahead of the GPU, so this is max(per-call CPU submission, GPU time per call).
  graph_us    the same `iters` calls captured once in a CUDA Graph (Day 03) and replayed: per-call CPU submission
              is gone, so this approaches GPU time per call (kernels + small gaps). n/a where capture is impossible.
  loop_us - graph_us  > 0 means the loop was limited by host submission (launch overhead) for this shape.

Bandwidth, one definition for every row:
  bytes  = 2 * M * N * element_size     (each input element read once + each output element written once: the
                                          minimum traffic of ANY softmax, fused or not)
  eff_GBps = bytes / time               "effective bandwidth". It is a property of the workload and the time, not a
                                          measurement of DRAM traffic. The naive composition actually moves about
                                          4x these bytes (5 reads + 3 writes per element), so its effective GB/s is
                                          low by construction. `copy` gives the achieved bandwidth of a plain copy
                                          of the same bytes; the A100-40GB data-sheet peak is 1555 GB/s. Neither is a
                                          hardware counter: measuring actual DRAM bytes needs Nsight Compute.
  L2 flag: when bytes <= the L2 size (40 MiB on A100), back-to-back calls can hit in L2 and eff_GBps can exceed DRAM
  bandwidth; those rows are not DRAM measurements.

Stages (--stage):
  rows      fixed 2^25 elements (128 MiB fp32 in), N = 128 ... 65536, fp32 and bf16: short vs long rows
  count     fixed N (1024 and 16384), M = 1 ... 32768 rows: how the number of rows changes the result
  padding   M = 16384, N = 1000, 1024, 1025, 2048: what a non-power-of-two N costs (BLOCK doubles at 1025)
  warps     Triton only, num_warps 1 ... 32 at N = 1024, 16384, 65536
  launch    (optional) tiny shapes with compile_ro: CUDA Graphs through torch.compile where launches dominate
  quick     a few shapes for checking your kernel; core = rows + count + padding + warps; all = core + launch
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from day10_common import DTYPES, OUT_DIR, PROJECT, TOLERANCES, env_info, is_not_implemented, load_impl, write_json  # noqa: E402,E501

TOTAL = 2**25
STAGES = {
    "rows": [("fp32", TOTAL // n, n) for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)]
    + [("bf16", TOTAL // n, n) for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)],
    "count": [("fp32", m, 1024) for m in (1, 16, 108, 432, 1728, 6912, 32768)]
    + [("fp32", m, 16384) for m in (1, 16, 108, 432, 1728)],
    "padding": [("fp32", 16384, n) for n in (1000, 1024, 1025, 2048)],
    "warps": [("fp32", 32768, 1024), ("fp32", 2048, 16384), ("fp32", 512, 65536)],
    "launch": [("fp32", 1, 128), ("fp32", 16, 1024), ("fp32", 108, 1024), ("fp32", 1, 16384)],
    "quick": [("fp32", TOTAL // 1024, 1024), ("fp32", TOTAL // 16384, 16384), ("fp32", 1, 1024),
              ("fp32", 108, 1024)],
}
IMPLS = {"rows": ["triton", "torch", "naive", "compile", "copy"],
         "count": ["triton", "torch", "naive", "compile", "copy"],
         "padding": ["triton", "torch", "compile", "copy"],
         "launch": ["triton", "triton_raw", "torch", "compile", "compile_ro", "compile_ro_static"],
         "quick": ["triton", "torch", "compile"]}
WARPS = {1024: (1, 2, 4, 8, 16, 32), 16384: (1, 2, 4, 8, 16, 32), 65536: (4, 8, 16, 32)}


def torch_softmax(t):
    return torch.softmax(t, dim=-1)


def naive_softmax(t):
    m = t.amax(dim=-1, keepdim=True)  # kernel 1: read x
    z = t - m  # kernel 2: read x, write z
    e = torch.exp(z)  # kernel 3: read z, write e
    s = e.sum(dim=-1, keepdim=True)  # kernel 4: read e
    return e / s  # kernel 5: read e, write y


def make_fn(name, impl, x, num_warps=None):
    """Return (callable, needs_mark_step). Compiled variants are rebuilt per shape after a Dynamo reset, so every
    shape compiles once and the recompile limit is never reached."""
    if name == "triton":
        return (lambda t: impl.softmax(t, num_warps=num_warps)), False
    if name == "torch":
        return torch_softmax, False
    if name == "naive":
        return naive_softmax, False
    if name == "copy":
        out = torch.empty_like(x)
        return (lambda t: out.copy_(t)), False
    if name == "triton_raw":
        # The same kernel launched directly: no input validation, no allocation (output preallocated), no reshape.
        # triton - triton_raw = the Python wrapper's host cost; triton_raw - graph = Triton's own launcher cost.
        x2, out2 = impl.prepare_rows(x)
        cfg = impl.launch_config(x2.shape[1], num_warps)
        grid, s_in, s_out, n = (x2.shape[0],), x2.stride(0), out2.stride(0), x2.shape[1]

        def raw(t):
            impl.KERNEL[grid](out2, t, s_in, s_out, n, BLOCK_SIZE=cfg.block_size, num_warps=cfg.num_warps)
            return out2

        return raw, False
    if name in ("compile", "compile_ro", "compile_ro_static"):
        torch._dynamo.reset()
        mode = None if name == "compile" else "reduce-overhead"
        fn = torch.compile(torch_softmax, dynamic=False, mode=mode)
        if name == "compile_ro_static":
            # cudagraph trees copy every non-static input into the graph's own buffer before each replay. Marking a
            # (separate, identical) input static removes that copy, isolating its cost from the rest of the
            # per-call bookkeeping.
            xs = x.clone()
            torch._dynamo.mark_static_address(xs)
            return (lambda t: fn(xs)), True
        return fn, name != "compile"
    raise ValueError(name)


def measure(fn, x, mark_step, target_ms, reps, can_graph):
    sync = torch.cuda.synchronize
    step = torch.compiler.cudagraph_mark_step_begin if mark_step else (lambda: None)
    sync()
    t0 = time.perf_counter()
    step()
    y = fn(x)
    sync()
    first_ms = (time.perf_counter() - t0) * 1e3
    warm = []
    for _ in range(10):
        t0 = time.perf_counter()
        step()
        y = fn(x)
        sync()
        warm.append((time.perf_counter() - t0) * 1e3)
    y_check = y.clone()  # compile_ro outputs live in the graph pool and are overwritten by the next replay
    per_call = []
    for _ in range(30):
        t0 = time.perf_counter()
        step()
        fn(x)
        sync()
        per_call.append((time.perf_counter() - t0) * 1e3)
    sync_ms = statistics.median(per_call)
    iters = int(min(2000, max(5, math.ceil(target_ms / max(sync_ms, 1e-3)))))
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    loops = []
    for _ in range(reps):
        start.record()
        for _ in range(iters):
            step()
            fn(x)
        end.record()
        sync()
        loops.append(start.elapsed_time(end) / iters)
    graph_ms, graph_err = None, None
    if can_graph:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(iters):
                    fn(x)
            g.replay()  # first replay can include one-time setup
            sync()
            reps_g = []
            for _ in range(reps):
                start.record()
                g.replay()
                end.record()
                sync()
                reps_g.append(start.elapsed_time(end) / iters)
            graph_ms = statistics.median(reps_g)
            del g
        except Exception as e:  # noqa: BLE001
            graph_err = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            torch.cuda.synchronize()
    return dict(first_ms=first_ms, warm_max_ms=max(warm), sync_us=sync_ms * 1e3, iters=iters,
                loop_us=statistics.median(loops) * 1e3, loop_us_min=min(loops) * 1e3, loop_us_max=max(loops) * 1e3,
                graph_us=None if graph_ms is None else graph_ms * 1e3, graph_error=graph_err), y_check


def gbps(nbytes, us):
    return None if us is None or us <= 0 else nbytes / (us * 1e-6) / 1e9


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def run_case(stage, impl, dname, m, n, names, l2_bytes, target_ms, reps, num_warps=None):
    dtype = DTYPES[dname]
    gen = torch.Generator(device="cuda").manual_seed(m * 100003 + n)
    x = torch.randn((m, n), generator=gen, device="cuda", dtype=torch.float32).to(dtype)
    ref = torch.softmax(x, dim=-1)
    nbytes = 2 * m * n * x.element_size()
    rows = []
    # rotate the order of implementations from shape to shape, so slow drift does not always favour the same one
    k = (m + n) % len(names)
    for name in names[k:] + names[:k]:
        fn, mark = make_fn(name, impl, x, num_warps)
        r, y = measure(fn, x, mark, target_ms, reps, can_graph=not name.startswith("compile_ro"))
        close = None if name == "copy" else bool(torch.allclose(y.float(), ref.float(), **TOLERANCES[dtype]))
        r.update(stage=stage, dtype=dname, m=m, n=n, impl=name, num_warps=num_warps, bytes=nbytes,
                 l2_resident_possible=nbytes <= l2_bytes, close_to_torch=close,
                 eff_gbps_loop=gbps(nbytes, r["loop_us"]), eff_gbps_graph=gbps(nbytes, r["graph_us"]))
        rows.append(r)
        del fn
        torch.cuda.empty_cache()
    return rows


HEADER = (f"{'stage':<7} {'dt':<4} {'M':>6} {'N':>6} {'MiB':>7} {'impl':<10} {'wrp':>3} {'first ms':>9} "
          f"{'warm ms':>8} {'sync us':>8} {'loop us':>9} {'graph us':>9} {'GB/s loop':>9} {'GB/s grph':>9} "
          f"{'ok':>3} {'L2':>2}")


def print_row(r):
    print(f"{r['stage']:<7} {r['dtype']:<4} {r['m']:>6} {r['n']:>6} {r['bytes'] / 2**20:>7.2f} {r['impl']:<10} "
          f"{fmt(r['num_warps'], 'd') if r['num_warps'] else '-':>3} {r['first_ms']:>9.1f} {r['warm_max_ms']:>8.2f} "
          f"{r['sync_us']:>8.1f} {r['loop_us']:>9.2f} {fmt(r['graph_us'], '.2f'):>9} "
          f"{fmt(r['eff_gbps_loop'], '.0f'):>9} {fmt(r['eff_gbps_graph'], '.0f'):>9} "
          f"{'-' if r['close_to_torch'] is None else ('y' if r['close_to_torch'] else 'NO'):>3} "
          f"{'L2' if r['l2_resident_possible'] else '':>2}", flush=True)


def print_pivot(rows, stage):
    """One line per shape: steady-state loop time per implementation and Triton's speedup over eager torch."""
    shapes = []
    for r in rows:
        if r["stage"] == stage and (r["dtype"], r["m"], r["n"]) not in shapes:
            shapes.append((r["dtype"], r["m"], r["n"]))
    impls = [i for i in ("triton", "triton_raw", "torch", "naive", "compile", "compile_ro", "compile_ro_static", "copy")
             if any(r["impl"] == i and r["stage"] == stage for r in rows)]
    print(f"\n-- {stage}: loop_us per call (effective GB/s) --")
    print(f"{'dt':<4} {'M':>6} {'N':>6} " + " ".join(f"{i[:17]:>17}" for i in impls) + f" {'torch/triton':>12}")
    for d, m, n in shapes:
        cells, t = [], {}
        for i in impls:
            r = next((r for r in rows if r["stage"] == stage and (r["dtype"], r["m"], r["n"], r["impl"]) ==
                      (d, m, n, i)), None)
            if r is None:
                cells.append(f"{'-':>17}")
                continue
            t[i] = r["loop_us"]
            cells.append(f"{r['loop_us']:>9.2f} ({fmt(r['eff_gbps_loop'], '>5.0f')})")
        ratio = t["torch"] / t["triton"] if "torch" in t and "triton" in t else None
        print(f"{d:<4} {m:>6} {n:>6} " + " ".join(cells) + f" {fmt(ratio, '>12.2f')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--stage", choices=["quick", "core", "all", "rows", "count", "padding", "warps", "launch"],
                    default="quick")
    ap.add_argument("--target-ms", type=float, default=20.0, help="GPU time per timed loop repeat")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--run-id", default=None, help="write outputs/day10_softmax/bench_<impl>_<stage>_<run-id>.{json,csv}")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    impl = load_impl(args.impl)
    env = env_info()
    l2 = int(env["l2_cache_mib"] * 2**20)
    stages = {"core": ["rows", "count", "padding", "warps"], "all": ["rows", "count", "padding", "warps", "launch"]}
    stages = stages.get(args.stage, [args.stage])
    print(f"impl={args.impl} stages={stages} | {env['gpu']} ({env['sm_count']} SMs, L2 {env['l2_cache_mib']} MiB) | "
          f"torch {env['torch']} triton {env['triton']} | CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']}")
    print("bytes = 2*M*N*elem (one read + one write); GB/s = bytes / loop_us (or graph_us); ok = matches torch.softmax"
          " within the dtype tolerance; L2 = bytes fit in L2, not a DRAM measurement\n")
    print(HEADER)
    all_rows = []
    t_start = time.perf_counter()
    try:
        with torch.no_grad():
            for stage in stages:
                for dname, m, n in STAGES[stage]:
                    if stage == "warps":
                        for w in WARPS[n]:
                            rows = run_case(stage, impl, dname, m, n, ["triton"], l2, args.target_ms, args.reps, w)
                            all_rows += rows
                            for r in rows:
                                print_row(r)
                        continue
                    rows = run_case(stage, impl, dname, m, n, IMPLS[stage], l2, args.target_ms, args.reps)
                    all_rows += rows
                    for r in rows:
                        print_row(r)
    except Exception as e:  # noqa: BLE001
        if is_not_implemented(e):
            print("\nimpl=mine is not implemented yet: fill in my_softmax.py first.")
            return 3
        raise
    for stage in stages:
        if stage != "warps":
            print_pivot(all_rows, stage)
    bad = [r for r in all_rows if r["close_to_torch"] is False and r["impl"] != "naive"]
    print(f"\n{len(all_rows)} timed cases in {time.perf_counter() - t_start:.0f} s. "
          f"Rows not matching torch (naive excluded): {len(bad)}")
    graph_errors = sorted({(r["impl"], r["graph_error"]) for r in all_rows if r["graph_error"]})
    for i, e in graph_errors:
        print(f"graph capture not possible for {i}: {e}")
    if args.run_id:
        stem = OUT_DIR / f"bench_{args.impl}_{args.stage}_{args.run_id}"
        write_json(stem.with_suffix(".json"), dict(env=env, impl=args.impl, stage=args.stage, target_ms=args.target_ms,
                                                    reps=args.reps, byte_definition="2*M*N*element_size",
                                                    rows=all_rows))
        with stem.with_suffix(".csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {stem.with_suffix('.csv').relative_to(PROJECT).as_posix()}")
    print("BENCH DONE" if not bad else "BENCH DONE WITH MISMATCHES")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
