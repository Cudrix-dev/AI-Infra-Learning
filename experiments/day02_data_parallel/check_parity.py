"""Day 02 - 1-process vs N-process parity check (the data-parallel analogue of Day 01's check_resume.py).

Runs ddp_manual.py as
  ref : ONE process, global batch 16                        (reference)
  dp* : `--nproc` processes, local batch 16/nproc each      (per_param, flat, ring variants)
  bug*: (with --with-bugs) the same with one classic bug injected: NEGATIVE CONTROLS

and compares what rank 0 saved:
  A. the gradient after all_reduce + average, step 0   <- the SHARP test: same math, different float summation order
  B. the parameters after ONE optimizer update          <- weaker: Adam normalises the gradient (see guide)
  C. the global loss at step 0, and the whole loss curve (informational drift)

For the negative controls, failing check A is the GOOD outcome ("DETECTED"). Those rows print lower-case pass/fail so
they cannot be confused with a failing test in the real variants (UPPER-case PASS/FAIL).

Tolerances are unchanged from the first version (A: rtol 1e-4 / atol 1e-6, B: 1e-5, C: 1e-5). Observed 2026-10-21 with
the earlier version of this harness (torch 2.14.1, tiny preset, 20 steps; max abs errors): gradients 2.8e-9 .. 3.7e-9,
first-update parameters 2.0e-7 .. 2.8e-7, step-0 loss 0 or 4.8e-7. They are NOT numbers measured with this revision:
record your own.

Every run gets a unique tag suffix (--run-id, default: a timestamp), so repeated runs never overwrite evidence.
--skip-run re-reads an earlier run and verifies its recorded settings (preset, steps, world size, batch, variant) first.

Usage (from the repo root):
  python experiments/day02_data_parallel/check_parity.py --device cuda                   # 1 GPU vs 2 GPUs
  python experiments/day02_data_parallel/check_parity.py --device cuda --with-bugs
  python experiments/day02_data_parallel/check_parity.py --device cuda --nproc 4
  python experiments/day02_data_parallel/check_parity.py --device cuda --run-id 20261021-a --skip-run
Choose GPUs with CUDA_VISIBLE_DEVICES. On the 2 x RTX 6000D host start with
  CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 NCCL_P2P_DISABLE=1   (host-specific workaround, see the guide).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch_utils import DDP, OUT, make_env, run, smoke_test, torchrun_cmd  # noqa: E402

GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-6
PARAM1_ATOL = 1e-5
LOSS0_ATOL = 1e-5


def load(tag: str) -> dict:
    path = OUT / tag / "result.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_meta(res: dict, tag: str, expect: dict) -> None:
    """Refuse to compare artifacts that were produced with different settings than the ones requested."""
    a, t, w = res["meta"]["args"], res["timing"], res["meta"]["world"]
    got = {"preset": a["preset"], "steps": a["steps"], "device": a["device"], "world": w,
           "global_batch": t["global_batch"], "sync": a["sync"], "allreduce": a["allreduce"], "bug": a["bug"]}
    bad = {k: (got[k], v) for k, v in expect.items() if got[k] != v}
    if bad:
        raise SystemExit(f"stale or mismatched result for tag '{tag}' (recorded vs requested): {bad}")


def max_diff(a: dict, b: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for n in a:
        d = (a[n] - b[n]).abs().max().item()
        if d > worst:
            worst, where = d, n
    return worst, where


def compare(ref: dict, dp: dict) -> dict:
    """Return {'A': (ok, text), 'B': ..., 'C': ..., 'info': [...]}."""
    res = {}
    g_ref, g_dp = ref["grads0"], dp["grads0"]
    gd, gname = max_diff(g_ref, g_dp)
    gscale = max(v.abs().max().item() for v in g_ref.values())
    ok_a = all(torch.allclose(g_dp[n], g_ref[n], rtol=GRAD_RTOL, atol=GRAD_ATOL) for n in g_ref)
    res["A"] = (ok_a, f"synced gradient, step 0      max|diff| = {gd:.3e}  (largest |grad| = {gscale:.3e}; worst: {gname})")

    pd, pname = max_diff(ref["params1"], dp["params1"])
    n_big = sum(int(((ref["params1"][n] - dp["params1"][n]).abs() > 1e-6).sum()) for n in ref["params1"])
    res["B"] = (pd <= PARAM1_ATOL,
                f"params after 1 update       max|diff| = {pd:.3e}  (lr at step 0 = {ref['meta']['lr0']:.2e}; "
                f"{n_big} elements differ by > 1e-6; worst: {pname})")

    l0 = abs(ref["losses"][0] - dp["losses"][0])
    res["C"] = (l0 <= LOSS0_ATOL, f"global loss, step 0         |diff| = {l0:.3e}")

    k = len(ref["losses"])
    marks = sorted({0, min(4, k - 1), min(9, k - 1), k - 1})
    drift = ", ".join(f"step {m + 1}: {abs(ref['losses'][m] - dp['losses'][m]):.1e}" for m in marks)
    fd, fname = max_diff(ref["final"], dp["final"])
    res["info"] = [f"loss |diff| over time      {drift}",
                   f"final params after {k} steps  max|diff| = {fd:.3e} (worst: {fname})  [informational: later drift is possible and is not a pass/fail condition]"]
    return res


def report(name: str, res: dict, ref: dict, dp: dict, control: bool = False) -> bool:
    m = dp["meta"]
    print(f"\n--- {name}  (world={m['world']}, backend={m['backend']}, sync={m['args']['sync']}, "
          f"allreduce={m['args']['allreduce']}, bug={m['args']['bug']}) ---")
    for key in ("A", "B", "C"):
        ok, text = res[key]
        word = ("pass" if ok else "fail") if control else ("PASS" if ok else "FAIL")
        print(f"{word}  {key}  {text}")
    for line in res["info"]:
        print(f"      i  {line}")
    t_ref, t_dp = ref["timing"], dp["timing"]
    print(f"      t  core step {t_ref['core_step_ms']:.1f} ms (1 proc) vs {t_dp['core_step_ms']:.1f} ms ({m['world']} procs); "
          f"sync phase median {t_dp['comm_ms']:.1f} ms  [one short run: not a benchmark; condition={m['condition']}]")
    return all(res[k][0] for k in ("A", "B", "C"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--preset", choices=["tiny", "medium"], default="tiny")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--omp-threads", type=int, default=1, help="OMP_NUM_THREADS for the reference AND the workers")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    ap.add_argument("--run-id", default=None, help="suffix for all tags (default: timestamp); required with --skip-run")
    ap.add_argument("--with-bugs", action="store_true", help="also run the negative controls")
    ap.add_argument("--skip-run", action="store_true", help="only compare an earlier run (needs --run-id); settings are verified")
    ap.add_argument("--no-smoke", action="store_true", help="skip the hello_dist.py smoke test (not recommended)")
    args = ap.parse_args()

    if args.nproc < 2:
        raise SystemExit("--nproc must be >= 2 (a parity check of 1 process against itself proves nothing)")
    if args.steps < 1 or args.omp_threads < 1 or args.global_batch < 1:
        raise SystemExit("--steps, --omp-threads and --global-batch must be >= 1")
    if args.global_batch % args.nproc != 0:
        raise SystemExit(f"--global-batch {args.global_batch} must be divisible by --nproc {args.nproc}")
    if args.skip_run and not args.run_id:
        raise SystemExit("--skip-run needs --run-id (it must name the earlier run)")
    rid = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    print(f"run id: {rid}   (results in {OUT}/<name>_{rid}/)")

    common = ["--preset", args.preset, "--steps", str(args.steps), "--device", args.device,
              "--global-batch", str(args.global_batch), "--condition", args.condition]
    variants = [("dp_per_param", ["--sync", "per_param"]),
                ("dp_flat", ["--sync", "flat"]),
                ("dp_ring", ["--sync", "flat", "--allreduce", "ring"])]
    bugs = [("bug_no_broadcast", ["--bug", "no_broadcast"]),
            ("bug_no_allreduce", ["--bug", "no_allreduce"]),
            ("bug_no_average", ["--bug", "no_average"])]
    tag = lambda name: f"{name}_{rid}"  # noqa: E731

    if not args.skip_run:
        if not args.no_smoke:
            smoke_test(args.nproc, args.device, args.omp_threads)
        env = make_env(args.omp_threads)
        cmd = [sys.executable, str(DDP), "--tag", tag("ref"), *common]
        print("\n>>", " ".join(cmd), flush=True)
        if run(cmd, env)[0] != 0:
            raise SystemExit("reference run failed")
        for name, extra in variants + (bugs if args.with_bugs else []):
            cmd = torchrun_cmd(args.nproc, DDP, ["--tag", tag(name), *common, *extra])
            print("\n>>", " ".join(cmd), flush=True)
            if run(cmd, env)[0] != 0:
                raise SystemExit(f"run '{name}' failed; not continuing")

    base = {"preset": args.preset, "steps": args.steps, "device": args.device, "global_batch": args.global_batch}
    ref = load(tag("ref"))
    validate_meta(ref, tag("ref"), {**base, "world": 1, "bug": "none"})
    failed = []
    for name, extra in variants:
        dp = load(tag(name))
        sync = extra[extra.index("--sync") + 1]
        algo = extra[extra.index("--allreduce") + 1] if "--allreduce" in extra else "builtin"
        validate_meta(dp, tag(name), {**base, "world": args.nproc, "bug": "none", "sync": sync, "allreduce": algo})
        if not report(name, compare(ref, dp), ref, dp):
            failed.append(name)

    n_detected = 0
    if args.with_bugs:
        print("\n=== NEGATIVE CONTROLS: each injected bug must make check A fail (lower-case 'fail' on A is the GOOD outcome) ===")
        for name, extra in bugs:
            dp = load(tag(name))
            validate_meta(dp, tag(name), {**base, "world": args.nproc, "bug": extra[1], "sync": "per_param", "allreduce": "builtin"})
            res = compare(ref, dp)
            report(name, res, ref, dp, control=True)
            caught = not res["A"][0]
            n_detected += caught
            print(f"      => {'DETECTED by check A (good)' if caught else 'MISSED by check A: the check is too weak!'}"
                  f"{'  | check B ' + ('passed: Adam hid this bug in the first update' if res['B'][0] else 'failed too') if caught else ''}")
            if not caught:
                failed.append(name + " (control missed)")

    print(f"\nreal variants: {len(variants) - sum(1 for f in failed if f in dict(variants))}/{len(variants)} passed"
          + (f" | negative controls: {n_detected}/{len(bugs)} detected" if args.with_bugs else ""))
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
