"""Day 03 - orchestrator: smoke test, parity, timing table, profiler traces. Launch with plain python (not torchrun).

  --stage parity   tiny model, 20 steps, global batch 16. 1 process (reference) vs 2 processes for every variant.
                   Checks A/B/C exactly as Day 02 (check_parity.compare, same tolerances), PLUS a bitwise check
                   against Day 02's flat_sync on the same 2 ranks (why bitwise is expected at W=2: Guide, Block 1.5).
                   Negative controls: early_launch must be DETECTED by check A; no_wait is a race, reported only.
  --stage timing   medium model, global batch 16 (= Day 02's strong scaling), --repeats R. Every repeat measures the
                   1-GPU baseline and `none` (no communication) too, so overhead = core(variant) - core(none) is
                   computed inside one repeat.
  --stage profile  medium model, a few profiled steps per main variant -> chrome traces + overlap table + SVG.
  --stage ablation medium model: split the overhead into causes (hooks without communication, DDP without its
                   per-forward buffer broadcast, NCCL limited to 2 channels).
  --stage regimes  medium model: none / flat_sync / mini_25_rebuild / torch_ddp where communication is slow (P2P
                   disabled) and where backward is GPU-bound (global batch 64): when does overlap pay off?
  --stage all      all of them, in that order.

  python experiments/day03_ddp_hooks/compare_ddp.py --stage all --condition shared --run-id "a100_${RID}"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
sys.path.insert(0, str(HERE))
from check_parity import compare  # noqa: E402  (Day 02's A/B/C checks and tolerances)
from launch_utils import make_env, run, smoke_test, torchrun_cmd  # noqa: E402

import analyze_trace  # noqa: E402

TRAIN = HERE / "train_ddp.py"
OUT = HERE.parents[1] / "outputs" / "day03_ddp_hooks"

# name -> extra flags for train_ddp.py. Parity uses a 0.25 MiB cap so the 1.7 MiB tiny model really has several buckets.
PARITY = [
    ("flat_sync", ["--impl", "flat_sync"]),
    ("mini_per_param", ["--impl", "mini_ddp", "--bucket-mb", "0"]),
    ("mini_bucket", ["--impl", "mini_ddp", "--bucket-mb", "0.25"]),
    ("mini_rebuild_view", ["--impl", "mini_ddp", "--bucket-mb", "0.25", "--first-bucket-mb", "0.0625",
                           "--rebuild", "--grad-as-view"]),
    ("torch_ddp", ["--impl", "torch_ddp", "--bucket-mb", "0.25"]),
    ("torch_ddp_view", ["--impl", "torch_ddp", "--bucket-mb", "0.25", "--grad-as-view"]),
]
CONTROLS = [
    ("bug_early_launch", ["--impl", "mini_ddp", "--bucket-mb", "0.25", "--bug", "early_launch"], True),  # counted
    ("bug_no_wait", ["--impl", "mini_ddp", "--bucket-mb", "0.25", "--bug", "no_wait"], False),  # race: report only
]
TIMING = [
    ("1gpu", None),  # single process: the strong-scaling baseline (global batch 16 on one GPU)
    ("none", ["--impl", "none"]),
    ("flat_sync", ["--impl", "flat_sync"]),
    ("mini_per_param", ["--impl", "mini_ddp", "--bucket-mb", "0"]),
    ("mini_25", ["--impl", "mini_ddp", "--bucket-mb", "25"]),
    ("mini_25_rebuild", ["--impl", "mini_ddp", "--bucket-mb", "25", "--first-bucket-mb", "1", "--rebuild"]),
    ("mini_25_rebuild_view", ["--impl", "mini_ddp", "--bucket-mb", "25", "--first-bucket-mb", "1", "--rebuild",
                              "--grad-as-view"]),
    ("torch_ddp", ["--impl", "torch_ddp"]),
    ("torch_ddp_view", ["--impl", "torch_ddp", "--grad-as-view"]),
]
PROFILE = ["flat_sync", "mini_per_param", "mini_25", "mini_25_rebuild", "torch_ddp"]
# Regimes: the same four variants where communication is expensive (P2P disabled: ~2.8 GB/s instead of ~150 GB/s, a
# stand-in for a slow interconnect) and where the GPU, not the CPU, bounds backward (local batch 32 instead of 8).
REGIME_VARIANTS = ["none", "flat_sync", "mini_25_rebuild", "torch_ddp"]
REGIMES = [
    ("default_b16", ["--global-batch", "16"], {}),
    ("slowlink_b16", ["--global-batch", "16"], {"NCCL_P2P_DISABLE": "1"}),
    ("default_b64", ["--global-batch", "64"], {}),
]
# Ablations that split an overhead into causes (Guide, Block 4). (name, flags, extra environment)
ABLATION = [
    ("none", ["--impl", "none"], {}),
    ("flat_sync", ["--impl", "flat_sync"], {}),
    ("mini_per_param", ["--impl", "mini_ddp", "--bucket-mb", "0"], {}),
    ("mini_per_param_nocomm", ["--impl", "mini_ddp", "--bucket-mb", "0", "--no-comm"], {}),
    ("mini_25_rebuild", ["--impl", "mini_ddp", "--bucket-mb", "25", "--first-bucket-mb", "1", "--rebuild"], {}),
    ("mini_25_rebuild_nocomm", ["--impl", "mini_ddp", "--bucket-mb", "25", "--first-bucket-mb", "1", "--rebuild",
                                "--no-comm"], {}),
    ("torch_ddp", ["--impl", "torch_ddp"], {}),
    ("torch_ddp_nobufsync", ["--impl", "torch_ddp", "--forward-sync-buffers", "0"], {}),
    ("torch_ddp_nocomm", ["--impl", "torch_ddp", "--no-comm"], {}),
    # Fewer NCCL channels = fewer thread blocks (SMs) held by each NCCL kernel, including while it waits for the peer
    ("flat_sync_2ch", ["--impl", "flat_sync"], {"NCCL_MAX_NCHANNELS": "2"}),
    ("mini_per_param_2ch", ["--impl", "mini_ddp", "--bucket-mb", "0"], {"NCCL_MAX_NCHANNELS": "2"}),
    ("mini_25_rebuild_2ch", ["--impl", "mini_ddp", "--bucket-mb", "25", "--first-bucket-mb", "1", "--rebuild"],
     {"NCCL_MAX_NCHANNELS": "2"}),
    ("torch_ddp_2ch", ["--impl", "torch_ddp"], {"NCCL_MAX_NCHANNELS": "2"}),
]


def load(tag: str) -> dict:
    return torch.load(OUT / tag / "result.pt", map_location="cpu", weights_only=True)


def launch(nproc: int, flags: list[str], env: dict) -> None:
    cmd = [sys.executable, str(TRAIN), *flags] if nproc == 1 else torchrun_cmd(nproc, TRAIN, flags)
    print("\n>>", " ".join(cmd), flush=True)
    if run(cmd, env)[0] != 0:
        raise SystemExit("run failed; not continuing")


def bitwise_same(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k][n], b[k][n]) for k in ("grads0", "params1", "final") for n in a[k])


def stage_parity(args, env, rid: str) -> list[str]:
    common = ["--preset", "tiny", "--steps", "20", "--global-batch", "16", "--condition", args.condition,
              "--device", "cuda"]
    tag = lambda n: f"par_{n}_{rid}"  # noqa: E731
    if not args.skip_run:
        launch(1, ["--impl", "flat_sync", "--tag", tag("ref"), *common], env)
        for name, extra, *_ in PARITY + CONTROLS:
            launch(2, ["--tag", tag(name), *common, *extra], env)
    ref, base = load(tag("ref")), load(tag("flat_sync"))
    failed = []
    print("\n=== PARITY vs 1 process (Day 02 checks A/B/C), and BITWISE vs flat_sync on the same 2 ranks ===")
    for name, _ in PARITY:
        dp = load(tag(name))
        res = compare(ref, dp)
        ok = all(res[k][0] for k in "ABC")
        same = bitwise_same(dp, base)
        print(f"\n--- {name} ---")
        for k in "ABC":
            print(f"{'PASS' if res[k][0] else 'FAIL'}  {k}  {res[k][1]}")
        print(f"      bitwise == flat_sync (grads0, params1, final params after 20 steps): {same}")
        if not ok:
            failed.append(name)
        if not same:
            print("      (not a pass/fail condition, but at W=2 it is expected: see Guide Block 1.5)")
    print("\n=== NEGATIVE CONTROLS (lower-case pass/fail; failing A is the GOOD outcome) ===")
    for name, _, counted in CONTROLS:
        res = compare(ref, load(tag(name)))
        for k in "ABC":
            print(f"{'pass' if res[k][0] else 'fail'}  {k}  {name}: {res[k][1]}")
        caught = not res["A"][0]
        verdict = "DETECTED by check A (good)" if caught else "MISSED by check A"
        print(f"      => {verdict}" + ("" if counted else "  [race: informational, not counted]"))
        if counted and not caught:
            failed.append(name + " (control missed)")
    print("\nPARITY: " + ("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}"))
    return failed


def stage_timing(args, env, rid: str) -> None:
    common = ["--preset", "medium", "--steps", str(args.steps), "--warmup", str(args.warmup), "--global-batch", "16",
              "--condition", args.condition, "--device", "cuda", "--check-every", "1000", "--log-every", "1000",
              "--no-tensors"]
    names = [n for n, _ in TIMING if not args.variants or n in args.variants or n in ("1gpu", "none")]
    res: dict[str, list[dict]] = {n: [] for n in names}
    for rep in range(args.repeats):
        for name, extra in TIMING:
            if name not in names:
                continue
            tag = f"time_{name}_r{rep}_{rid}"
            if not args.skip_run:
                if extra is None:
                    launch(1, ["--impl", "flat_sync", "--tag", tag, *common], env)
                else:
                    launch(2, ["--tag", tag, *common, *extra], env)
            res[name].append(json.loads((OUT / tag / "metrics.json").read_text())["timing"])
    rng = lambda v, f: f"{format(statistics.median(v), f)} [{format(min(v), f)}-{format(max(v), f)}]"  # noqa: E731
    print(f"\n=== TIMING: medium, global batch 16, {args.steps} steps ({args.warmup} warmup excluded), "
          f"repeats={args.repeats}, condition={args.condition}, run-id={rid} ===")
    print(f"{'variant':22s} {'core ms [min-max]':>24s} {'overhead vs none ms':>24s} {'tokens/s':>9s} "
          f"{'speedup vs 1gpu':>18s} {'peak MiB':>9s}")
    rows = []
    for n in names:
        core = [t["core_step_ms"] for t in res[n]]
        over = [t["core_step_ms"] - z["core_step_ms"] for t, z in zip(res[n], res["none"])] if n != "1gpu" else None
        spd = [o["core_step_ms"] / t["core_step_ms"] for t, o in zip(res[n], res["1gpu"])]
        tok = statistics.median(t["core_tokens_per_s"] for t in res[n])
        peak = statistics.median(t["peak_alloc_mib_max_rank"] for t in res[n])
        print(f"{n:22s} {rng(core, '.2f'):>24s} {(rng(over, '+.2f') if over else '-'):>24s} {tok:9,.0f} "
              f"{rng(spd, '.2f'):>18s} {peak:9.0f}")
        rows.append({"variant": n, "core_ms": core, "overhead_ms": over, "speedup": spd, "tokens_per_s": tok,
                     "peak_mib": peak})
    print("core = fwd+bwd+sync+optimizer, slowest rank per step, median over steps; columns: median [min-max] over\n"
          "repeats. overhead = core(variant) - core(none) of the SAME repeat (none = no communication, not training).\n"
          "speedup = core(1gpu)/core(variant) of the same repeat (strong scaling: same global batch).")
    (OUT / f"timing_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def stage_ablation(args, env, rid: str) -> None:
    common = ["--preset", "medium", "--steps", str(args.steps), "--warmup", str(args.warmup), "--global-batch", "16",
              "--condition", args.condition, "--device", "cuda", "--check-every", "1000", "--log-every", "1000",
              "--no-tensors"]
    ablation = [a for a in ABLATION if not args.variants or a[0] in args.variants or a[0] == "none"]
    res: dict[str, list[dict]] = {n: [] for n, _, _ in ablation}
    for rep in range(args.repeats):
        for name, flags, extra_env in ablation:
            tag = f"abl_{name}_r{rep}_{rid}"
            if not args.skip_run:
                launch(2, ["--tag", tag, *common, *flags], {**env, **extra_env})
            res[name].append(json.loads((OUT / tag / "metrics.json").read_text())["timing"])
    rng = lambda v, f: f"{format(statistics.median(v), f)} [{format(min(v), f)}-{format(max(v), f)}]"  # noqa: E731
    print(f"\n=== ABLATION: medium, global batch 16, 2 GPUs, {args.steps} steps, repeats={args.repeats}, run-id={rid} ===")
    print(f"{'variant':24s} {'env':22s} {'core ms [min-max]':>24s} {'overhead vs none ms':>24s}")
    rows = []
    for name, _, extra_env in ablation:
        core = [t["core_step_ms"] for t in res[name]]
        over = [t["core_step_ms"] - z["core_step_ms"] for t, z in zip(res[name], res["none"])]
        e = " ".join(f"{k}={v}" for k, v in extra_env.items()) or "-"
        print(f"{name:24s} {e:22s} {rng(core, '.2f'):>24s} {rng(over, '+.2f'):>24s}")
        rows.append({"variant": name, "env": extra_env, "core_ms": core, "overhead_ms": over})
    print("nocomm = hooks + copies without any all_reduce (timing only). 2ch = NCCL_MAX_NCHANNELS=2 for that run only.")
    (OUT / f"ablation_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def stage_regimes(args, env, rid: str) -> None:
    flags = {n: f for n, f, _ in ABLATION}
    common = ["--preset", "medium", "--steps", str(args.steps), "--warmup", str(args.warmup), "--condition",
              args.condition, "--device", "cuda", "--check-every", "1000", "--log-every", "1000",
              "--no-tensors"]
    res = {(r, v): [] for r, _, _ in REGIMES for v in REGIME_VARIANTS}
    for rep in range(args.repeats):
        for reg, rflags, renv in REGIMES:
            for v in REGIME_VARIANTS:
                tag = f"reg_{reg}_{v}_r{rep}_{rid}"
                if not args.skip_run:
                    launch(2, ["--tag", tag, *common, *rflags, *flags[v]], {**env, **renv})
                res[(reg, v)].append(json.loads((OUT / tag / "metrics.json").read_text())["timing"])
    rng = lambda v, f: f"{format(statistics.median(v), f)} [{format(min(v), f)}-{format(max(v), f)}]"  # noqa: E731
    print(f"\n=== REGIMES: medium, 2 GPUs, {args.steps} steps, repeats={args.repeats}, run-id={rid} ===")
    print(f"{'regime':14s} {'variant':18s} {'core ms [min-max]':>24s} {'overhead vs none ms':>24s}")
    rows = []
    for reg, _, renv in REGIMES:
        for v in REGIME_VARIANTS:
            core = [t["core_step_ms"] for t in res[(reg, v)]]
            over = [t["core_step_ms"] - z["core_step_ms"] for t, z in zip(res[(reg, v)], res[(reg, "none")])]
            print(f"{reg:14s} {v:18s} {rng(core, '.2f'):>24s} {rng(over, '+.2f'):>24s}")
            rows.append({"regime": reg, "env": renv, "variant": v, "core_ms": core, "overhead_ms": over})
    print("slowlink = NCCL_P2P_DISABLE=1 for that run only (Day 02: 256 MiB all-reduce 97 ms instead of 1.75 ms).\n"
          "b64 = global batch 64, local 32: more GPU work per step for the same 105 MiB of gradients.")
    (OUT / f"regimes_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def stage_profile(args, env, rid: str) -> None:
    flags = dict(TIMING)
    common = ["--preset", "medium", "--steps", "12", "--warmup", "5", "--global-batch", "16", "--profile",
              "--profile-steps", "3", "--condition", args.condition, "--device", "cuda", "--check-every", "1000",
              "--log-every", "1000", "--no-tensors"]
    traces, labels = [], []
    for name in PROFILE:
        tag = f"prof_{name}_{rid}"
        if not args.skip_run:
            launch(2, ["--tag", tag, *common, *flags[name]], env)
        traces.append(OUT / tag / "trace_rank0.json")
        labels.append(name)
    print("\n=== PROFILE (rank 0) ===")
    sys.argv = ["analyze_trace.py", *map(str, traces), "--labels", *labels,
                "--svg", str(OUT / f"timeline_{rid}.svg"), "--json", str(OUT / f"profile_summary_{rid}.json")]
    analyze_trace.main()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["parity", "timing", "profile", "ablation", "regimes", "all"], default="all")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--variants", nargs="*", default=None, help="timing: subset of variant names (1gpu, none always run)")
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--skip-run", action="store_true", help="re-read results of an earlier --run-id")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    if args.skip_run and not args.run_id:
        raise SystemExit("--skip-run needs --run-id")
    rid = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    env = make_env(args.omp_threads)
    print(f"run id {rid}; results in {OUT}/")
    if not args.skip_run and not args.no_smoke:
        smoke_test(2, "cuda", args.omp_threads)
    failed = []
    if args.stage in ("parity", "all"):
        failed = stage_parity(args, env, rid)
        if failed:
            print("parity failed: not timing a wrong implementation")
            return 1
    if args.stage in ("timing", "all"):
        stage_timing(args, env, rid)
    if args.stage in ("profile", "all"):
        stage_profile(args, env, rid)
    if args.stage in ("ablation", "all"):
        stage_ablation(args, env, rid)
    if args.stage in ("regimes", "all"):
        stage_regimes(args, env, rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
