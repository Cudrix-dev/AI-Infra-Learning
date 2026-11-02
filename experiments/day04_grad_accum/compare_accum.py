"""Day 04 - orchestrator: parity, collective counts, timing vs K, slow-link regime. Launch with plain python.

  --stage parity    tiny model, 6 optimizer steps of G = 32 (2 ranks x K=4 x m=4), the LAST window short (G = 16:
                    2 microbatches per rank). Reference: 1 process, one big batch per optimizer step (no accumulation).
                    Checks A/B/C exactly as Day 02 (check_parity.compare), plus
                      D  LR used at every optimizer step and Adam's step counter equal the reference (exact)
                      E  gradient of the LAST (short) window matches the reference (tolerance of check A)
                    and bitwise comparisons inside the nosync group and inside the every group.
                    Negative controls (torch_ddp): each must be DETECTED by its own check.
  --stage count     medium model, K = 4, a profiled run per variant -> NCCL kernels per optimizer step.
  --stage timing    medium model, m = 8 per rank, K in --ks: ms per optimizer step for every variant, + 1 GPU and
                    DDP with one big batch (no accumulation) as references.
  --stage slowlink  timing with NCCL_P2P_DISABLE=1 (communication ~50x slower): where no_sync really pays.
  --stage all       all of them, in that order.

  python experiments/day04_grad_accum/compare_accum.py --stage all --condition shared --run-id "a100_${RID}"
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
from check_parity import GRAD_ATOL, GRAD_RTOL, compare  # noqa: E402  (Day 02's A/B/C checks and tolerances)
from launch_utils import make_env, run, smoke_test, torchrun_cmd  # noqa: E402

import count_collectives  # noqa: E402

TRAIN = HERE / "train_accum.py"
OUT = HERE.parents[1] / "outputs" / "day04_grad_accum"

# ---- parity: tiny, 0.25 MiB buckets so the 1.7 MiB model really has several
P_SMALL = ["--bucket-mb", "0.25"]
PARITY = [  # (name, flags, group for the bitwise comparison)
    ("flat_nosync", ["--impl", "flat_sync", "--sync", "nosync"], "nosync"),
    ("mini_nosync", ["--impl", "mini_ddp", "--sync", "nosync", *P_SMALL, "--first-bucket-mb", "0.0625"], "nosync"),
    ("mini_nosync_view", ["--impl", "mini_ddp", "--sync", "nosync", *P_SMALL, "--first-bucket-mb", "0.0625",
                          "--grad-as-view"], "nosync"),
    ("ddp_nosync", ["--impl", "torch_ddp", "--sync", "nosync", *P_SMALL], "nosync"),
    ("ddp_nosync_view", ["--impl", "torch_ddp", "--sync", "nosync", *P_SMALL, "--grad-as-view"], "nosync"),
    ("flat_every", ["--impl", "flat_sync", "--sync", "every"], "every"),
    ("mini_every", ["--impl", "mini_ddp", "--sync", "every", *P_SMALL, "--first-bucket-mb", "0.0625"], "every"),
    ("ddp_every", ["--impl", "torch_ddp", "--sync", "every", *P_SMALL], "every"),
    ("ddp_bwd_only_nosync", ["--impl", "torch_ddp", "--sync", "bwd_only_nosync", *P_SMALL], "every"),
]
CONTROLS = [  # (bug, check that must fail)
    ("no_loss_scale", "A"),
    ("zero_every_micro", "A"),
    ("sched_per_micro", "D"),
    ("fixed_k_scale", "E"),
    ("modulo_last_sync", "E"),
]

# ---- timing / count: medium. Variant name -> flags (impl + sync); K and batch sizes are added per run.
TIMED = {
    "none": ["--impl", "none"],
    "flat_nosync": ["--impl", "flat_sync", "--sync", "nosync"],
    "flat_every": ["--impl", "flat_sync", "--sync", "every"],
    "mini_nosync": ["--impl", "mini_ddp", "--sync", "nosync"],
    "mini_every": ["--impl", "mini_ddp", "--sync", "every"],
    "ddp_nosync": ["--impl", "torch_ddp", "--sync", "nosync"],
    "ddp_every": ["--impl", "torch_ddp", "--sync", "every"],
    "ddp_bwd_only_nosync": ["--impl", "torch_ddp", "--sync", "bwd_only_nosync"],
}
COUNT = ["flat_nosync", "flat_every", "mini_nosync", "mini_every", "ddp_nosync", "ddp_every", "ddp_bwd_only_nosync"]
TIMING = ["none", "flat_nosync", "flat_every", "mini_nosync", "mini_every", "ddp_nosync", "ddp_every"]
SLOWLINK = ["none", "flat_nosync", "flat_every", "ddp_nosync", "ddp_every"]
M = 8  # rows per microbatch per rank (Day 03's local batch)


def load(tag: str) -> dict:
    return torch.load(OUT / tag / "result.pt", map_location="cpu", weights_only=True)


def launch(nproc: int, flags: list[str], env: dict) -> None:
    cmd = [sys.executable, str(TRAIN), *flags] if nproc == 1 else torchrun_cmd(nproc, TRAIN, flags)
    print("\n>>", " ".join(cmd), flush=True)
    if run(cmd, env)[0] != 0:
        raise SystemExit("run failed; not continuing")


def bitwise_same(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k][n], b[k][n]) for k in ("grads0", "params1", "grads_last", "final") for n in a[k])


def max_abs(a: dict, b: dict, key: str) -> float:
    return max((a[key][n] - b[key][n]).abs().max().item() for n in a[key])


def extra_checks(ref: dict, dp: dict) -> dict:
    """D: step counters / LR schedule (exact). E: gradient of the short final window (check A's tolerance)."""
    mr, md = ref["meta"], dp["meta"]
    ok_d = mr["lr_used"] == md["lr_used"] and mr["adam_step"] == md["adam_step"]
    first_bad = next((i for i, (a, b) in enumerate(zip(mr["lr_used"], md["lr_used"])) if a != b), None)
    text_d = (f"LR per optimizer step and Adam step counter equal the reference: {ok_d} (adam step "
              f"{md['adam_step']} vs {mr['adam_step']}" + (f"; first LR mismatch at step {first_bad}: "
              f"{md['lr_used'][first_bad]:.3e} vs {mr['lr_used'][first_bad]:.3e}" if first_bad is not None else "")
              + ")")
    gl_r, gl_d = ref["grads_last"], dp["grads_last"]
    ok_e = all(torch.allclose(gl_d[n], gl_r[n], rtol=GRAD_RTOL, atol=GRAD_ATOL) for n in gl_r)
    ratio = (sum(gl_d[n].norm() ** 2 for n in gl_d) ** 0.5 / sum(gl_r[n].norm() ** 2 for n in gl_r) ** 0.5).item()
    text_e = (f"gradient of the short final window  max|diff| = {max_abs(ref, dp, 'grads_last'):.3e}  "
              f"(|g| ratio dp/ref = {ratio:.3f}; windows {md['n_micro_per_window']})")
    return {"D": (ok_d, text_d), "E": (ok_e, text_e)}


def stage_parity(args, env, rid: str) -> list[str]:
    common = ["--preset", "tiny", "--opt-steps", "6", "--global-batch", "32", "--last-global-batch", "16",
              "--condition", args.condition, "--device", "cuda", "--check-every", "1"]
    tag = lambda n: f"par_{n}_{rid}"  # noqa: E731
    if not args.skip_run:
        launch(1, ["--impl", "flat_sync", "--micro-batch", "32", "--tag", tag("ref"), *common], env)  # K = 1
        launch(1, ["--impl", "flat_sync", "--micro-batch", "4", "--tag", tag("ref_accum8"), *common], env)  # K = 8
        for name, flags, _ in PARITY:
            launch(2, ["--micro-batch", "4", "--tag", tag(name), *common, *flags], env)
        for bug, _ in CONTROLS:
            launch(2, ["--micro-batch", "4", "--tag", tag(f"bug_{bug}"), *common, "--impl", "torch_ddp",
                       "--sync", "nosync", *P_SMALL, "--bug", bug], env)
    ref = load(tag("ref"))
    failed = []
    print("\n=== PARITY vs 1 process, one big batch per optimizer step (Day 02 A/B/C + D step counters + "
          "E short final window) ===")
    results = {}
    for name in ["ref_accum8"] + [n for n, _, _ in PARITY]:
        dp = load(tag(name))
        results[name] = dp
        res = {**compare(ref, dp), **extra_checks(ref, dp)}
        ok = all(res[k][0] for k in "ABCDE") and dp["meta"]["in_sync_final"]
        m = dp["meta"]
        print(f"\n--- {name} (world {m['world']}, K {m['accum']}, m {m['micro_batch']}, windows "
              f"{m['n_micro_per_window']}; counters {m['counters'] and m['counters'].get('per_opt_step')}) ---")
        for k in "ABCDE":
            print(f"{'PASS' if res[k][0] else 'FAIL'}  {k}  {res[k][1]}")
        print(f"      ranks bitwise in sync at the end: {m['in_sync_final']}")
        if not ok:
            failed.append(name)
    print("\n=== BITWISE (grads0, params1, grads_last, final params) ===")
    for group in ("nosync", "every"):
        members = [n for n, _, g in PARITY if g == group]
        base = members[0]
        for n in members[1:]:
            print(f"  [{group}] {n:22s} == {base:14s}: {bitwise_same(results[n], results[base])}")
    a, b = "flat_nosync", "flat_every"
    print(f"  [across] {b} == {a}: {bitwise_same(results[b], results[a])}   "
          f"(max |diff| final params {max_abs(results[a], results[b], 'final'):.3e})")
    print("  not pass/fail conditions: see Guide Block 1.4 for the predictions")

    print("\n=== NEGATIVE CONTROLS (torch_ddp + no_sync; lower-case pass/fail; the named check must FAIL) ===")
    for bug, must_fail in CONTROLS:
        dp = load(tag(f"bug_{bug}"))
        res = {**compare(ref, dp), **extra_checks(ref, dp)}
        print(f"\n--- bug_{bug} (expected detector: {must_fail}) ---")
        for k in "ABCDE":
            print(f"{'pass' if res[k][0] else 'fail'}  {k}  {res[k][1]}")
        print(f"      ranks bitwise in sync at the end: {dp['meta']['in_sync_final']}")
        caught = not res[must_fail][0]
        print(f"      => {'DETECTED by check ' + must_fail if caught else 'MISSED by check ' + must_fail}")
        if not caught:
            failed.append(f"bug_{bug} (control missed)")
    print("\nPARITY: " + ("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}"))
    return failed


def medium_flags(name: str, k: int, args, steps: int, warmup: int) -> tuple[int, list[str]]:
    """(nproc, flags) for a medium run with K microbatches of M rows per rank (global batch 2*M*K)."""
    common = ["--preset", "medium", "--opt-steps", str(steps), "--warmup", str(warmup), "--condition", args.condition,
              "--device", "cuda", "--check-every", "1000", "--log-every", "1000", "--no-tensors"]
    g = 2 * M * k
    if name == "1gpu":  # one GPU, same global batch, same K: microbatches of 2*M rows
        return 1, ["--impl", "flat_sync", "--global-batch", str(g), "--micro-batch", str(2 * M), *common]
    if name == "ddp_bigbatch":  # same global batch in ONE microbatch per rank (no accumulation): K = 1
        return 2, ["--impl", "torch_ddp", "--global-batch", str(g), "--micro-batch", str(M * k), *common]
    return 2, [*TIMED[name], "--global-batch", str(g), "--micro-batch", str(M), *common]


def stage_count(args, env, rid: str) -> None:
    k = 4
    traces = []
    for name in COUNT:
        tag = f"cnt_{name}_k{k}_{rid}"
        if not args.skip_run:
            nproc, flags = medium_flags(name, k, args, steps=8, warmup=3)
            launch(nproc, ["--tag", tag, *flags, "--profile", "--profile-steps", "2"], env)
        traces.append(OUT / tag / "trace_rank0.json")
    print(f"\n=== COLLECTIVES per optimizer step: medium, K = {k}, rank 0 trace ===")
    sys.argv = ["count_collectives.py", *map(str, traces), "--labels", *COUNT,
                "--json", str(OUT / f"count_summary_{rid}.json")]
    count_collectives.main()


def timing_table(title: str, names_by_k: dict[int, list[str]], prefix: str, args, env, rid: str,
                 extra_env: dict | None = None) -> list[dict]:
    res: dict[tuple[str, int], list[dict]] = {(n, k): [] for k, ns in names_by_k.items() for n in ns}
    for rep in range(args.repeats):
        for k, names in names_by_k.items():
            for name in names:
                tag = f"{prefix}_{name}_k{k}_r{rep}_{rid}"
                if not args.skip_run:
                    nproc, flags = medium_flags(name, k, args, steps=args.steps, warmup=args.warmup)
                    launch(nproc, ["--tag", tag, *flags], {**env, **(extra_env or {})})
                metrics = json.loads((OUT / tag / "metrics.json").read_text())
                res[(name, k)].append({**metrics["timing"], "counters": metrics["meta"]["counters"]})
    rng = lambda v, f: f"{format(statistics.median(v), f)} [{format(min(v), f)}-{format(max(v), f)}]"  # noqa: E731
    print(f"\n=== {title}: medium, m = {M} rows/microbatch/rank, {args.steps} optimizer steps ({args.warmup} warmup "
          f"excluded), repeats={args.repeats}, condition={args.condition}, run-id={rid} ===")
    print(f"{'K':>2s} {'G':>4s} {'variant':14s} {'ms / opt step [min-max]':>24s} {'ms/micro':>8s} "
          f"{'overhead vs none':>22s} {'tokens/s':>9s} {'vs 1gpu':>7s} {'peak MiB':>8s} {'coll/step':>9s}")
    rows = []
    for k, names in names_by_k.items():
        for n in names:
            t = res[(n, k)]
            core = [x["core_step_ms"] for x in t]
            none = res.get(("none", k))
            over = [x["core_step_ms"] - z["core_step_ms"] for x, z in zip(t, none)] if none and n not in (
                "none", "1gpu") else None
            ref1 = res.get(("1gpu", k))
            spd = [o["core_step_ms"] / x["core_step_ms"] for x, o in zip(t, ref1)] if ref1 else None
            c = t[0]["counters"]
            coll = f"{c['per_opt_step']['collectives']:.0f}" if c else "-"
            print(f"{k:2d} {t[0]['global_batch']:4d} {n:14s} {rng(core, '.2f'):>24s} "
                  f"{statistics.median(core) / k:8.2f} {(rng(over, '+.2f') if over else '-'):>22s} "
                  f"{statistics.median(x['core_tokens_per_s'] for x in t):9,.0f} "
                  f"{(format(statistics.median(spd), '.2f') if spd else '-'):>7s} "
                  f"{statistics.median(x['peak_alloc_mib_max_rank'] for x in t):8.0f} {coll:>9s}")
            rows.append({"variant": n, "accum": k, "global_batch": t[0]["global_batch"], "core_ms": core,
                         "overhead_ms": over, "speedup_vs_1gpu": spd,
                         "tokens_per_s": statistics.median(x["core_tokens_per_s"] for x in t),
                         "peak_mib": statistics.median(x["peak_alloc_mib_max_rank"] for x in t), "counters": c})
        print()
    print("ms / opt step = all K fwd+bwd + syncs + optimizer, slowest rank per step, median over steps; [min-max] over\n"
          "repeats. overhead = core(variant) - core(none) at the same K in the SAME repeat (none = no communication).\n"
          "vs 1gpu = core(1 GPU, same G, same K) / core(variant). coll/step: in-process counter (flat, MiniDDP); DDP:\n"
          "see --stage count. ddp_bigbatch: same G in one microbatch per rank (no accumulation).")
    return rows


def stage_timing(args, env, rid: str) -> None:
    names_by_k = {}
    for k in args.ks:
        names = ["1gpu", *TIMING] if k == 1 else ["1gpu", *TIMING, "ddp_bigbatch"]
        if k == 1:  # with one microbatch per window, nosync == every: keep one of each pair
            names = [n for n in names if not n.endswith("_every")]
        names_by_k[k] = names
    rows = timing_table("TIMING vs K", names_by_k, "time", args, env, rid)
    (OUT / f"timing_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def stage_slowlink(args, env, rid: str) -> None:
    names_by_k = {k: ([n for n in SLOWLINK if not n.endswith("_every")] if k == 1 else SLOWLINK)
                  for k in args.slow_ks}
    rows = timing_table("SLOW LINK (NCCL_P2P_DISABLE=1)", names_by_k, "slow", args, env, rid,
                        extra_env={"NCCL_P2P_DISABLE": "1"})
    (OUT / f"slowlink_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["parity", "count", "timing", "slowlink", "all"], default="all")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--steps", type=int, default=25, help="optimizer steps per timing run")
    ap.add_argument("--warmup", type=int, default=5, help="optimizer steps excluded from timing")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8], help="timing: microbatches per window")
    ap.add_argument("--slow-ks", type=int, nargs="+", default=[1, 4], help="slowlink: microbatches per window")
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--skip-run", action="store_true", help="re-read results of an earlier --run-id")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    if args.skip_run and not args.run_id:
        raise SystemExit("--skip-run needs --run-id")
    rid = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    env = make_env(args.omp_threads)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"run id {rid}; results in {OUT}/")
    if not args.skip_run and not args.no_smoke:
        smoke_test(2, "cuda", args.omp_threads)
    if args.stage in ("parity", "all"):
        if stage_parity(args, env, rid):
            print("parity failed: not timing a wrong implementation")
            return 1
    if args.stage in ("count", "all"):
        stage_count(args, env, rid)
    if args.stage in ("timing", "all"):
        stage_timing(args, env, rid)
    if args.stage in ("slowlink", "all"):
        stage_slowlink(args, env, rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
