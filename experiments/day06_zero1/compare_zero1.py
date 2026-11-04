"""Day 06 - orchestrator: parity, state per rank, collective counts, timing, slow link. Launch with plain python.

  --stage parity    tiny model, Day 04's setting: 6 optimizer steps of G = 32 (2 ranks x K 4 x m 4), the LAST window
                    short (G = 16). Reference: 1 process, one big batch per optimizer step, Day 01's AdamW.
                      A B C   Day 02's checks (check_parity.compare): step-0 gradient, params after 1 update, loss
                      D E     Day 04's checks (compare_accum.extra_checks): LR + Adam step counter, short final window
                      R       replicas: params bitwise equal across ranks after EVERY optimizer step
                      S       optimizer state: AdamW's exp_avg / exp_avg_sq gathered from all ranks match the UNSHARDED
                              DDP + AdamW run (max|diff| <= 1e-4 x max|ddp| per kind) and Adam's step counter is equal.
                              Against the 1-process reference the state is printed as information only: it integrates
                              6 steps of a slightly different trajectory (even ddp drifts to ~2e-4 relative there)
                    plus BITWISE against the unsharded DDP + AdamW run (gradients, params, gathered state): required for
                    the "boundary" group, reported for zero1_rs_every (predicted NOT bitwise, see the guide).
                    Negative controls: each must be DETECTED by its named check.
  --stage count     medium, K = 4, one profiled run per variant -> NCCL kernels per optimizer step, by kind.
  --stage timing    medium, m = 8, K in --ks: ms per optimizer step and its phases, peak / between-microbatch memory,
                    optimizer state per rank.
  --stage slowlink  timing at K = 4 with NCCL_P2P_DISABLE=1: what K reduce-scatters per window cost on a slow link.
  --stage all       all of them, in that order.

  python experiments/day06_zero1/compare_zero1.py --stage all --condition shared --run-id "a100_${RID}"
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
sys.path.insert(0, str(HERE.parent / "day04_grad_accum"))
from check_parity import compare  # noqa: E402  (Day 02's A/B/C checks and tolerances)
from compare_accum import extra_checks, max_abs  # noqa: E402  (Day 04's D/E checks)
from count_collectives import count  # noqa: E402  (Day 04's NCCL kernel counter)
from launch_utils import make_env, run, smoke_test, torchrun_cmd  # noqa: E402

TRAIN = HERE / "train_zero1.py"
OUT = HERE.parents[1] / "outputs" / "day06_zero1"
MiB = 2**20
STATE_RTOL = 1e-4

P_SMALL = ["--bucket-mb", "0.25"]  # tiny model (1.7 MiB): several DDP buckets, as on Day 04
PARITY = [  # (name, flags, bitwise group vs ddp)
    ("ddp", ["--impl", "ddp", *P_SMALL], "boundary"),
    ("torch_zero", ["--impl", "torch_zero", *P_SMALL], "boundary"),
    ("torch_zero_bucket", ["--impl", "torch_zero_bucket", *P_SMALL], "boundary"),
    ("flat_ar", ["--impl", "flat_ar"], "boundary"),
    ("zero1", ["--impl", "zero1"], "boundary"),
    ("zero1_rs_every", ["--impl", "zero1_rs_every"], "per_micro"),
]
CONTROLS = [  # (bug, impl, check that must fail)
    ("no_reduce_scatter", "zero1", "A"),
    ("no_allgather", "zero1", "R"),
    ("grad_set_to_none", "zero1", "A"),
    ("rs_overwrite", "zero1_rs_every", "A"),
]
TIMING = ["ddp", "torch_zero", "flat_ar", "zero1", "zero1_rs_every"]
COUNT = ["ddp", "torch_zero", "torch_zero_bucket", "flat_ar", "zero1", "zero1_rs_every"]
SLOWLINK = ["ddp", "flat_ar", "zero1", "zero1_rs_every"]
KINDS = ["AllReduce", "ReduceScatter", "AllGather", "Broadcast"]
M = 8  # rows per microbatch per rank (Days 03-04)


def load(tag: str) -> dict:
    return torch.load(OUT / tag / "result.pt", map_location="cpu", weights_only=True)


def launch(nproc: int, flags: list[str], env: dict) -> None:
    cmd = [sys.executable, str(TRAIN), *flags] if nproc == 1 else torchrun_cmd(nproc, TRAIN, flags)
    print("\n>>", " ".join(cmd), flush=True)
    if run(cmd, env)[0] != 0:
        raise SystemExit("run failed; not continuing")


# ---------------------------------------------------------------- the two new checks
def check_r(dp: dict) -> tuple[bool, str]:
    steps = dp["meta"]["in_sync_steps"]
    bad = [s for s, ok in steps if not ok]
    ok = not bad and dp["meta"]["in_sync_final"]
    return ok, (f"replicas bitwise equal after {len(steps)} checked optimizer steps: {ok}"
                + (f" (first divergence after step {bad[0]})" if bad else ""))


def state_diff(a: dict, b: dict) -> dict:
    """max|diff| and max|b| per state kind, over all parameters."""
    out = {}
    for kind in ("exp_avg", "exp_avg_sq"):
        d = max((a[n][kind] - b[n][kind]).abs().max().item() for n in b)
        scale = max(b[n][kind].abs().max().item() for n in b)
        out[kind] = (d, scale)
    return out


def check_s(base: dict, dp: dict) -> tuple[bool, str]:
    """S: gathered optimizer state vs the unsharded DDP + AdamW run (same 2-rank trajectory)."""
    sd = state_diff(dp["state_final"], base["state_final"])
    ok_t = all(d <= STATE_RTOL * scale for d, scale in sd.values())
    ok_step = dp["meta"]["adam_step"] == base["meta"]["adam_step"]
    text = "  ".join(f"{k} max|diff| {d:.3e} (max|ddp| {s:.3e})" for k, (d, s) in sd.items())
    return ok_t and ok_step, f"gathered AdamW state vs DDP + AdamW: {text}; step {dp['meta']['adam_step']}"


def state_vs_ref(ref: dict, dp: dict) -> str:
    sd = state_diff(dp["state_final"], ref["state_final"])
    return ("state vs 1-process reference after 6 steps (information: trajectory drift)  "
            + "  ".join(f"{k} {d / s:.1e} relative" for k, (d, s) in sd.items()))


def bitwise(a: dict, b: dict) -> dict:
    """Which saved quantities are bitwise identical between two runs."""
    res = {k: all(torch.equal(a[k][n], b[k][n]) for n in b[k]) for k in ("grads0", "params1", "grads_last", "final")}
    res["state"] = all(torch.equal(a["state_final"][n][k], b["state_final"][n][k])
                       for n in b["state_final"] for k in ("exp_avg", "exp_avg_sq"))
    return res


def all_checks(ref: dict, base: dict, dp: dict) -> dict:
    return {**compare(ref, dp), **extra_checks(ref, dp), "R": check_r(dp), "S": check_s(base, dp)}


def state_line(r: dict) -> str:
    sb = r["state_bytes"]
    return (f"rank {r['rank']}: owns {r['owned_real_numel']:>10,} elements"
            + (f" in {r['owned_tensors']:2d} tensors" if "owned_tensors" in r else "              ")
            + f" | exp_avg + exp_avg_sq {(sb.get('exp_avg', 0) + sb.get('exp_avg_sq', 0)) / MiB:7.2f} MiB"
            f" | delta first step {r['delta_first_step_mib']:7.2f} MiB")


# ---------------------------------------------------------------- stages
def stage_parity(args, env, rid: str) -> list[str]:
    common = ["--preset", "tiny", "--opt-steps", "6", "--global-batch", "32", "--last-global-batch", "16",
              "--condition", args.condition, "--device", "cuda", "--check-every", "1"]
    tag = lambda n: f"par_{n}_{rid}"  # noqa: E731
    if not args.skip_run:
        launch(1, ["--impl", "single", "--micro-batch", "32", "--tag", tag("ref"), *common], env)  # K = 1
        for name, flags, _ in PARITY:
            launch(2, ["--micro-batch", "4", "--tag", tag(name), *common, *flags], env)
        for bug, impl, _ in CONTROLS:
            launch(2, ["--micro-batch", "4", "--tag", tag(f"bug_{bug}"), *common, "--impl", impl, "--bug", bug], env)
    ref = load(tag("ref"))
    results = {name: load(tag(name)) for name, _, _ in PARITY}
    base = results["ddp"]  # the unsharded baseline: DDP + Day 01's AdamW on every rank
    failed = []
    print("\n=== PARITY vs 1 process, one big batch per optimizer step (A B C: Day 02 | D E: Day 04 | R replicas | "
          "S optimizer state vs DDP + AdamW) ===")
    for name, _, _ in PARITY:
        dp = results[name]
        res = all_checks(ref, base, dp)
        m = dp["meta"]
        print(f"\n--- {name} (world {m['world']}, K {m['accum']}, windows {m['n_micro_per_window']}; FlatZeroAdamW "
              f"collectives {m['counters']}) ---")
        for k in "ABCDERS":
            print(f"{'PASS' if res[k][0] else 'FAIL'}  {k}  {res[k][1]}" + ("  [S reference: this run]"
                                                                           if k == "S" and name == "ddp" else ""))
        print(f"      i  {state_vs_ref(ref, dp)}")
        for r in m["per_rank"]:
            print("      state " + state_line(r))
        if not all(res[k][0] for k in "ABCDERS"):
            failed.append(name)

    print("\n=== BITWISE vs the unsharded DDP + AdamW run (grads0, params1, grads_last, final params, gathered state) ===")
    for name, _, group in PARITY[1:]:
        bw = bitwise(results[name], base)
        same = all(bw.values())
        diff = max_abs(results[name], base, "final")
        sd = state_diff(results[name]["state_final"], base["state_final"])
        print(f"  [{group:9s}] {name:18s} == ddp: {same}  {bw}  (final params max|diff| {diff:.3e}; "
              f"exp_avg {sd['exp_avg'][0]:.3e}, exp_avg_sq {sd['exp_avg_sq'][0]:.3e})")
        if group == "boundary" and not same:
            failed.append(f"{name} (not bitwise equal to ddp)")
    print("  boundary group: bitwise equality is REQUIRED (one reduction of the window sum, two operands at W = 2, "
          "elementwise AdamW). per_micro: predicted NOT bitwise, reported only (guide 1.6)")

    print("\n=== NEGATIVE CONTROLS (lower-case pass/fail; the named check must FAIL) ===")
    for bug, impl, must_fail in CONTROLS:
        dp = load(tag(f"bug_{bug}"))
        res = all_checks(ref, base, dp)
        print(f"\n--- bug_{bug} on {impl} (expected detector: {must_fail}) ---")
        for k in "ABCDERS":
            print(f"{'pass' if res[k][0] else 'fail'}  {k}  {res[k][1]}")
        caught = not res[must_fail][0]
        print(f"      => {'DETECTED by check ' + must_fail if caught else 'MISSED by check ' + must_fail}"
              + ("   (note: R passes, the replicas agree on a WRONG model)" if res["R"][0] and not res["A"][0] else ""))
        if not caught:
            failed.append(f"bug_{bug} (control missed)")
    print("\nPARITY: " + ("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}"))
    return failed


def medium_flags(name: str, k: int, args, steps: int, warmup: int) -> list[str]:
    common = ["--preset", "medium", "--opt-steps", str(steps), "--warmup", str(warmup), "--condition", args.condition,
              "--device", "cuda", "--check-every", "1000", "--log-every", "1000", "--no-tensors"]
    return ["--impl", name, "--global-batch", str(2 * M * k), "--micro-batch", str(M), *common]


def stage_count(args, env, rid: str) -> list[dict]:
    k, rows = 4, []
    for name in COUNT:
        tag = f"cnt_{name}_k{k}_{rid}"
        if not args.skip_run:
            launch(2, ["--tag", tag, *medium_flags(name, k, args, steps=8, warmup=3), "--profile",
                       "--profile-steps", "2"], env)
        c = count(OUT / tag / "trace_rank0.json")
        st = c["profile_steps"] or 1
        row = {"variant": name, "backwards_per_step": c["bwd_ranges"] / st,
               **{f"{kind}_per_step": c["kernels"][kind] / st for kind in KINDS},
               "other_per_step": sum(v for kk, v in c["kernels"].items() if kk not in KINDS) / st,
               "raw": dict(c["kernels"])}
        rows.append(row)
    print(f"\n=== COLLECTIVES per optimizer step: medium, K = {k}, rank-0 trace, profiled steps only ===")
    print(f"{'variant':18s} {'backwards':>9s} " + " ".join(f"{kind:>13s}" for kind in KINDS) + f" {'other':>6s}")
    for r in rows:
        print(f"{r['variant']:18s} {r['backwards_per_step']:9.1f} "
              + " ".join(f"{r[kind + '_per_step']:13.1f}" for kind in KINDS) + f" {r['other_per_step']:6.1f}")
    print("one ncclDevKernel_* per collective (Day 04's count_collectives.count); per step = kernels / profiled steps")
    (OUT / f"count_summary_{rid}.json").write_text(json.dumps(rows, indent=2))
    return rows


def timing_table(title: str, names_by_k: dict[int, list[str]], prefix: str, args, env, rid: str,
                 extra_env: dict | None = None) -> list[dict]:
    res = {(n, k): [] for k, ns in names_by_k.items() for n in ns}
    for rep in range(args.repeats):
        for k, names in names_by_k.items():
            for name in names:
                tag = f"{prefix}_{name}_k{k}_r{rep}_{rid}"
                if not args.skip_run:
                    launch(2, ["--tag", tag, *medium_flags(name, k, args, args.steps, args.warmup)],
                           {**env, **(extra_env or {})})
                res[(name, k)].append(json.loads((OUT / tag / "metrics.json").read_text()))
    med = statistics.median
    rng = lambda v: f"{med(v):.2f} [{min(v):.2f}-{max(v):.2f}]"  # noqa: E731
    print(f"\n=== {title}: medium, m = {M} rows/microbatch/rank, {args.steps} optimizer steps ({args.warmup} warmup "
          f"excluded), repeats={args.repeats}, condition={args.condition}, run-id={rid} ===")
    print(f"{'K':>2s} {'variant':15s} {'ms / opt step [min-max]':>24s} {'fwd+bwd':>8s} {'reduce max/last':>15s} "
          f"{'optim':>6s} {'gather max/last':>15s} {'peak MiB':>8s} {'between':>8s} {'state/rank MiB':>15s}")
    rows = []
    for k, names in names_by_k.items():
        for n in names:
            t = [x["timing"] for x in res[(n, k)]]
            pr = res[(n, k)][0]["meta"]["per_rank"]
            core = [x["core_step_ms"] for x in t]
            ph = {p: med(x[p] for x in t) for p in ("fwd_bwd_ms", "reduce_ms", "optim_ms", "gather_ms",
                                                     "reduce_ms_min_rank", "optim_ms_min_rank", "gather_ms_min_rank")}
            state = [sum(r["state_bytes"].get(kk, 0) for kk in ("exp_avg", "exp_avg_sq")) / MiB for r in pr]
            between = max(r.get("alloc_between_micro_mib", float("nan")) for r in pr)
            peak = med(x["peak_alloc_mib_max_rank"] for x in t)
            red = f"{ph['reduce_ms']:.2f}/{ph['reduce_ms_min_rank']:.2f}"
            gat = f"{ph['gather_ms']:.2f}/{ph['gather_ms_min_rank']:.2f}"
            print(f"{k:2d} {n:15s} {rng(core):>24s} {ph['fwd_bwd_ms']:8.2f} {red:>15s} {ph['optim_ms']:6.2f} "
                  f"{gat:>15s} {peak:8.0f} {between:8.1f} {' / '.join(f'{s:.1f}' for s in state):>15s}")
            rows.append({"variant": n, "accum": k, "core_ms": core, "phases_ms": ph, "peak_mib": peak,
                         "between_micro_mib": between, "state_mib_per_rank": state, "per_rank": pr})
        print()
    print("ms / opt step: all K fwd+bwd, the gradient reduction, optimizer, all-gather + scheduler; synchronized phase\n"
          "boundaries; slowest rank per step, median over steps; [min-max] over repeats; phases: median of repeats.\n"
          "max/last: the phase on the slowest rank (includes waiting for the other rank = rank skew) / on the rank that\n"
          "arrived last (min over ranks: the collective itself).\n"
          "ddp / torch_zero: the bucket all-reduces overlap backward (inside fwd+bwd); torch_zero's broadcasts are\n"
          "inside optim. between = memory_allocated() after microbatch 0 (K >= 2), max over ranks; state/rank =\n"
          "exp_avg + exp_avg_sq on rank 0 / rank 1 (repeat 0).")
    return rows


def stage_timing(args, env, rid: str) -> None:
    names_by_k = {k: [n for n in TIMING if not (k == 1 and n == "zero1_rs_every")] for k in args.ks}
    rows = timing_table("TIMING", names_by_k, "time", args, env, rid)
    (OUT / f"timing_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def stage_slowlink(args, env, rid: str) -> None:
    rows = timing_table("SLOW LINK (NCCL_P2P_DISABLE=1)", {4: SLOWLINK}, "slow", args, env, rid,
                        extra_env={"NCCL_P2P_DISABLE": "1"})
    (OUT / f"slowlink_summary_{rid}.json").write_text(json.dumps(rows, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["parity", "count", "timing", "slowlink", "all"], default="all")
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=25, help="optimizer steps per timing run")
    ap.add_argument("--warmup", type=int, default=5, help="optimizer steps excluded from timing")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4], help="timing: microbatches per window")
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
    if args.stage in ("parity", "all") and stage_parity(args, env, rid):
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
