"""Day 09 orchestrator. Plain `python`; it launches train_amp.py itself (one process, or two for --stage ddp).

--stage matrix  (required) one GPU, medium, G = 32 rows per optimizer step, K = 1 (m = 32):
                  precision {fp32, bf16, fp16} x checkpointing {off, on}, same seed, same batches, same clipping.
                  Reports step time (fwd / bwd / opt), tokens/s, memory (start of step, activations held after
                  forward, per-phase peaks), final loss and skipped steps. Checks:
                    * checkpointing changes nothing numerically: per precision, losses over every step and the final
                      parameter SHA-256 are equal with ckpt on and off
                    * drift of bf16 and fp16 from fp32 (loss curve and final parameters)
                Optional rows (skipped with --core): bf16 at K = 4 x m = 8 (same G: accumulation as the other way to
                cut activation memory), and bf16 with SDPA 'auto' (flash attention) with ckpt off / on.
--stage ddp     (optional, two GPUs) one distributed question: under bf16 autocast, in which dtype does DDP all-reduce
                  the gradients, how many bytes per step, and what does bf16_compress_hook change? Profiled runs.
--stage all     both

  python experiments/day09_mixed_precision/compare_amp.py --stage matrix --run-id "$RID" --condition shared
  python experiments/day09_mixed_precision/compare_amp.py --stage ddp    --run-id "$RID" --condition shared
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from launch_utils import make_env, run, smoke_test, torchrun_cmd  # noqa: E402

TRAIN = HERE / "train_amp.py"
OUT = HERE.parents[1] / "outputs" / "day09_amp"
MiB = 2**20

# (name, flags, required)
MATRIX = [
    ("fp32_off", ["--precision", "fp32", "--ckpt", "off"], True),
    ("fp32_on", ["--precision", "fp32", "--ckpt", "on"], True),
    ("bf16_off", ["--precision", "bf16", "--ckpt", "off"], True),
    ("bf16_on", ["--precision", "bf16", "--ckpt", "on"], True),
    ("fp16_off", ["--precision", "fp16", "--ckpt", "off"], True),
    ("fp16_on", ["--precision", "fp16", "--ckpt", "on"], True),
    ("bf16_off_k4", ["--precision", "bf16", "--ckpt", "off", "--micro-batch", "8"], False),
    ("bf16_off_flash", ["--precision", "bf16", "--ckpt", "off", "--sdpa", "auto"], False),
    ("bf16_on_flash", ["--precision", "bf16", "--ckpt", "on", "--sdpa", "auto"], False),
]


def load(tag: str) -> dict:
    return json.loads((OUT / tag / "metrics.json").read_text())


def launch(nproc: int, flags: list[str], env: dict) -> None:
    cmd = [sys.executable, str(TRAIN), *flags] if nproc == 1 else torchrun_cmd(nproc, TRAIN, flags)
    print("\n>>", " ".join(cmd), flush=True)
    if run(cmd, env)[0] != 0:
        raise SystemExit("run failed; not continuing")


def theta0() -> dict:
    """Day 01 initializes on the CPU from seed 42 before .to(device): rebuilding on the CPU gives the same init."""
    sys.path.insert(0, str(HERE.parent / "day01_single_gpu"))
    from train import PRESETS, Config, TinyTransformer, setup_reproducibility
    cfg = Config(**PRESETS["medium"])
    setup_reproducibility(cfg)
    return {k: v.detach().float() for k, v in TinyTransformer(cfg).named_parameters()}


def dist_(a: dict, b: dict) -> float:
    return sum(float((a[k] - b[k]).double().pow(2).sum()) for k in a) ** 0.5


# ============================================================================ matrix
def stage_matrix(args, env, rid: str) -> dict:
    rows = [r for r in MATRIX if r[2] or not args.core]
    common = ["--global-batch", "32", "--micro-batch", "32", "--opt-steps", str(args.steps), "--warmup",
              str(args.warmup), "--condition", args.condition, "--impl", args.impl]
    tags = {}
    for rep in range(args.repeats):
        order = rows if rep % 2 == 0 else rows[::-1]  # alternate the order: drift on a shared host hits both ends
        for name, flags, req in order:
            tag = f"mx_{name}_r{rep}_{rid}"
            extra = ["--save-params"] if rep == 0 and name.endswith("_off") and req else []
            launch(1, [*common, *flags, *extra, "--tag", tag], env)
            tags.setdefault(name, []).append(tag)

    res = {name: [load(t) for t in ts] for name, ts in tags.items()}
    print(f"\n=== MATRIX (medium, G = 32 rows/step, 1 GPU, {args.steps} steps, timing = median of steps >= "
          f"{args.warmup}, {args.repeats} repeats) ===")
    print(f"{'row':16s} {'K x m':>6s} {'ms/step':>16s} {'fwd':>7s} {'bwd':>7s} {'opt':>6s} {'tok/s':>9s} "
          f"{'start':>7s} {'act held':>9s} {'pk fwd':>8s} {'pk bwd':>8s} {'peak':>8s} {'loss':>8s} {'skip':>5s}")
    table = {}
    for name, flags, req in rows:
        ms = [r["timing"]["core_step_ms"] for r in res[name]]
        r0 = res[name][0]
        t, m, meta = r0["timing"], r0["memory"], r0["meta"]
        row = {"required": req, "ms_per_step_repeats": ms, "ms_per_step": statistics.median(ms),
               "fwd_ms": statistics.median(r["timing"]["fwd_ms"] for r in res[name]),
               "bwd_ms": statistics.median(r["timing"]["bwd_ms"] for r in res[name]),
               "opt_ms": statistics.median(r["timing"]["opt_ms"] for r in res[name]),
               "tokens_per_s": statistics.median(r["timing"]["tokens_per_s"] for r in res[name]),
               **{k: m[k] for k in m}, "final_loss": r0["losses"][-1], "skipped": r0["skipped_steps"],
               "K": meta["accum"], "m": meta["micro_batch"], "param_sha256": r0["param_sha256"],
               "grad_dtypes": meta["grad_dtypes"], "param_dtypes": meta["param_dtypes"]}
        table[name] = row
        print(f"{name + ('' if req else ' [opt]'):16s} {row['K']:>2d}x{row['m']:<3d} "
              f"{'/'.join(f'{x:.1f}' for x in ms):>16s} {row['fwd_ms']:7.1f} {row['bwd_ms']:7.1f} {row['opt_ms']:6.1f} "
              f"{row['tokens_per_s']:9,.0f} {m['start_of_step_mib']:7.0f} {m['act_held_mib']:9.0f} "
              f"{m['peak_fwd_mib']:8.0f} {m['peak_bwd_mib']:8.0f} {m['peak_mib']:8.0f} {row['final_loss']:8.4f} "
              f"{row['skipped']:5d}")

    print("\n-- checkpointing at the same precision: on / off --")
    ck = {}
    for p in ("fp32", "bf16", "fp16"):
        off, on = res[f"{p}_off"][0], res[f"{p}_on"][0]
        loss_eq = off["losses"] == on["losses"]
        sha_eq = off["param_sha256"] == on["param_sha256"]
        dmax = max(abs(a - b) for a, b in zip(off["losses"], on["losses"]))
        a, b = table[f"{p}_off"], table[f"{p}_on"]
        ck[p] = {"losses_equal": loss_eq, "params_bitwise": sha_eq, "max_abs_loss_diff": dmax,
                 "peak_ratio": b["peak_mib"] / a["peak_mib"], "act_held_ratio": b["act_held_mib"] / a["act_held_mib"],
                 "time_ratio": b["ms_per_step"] / a["ms_per_step"], "bwd_time_ratio": b["bwd_ms"] / a["bwd_ms"],
                 "peak_saved_mib": a["peak_mib"] - b["peak_mib"], "ms_added": b["ms_per_step"] - a["ms_per_step"]}
        c = ck[p]
        print(f"{p}: losses equal over {len(off['losses'])} steps {loss_eq} (max |diff| {dmax:.2e}), final params "
              f"bitwise {sha_eq} | peak {a['peak_mib']:.0f} -> {b['peak_mib']:.0f} MiB (x{c['peak_ratio']:.2f}), act "
              f"held {a['act_held_mib']:.0f} -> {b['act_held_mib']:.0f} MiB | step {a['ms_per_step']:.1f} -> "
              f"{b['ms_per_step']:.1f} ms (x{c['time_ratio']:.2f}; backward x{c['bwd_time_ratio']:.2f})")
    ck_pass = all(c["losses_equal"] and c["params_bitwise"] for c in ck.values())
    print(f"CKPT EQUALITY: {'PASSED' if ck_pass else 'FAILED'} (losses and final parameters bitwise equal, all "
          f"three precisions)")

    print("\n-- drift from fp32 (ckpt off, repeat 0, same init and batches) --")
    t0 = theta0()
    p32 = torch.load(OUT / tags["fp32_off"][0] / "params_final.pt", map_location="cpu", weights_only=True)
    move = dist_(p32, t0)
    l32 = res["fp32_off"][0]["losses"]
    drift = {}
    for p in ("bf16", "fp16"):
        pp = torch.load(OUT / tags[f"{p}_off"][0] / "params_final.pt", map_location="cpu", weights_only=True)
        lp = res[f"{p}_off"][0]["losses"]
        diffs = [abs(a - b) for a, b in zip(lp, l32)]
        drift[p] = {"mean_abs_loss_diff_last10": statistics.fmean(diffs[-10:]), "max_abs_loss_diff": max(diffs),
                    "final_loss": lp[-1], "final_loss_fp32": l32[-1],
                    "param_dist_over_fp32_move": dist_(pp, p32) / move}
        d = drift[p]
        print(f"{p}: |loss - loss_fp32| mean over the last 10 steps {d['mean_abs_loss_diff_last10']:.2e}, max "
              f"{d['max_abs_loss_diff']:.2e} | final loss {d['final_loss']:.4f} vs {l32[-1]:.4f} | "
              f"||theta - theta_fp32|| / ||theta_fp32 - theta_0|| = {d['param_dist_over_fp32_move']:.3e}")
    print(f"(fp32 training moved the parameters by ||theta_fp32 - theta_0|| = {move:.3f}; loss "
          f"{l32[0]:.3f} -> {l32[-1]:.3f})")
    if not args.keep_params:
        for name in ("fp32_off", "bf16_off", "fp16_off"):
            (OUT / tags[name][0] / "params_final.pt").unlink()
        print("removed params_final.pt (3 x 105 MiB; pass --keep-params to keep them)")
    summary = {"table": table, "ckpt": ck, "ckpt_pass": ck_pass, "drift": drift, "fp32_move": move,
               "tags": tags, "steps": args.steps, "warmup": args.warmup, "repeats": args.repeats}
    (OUT / f"matrix_summary_{rid}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT / f'matrix_summary_{rid}.json'}")
    return summary


# ============================================================================ ddp
def stage_ddp(args, env, rid: str) -> dict:
    smoke_test(2, "cuda", 1)
    common = ["--precision", "bf16", "--global-batch", "32", "--opt-steps", "8", "--warmup", "3", "--condition",
              args.condition, "--save-grads0", "--impl", args.impl]
    plan = [("single", 1, ["--micro-batch", "32"]),
            ("ddp", 2, ["--micro-batch", "16", "--profile"]),
            ("ddp_bf16hook", 2, ["--micro-batch", "16", "--profile", "--comm-hook", "bf16"])]
    tags = {}
    for name, n, flags in plan:
        tags[name] = f"ddp_{name}_{rid}"
        launch(n, [*common, *flags, "--tag", tags[name]], env)
    res = {k: load(t) for k, t in tags.items()}
    g = {k: torch.load(OUT / t / "grads0.pt", map_location="cpu", weights_only=True) for k, t in tags.items()}

    def rel(a, b):
        return dist_(a, b) / sum(float(v.double().pow(2).sum()) for v in b.values()) ** 0.5

    print("\n=== DDP question: under bf16 autocast, what does DDP all-reduce? (2 GPUs, G = 32 = 2 x 16, profiled) ===")
    out = {}
    for name in ("ddp", "ddp_bf16hook"):
        c = res[name]["comms"]
        tot_bytes = 0.0  # gradient all-reduces only (the Bool broadcast is DDP re-sending the causal-mask buffer)
        for k, v in c["collectives"].items():
            if k.startswith("allreduce"):
                tot_bytes += v["elements_per_step"] * {"Float": 4, "BFloat16": 2, "Half": 2}[k.split()[-1]]
        out[name] = {"collectives": c["collectives"], "nccl_kernels_per_step": c["nccl_kernels_per_step"],
                     "bytes_per_step_mib": tot_bytes / MiB, "grad_dtypes": res[name]["meta"]["grad_dtypes"],
                     "grad_rel_err_vs_single": rel(g[name], g["single"])}
        print(f"\n[{name}] p.grad dtype {res[name]['meta']['grad_dtypes']}")
        for k, v in c["collectives"].items():
            print(f"  {k}: {v['calls_per_step']:.0f} calls/step, {v['elements_per_step']:,.0f} elements/step "
                  f"(bucket sizes {v['sizes']})")
        print(f"  gradient all-reduce payload per step = {tot_bytes / MiB:.1f} MiB | NCCL kernels/step: "
              f"{c['nccl_kernels_per_step']}")
        print(f"  step-0 gradient vs 1-GPU bf16 on the same 32 rows: rel err {out[name]['grad_rel_err_vs_single']:.3e}")
    out["hook_vs_plain_rel_err"] = rel(g["ddp_bf16hook"], g["ddp"])
    print(f"\nbf16 hook vs plain DDP gradients: rel err {out['hook_vs_plain_rel_err']:.3e}")
    if not args.keep_params:
        for t in tags.values():
            (OUT / t / "grads0.pt").unlink()
    first = res["ddp"]["comms"]["first_record_param_comms_args"]
    print(f"(first record_param_comms args in the trace, for reference: {first})")
    (OUT / f"ddp_summary_{rid}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT / f'ddp_summary_{rid}.json'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["matrix", "ddp", "all"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="shared")
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--core", action="store_true", help="required matrix rows only")
    ap.add_argument("--keep-params", action="store_true", help="keep params_final.pt / grads0.pt files")
    args = ap.parse_args()
    env = make_env(1)
    if args.stage in ("matrix", "all"):
        stage_matrix(args, env, args.run_id)
    if args.stage in ("ddp", "all"):
        stage_ddp(args, env, args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
