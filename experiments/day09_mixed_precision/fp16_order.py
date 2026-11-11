"""Day 09 debugging exercise: the order of loss scaling, accumulation, unscaling, clipping and the update in fp16.

One GPU, one process. Day 01's medium model, one window of K = 4 microbatches (G = 16, m = 4, Day 04's weights),
clip_grad_norm_(max_norm = 1.0), AdamW. Three runs from the same init on the same batches:

  ref   fp32, no autocast, no scaler            the gradient and update we want
  ok    fp16 autocast + GradScaler, your order   scale -> backward x K -> unscale_ -> clip -> step -> update
  bug   fp16, bug="clip_before_unscale"          scale -> backward x K -> clip -> unscale_ -> step -> update

Checks (step 0 unless stated):
  C0  contract     per window: scale() K times, unscale_ once, update once; the scale is constant over the window
  C1  clip input   the norm clip_grad_norm_ returned (what it SAW) / the true fp32 norm    ok ~ 1, bug ~ S
  C2  clip output  ||g_handed_to_AdamW - g_ref|| / ||g_ref||  (after clipping both)        ok small, bug ~ 1
  C3  training     ||theta_N - theta_0|| / ||theta_N,ref - theta_0|| after --steps windows  ok ~ 1, bug << 1
The control must FAIL C1 and C2 ("DETECTED"), and C3 shows the consequence: the bug does not crash, it just makes
every update tiny (AdamW divides by sqrt(v) + eps, and with gradients near 1e-10 the eps = 1e-8 dominates).

  python experiments/day09_mixed_precision/fp16_order.py                 # reference implementation
  python experiments/day09_mixed_precision/fp16_order.py --impl mine     # YOUR window_step (my_amp.py)
  python experiments/day09_mixed_precision/fp16_order.py --underflow     # optional: fp16 WITHOUT a scaler
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day01_single_gpu"))
sys.path.insert(0, str(HERE))
from train import PRESETS, Config, build, get_batch, setup_reproducibility  # noqa: E402

from amp_setup import AmpNet, load_impl, make_scaler, sdpa_context  # noqa: E402

OUT = HERE.parents[1] / "outputs" / "day09_amp"
TOL_C1, TOL_C2, TOL_C3 = 0.01, 0.02, 0.10


class CountingScaler(torch.amp.GradScaler):
    """GradScaler that counts its calls, so C0 can check the per-window contract of YOUR window_step.
    (step() calls unscale_ itself when you did not: that call is counted too, as it should be.)"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = Counter()

    def scale(self, outputs):
        self.calls["scale"] += 1
        return super().scale(outputs)

    def unscale_(self, optimizer):
        self.calls["unscale_"] += 1
        return super().unscale_(optimizer)

    def update(self, new_scale=None):
        self.calls["update"] += 1
        return super().update(new_scale)


def flat(params) -> torch.Tensor:
    return torch.cat([p.grad.detach().float().reshape(-1) for p in params])


def run_variant(cfg, device, impl, precision: str, bug: str, steps: int, m: int, scaler_on: bool = True) -> dict:
    setup_reproducibility(cfg)
    model, optimizer, scheduler = build(cfg, device)
    params = list(model.parameters())
    theta0 = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    net = AmpNet(model, False, impl.model_forward)
    scaler = make_scaler(precision, enabled=scaler_on, scaler_cls=CountingScaler)
    rec = {"windows": []}
    for s in range(steps):
        x, y = get_batch(cfg, s, device)  # G rows; K = G / m microbatches
        micro = [(a, b, a.size(0) / x.size(0)) for a, b in zip(x.split(m), y.split(m))]
        w = {"scale_at_bwd": [], "calls_before": dict(scaler.calls) if scaler else {}}

        def on(event, w=w, s=s):
            if event == "bwd" and scaler is not None:
                w["scale_at_bwd"].append(scaler.get_scale())
            if s != 0:
                return
            if event == "pre_clip":
                g = flat(params)
                w["pre_clip_norm"] = float(g.norm())
                w["pre_clip_zero_frac"] = float((g == 0).float().mean())
                w["pre_clip_grad"] = g  # kept on the GPU: 105 MiB, four runs
            elif event == "pre_step":
                w["pre_step_grad"] = flat(params)

        with sdpa_context("math"):
            out = impl.window_step(net, optimizer, scaler, micro, precision=precision, max_norm=1.0, bug=bug, on=on)
        scheduler.step()
        w.update(loss=out["loss"].item(), norm_seen_by_clip=float(out["norm"]), scale=out["scale"],
                 skipped=out["skipped"])
        if scaler is not None:
            w["calls"] = {k: scaler.calls[k] - w["calls_before"].get(k, 0) for k in ("scale", "unscale_", "update")}
        rec["windows"].append(w)
    rec["move"] = float((torch.cat([p.detach().reshape(-1) for p in params]) - theta0).norm())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4, help="optimizer steps (windows) for C3")
    ap.add_argument("--underflow", action="store_true", help="optional: also run fp16 with the scaler disabled")
    ap.add_argument("--run-id", default="local")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    impl = load_impl(args.impl)
    cfg = replace(Config(**PRESETS["medium"]), batch_size=args.global_batch, total_steps=args.steps)
    K = args.global_batch // args.micro_batch
    print(f"torch {torch.__version__} | impl={args.impl} | medium, G={args.global_batch} = K {K} x m "
          f"{args.micro_batch}, max_norm=1.0, {args.steps} optimizer steps, SDPA math")

    runs = {"ref": run_variant(cfg, device, impl, "fp32", "none", args.steps, args.micro_batch),
            "ok": run_variant(cfg, device, impl, "fp16", "none", args.steps, args.micro_batch),
            "bug": run_variant(cfg, device, impl, "fp16", "clip_before_unscale", args.steps, args.micro_batch)}
    if args.underflow:
        runs["noscale"] = run_variant(cfg, device, impl, "fp16", "none", args.steps, args.micro_batch,
                                      scaler_on=False)
    ref0 = runs["ref"]["windows"][0]
    true_norm = ref0["pre_clip_norm"]
    g_ref = ref0["pre_step_grad"]
    print(f"\nfp32 reference, step 0: true gradient norm {true_norm:.4f} -> clipping "
          f"{'ACTIVE' if true_norm > 1.0 else 'inactive'} (max_norm 1.0); ||g after clip|| {float(g_ref.norm()):.4f}")

    print(f"\n{'variant':9s} {'scale S':>9s} {'C1 norm seen by clip':>21s} {'/ true':>10s} {'C2 rel err after clip':>22s}"
          f" {'||g to AdamW||':>15s} {'C3 move ratio':>14s} {'skipped':>8s}")
    summary = {}
    for name, r in runs.items():
        w0 = r["windows"][0]
        seen = w0["norm_seen_by_clip"]
        c1 = seen / true_norm
        c2 = float((w0["pre_step_grad"] - g_ref).norm() / g_ref.norm())
        c3 = r["move"] / runs["ref"]["move"]
        skipped = sum(w["skipped"] for w in r["windows"])
        summary[name] = {"scale": w0["scale"], "norm_seen_by_clip": seen, "c1_ratio": c1, "c2_rel_err": c2,
                         "grad_to_adamw_norm": float(w0["pre_step_grad"].norm()), "c3_move_ratio": c3,
                         "skipped": skipped, "loss": [w["loss"] for w in r["windows"]],
                         "pre_clip_zero_frac": w0["pre_clip_zero_frac"],
                         "scale_at_bwd": w0["scale_at_bwd"], "calls": [w.get("calls") for w in r["windows"]]}
        print(f"{name:9s} {w0['scale']:9.0f} {seen:21.4e} {c1:10.3e} {c2:22.3e} "
              f"{summary[name]['grad_to_adamw_norm']:15.4e} {c3:14.3e} {skipped:8d}")

    # C0: the per-window contract of the scaler calls (ok and bug use the same call counts, different order)
    c0 = {}
    for name in ("ok", "bug"):
        ws = runs[name]["windows"]
        calls_ok = all(w["calls"] == {"scale": K, "unscale_": 1, "update": 1} for w in ws)
        const = all(len(set(w["scale_at_bwd"])) == 1 for w in ws)
        c0[name] = calls_ok and const
        print(f"C0 {name:4s}: per window scale() x{ws[0]['calls']['scale']}, unscale_ x{ws[0]['calls']['unscale_']}, "
              f"update x{ws[0]['calls']['update']}; scale constant within every window: {const}  =>  "
              f"{'pass' if c0[name] else 'FAIL'}")

    ok = summary["ok"]
    bug = summary["bug"]
    ok_pass = {"C0": c0["ok"], "C1": abs(ok["c1_ratio"] - 1) < TOL_C1, "C2": ok["c2_rel_err"] < TOL_C2,
               "C3": abs(ok["c3_move_ratio"] - 1) < TOL_C3}
    bug_fail = {"C1": abs(bug["c1_ratio"] - 1) >= TOL_C1, "C2": bug["c2_rel_err"] >= TOL_C2,
                "C3": abs(bug["c3_move_ratio"] - 1) >= TOL_C3}
    print(f"\nok  : " + " ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in ok_pass.items()))
    print(f"bug : " + " ".join(f"{k} {'fail' if v else 'pass'}" for k, v in bug_fail.items())
          + f"  =>  {'DETECTED by C1 and C2' if bug_fail['C1'] and bug_fail['C2'] else 'NOT DETECTED'}")
    print(f"      clip saw S * ||g|| = {bug['norm_seen_by_clip']:.3e} (S = {bug['scale']:.0f}) and scaled the gradient "
          f"by about 1 / (S * ||g||); after unscale_ AdamW received ||g|| = {bug['grad_to_adamw_norm']:.3e} instead of "
          f"{float(g_ref.norm()):.3e}")
    if "noscale" in summary:
        ns = summary["noscale"]
        print(f"\noptional, fp16 without GradScaler: gradient elements exactly 0 before clipping: "
              f"{ns['pre_clip_zero_frac']:.2%} vs {ok['pre_clip_zero_frac']:.2%} with S = {ok['scale']:.0f} and "
              f"{summary['ref']['pre_clip_zero_frac']:.2%} in fp32; rel err vs fp32 after clip {ns['c2_rel_err']:.3e} "
              f"(with scaler {ok['c2_rel_err']:.3e})")
    passed = all(ok_pass.values()) and bug_fail["C1"] and bug_fail["C2"]
    print(f"\nFP16 ORDER: {'ALL CHECKS PASSED' if passed else 'CHECKS FAILED'} "
          f"(ok passes C0-C3, clip_before_unscale {'DETECTED' if bug_fail['C1'] and bug_fail['C2'] else 'missed'})")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fp16_order_{args.impl}_{args.run_id}.json"
    path.write_text(json.dumps({"true_norm": true_norm, "summary": summary, "c0": c0, "ok_pass": ok_pass,
                                "bug_fail": bug_fail, "passed": passed}, indent=2))
    print(f"wrote {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
