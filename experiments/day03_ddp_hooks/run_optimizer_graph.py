"""Run the optimizer-only graph follow-up on an otherwise unchanged Day03 model.

Run in /venv/main: python experiments/day03_ddp_hooks/run_optimizer_graph.py --run-id NAME
Results use outputs/day03_cuda_graph/NAME, never replace the original Day03 run.
With --bucket-view-study, compare independent/view Graph runs instead, under
outputs/day03_cuda_graph_view/NAME, after validating matched eager controls.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))
from check_parity import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--stage", choices=["parity", "timing", "profile", "all"], default="all")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--bucket-view-study", action="store_true", help="compare graph replay with independent grads vs stable bucket views")
    a = p.parse_args()
    out = ROOT / "outputs" / ("day03_cuda_graph_view" if a.bucket_view_study else "day03_cuda_graph") / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    modes = ["original", "ready_eager", "graph"]
    if a.bucket_view_study:
        modes = ["ready_eager", "graph", "ready_eager_view", "graph_view"]
    impls = ["mini_ddp", "torch_ddp"]

    def run(tag, impl, mode, preset, steps, warmup, profile=False, tensors=False):
        view = mode.endswith("_view")
        optimizer_mode = mode.removesuffix("_view")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               str(HERE / "train_ddp.py"), "--impl", impl, "--optimizer-mode", optimizer_mode,
               "--preset", preset, "--steps", str(steps), "--warmup", str(warmup),
               "--global-batch", "16", "--output-root", str(out), "--tag", tag,
               "--condition", "unknown", "--check-every", str(steps), "--log-every", str(steps)]
        if impl == "mini_ddp":
            cmd += ["--rebuild", "--bucket-mb", "0.25" if preset == "tiny" else "25",
                    "--first-bucket-mb", "0.0625" if preset == "tiny" else "1"]
        elif preset == "tiny" and impl == "torch_ddp":
            cmd += ["--bucket-mb", "0.25"]
        if profile:
            cmd += ["--profile", "--profile-steps", "3"]
        if view:
            cmd += ["--grad-as-view"]
        if not tensors:
            cmd += ["--no-tensors"]
        print(f"\nSTART {tag}: {' '.join(cmd)}", flush=True)
        with (out / f"{tag}.log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print((out / f"{tag}.log").read_text()[-4000:], flush=True)
        result.check_returncode()
        return torch.load(out / tag / "result.pt", map_location="cpu", weights_only=False)

    if a.stage in ("parity", "all"):
        results = {}
        reference = run("parity_flat_original", "flat_sync", "original", "tiny", 24, 5, tensors=True)
        for impl in impls:
            for mode in modes:
                tag = f"parity_{impl}_{mode}"
                r = run(tag, impl, mode, "tiny", 24, 5, tensors=True)
                # Existing Day02/03 numerical acceptance tests.
                checks = compare(reference, r)
                print(json.dumps(checks, indent=2), flush=True)
                assert all(checks[k][0] for k in ("A", "B", "C")), f"parity failed: {tag}"
                meta = r["meta"]
                assert set(meta["adam_steps"]) == {24.0}, meta["adam_steps"]
                assert meta["scheduler_last_epoch"] == 24
                from train_ddp import Config, PRESETS, lr_lambda
                cfg = Config(**PRESETS["tiny"])
                cfg.total_steps = 24
                assert all(math.isclose(v, cfg.lr * lr_lambda(i, cfg), rel_tol=2e-6)
                           for i, v in enumerate(meta["lr_used"]))
                results[tag] = r
        matched = {}
        for impl in impls:
            for suffix in (["", "_view"] if a.bucket_view_study else [""]):
                eager = results[f"parity_{impl}_ready_eager{suffix}"]
                graph = results[f"parity_{impl}_graph{suffix}"]
                keyname = impl + suffix
                matched[keyname] = {key: all(torch.equal(eager[key][n], graph[key][n]) for n in eager[key])
                                       for key in ("grads0", "params1", "final")}
                matched[keyname]["losses"] = eager["losses"] == graph["losses"]
                assert all(matched[keyname].values()), matched
            if a.bucket_view_study:
                independent, view = [results[f"parity_{impl}_{mode}"] for mode in ("graph", "graph_view")]
                matched[impl + "_view_vs_independent"] = {
                    key: all(torch.equal(independent[key][n], view[key][n]) for n in independent[key])
                    for key in ("grads0", "params1", "final")}
                matched[impl + "_view_vs_independent"]["losses"] = independent["losses"] == view["losses"]
                assert all(matched[impl + "_view_vs_independent"].values()), matched
        (out / "parity_summary.json").write_text(json.dumps({"all_passed": True, "matched_bitwise": matched,
            "steps": 24, "lr_schedule_checked": True, "adam_counters_checked": True}, indent=2))

    if a.stage in ("timing", "all"):
        if a.bucket_view_study:
            modes = ["graph", "graph_view"]
        rows = []
        # Rotate order to reduce systematic run-order bias; no profiler for these jobs.
        for repeat in range(a.repeats):
            for impl in impls[repeat % 2:] + impls[:repeat % 2]:
                for mode in modes[repeat % len(modes):] + modes[:repeat % len(modes)]:
                    tag = f"timing_r{repeat}_{impl}_{mode}"
                    r = run(tag, impl, mode, "medium", 60, 10)
                    rows.append(dict(repeat=repeat, impl=impl, mode=mode, **r["timing"]))
        summary = {}
        for impl in impls:
            summary[impl] = {mode: {"median_ms": statistics.median(
                r["core_step_ms"] for r in rows if r["impl"] == impl and r["mode"] == mode),
                "repeats_ms": [r["core_step_ms"] for r in rows if r["impl"] == impl and r["mode"] == mode]}
                for mode in modes}
        (out / "timing_summary.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))

    if a.stage in ("profile", "all"):
        if a.bucket_view_study:
            modes = ["graph", "graph_view"]
        for impl in impls:
            for mode in modes:
                run(f"profile_{impl}_{mode}", impl, mode, "medium", 14, 5, profile=True)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
