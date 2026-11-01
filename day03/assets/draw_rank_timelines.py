"""Draw both ranks with matching step IDs and a common scale from saved Day 03 traces.

Select the median-length rank-0 step for each variant, then select that same
ProfilerStep ID on rank 1. Each panel's zero remains its own local step start;
the separate figures are not globally clock-aligned. An additional combined
figure uses a shared origin within each variant, after verifying the host and
trace clock-base metadata match. Original server SVGs stay untouched.

  python day03/assets/draw_rank_timelines.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/day03_ddp_hooks"))
import analyze_trace as analyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="a100_20261028T040220Z")
    args = parser.parse_args()
    out = ROOT / "outputs/day03_ddp_hooks"
    panels = {0: [], 1: []}
    aligned_panels, aligned_origins = [], []
    aligned_span = 0.0
    selections = []
    variants = ("flat_sync", "mini_per_param", "mini_25", "mini_25_rebuild", "torch_ddp")
    for variant in variants:
        rank0_path = out / f"prof_{variant}_{args.run_id}" / "trace_rank0.json"
        _, chosen, gpu0, marks0 = analyzer.summarize(rank0_path)
        panels[0].append((f"{variant} | rank 0 | {chosen['name']}", chosen, gpu0, marks0))
        rank1_path = rank0_path.with_name("trace_rank1.json")
        steps1, gpu1, marks1 = analyzer.load_events(rank1_path)
        matches = [s for s in steps1 if s["name"] == chosen["name"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one matching {chosen['name']} in {rank1_path}")
        panels[1].append((f"{variant} | rank 1 | {chosen['name']}", matches[0], gpu1, marks1))
        raw0, raw1 = (json.loads(p.read_text(encoding="utf-8")) for p in (rank0_path, rank1_path))
        for key in ("host_name", "baseTimeNanoseconds"):
            if raw0.get(key) is None or raw0[key] != raw1.get(key):
                raise ValueError(f"Cannot align {variant}: trace metadata {key} differs or is missing")
        origin = min(chosen["ts"], matches[0]["ts"])
        for rank, step, gpu, marks in ((0, chosen, gpu0, marks0), (1, matches[0], gpu1, marks1)):
            offset = (step["ts"] - origin) / 1000
            aligned_panels.append((f"{variant} | rank {rank} | {step['name']} | start +{offset:.2f} ms", step, gpu, marks))
            aligned_origins.append(origin)
            aligned_span = max(aligned_span, step["ts"] + step["dur"] - origin)
        selections.append({"variant": variant, "step": chosen["name"],
                           "rank1_step_start_minus_rank0_ms": (matches[0]["ts"] - chosen["ts"]) / 1000,
                           "rank0": analyzer.step_metrics(chosen, gpu0),
                           "rank1": analyzer.step_metrics(matches[0], gpu1)})
    span = max(step["dur"] for rank in panels.values() for _, step, _, _ in rank)
    for rank in (0, 1):
        artifact = out / f"timeline_rank{rank}_{args.run_id}.svg"
        analyzer.svg_panels(panels[rank], artifact, span_us=span)
        asset = Path(__file__).parent / ("overlap_timeline.svg" if rank == 0 else "overlap_timeline_rank1.svg")
        shutil.copyfile(artifact, asset)
    aligned_artifact = out / f"timeline_both_ranks_aligned_{args.run_id}.svg"
    analyzer.svg_panels(aligned_panels, aligned_artifact, span_us=aligned_span, origins_us=aligned_origins)
    shutil.copyfile(aligned_artifact, Path(__file__).parent / "overlap_timeline_both_ranks.svg")
    (out / f"timeline_selection_{args.run_id}.json").write_text(json.dumps({
        "run_id": args.run_id, "span_us": span,
        "selection": "median-length rank-0 step; same step ID on rank 1",
        "origin": "local start of each step, not a shared absolute clock",
        "combined_figure_origin": "earlier rank step start within each variant; common host and baseTimeNanoseconds verified",
        "panels": selections,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
