"""Recompute descriptive interference evidence from both ranks' saved traces.

No causal penalty or channel count is inferred from the correlation.
Run: python experiments/day03_ddp_hooks/analyze_interference.py
"""
import argparse
import json
from pathlib import Path
import statistics

from analyze_trace import load_events, step_metrics

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="a100_20261028T040220Z")
    args = parser.parse_args()
    rows = []
    for variant in ("flat_sync", "mini_per_param", "mini_25", "mini_25_rebuild", "torch_ddp"):
        for rank in (0, 1):
            path = ROOT / "outputs/day03_ddp_hooks" / f"prof_{variant}_{args.run_id}" / f"trace_rank{rank}.json"
            steps, gpu, _ = load_events(path)
            for step in steps:
                rows.append(dict(variant=variant, rank=rank, step=step["name"], **step_metrics(step, gpu)))
    result = {"run_id": args.run_id, "n_rank_steps": len(rows),
              "pearson_overlap_vs_non_nccl": statistics.correlation(
                  [r["overlapped_ms"] for r in rows], [r["gpu_compute_ms"] for r in rows]),
              "scope": "Descriptive association across profiled variants and ranks; not independent trials or a causal penalty.",
              "rows": rows}
    out = ROOT / "day03/assets/profile_interference.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} rank-steps; Pearson r={result['pearson_overlap_vs_non_nccl']:.6f}; wrote {out}")


if __name__ == "__main__":
    main()
