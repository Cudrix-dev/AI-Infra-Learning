"""Trace-level check for run_profiler_overhead.py: optimizer-Graph end offsets between ranks, per profiled run.

Each training step replays the optimizer graph once; its kernels carry a nonzero "graph id" in both CUDA-only and
CPU+CUDA traces (CUDA-only traces have no ProfilerStep ranges, so replays identify the steps). Per rank, kernels with
gaps under 2 ms are one replay (a replay spans ~4.6 ms with ~0.1 ms internal gaps; steps are 35+ ms apart). The last
kernel end of each replay is compared between ranks, replay by replay, after checking that both traces share host
and baseTimeNanoseconds. If these offsets equal the perf_counter end offsets in summary.json, the trace records the
offset that existed in that profiled run; it does not create it in post-processing.

  python experiments/day03_ddp_hooks/analyze_profiler_overhead.py outputs/day03_profiler_overhead/RUN_ID
Writes RUN_ID/trace_offsets.json. Pure Python, no torch.
"""
import json
import statistics
import sys
from pathlib import Path


def replay_ends(path: Path) -> tuple[dict, list[float]]:
    raw = json.loads(path.read_text())
    kernels = sorted((e["ts"], e["ts"] + e["dur"]) for e in raw["traceEvents"]
                     if e.get("ph") == "X" and e.get("cat") == "kernel" and e.get("args", {}).get("graph id", 0))
    replays: list[list[float]] = []
    for a, b in kernels:
        if replays and a - replays[-1][1] < 2000:  # microseconds
            replays[-1][1] = max(replays[-1][1], b)
        else:
            replays.append([a, b])
    return raw, [b for _, b in replays]


def main() -> None:
    run = Path(sys.argv[1])
    summary = json.loads((run / "summary.json").read_text())
    pooled: dict[str, list[float]] = {}
    per_run = []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and (p / "trace_rank0.json").exists()):
        (raw0, e0), (raw1, e1) = replay_ends(d / "trace_rank0.json"), replay_ends(d / "trace_rank1.json")
        for k in ("host_name", "baseTimeNanoseconds"):
            if raw0.get(k) is None or raw0[k] != raw1.get(k):
                raise SystemExit(f"{d.name}: trace metadata {k} differs or is missing; cannot compare ranks")
        if len(e0) != len(e1) or len(e0) != len(summary["window_steps"]):
            raise SystemExit(f"{d.name}: {len(e0)}/{len(e1)} replays, expected {len(summary['window_steps'])}")
        offsets = [(b - a) / 1e3 for a, b in zip(e0, e1)]  # rank1 - rank0, ms
        key = d.name.split("_", 1)[1]  # r0_mini_ddp_cuda -> mini_ddp_cuda
        pooled.setdefault(key, []).extend(abs(x) for x in offsets)
        per_run.append({"run": d.name, "graph_end_offset_ms": offsets})
    rows = []
    print(f"{'impl/config':22s} {'n':>3s} {'trace |end| median':>18s} {'max':>7s} {'boundary |end| median':>22s}")
    for r in summary["summary"]:
        key = f"{r['impl']}_{r['config']}"
        if key not in pooled:
            continue
        v = pooled[key]
        rows.append({"impl": r["impl"], "config": r["config"], "n": len(v), "median_abs_ms": statistics.median(v),
                     "max_abs_ms": max(v), "boundary_median_abs_end_ms": r["abs_end_offset_ms"]["median"]})
        print(f"{key:22s} {len(v):3d} {statistics.median(v):18.3f} {max(v):7.3f} {r['abs_end_offset_ms']['median']:22.3f}")
    (run / "trace_offsets.json").write_text(json.dumps({"method": __doc__.split("\n\n")[0], "summary": rows,
                                                        "runs": per_run}, indent=2))
    print(f"wrote {run / 'trace_offsets.json'}")


if __name__ == "__main__":
    main()
