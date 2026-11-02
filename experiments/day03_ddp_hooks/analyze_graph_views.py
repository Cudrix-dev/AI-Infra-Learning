"""Analyze graph-view ablation: copy-back, optimizer completion and rank skew.

Pure Python: python experiments/day03_ddp_hooks/analyze_graph_views.py RUN_DIR
"""
import argparse
import json
from pathlib import Path
import statistics

from analyze_optimizer_graph import optimizer_metrics
from analyze_trace import load_events, length, union, svg_panels


def copy_metrics(raw, step, impl):
    xs = [e for e in raw["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    inside = lambda e: step["ts"] <= e["ts"] < step["ts"] + step["dur"]
    if impl == "mini_ddp":
        scopes = [e for e in xs if inside(e) and e.get("cat") == "user_annotation"
                  and e["name"] == "## finish_grad_sync"]
    else:
        scopes = [e for e in xs if inside(e) and e.get("cat") == "cpu_op"
                  and "copy_bucket_to_grad" in e["name"]]
    ids = {e.get("args", {}).get("External id") for e in xs
           if e.get("cat") == "cpu_op" and e["name"] == "aten::copy_"
           and any(s["ts"] <= e["ts"] < s["ts"] + s["dur"] for s in scopes)}
    ids.discard(None)
    copies = [e for e in xs if inside(e) and e.get("cat") == "gpu_memcpy"
              and e.get("args", {}).get("External id") in ids]
    all_copies = [e for e in xs if inside(e) and e.get("cat") == "gpu_memcpy"]
    return {"copyback_count": len(copies),
            "copyback_mib": sum(e.get("args", {}).get("bytes", 0) for e in copies) / 2**20,
            "copyback_busy_ms": length(union([(e["ts"], e["ts"] + e["dur"]) for e in copies])) / 1000,
            "all_memcpy_count": len(all_copies),
            "all_memcpy_mib": sum(e.get("args", {}).get("bytes", 0) for e in all_copies) / 2**20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    profile, selections = [], []
    for impl in ("mini_ddp", "torch_ddp"):
        panels, origins, span = [], [], 0
        for mode in ("graph", "graph_view"):
            folder = root / f"profile_{impl}_{mode}"
            raw = [json.loads((folder / f"trace_rank{r}.json").read_text()) for r in (0, 1)]
            for key in ("host_name", "baseTimeNanoseconds"):
                assert raw[0].get(key) is not None and raw[0][key] == raw[1][key]
            traces = [load_events(folder / f"trace_rank{r}.json") for r in (0, 1)]
            selected = sorted(traces[0][0], key=lambda s: s["dur"])[len(traces[0][0]) // 2]["name"]
            steps = [next(s for s in trace[0] if s["name"] == selected) for trace in traces]
            origin = min(s["ts"] for s in steps)
            selected_ends = []
            for rank, (data, trace, step) in enumerate(zip(raw, traces, steps)):
                rows = []
                for s in trace[0]:
                    m = {**optimizer_metrics(data, s, "graph"), **copy_metrics(data, s, impl)}
                    assert m["copyback_count"] == (0 if mode == "graph_view" else 78), (impl, mode, m)
                    rows.append(m)
                med = {k: statistics.median(r[k] for r in rows) for k in rows[0]
                       if k not in ("step", "optimizer_gpu_start_ts", "optimizer_gpu_end_ts")}
                profile.append(dict(impl=impl, mode=mode, rank=rank, median=med, per_step=rows))
                selected_ends.append(next(r["optimizer_gpu_end_ts"] for r in rows if r["step"] == selected))
                panels.append((f"{mode} | rank {rank} | {selected}", step, trace[1], trace[2]))
                origins.append(origin)
                span = max(span, step["ts"] + step["dur"] - origin)
            selections.append(dict(impl=impl, mode=mode, step=selected,
                                   rank1_start_minus_rank0_ms=(steps[1]["ts"] - steps[0]["ts"]) / 1000,
                                   rank1_optimizer_end_minus_rank0_ms=(selected_ends[1] - selected_ends[0]) / 1000))
        svg_panels(panels, root / f"timeline_view_{impl}.svg", origins_us=origins, span_us=span, width=1400)

    offsets = []
    for folder in sorted(root.glob("timing_r*")):
        if not folder.is_dir():
            continue
        meta = json.loads((folder / "metrics.json").read_text())["meta"]
        warm = meta["warmup_steps"]
        ranks = [json.loads((folder / f"boundaries_rank{r}.json").read_text()) for r in (0, 1)]
        d = [(b["core_start"] - a["core_start"]) * 1000 for a, b in zip(ranks[0][warm:], ranks[1][warm:])]
        end = [(b["core_end"] - a["core_end"]) * 1000 for a, b in zip(ranks[0][warm:], ranks[1][warm:])]
        offsets.append(dict(tag=folder.name, median_abs_start_ms=statistics.median(map(abs, d)),
                            max_abs_start_ms=max(map(abs, d)), median_abs_end_ms=statistics.median(map(abs, end)),
                            signed_start_ms=d, signed_end_ms=end))
    result = dict(profile=profile, selection=selections, unprofiled_offsets=offsets)
    (root / "view_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "per_step"} for r in profile], indent=2))


if __name__ == "__main__":
    main()
