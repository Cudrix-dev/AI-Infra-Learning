"""Offline, torch-free analysis of optimizer graph traces and rank offsets."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_trace import load_events, union, length, svg_panels


def optimizer_metrics(raw, step, mode):
    events = raw["traceEvents"]
    xs = [e for e in events if e.get("ph") == "X" and "dur" in e]
    inside = lambda e: step["ts"] <= e["ts"] < step["ts"] + step["dur"]
    scope = next(e for e in xs if inside(e) and e.get("cat") == "user_annotation"
                 and e.get("name") == "## optimizer")
    launches = [e for e in xs if scope["ts"] <= e["ts"] < scope["ts"] + scope["dur"]
                and e.get("cat") == "cuda_runtime"]
    adam = [e for e in xs if inside(e) and e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("Optimizer.step#AdamW.step")]
    if mode == "graph":
        # Only optimizer.step is captured. Graph nodes may share one external ID;
        # their graph ID identifies membership without guessing CPU ancestry.
        kernels = [e for e in xs if inside(e) and e.get("cat") == "kernel"
                   and e.get("args", {}).get("graph id", 0)]
    else:
        assert len(adam) == 1, adam
        a = adam[0]
        ids = {e.get("args", {}).get("External id") for e in xs if e.get("cat") == "cpu_op"
               and a["ts"] <= e["ts"] < a["ts"] + a["dur"]}
        ids.discard(None)
        kernels = [e for e in xs if inside(e) and e.get("cat") == "kernel"
                   and e.get("args", {}).get("External id") in ids]
    assert kernels, f"No optimizer kernels matched {mode} {step['name']}"
    intervals = union([(e["ts"], e["ts"] + e["dur"]) for e in kernels])
    busy = length(intervals)
    start, end = intervals[0][0], intervals[-1][1]
    return {"step": step["name"], "optimizer_cpu_ms": scope["dur"] / 1000,
            "optimizer_kernels": len(kernels), "optimizer_gpu_span_ms": (end - start) / 1000,
            "optimizer_gpu_busy_ms": busy / 1000, "optimizer_gpu_gaps_ms": (end - start - busy) / 1000,
            "cuda_launch_kernel_calls": sum(e["name"].startswith("cudaLaunchKernel") for e in launches),
            "cuda_graph_launch_calls": sum(e["name"].startswith("cudaGraphLaunch") for e in launches),
            "optimizer_gpu_start_ts": start, "optimizer_gpu_end_ts": end}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    a = p.parse_args()
    profile_rows, selection, skew_rows = [], [], []
    for impl in ("mini_ddp", "torch_ddp"):
        panels, origins, span = [], [], 0
        for mode in ("original", "ready_eager", "graph"):
            folder = a.run_dir / f"profile_{impl}_{mode}"
            raws = [json.loads((folder / f"trace_rank{r}.json").read_text()) for r in (0, 1)]
            for key in ("host_name", "baseTimeNanoseconds"):
                assert raws[0].get(key) is not None and raws[0][key] == raws[1][key], key
            ranks = [load_events(folder / f"trace_rank{r}.json") for r in (0, 1)]
            selected = sorted(ranks[0][0], key=lambda s: s["dur"])[len(ranks[0][0]) // 2]["name"]
            chosen = [next(s for s in rank[0] if s["name"] == selected) for rank in ranks]
            origin = min(s["ts"] for s in chosen)
            for rank, ((steps, gpu, marks), raw, step) in enumerate(zip(ranks, raws, chosen)):
                per = [optimizer_metrics(raw, s, mode) for s in steps]
                numeric = [k for k in per[0] if k not in ("step", "optimizer_gpu_start_ts", "optimizer_gpu_end_ts")]
                profile_rows.append({"impl": impl, "mode": mode, "rank": rank,
                                     "median": {k: statistics.median(s[k] for s in per) for k in numeric},
                                     "per_step": per})
                panels.append((f"{mode} | rank {rank} | {selected}", step, gpu, marks))
                origins.append(origin)
                span = max(span, step["ts"] + step["dur"] - origin)
            selection.append({"impl": impl, "mode": mode, "step": selected,
                              "rank1_start_minus_rank0_ms": (chosen[1]["ts"] - chosen[0]["ts"]) / 1000})
        svg_panels(panels, a.run_dir / f"timeline_{impl}.svg", origins_us=origins, span_us=span, width=1400)

    for folder in sorted(a.run_dir.glob("timing_r*")):
        if not folder.is_dir():
            continue
        b0, b1 = [json.loads((folder / f"boundaries_rank{r}.json").read_text()) for r in (0, 1)]
        # Same Linux host, perf_counter uses the common CLOCK_MONOTONIC clock.
        offsets = [(v["core_start"] - u["core_start"]) * 1000 for u, v in zip(b0[10:], b1[10:])]
        skew_rows.append({"tag": folder.name, "median_abs_core_start_offset_ms": statistics.median(map(abs, offsets)),
                          "max_abs_core_start_offset_ms": max(map(abs, offsets)), "signed_offsets_ms": offsets})
    (a.run_dir / "trace_analysis.json").write_text(json.dumps({"profile": profile_rows, "selection": selection,
                                                             "unprofiled_rank_offsets": skew_rows}, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != "per_step"} for r in profile_rows], indent=2))


if __name__ == "__main__":
    main()
