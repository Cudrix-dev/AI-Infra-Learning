"""Day 03 - read torch.profiler chrome traces and measure how much gradient communication was hidden.

For every profiled step (the ProfilerStep#N ranges) and every trace given:
  gpu_compute_ms   union of the time intervals of all non-NCCL GPU activity (kernels, memcpy, memset)
  nccl_ms          union of the NCCL kernel intervals
  overlapped_ms    time when an NCCL kernel AND some other GPU work were running at the same time
  exposed_nccl_ms  nccl_ms - overlapped_ms: communication nothing else was hiding
  n_nccl           number of NCCL kernels (78 for per-parameter on medium, a handful for buckets)
Then the median over the profiled steps. Optionally one SVG timeline per trace (the median-length step).

Read these numbers with two caveats (Guide, Block 4):
  * an NCCL kernel's duration includes time spent WAITING for the peer rank, so it is not transfer time
  * "overlapped" means "running concurrently", not "free": NCCL kernels occupy SMs and memory bandwidth, which can
    slow the compute kernels next to them. Compare the core step time against --impl none to see the net effect.

Pure Python, no torch needed: works on your laptop with downloaded traces too.
  python experiments/day03_ddp_hooks/analyze_trace.py outputs/day03_ddp_hooks/prof_mini_ddp_*/trace_rank0.json \
      --labels mini --svg day03/assets/timeline.svg
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load_events(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    ev = json.loads(path.read_text())
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    xs = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    # Each ProfilerStep#N and record_function range appears twice in torch 2.14 traces: as a CPU range
    # (cat "user_annotation") and as its GPU projection (cat "gpu_user_annotation"). Step windows = the CPU ranges.
    steps = sorted((e for e in xs if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("ProfilerStep#")),
                   key=lambda e: e["ts"])
    gpu = [e for e in xs if e.get("cat") in GPU_CATS]
    marks = [e for e in xs if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("## ")]
    return steps, gpu, marks


def is_nccl(e: dict) -> bool:
    return "nccl" in e["name"].lower()


def union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def length(iv: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in iv)


def intersect(u: list[tuple[float, float]], v: list[tuple[float, float]]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(u) and j < len(v):
        a, b = max(u[i][0], v[j][0]), min(u[i][1], v[j][1])
        tot += max(0.0, b - a)
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return tot


def step_metrics(step: dict, gpu: list[dict]) -> dict:
    t0, t1 = step["ts"], step["ts"] + step["dur"]
    inside = [e for e in gpu if t0 <= e["ts"] < t1]
    comp = union([(e["ts"], e["ts"] + e["dur"]) for e in inside if not is_nccl(e)])
    comm = union([(e["ts"], e["ts"] + e["dur"]) for e in inside if is_nccl(e)])
    ov = intersect(comp, comm)
    return {"step_ms": step["dur"] / 1e3, "gpu_compute_ms": length(comp) / 1e3, "nccl_ms": length(comm) / 1e3,
            "overlapped_ms": ov / 1e3, "exposed_nccl_ms": (length(comm) - ov) / 1e3,
            "n_nccl": sum(1 for e in inside if is_nccl(e)), "n_gpu_ops": len(inside)}


def summarize(path: Path) -> tuple[dict, dict, list[dict], list[dict]]:
    steps, gpu, marks = load_events(path)
    if not steps:
        raise SystemExit(f"{path}: no ProfilerStep# ranges (was the trace written by train_ddp.py --profile?)")
    per = [step_metrics(s, gpu) for s in steps]
    med = {k: statistics.median(m[k] for m in per) for k in per[0]}
    med["n_steps"] = len(per)
    pick = sorted(range(len(per)), key=lambda k: per[k]["step_ms"])[len(per) // 2]  # median-length step for the SVG
    return med, steps[pick], gpu, marks


# ----------------------------------------------------------------------------- SVG
COL = {"compute": "#4e79a7", "nccl": "#f28e2b", "copy": "#9c9c9c"}
MARK_COL = {"## forward": "#b6d7a8", "## backward": "#a4c2f4", "## finish_grad_sync": "#f9cb9c", "## optimizer": "#d5a6bd"}


def svg_panels(panels: list[tuple[str, dict, list[dict], list[dict]]], out: Path, width: int = 1100,
               span_us: float | None = None, origins_us: list[float] | None = None) -> None:
    left, right, row_h, gap = 150, 20, 18, 4
    # A shared span lets separate rank figures use the same pixels-per-millisecond scale.
    span = max(s["dur"] for _, s, _, _ in panels) if span_us is None else span_us
    sx = (width - left - right) / span  # px per microsecond, SAME for every panel so lengths compare
    parts, y = [], 10
    if origins_us is not None and len(origins_us) != len(panels):
        raise ValueError("Expected one plot origin per panel")
    for panel_index, (label, step, gpu, marks) in enumerate(panels):
        t0, t1 = step["ts"], step["ts"] + step["dur"]
        origin = t0 if origins_us is None else origins_us[panel_index]
        inside = [e for e in gpu if t0 <= e["ts"] < t1]
        streams = sorted({(e.get("args", {}).get("device", 0), e.get("args", {}).get("stream", e.get("tid")))
                          for e in inside}, key=lambda s: (str(s[0]), str(s[1])))
        key = lambda e: (e.get("args", {}).get("device", 0), e.get("args", {}).get("stream", e.get("tid")))  # noqa: E731
        nccl_streams = {key(e) for e in inside if is_nccl(e)}
        comp_streams = {key(e) for e in inside if not is_nccl(e)}
        parts.append(f'<text x="4" y="{y + 12}" font-size="13" font-weight="bold">{label}  '
                     f'(step {step["dur"] / 1e3:.1f} ms)</text>')
        y += 18
        cpu = [m for m in marks if t0 <= m["ts"] < t1 and m.get("tid") == step.get("tid")]
        parts.append(f'<text x="4" y="{y + 13}" font-size="11">CPU ranges</text>')
        for m in cpu:
            x, w = left + (m["ts"] - origin) * sx, max(m["dur"] * sx, 0.5)
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h}" fill="{MARK_COL.get(m["name"], "#ddd")}">'
                         f'<title>{m["name"][3:]} {m["dur"] / 1e3:.2f} ms</title></rect>')
            if w > 60:
                parts.append(f'<text x="{x + 3:.1f}" y="{y + 13}" font-size="10">{m["name"][3:]}</text>')
        y += row_h + gap
        for st in streams:
            # a synchronous all_reduce (async_op=False, flat_sync) runs its NCCL kernel on the CURRENT stream
            kind = "+".join(k for k, s_ in (("compute", comp_streams), ("NCCL", nccl_streams)) if st in s_)
            parts.append(f'<text x="4" y="{y + 13}" font-size="11">stream {st[1]} ({kind})</text>')
            for e in inside:
                if (e.get("args", {}).get("device", 0), e.get("args", {}).get("stream", e.get("tid"))) != st:
                    continue
                c = COL["nccl"] if is_nccl(e) else (COL["compute"] if e.get("cat") == "kernel" else COL["copy"])
                x, w = left + (e["ts"] - origin) * sx, max(e["dur"] * sx, 0.6)
                name = e["name"][:80].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # cut, then escape
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h}" fill="{c}">'
                             f'<title>{name} {e["dur"]:.0f} us</title></rect>')
            y += row_h + gap
        y += 14
    ticks = []
    step_ms = 5 if span / 1e3 > 20 else 1
    for k in range(0, int(span / 1e3) + 1, step_ms):
        x = left + k * 1e3 * sx
        ticks.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{y}" stroke="#e0e0e0"/>'
                     f'<text x="{x + 2:.1f}" y="{y + 12}" font-size="10" fill="#666">{k} ms</text>')
    legend = (f'<rect x="{left}" y="{y + 20}" width="12" height="12" fill="{COL["compute"]}"/>'
              f'<text x="{left + 16}" y="{y + 30}" font-size="11">compute kernel</text>'
              f'<rect x="{left + 120}" y="{y + 20}" width="12" height="12" fill="{COL["nccl"]}"/>'
              f'<text x="{left + 136}" y="{y + 30}" font-size="11">NCCL kernel (includes waiting for the peer)</text>'
              f'<rect x="{left + 420}" y="{y + 20}" width="12" height="12" fill="{COL["copy"]}"/>'
              f'<text x="{left + 436}" y="{y + 30}" font-size="11">memcpy / memset</text>')
    h = y + 44
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}" '
                   f'font-family="sans-serif"><rect width="100%" height="100%" fill="white"/>'
                   + "".join(ticks) + "".join(parts) + legend + "</svg>")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--svg", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    args = ap.parse_args()
    labels = args.labels or [p.parent.name for p in args.traces]
    if len(labels) != len(args.traces):
        raise SystemExit("--labels must have one entry per trace")
    rows, panels = [], []
    print(f"{'trace':28s} {'step ms':>8s} {'gpu compute':>11s} {'nccl':>7s} {'overlapped':>10s} {'exposed':>8s} "
          f"{'#nccl':>5s}   (medians over profiled steps, ms)")
    for label, p in zip(labels, args.traces):
        med, step, gpu, marks = summarize(p)
        rows.append({"label": label, "trace": str(p), **med})
        panels.append((label, step, gpu, marks))
        print(f"{label:28s} {med['step_ms']:8.2f} {med['gpu_compute_ms']:11.2f} {med['nccl_ms']:7.2f} "
              f"{med['overlapped_ms']:10.2f} {med['exposed_nccl_ms']:8.2f} {med['n_nccl']:5.0f}")
    print("profiled runs are slower than unprofiled ones: compare these numbers with each other, not with timing runs")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
    if args.svg:
        svg_panels(panels, args.svg)


if __name__ == "__main__":
    main()
