"""Day 13 follow-up - join stall_probe.py's stalls with host_sampler.py's samples (GUIDE Appendix B).

For every F/B call that took more than median + --stall-ms on its rank, and for a same-sized random sample of
normal calls, it reports what the rank's main thread did during the call: fraction of samples in state R
(on CPU or runnable), S (sleeping), D; its CPU time per wall ms; involuntary context switches (preempted by
another task); page faults; and container-wide cgroup throttling and PSI stall time. It also joins GPU clocks and
throttle reasons from an `nvidia-smi -lms` CSV when given.

Usage: python stall_correlate.py <probe-run-dir> <host.jsonl> [--smi smi.csv] [--stall-ms 10]
"""

import argparse
import datetime as dt
import json
import random
import statistics
from collections import Counter
from pathlib import Path

NS = 1e-6


def load_samples(path):
    lines = Path(path).read_text().splitlines()
    head = json.loads(lines[0])
    return head, [json.loads(x) for x in lines[1:]]


def window(samples, t0, t1):
    """Samples inside [t0, t1] plus the one just before and just after (for counter deltas)."""
    inside = [s for s in samples if t0 <= s["t"] <= t1]
    before = [s for s in samples if s["t"] < t0][-1:]
    after = [s for s in samples if s["t"] > t1][:1]
    return before + inside + after, inside


def describe(samples, rank, t0, t1):
    edge, inside = window(samples, t0, t1)
    if len(edge) < 2:
        return None
    a, b = edge[0], edge[-1]
    ra, rb = a["ranks"].get(str(rank)), b["ranks"].get(str(rank))
    if ra is None or rb is None:
        return None
    span_ms = (b["t"] - a["t"]) * NS
    states = Counter(s["ranks"][str(rank)]["state"] for s in inside if s["ranks"].get(str(rank)))
    wchan = Counter(s["ranks"][str(rank)]["wchan"] for s in inside if s["ranks"].get(str(rank)))
    n = max(1, sum(states.values()))
    others = Counter()   # (comm, state) of other threads seen busy inside the window, with cpu ticks
    for s in inside:
        for comm, st, d in (s["ranks"].get(str(rank)) or {}).get("others", {}).values():
            others[f"{comm}:{st}"] += max(d, 1)
    return {
        "wall_ms": round((t1 - t0) * NS, 1), "n_samples": len(inside),
        "R": round(states.get("R", 0) / n, 2), "S": round(states.get("S", 0) / n, 2), "D": round(states.get("D", 0) / n, 2),
        "cpu_per_ms": round((rb["utime_ms"] + rb["stime_ms"] - ra["utime_ms"] - ra["stime_ms"]) / span_ms, 2),
        "sys_per_ms": round((rb["stime_ms"] - ra["stime_ms"]) / span_ms, 2),
        "nvcs": rb["nvcs"] - ra["nvcs"], "vcs": rb["vcs"] - ra["vcs"],
        "minflt": rb["minflt"] - ra["minflt"], "majflt": rb["majflt"] - ra["majflt"],
        "cg_throttled_us": b["cg"]["throttled_us"] - a["cg"]["throttled_us"],
        "cg_cpu_per_ms": round((b["cg"]["usage_us"] - a["cg"]["usage_us"]) / 1e3 / span_ms, 2),
        "psi_cpu_some_us": b["psi"]["cpu"].get("some", 0) - a["psi"]["cpu"].get("some", 0),
        "psi_mem_some_us": b["psi"]["memory"].get("some", 0) - a["psi"]["memory"].get("some", 0),
        "psi_io_some_us": b["psi"]["io"].get("some", 0) - a["psi"]["io"].get("some", 0),
        "wchan": dict(wchan.most_common(3)), "others": dict(others.most_common(6)),
    }


def load_smi(path, head):
    """nvidia-smi CSV (timestamp, index, clocks.sm, power, temp, throttle reasons) -> monotonic ns rows."""
    off = head["mono_ns"] - head["wall_ns"]   # monotonic = wall + off
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6:
            continue
        try:
            wall = dt.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
        rows.append({"t": int(wall * 1e9) + off, "gpu": int(f[1]), "sm_mhz": f[2], "power": f[3], "temp": f[4],
                     "throttle": f[5]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("host")
    ap.add_argument("--smi", default="")
    ap.add_argument("--stall-ms", type=float, default=10.0)
    a = ap.parse_args()
    head, samples = load_samples(a.host)
    smi = load_smi(a.smi, head) if a.smi else []
    random.seed(0)
    out = {"period_ms": head["period_ms"], "n_samples": len(samples), "ranks": {}}
    lines = [f"host samples: {len(samples)} at {head['period_ms']} ms"]
    for f in sorted(Path(a.run_dir).glob("rank*.json")):
        rec = json.loads(f.read_text())
        r = rec["rank"]
        ev = rec["events"]
        gcs = [e for e in ev if e["kind"] == "gc"]
        calls = [e for e in ev if e["kind"] in ("F", "B") and e["step"] >= 5]
        med = {k: statistics.median((e["t1"] - e["t0"]) * NS for e in calls if e["kind"] == k) for k in ("F", "B")}
        stalls = [e for e in calls if (e["t1"] - e["t0"]) * NS > med[e["kind"]] + a.stall_ms
                  and not any(g["t0"] < e["t1"] and g["t1"] > e["t0"] for g in gcs)]
        normal = random.sample([e for e in calls if (e["t1"] - e["t0"]) * NS <= med[e["kind"]] + 2],
                               min(200, len(calls)))
        rs = [dict(step=e["step"], name=e["name"], **(describe(samples, r, e["t0"], e["t1"]) or {})) for e in stalls]
        rn = [d for d in (describe(samples, r, e["t0"], e["t1"]) for e in normal) if d]
        # normal calls are short (a few ms): aggregate them instead of listing them
        agg = {k: round(statistics.mean(d[k] for d in rn), 3) for k in ("R", "S", "D", "cpu_per_ms", "sys_per_ms")} if rn else {}
        base_others = Counter()
        for d in rn:
            base_others.update(d["others"])
        agg["others_top"] = dict(base_others.most_common(6))
        agg["cg_cpu_per_ms"] = round(statistics.mean(d["cg_cpu_per_ms"] for d in rn), 2) if rn else None
        agg.update({k + "_per_ms": round(sum(d[k] for d in rn) / max(1e-9, sum(d["wall_ms"] for d in rn)), 3)
                    for k in ("nvcs", "vcs", "minflt")} if rn else {})
        # allocator counters: did any step after warmup call cudaMalloc/cudaFree?
        alloc = [(s["step"], s.get("alloc", {})) for s in rec["steps"]]
        grew = [st for (st, al), (_, prev) in zip(alloc[1:], alloc[:-1])
                if st >= 5 and al and any(al[k] != prev.get(k) for k in al)]
        out["ranks"][r] = {"median_ms": med, "stalls": rs, "normal": agg, "alloc_change_steps": grew}
        lines.append(f"\nrank {r}: medians F {med['F']:.2f} B {med['B']:.2f} ms; {len(stalls)} non-GC stalls; "
                     f"allocator changed at steps {grew[:10]}")
        lines.append(f"  normal calls (mean): {agg}")
        for d in rs:
            lines.append(f"  STALL step {d['step']} {d['name']} {d.get('wall_ms')} ms: R {d.get('R')} S {d.get('S')} "
                         f"D {d.get('D')} cpu/ms {d.get('cpu_per_ms')} sys/ms {d.get('sys_per_ms')} nvcs {d.get('nvcs')} "
                         f"vcs {d.get('vcs')} minflt {d.get('minflt')} majflt {d.get('majflt')} thr {d.get('cg_throttled_us')} "
                         f"cg_cpu/ms {d.get('cg_cpu_per_ms')} psi cpu/mem/io {d.get('psi_cpu_some_us')}/"
                         f"{d.get('psi_mem_some_us')}/{d.get('psi_io_some_us')} wchan {d.get('wchan')}")
            lines.append(f"    other busy threads: {d.get('others')}")
            e = next(x for x in stalls if x["step"] == d["step"] and x["name"] == d["name"])
            near = [s for s in smi if e["t0"] - 60e6 <= s["t"] <= e["t1"] + 60e6]
            if near:
                lines.append("    smi: " + "; ".join(f"g{s['gpu']} {s['sm_mhz']} {s['power']} {s['throttle']}"
                                                     for s in near[:12]))
    rep = "\n".join(lines)
    print(rep)
    Path(a.run_dir, "correlate.txt").write_text(rep)
    Path(a.run_dir, "correlate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
