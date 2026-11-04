"""Day 06 follow-up - analyzer for run_comm_sync.py. Pure python + json (torch only for the correctness section).

  python experiments/day06_comm_sync/analyze_comm_sync.py --root outputs/day06_comm_sync/<run-id>

Writes into --root:
  summary.json   everything below, machine-readable
  rows.csv       one PAIRED row per (job, step, operation): both ranks' values side by side, never aggregated apart
  tables.md      the comparison tables used by report.md

Pairing rules. Host timestamps are time.perf_counter_ns() (CLOCK_MONOTONIC, one clock for all processes on the host),
so rank 0 and rank 1 values of the same step are directly comparable. Trace kernels are paired by construction: the
CPU range "## s<step> TARGET <op>" on the main thread contains exactly one cuLaunchKernel(Ex)/cudaLaunchKernel, whose
correlation id names exactly one NCCL kernel. Traces are compared across ranks only if both files carry the same
baseTimeNanoseconds; otherwise kernel gaps are reported as uncomparable. Any (step, op) whose launch cannot be found
this way is reported as unmapped, not guessed.

Statistics. Per job: median over its steps (unprofiled: steps 4..11; profiled: the recorded steps 6..8). Per
configuration: median [min-max] over the repeat jobs' medians. "arrival gap" = rank 1's dispatch time minus rank 0's
(positive: rank 1 arrived later); "|gap|" its absolute value. "step incl." = per step, max over ranks of the whole
iteration (it_begin -> it_end), INCLUDING every added barrier, the injected delay and prof.step().
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

MS = 1e-6  # ns -> ms
TIMED_FROM = 4
OPS = {"reduce": ("red_disp0", "red_disp1", "red_done", "red_bar"), "gather": ("gat_disp0", "gat_disp1", "gat_done",
                                                                               "gat_bar")}


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def rng(v):
    v = [x for x in v if x is not None]
    return (statistics.median(v), min(v), max(v)) if v else None


def fmt(t, nd=2):
    if t is None:
        return "—"
    if isinstance(t, tuple):
        m, lo, hi = t
        return f"{m:.{nd}f} [{lo:.{nd}f}–{hi:.{nd}f}]" if hi - lo > 10 ** -nd / 2 else f"{m:.{nd}f}"
    return f"{t:.{nd}f}"


def load_rows(job: Path) -> tuple[list[dict], list[dict], dict] | None:
    files = [job / f"rows_rank{r}.jsonl" for r in (0, 1)]
    if not all(f.exists() for f in files):
        return None
    r0, r1 = ([json.loads(x) for x in f.read_text().splitlines()] for f in files)
    meta = json.loads((job / "meta_rank0.json").read_text())
    return r0, r1, meta


# ---------------------------------------------------------------- traces
def trace_kernels(path: Path) -> tuple[int | None, dict]:
    """{(step, op): (ts_us, dur_us, name)} for the NCCL kernel launched inside each TARGET range."""
    t = json.loads(path.read_text())
    ev = t["traceEvents"]
    kernels = {e["args"]["correlation"]: e for e in ev
               if e.get("cat") == "kernel" and e.get("name", "").startswith("nccl")}
    launches = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "LaunchKernel" in e.get("name", "") and e.get("args", {}).get("correlation") in kernels]
    out, unmapped = {}, []
    for a in ev:
        mt = re.match(r"## s(\d+) TARGET (\w+)", a.get("name", ""))
        if a.get("cat") != "user_annotation" or not mt:
            continue
        step, op = int(mt.group(1)), ("gather" if mt.group(2) == "all_gather" else "reduce")
        inside = [x for x in launches if x["tid"] == a["tid"] and a["ts"] <= x["ts"] <= a["ts"] + a["dur"]]
        if len(inside) != 1:
            unmapped.append((step, op, len(inside)))
            continue
        k = kernels[inside[0]["args"]["correlation"]]
        out[(step, op)] = (k["ts"], k["dur"], k["name"].split("(")[0])
    extra = len(kernels) - len(out)
    return t.get("baseTimeNanoseconds"), {"kernels": out, "unmapped": unmapped, "unpaired_nccl_kernels": extra}


# ---------------------------------------------------------------- per job
def job_records(job: Path) -> dict | None:
    loaded = load_rows(job)
    if loaded is None:
        return None
    r0, r1, meta = loaded
    a = meta["args"]
    transport = "nop2p" if meta["env"].get("NCCL_P2P_DISABLE") == "1" else "p2p"
    profiled = bool(a["profile"])
    rec_steps = meta["profile_schedule"]["recorded_steps"] if profiled else None
    tk = None
    if profiled and all((job / f"trace_rank{r}.json").exists() for r in (0, 1)):
        b0, k0 = trace_kernels(job / "trace_rank0.json")
        b1, k1 = trace_kernels(job / "trace_rank1.json")
        tk = {"base_equal": b0 is not None and b0 == b1, "base": [b0, b1], "rank0": k0, "rank1": k1}
    rows = []
    for x0, x1 in zip(r0, r1):
        s = x0["step"]
        assert x1["step"] == s
        base = {"job": job.name, "impl": a["impl"], "align": a["align"], "transport": transport, "profiled": profiled,
                "inject_ms": a["inject_delay_ms"], "step": s, "timed": s >= TIMED_FROM,
                "recorded": bool(rec_steps and s in rec_steps),
                "step_incl_ms": max(x["it_end"] - x["it_begin"] for x in (x0, x1)) * MS,
                "fwd_bwd_ms_r0": (x0["t1"] - x0["t0"]) * MS, "fwd_bwd_ms_r1": (x1["t1"] - x1["t0"]) * MS,
                "grads_ready_gap_ms": (x1["t1"] - x0["t1"]) * MS,
                "prev_bar_wait_ms_r0": (x0["prev_bar_out"] - x0["prev_bar_in"]) * MS if "prev_bar_in" in x0 else None,
                "prev_bar_wait_ms_r1": (x1["prev_bar_out"] - x1["prev_bar_in"]) * MS if "prev_bar_in" in x1 else None,
                "prev_bar_exit_gap_ms": (x1["prev_bar_out"] - x0["prev_bar_out"]) * MS if "prev_bar_in" in x0 else None,
                "it_begin_gap_ms": (x1["it_begin"] - x0["it_begin"]) * MS,
                "t0_gap_ms": (x1["t0"] - x0["t0"]) * MS,
                "optim_ms_r0": (x0["opt_done"] - x0["red_done"]) * MS, "optim_ms_r1": (x1["opt_done"] - x1["red_done"]) * MS,
                "sched_ms_r0": (x0["sched_done"] - x0.get("gat_done", x0["opt_done"])) * MS,
                "sched_ms_r1": (x1["sched_done"] - x1.get("gat_done", x1["opt_done"])) * MS,
                "prof_step_ms_r0": (x0["prof_step1"] - x0["prof_step0"]) * MS if "prof_step0" in x0 else None,
                "prof_step_ms_r1": (x1["prof_step1"] - x1["prof_step0"]) * MS if "prof_step0" in x1 else None,
                "cpu_r0": [x0["cpu_begin"], x0["cpu_pre_reduce"]], "cpu_r1": [x1["cpu_begin"], x1["cpu_pre_reduce"]]}
        for op, (d0, d1, done, bar) in OPS.items():
            if d0 not in x0:
                continue
            row = {**base, "op": op,
                   "arrival_gap_ms": (x1[d0] - x0[d0]) * MS,
                   "host_phase_ms_r0": (x0[done] - x0[d0]) * MS, "host_phase_ms_r1": (x1[done] - x1[d0]) * MS,
                   "dispatch_ms_r0": (x0[d1] - x0[d0]) * MS, "dispatch_ms_r1": (x1[d1] - x1[d0]) * MS,
                   "bar_wait_ms_r0": (x0[bar + "_out"] - x0[bar + "_in"]) * MS if bar + "_in" in x0 else None,
                   "bar_wait_ms_r1": (x1[bar + "_out"] - x1[bar + "_in"]) * MS if bar + "_in" in x1 else None,
                   "bar_exit_gap_ms": (x1[bar + "_out"] - x0[bar + "_out"]) * MS if bar + "_in" in x0 else None,
                   "done_gap_ms": (x1[done] - x0[done]) * MS}
            k = None
            if tk and row["recorded"]:
                ka, kb = tk["rank0"]["kernels"].get((s, op)), tk["rank1"]["kernels"].get((s, op))
                if ka and kb:
                    k = {"kernel": ka[2], "kdur_ms_r0": ka[1] / 1e3, "kdur_ms_r1": kb[1] / 1e3,
                         "kstart_gap_ms": (kb[0] - ka[0]) / 1e3 if tk["base_equal"] else None,
                         "kend_gap_us": ((kb[0] + kb[1]) - (ka[0] + ka[1])) if tk["base_equal"] else None}
            row.update(k or {"kernel": None, "kdur_ms_r0": None, "kdur_ms_r1": None, "kstart_gap_ms": None,
                             "kend_gap_us": None})
            rows.append(row)
    s0 = [x for x in r0 if x["step"] >= TIMED_FROM]
    s1 = [x for x in r1 if x["step"] >= TIMED_FROM]
    loop = max((x[-1]["it_end"] - x[0]["it_begin"]) * MS / len(x) for x in (s0, s1))
    return {"job": job.name, "meta": {"impl": a["impl"], "align": a["align"], "transport": transport,
                                      "profiled": profiled, "inject_ms": a["inject_delay_ms"],
                                      "counters": meta["counters"], "in_sync_final": meta["in_sync_final"],
                                      "affinity_r0": [min(meta["affinity"]), max(meta["affinity"])],
                                      "nccl_env": {k: v for k, v in meta["env"].items() if k.startswith("NCCL")}},
            "loop_ms_per_step": loop, "rows": rows,
            "trace": None if tk is None else {"base_equal": tk["base_equal"], "base": tk["base"],
                                              "unmapped": [tk["rank0"]["unmapped"], tk["rank1"]["unmapped"]],
                                              "unpaired_nccl_kernels": [tk["rank0"]["unpaired_nccl_kernels"],
                                                                        tk["rank1"]["unpaired_nccl_kernels"]]}}


METRICS = ["arrival_gap_ms", "abs_gap_ms", "bar_wait_ms_r0", "bar_wait_ms_r1", "bar_exit_gap_ms", "host_phase_ms_r0",
           "host_phase_ms_r1", "kdur_ms_r0", "kdur_ms_r1", "kstart_gap_ms", "kend_gap_us", "step_incl_ms"]


def job_summary(rec: dict, op: str, steps: str) -> dict:
    sel = [r for r in rec["rows"] if r["op"] == op and (r["recorded"] if steps == "recorded" else r["timed"])]
    for r in sel:
        r["abs_gap_ms"] = abs(r["arrival_gap_ms"])
    out = {m: med([r.get(m) for r in sel]) for m in METRICS}
    out["max_abs_gap_ms"] = max((r["abs_gap_ms"] for r in sel), default=None)
    out["n_steps"] = len(sel)
    return out


# ---------------------------------------------------------------- correctness
def correctness(root: Path) -> dict:
    try:
        import torch
    except ImportError:
        return {"skipped": "torch not importable here"}
    jd = root / "jobs"
    res = {}

    def load(p):
        return torch.load(p, map_location="cpu", weights_only=True)

    for impl, aligns in (("zero1", ["none", "prev_step", "pre_collective"]), ("flat_ar", ["pre_collective"])):
        ref_p = jd / f"tiny_ref_train_zero1_{impl}" / "result.pt"
        if not ref_p.exists():
            res[impl] = {"missing": str(ref_p)}
            continue
        ref = load(ref_p)
        for a in aligns:
            job = jd / f"tiny_{impl}_{a}"
            if not (job / "result.pt").exists():
                res[f"{impl}_{a}"] = {"missing": str(job)}
                continue
            d = load(job / "result.pt")
            meta = json.loads((job / "meta_rank0.json").read_text())
            res[f"{impl}_{a}"] = {
                "final_params_bitwise": all(torch.equal(d["final"][n], ref["final"][n]) for n in ref["final"]),
                "adamw_state_bitwise": all(torch.equal(d["state_final"][n][k], ref["state_final"][n][k])
                                           for n in ref["state_final"] for k in ("exp_avg", "exp_avg_sq")),
                "adam_step_equal": d["adam_step"] == ref["meta"]["adam_step"][0],
                "replicas_every_step": all(ok for _, ok in meta["in_sync_steps"]) and meta["in_sync_final"],
                "reference_replicas": ref["meta"]["in_sync_final"],
                "counters": meta["counters"], "reference_counters": ref["meta"]["counters"]}
    res["all_pass"] = all(v.get("final_params_bitwise") and v.get("adamw_state_bitwise") and v.get("adam_step_equal")
                          and v.get("replicas_every_step") for k, v in res.items() if isinstance(v, dict))
    return res


# ---------------------------------------------------------------- communication-only benchmark and route logs
def comm_bench(root: Path) -> dict:
    out = {}
    for t in ("p2p", "nop2p"):
        d = root / "comm" / t
        if not (d / "bench_rank0.jsonl").exists():
            continue
        r0, r1 = ([json.loads(x) for x in (d / f"bench_rank{r}.jsonl").read_text().splitlines()] for r in (0, 1))
        per = {}
        for a, b in zip(r0, r1):
            assert a["op"] == b["op"] and a["iter"] == b["iter"]
            if a["warmup"]:
                continue
            p = per.setdefault(a["op"], {k: [] for k in ("host_ms_r0", "host_ms_r1", "event_ms_r0", "event_ms_r1",
                                                         "arrival_gap_ms", "bar_exit_gap_ms", "valid")})
            p["host_ms_r0"].append((a["done"] - a["disp0"]) * MS)
            p["host_ms_r1"].append((b["done"] - b["disp0"]) * MS)
            p["event_ms_r0"].append(a["event_ms"])
            p["event_ms_r1"].append(b["event_ms"])
            p["arrival_gap_ms"].append((b["disp0"] - a["disp0"]) * MS)
            p["bar_exit_gap_ms"].append((b["bar_out"] - a["bar_out"]) * MS)
            p["valid"].append(a["valid"] and b["valid"])
        out[t] = {op: {**{k: rng(v) for k, v in p.items() if k != "valid"}, "n": len(p["valid"]),
                       "all_valid": all(p["valid"])} for op, p in per.items()}
        h = lambda op: max(out[t][op]["host_ms_r0"][0], out[t][op]["host_ms_r1"][0])  # noqa: E731
        if all(op in out[t] for op in ("all_reduce", "reduce_scatter", "all_gather", "rs_ag_pair")):
            out[t]["derived"] = {"rs_plus_ag_ms": h("reduce_scatter") + h("all_gather"), "ar_ms": h("all_reduce"),
                                 "pair_ms": h("rs_ag_pair"),
                                 "rs_plus_ag_minus_ar_ms": h("reduce_scatter") + h("all_gather") - h("all_reduce"),
                                 "pair_minus_ar_ms": h("rs_ag_pair") - h("all_reduce"),
                                 "note": "max over ranks of the per-rank median host duration"}
    return out


def route_logs(root: Path) -> dict:
    out = {}
    pats = {"via": re.compile(r"Channel \d+(?:/\d+)? : \d+\[\d+\] -> \d+\[\d+\] (?:\[\w+\] )?via (\S+)"),
            "coll": re.compile(r"(AllReduce|ReduceScatter|AllGather): opCount \w+.*"),
            "algo": re.compile(r"((?:AllReduce|ReduceScatter|AllGather): \d+ Bytes -> Algo \w+ proto \w+ \S+)"),
            "channels": re.compile(r"(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p "
                                   r"channels, (\d+) p2p channels per peer"),
            "shm": re.compile(r".*SHM.*"), "p2p_disable": re.compile(r".*NCCL_P2P_DISABLE.*")}
    for t in ("p2p", "nop2p"):
        files = sorted((root / "route" / t).glob("nccl.*.log"))
        if not files:
            continue
        text = "\n".join(f.read_text(errors="replace") for f in files)
        out[t] = {"files": [f.name for f in files],
                  "via": sorted(set(pats["via"].findall(text))),
                  "channels": sorted(set(pats["channels"].findall(text))),
                  "coll_lines": [m.group(0)[:240] for m in pats["coll"].finditer(text)][:12],
                  "algo_proto_lines": sorted(set(m.group(1)[:200] for m in pats["algo"].finditer(text)))[:12],
                  "p2p_disable_lines": sorted(set(m.group(0)[-160:] for m in pats["p2p_disable"].finditer(text)))[:4],
                  "version_lines": sorted(set(re.findall(r"NCCL version [^\n]*", text)))[:2]}
    return out


# ---------------------------------------------------------------- tables
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root)
    recs = [r for r in (job_records(j) for j in sorted((root / "jobs").iterdir()) if j.is_dir()) if r]
    with (root / "rows.csv").open("w", newline="") as f:
        cols = sorted({k for r in recs for row in r["rows"] for k in row})
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            for row in r["rows"]:
                w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})

    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        m = r["meta"]
        kind = "injected" if m["inject_ms"] else ("profiled" if m["profiled"] else "timing")
        if r["job"].startswith("tiny"):
            continue
        groups.setdefault((kind, m["impl"], m["transport"], m["align"], m["profiled"]), []).append(r)

    table, md = [], []
    for (kind, impl, transport, align, profiled), rs in sorted(groups.items()):
        for op in ("reduce", "gather"):
            if op == "gather" and impl != "zero1":
                continue
            for steps in (["recorded", "timed"] if profiled else ["timed", "steps6_8"]):
                sums = []
                for r in rs:
                    if steps == "steps6_8":  # unprofiled jobs on the profiled jobs' step indices (perturbation check)
                        for row in r["rows"]:
                            row["recorded_unprof"] = 6 <= row["step"] <= 8
                        sel = {"rows": [dict(x, recorded=x["recorded_unprof"]) for x in r["rows"]]}
                        sums.append(job_summary(sel, op, "recorded"))
                    else:
                        sums.append(job_summary(r, op, steps))
                row = {"kind": kind, "impl": impl, "transport": transport, "align": align, "profiled": profiled,
                       "op": op, "steps": steps, "jobs": len(rs),
                       **{m: rng([s[m] for s in sums]) for m in METRICS + ["max_abs_gap_ms"]},
                       "loop_ms_per_step": rng([r["loop_ms_per_step"] for r in rs]) if steps == "timed" else None,
                       "n_steps_per_job": sums[0]["n_steps"]}
                table.append(row)

    def mdtable(rows, title, note):
        md.append(f"\n### {title}\n\n{note}\n")
        md.append("| kind | impl | transport | align | op | steps | jobs | arrival gap r1−r0 | max abs gap | barrier wait "
                  "r0 / r1 | barrier exit gap | host phase r0 / r1 | NCCL kernel r0 / r1 | kernel start gap | "
                  "kernel end gap µs | step incl. | loop ms/step |")
        md.append("|" + " --- |" * 17)
        for r in rows:
            md.append(f"| {r['kind']} | {r['impl']} | {r['transport']} | {r['align']} | {r['op']} | {r['steps']} | "
                      f"{r['jobs']} | {fmt(r['arrival_gap_ms'])} | {fmt(r['max_abs_gap_ms'])} | "
                      f"{fmt(r['bar_wait_ms_r0'])} / {fmt(r['bar_wait_ms_r1'])} | {fmt(r['bar_exit_gap_ms'], 3)} | "
                      f"{fmt(r['host_phase_ms_r0'])} / {fmt(r['host_phase_ms_r1'])} | "
                      f"{fmt(r['kdur_ms_r0'], 3)} / {fmt(r['kdur_ms_r1'], 3)} | {fmt(r['kstart_gap_ms'], 3)} | "
                      f"{fmt(r['kend_gap_us'], 1)} | {fmt(r['step_incl_ms'])} | {fmt(r['loop_ms_per_step'])} |")

    md.append("# Comparison tables (generated by analyze_comm_sync.py)\n\nms unless marked. Per job: median over steps; "
              "then median [min–max] over jobs (a single value when all jobs agree to 0.01). '—': not available "
              "(no trace in unprofiled jobs, no barrier in that mode, or not applicable).")
    mdtable([r for r in table if r["kind"] == "timing" and r["steps"] == "timed"], "Unprofiled timing (steps 4–11)",
            "3 repeat jobs per row, configuration order rotated.")
    mdtable([r for r in table if r["kind"] == "profiled" and r["steps"] == "recorded"],
            "Profiled paired traces (recorded steps 6–8)", "1 job per row; kernel columns from both ranks' traces.")
    mdtable([r for r in table if r["kind"] == "timing" and r["steps"] == "steps6_8"],
            "Unprofiled jobs on steps 6–8 (same indices as the traces: profiler perturbation)", "")
    mdtable([r for r in table if r["kind"] == "injected"], "Injected-delay control (rank 1 sleeps before the "
            "reduction; not part of any performance median)", "zero1, normal P2P; 1 unprofiled + 1 profiled job.")

    # profiler transitions: prof.step() host time per step index
    trans = {}
    for r in recs:
        if r["meta"]["profiled"]:
            for row in r["rows"]:
                if row["op"] == "reduce":
                    trans.setdefault(row["step"], []).append(max(row["prof_step_ms_r0"], row["prof_step_ms_r1"]))
    trans = {s: rng(v) for s, v in sorted(trans.items())}
    md.append("\n### prof.step() host duration by step (max over ranks; median [min–max] over profiled jobs)\n")
    md.append("| step | " + " | ".join(str(s) for s in trans) + " |")
    md.append("|" + " --- |" * (len(trans) + 1))
    md.append("| ms | " + " | ".join(fmt(v) for v in trans.values()) + " |")

    cb = comm_bench(root)
    md.append("\n### Communication only (no model; 104.6875 MiB FP32; barrier-aligned; 20 timed iterations)\n")
    md.append("| transport | op | host r0 / r1 | event r0 / r1 | arrival gap | barrier exit gap | valid |")
    md.append("|" + " --- |" * 7)
    for t, ops in cb.items():
        for op, v in ops.items():
            if op == "derived":
                continue
            md.append(f"| {t} | {op} | {fmt(v['host_ms_r0'])} / {fmt(v['host_ms_r1'])} | {fmt(v['event_ms_r0'])} / "
                      f"{fmt(v['event_ms_r1'])} | {fmt(v['arrival_gap_ms'], 3)} | {fmt(v['bar_exit_gap_ms'], 3)} | "
                      f"{v['all_valid']} ({v['n']}) |")
    for t, ops in cb.items():
        if "derived" in ops:
            d = ops["derived"]
            md.append(f"\n{t}: RS + AG {d['rs_plus_ag_ms']:.2f} vs AR {d['ar_ms']:.2f} ms (difference "
                      f"{d['rs_plus_ag_minus_ar_ms']:+.2f}); RS→AG pair {d['pair_ms']:.2f} ms ({d['pair_minus_ar_ms']:+.2f}"
                      " vs AR). Median of [min–max] above: per-rank median host duration, max over ranks.")

    corr = correctness(root)
    routes = route_logs(root)
    traces = {r["job"]: r["trace"] for r in recs if r["trace"]}
    md.append("\n### Correctness (tiny, 12 steps) and trace integrity\n")
    md.append("```\n" + json.dumps(corr, indent=1)[:4000] + "\n```")
    md.append("trace clock bases equal in every profiled job: "
              f"{all(t['base_equal'] for t in traces.values())}; unmapped TARGET ranges: "
              f"{sum(len(u) for t in traces.values() for u in t['unmapped'])}; NCCL kernels outside TARGET ranges: "
              f"{sum(sum(t['unpaired_nccl_kernels']) for t in traces.values())}")
    (root / "tables.md").write_text("\n".join(md) + "\n")
    (root / "summary.json").write_text(json.dumps({
        "table": table, "prof_step_ms_by_step": trans, "comm_bench": cb, "correctness": corr, "routes": routes,
        "traces": traces, "jobs": {r["job"]: {"meta": r["meta"], "loop_ms_per_step": r["loop_ms_per_step"]}
                                   for r in recs}}, indent=1, default=str))
    print((root / "tables.md").read_text())
    print(json.dumps(routes, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
