"""Day 13 - read the profiled step of train_gpt.py: every NCCL kernel tied to its group, phase and Megatron frame,
the 1F1B order on each rank, and where each rank's compute stream sat idle. Standard library only (runs on Windows).

    python experiments/day13_megatron/analyze_trace.py outputs/day13_megatron/run_<RID>/tp2pp2 \
        --megatron /workspace/Megatron-LM          # optional: resolve call lines in the Megatron source

How a kernel is tied to its source (GUIDE Block 4, "How to read the trace")
--------------------------------------------------------------------------
  GPU kernel  --args.correlation-->  cudaLaunchKernel on a CPU thread (cat cuda_runtime, same correlation)
  launch      --time containment, same thread-->  record_param_comms (collective name, group description and
              global ranks, element count, dtype, tensor dims, peer rank for send/recv)
  launch      --time containment, same thread-->  python_function frames (with_stack=True). Their names are
              "file(DEF line): function": the line is where the function is DEFINED, not the executing line.
              With --megatron, the line where the parent calls the child is found by a text search inside the
              parent's body (labelled "call ~line"; it is a search result, not profiler data).
  launch      --time containment, main thread-->  our ranges "day13/F mb=k" / "day13/B mb=k" (the phase).
              Backward TP all-reduces are launched from autograd's device thread, not the main thread, so the
              phase lookup always uses the MAIN thread's ranges (the main thread is inside backward_step then).

Batched P2P (batch_isend_irecv) launches ONE ncclDevKernel_SendRecv for a send+recv pair; its record_param_comms
events are the c10d::send / c10d::recv_ ops posted inside the same batch_isend_irecv frame.

Idle time: the compute stream is the stream with the most non-NCCL kernels. Inside the step's window, every gap
between its kernels is idle; a gap is attributed to "waiting for P2P" where a SendRecv kernel is running at the
same time (NCCL's kernel spins until the peer arrives), to "TP backward all-reduce" where an all-reduce runs on
another stream, and to "other (CPU launch gaps)" otherwise. [measured on the profiled step; profiling adds CPU
overhead, so absolute ms are inflated - read the order and the attribution, not the ms]
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FRAME_RE = re.compile(r"^(?P<file>.+?)\((?P<line>\d+)\): (?P<func>.+)$")


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)["traceEvents"]


class Intervals:
    """X events of one thread sorted by start; `containing(t)` = the events with ts <= t <= ts + dur."""

    def __init__(self, events):
        self.ev = sorted(events, key=lambda e: e["ts"])
        self.ts = [e["ts"] for e in self.ev]

    def containing(self, t):
        i = bisect.bisect_right(self.ts, t)
        return [e for e in self.ev[:i] if e["ts"] + e.get("dur", 0) >= t]


class SourceIndex:
    """Find where `parent` (a def at a known line) calls `child`, by text search in the parent's body."""

    def __init__(self, root: Path | None):
        self.root, self.cache = root, {}

    def call_line(self, file: str, def_line: int, child: str) -> int | None:
        if self.root is None or not file.startswith("megatron/"):
            return None
        lines = self.cache.get(file)
        if lines is None:
            p = self.root / file
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.is_file() else []
            self.cache[file] = lines
        if not lines or def_line > len(lines):
            return None
        indent = len(lines[def_line - 1]) - len(lines[def_line - 1].lstrip())
        pat = re.compile(rf"(?<![\w]){re.escape(child)}\s*\(")
        for i in range(def_line, len(lines)):
            s = lines[i]
            if s.strip() and len(s) - len(s.lstrip()) <= indent and re.match(r"\s*(def|class|@)", s):
                break   # left the parent's body
            if pat.search(s):
                return i + 1
        return None


def frames_at(py: Intervals, t: float) -> list[dict]:
    out = []
    for e in py.containing(t):
        m = FRAME_RE.match(e["name"])
        if m:
            out.append({"file": m["file"], "def_line": int(m["line"]), "func": m["func"], "ts": e["ts"]})
    return out   # outermost first


WRAPPERS = {"wrapper", "_fn", "decorate_context", "inner", "apply", "_call_impl", "_wrapped_call_impl"}


def describe_stack(frames: list[dict], src: SourceIndex, keep=("megatron/", "train_gpt.py", "my_step.py"),
                   limit=14) -> list[str]:
    """Megatron (and our) frames, innermost first, each with its def line and the line where it calls the next
    named frame (decorator/dispatch wrappers such as c10d's `wrapper` or nn.Module's `_call_impl` are skipped)."""
    rows = []
    for i, f in enumerate(frames):
        if not f["file"].startswith(keep) and not any(k in f["file"] for k in keep):
            continue
        child = next((g["func"] for g in frames[i + 1:] if g["func"] not in WRAPPERS), None)
        if child == "forward" and any(g["func"] == "_call_impl" for g in frames[i + 1:i + 4]):
            child = "self"   # module call: `self.linear_fc2(...)` etc.; not searchable by name
        if child in ("self", None):
            rows.append(f"{f['file']}:{f['def_line']} {f['func']}")
            continue
        call = src.call_line(f["file"], f["def_line"], child.split(".")[-1]) if child else None
        rows.append(f"{f['file']}:{f['def_line']} {f['func']}" + (f"  (calls {child} at ~line {call})" if call else ""))
    return rows[::-1][:limit]


def analyze_rank(path: Path, src: SourceIndex, ev: list[dict] | None = None) -> dict:
    """`ev`: events already loaded (the nsys reader, nsys_trace.events); default: load the PyTorch trace."""
    ev = load(path) if ev is None else ev
    xs = [e for e in ev if e.get("ph") == "X"]
    kernels = [e for e in xs if e.get("cat") == "kernel"]
    # NCCL launches its kernels through the driver API (cuLaunchKernelEx), PyTorch's own through the runtime API.
    runtime = {e["args"]["correlation"]: e for e in xs if e.get("cat") in ("cuda_runtime", "cuda_driver")
               and "correlation" in e.get("args", {})}
    by_tid_py, by_tid_rpc, cpu_ann = defaultdict(list), defaultdict(list), []
    for e in xs:
        if e.get("cat") == "python_function":
            by_tid_py[e["tid"]].append(e)
        elif e.get("name") == "record_param_comms":
            by_tid_rpc[e["tid"]].append(e)
        elif e.get("cat") == "user_annotation" and e["name"].startswith("day13/"):
            cpu_ann.append(e)
    py = {t: Intervals(v) for t, v in by_tid_py.items()}
    rpc = {t: Intervals(v) for t, v in by_tid_rpc.items()}
    step_cpu = next(e for e in cpu_ann if e["name"].startswith("day13/step"))
    main_tid = step_cpu["tid"]
    fb_cpu = sorted((e for e in cpu_ann if not e["name"].startswith("day13/step")), key=lambda e: e["ts"])
    phases = Intervals(fb_cpu)

    def phase_of(t):
        hit = phases.containing(t) if t is not None else []
        return hit[-1]["name"].removeprefix("day13/") if hit else None

    # Every kernel gets the phase of its launch. The GPU-side ranges (gpu_user_annotation) cannot be used for B:
    # backward kernels are launched from autograd's thread, outside the main thread's range, so "B" GPU ranges
    # come out empty. Instead: launch time -> main-thread range -> phase.
    for k in kernels:
        launch = runtime.get(k["args"].get("correlation"))
        k["_launch"] = launch
        k["_phase"] = phase_of(launch["ts"]) if launch else None
    step_kernels = [k for k in kernels if k["_launch"] and step_cpu["ts"] <= k["_launch"]["ts"]
                    <= step_cpu["ts"] + step_cpu["dur"]]
    t0 = min(k["ts"] for k in step_kernels)
    ms = lambda t: (t - t0) / 1e3  # noqa: E731

    nccl = [k for k in kernels if k["name"].startswith("ncclDevKernel")]
    compute_stream = Counter(k["args"]["stream"] for k in kernels if not k["name"].startswith("ncclDevKernel")).most_common(1)[0][0]

    comms = []
    for k in sorted(nccl, key=lambda e: e["ts"]):
        launch = k["_launch"]
        tid, t = (launch["tid"], launch["ts"]) if launch else (None, None)
        frames = frames_at(py[tid], t) if launch and tid in py else []
        ops = [e for e in rpc[tid].containing(t)] if launch and tid in rpc else []
        if not ops and launch and "SendRecv" in k["name"]:
            batch = next((f for f in reversed(frames) if f["func"] == "batch_isend_irecv"), None)
            if batch is not None:   # the c10d::send / recv_ ops posted inside the same batch_isend_irecv call
                end = next(e for e in py[tid].containing(t) if e["ts"] == batch["ts"])
                ops = [e for e in rpc[tid].ev if batch["ts"] <= e["ts"] <= end["ts"] + end["dur"]]
        ops = [o for o in ops if o["args"].get("Collective name") != "wait"]
        a = ops[0]["args"] if ops else k["args"]
        p2p_fn = next((f["func"] for f in reversed(frames) if f["file"].endswith("p2p_communication.py")
                       and f["func"] in ("send_forward", "send_backward", "recv_forward", "recv_backward",
                                         "send_forward_recv_backward", "send_backward_recv_forward")), None)
        comms.append({
            "kernel": k["name"].split("(")[0], "stream": k["args"]["stream"],
            "on_compute_stream": k["args"]["stream"] == compute_stream,
            "start_ms": ms(k["ts"]), "dur_us": k["dur"],
            "collective": "+".join(o["args"].get("Collective name", "?") for o in ops) or k["args"].get("Collective name"),
            "group": a.get("Process Group Description"), "group_ranks": a.get("Process Group Ranks"),
            "group_size": a.get("Group size"), "nelems": a.get("In msg nelems"), "dtype": a.get("dtype"),
            "dims": [o["args"].get("Input Dims", [[None]])[0][0] if o["args"].get("Input Dims") else None for o in ops],
            "peers": [o["args"].get("Dst Rank", o["args"].get("Src Rank")) for o in ops],
            "launch_thread": "main" if tid == main_tid else ("autograd" if tid is not None else "?"),
            "phase": k["_phase"] or "outside F/B", "p2p_call": p2p_fn,
            "megatron_function": next((f"{f['file']}:{f['def_line']} {f['func']}" for f in reversed(frames)
                                       if f["file"].startswith("megatron/")), None),
            "stack": describe_stack(frames, src),
        })

    # ---- 1F1B order and idle time on the compute stream --------------------------------------------------
    # Order: the main thread's F/B ranges in time order (exactly the order Megatron called them).
    # GPU span of an op: first start .. last end of the kernels launched inside that op's range (any thread).
    order, spans = [], []
    for e in fb_cpu:
        name = e["name"].removeprefix("day13/")
        ks = [k for k in step_kernels if k["_phase"] == name]
        order.append(name.replace(" mb=", ""))
        if ks:
            spans.append({"op": order[-1], "start_ms": ms(min(k["ts"] for k in ks)),
                          "end_ms": ms(max(k["ts"] + k["dur"] for k in ks)), "kernels": len(ks)})
    win_lo = t0
    win_hi = max(k["ts"] + k["dur"] for k in step_kernels if k["_phase"])
    busy = sorted((k["ts"], k["ts"] + k["dur"]) for k in kernels if k["args"]["stream"] == compute_stream
                  and k["ts"] + k["dur"] > win_lo and k["ts"] < win_hi)
    gaps, cur = [], win_lo
    for a, b in busy:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if win_hi > cur:
        gaps.append((cur, win_hi))
    sendrecv = [(k["ts"], k["ts"] + k["dur"]) for k in nccl if "SendRecv" in k["name"]]
    other_nccl = [(k["ts"], k["ts"] + k["dur"]) for k in nccl if "SendRecv" not in k["name"]
                  and k["args"]["stream"] != compute_stream]

    def overlap(gap, ivs):
        return sum(max(0.0, min(gap[1], b) - max(gap[0], a)) for a, b in ivs)

    idle = sum(b - a for a, b in gaps)
    p2p_wait = sum(min(b - a, overlap((a, b), sendrecv)) for a, b in gaps)
    tp_wait = sum(min(b - a - min(b - a, overlap((a, b), sendrecv)), overlap((a, b), other_nccl)) for a, b in gaps)
    compute_busy = sum(k["dur"] for k in kernels if k["args"]["stream"] == compute_stream and
                       not k["name"].startswith("ncclDevKernel") and win_lo <= k["ts"] < win_hi)
    nccl_on_compute = sum(k["dur"] for k in nccl if k["args"]["stream"] == compute_stream and win_lo <= k["ts"] < win_hi)
    long_gaps = sorted(gaps, key=lambda g: g[0] - g[1])[:6]
    return {
        "trace": path.name, "with_stack": bool(by_tid_py), "compute_stream": compute_stream, "order": order,
        "spans": spans, "comms": comms,
        "window_ms": (win_hi - win_lo) / 1e3, "compute_kernels_ms": compute_busy / 1e3,
        "nccl_on_compute_stream_ms": nccl_on_compute / 1e3, "idle_ms": idle / 1e3,
        "idle_waiting_p2p_ms": p2p_wait / 1e3, "idle_during_other_nccl_ms": tp_wait / 1e3,
        "idle_other_ms": (idle - p2p_wait - tp_wait) / 1e3,
        "longest_idle_gaps": [{"start_ms": ms(a), "len_ms": (b - a) / 1e3,
                               "after": next((s["op"] for s in reversed(spans) if s["start_ms"] <= ms(a)), "start"),
                               "p2p_overlap_ms": overlap((a, b), sendrecv) / 1e3} for a, b in long_gaps],
    }


def summarize(rank: int, r: dict) -> str:
    out = [f"=== rank {rank} ({r['trace']}) ==="]
    out.append(f"F/B order (GPU ranges): {' '.join(r['order'])}")
    out.append("F/B spans, ms from the step's first GPU range: " +
               "  ".join(f"{s['op']} {s['start_ms']:.2f}-{s['end_ms']:.2f}" for s in r["spans"]))
    out.append(f"compute stream {r['compute_stream']}: window {r['window_ms']:.2f} ms = compute kernels "
               f"{r['compute_kernels_ms']:.2f} + NCCL on compute stream {r['nccl_on_compute_stream_ms']:.2f} + idle "
               f"{r['idle_ms']:.2f} (waiting for P2P {r['idle_waiting_p2p_ms']:.2f}, during other-stream NCCL "
               f"{r['idle_during_other_nccl_ms']:.2f}, other/CPU {r['idle_other_ms']:.2f})")
    out.append("longest idle gaps: " + "; ".join(f"{g['len_ms']:.2f} ms after {g['after']} (P2P overlap "
                                                   f"{g['p2p_overlap_ms']:.2f})" for g in r["longest_idle_gaps"]))
    cat = Counter()
    for c in r["comms"]:
        kind = c["p2p_call"] or c["megatron_function"]
        ph = c["phase"].split()[0] if c["phase"] != "outside F/B" else "outside F/B"
        cat[(c["kernel"], c["collective"], c["group"], c["group_ranks"], c["nelems"], c["launch_thread"],
             c["on_compute_stream"], ph, kind)] += 1
    out.append(f"NCCL kernels in the step: {len(r['comms'])}")
    out.append("  count  kernel / collective / group ranks / elems / thread / on compute stream / phase / Megatron frame")
    for (kern, coll, grp, ranks, n, th, oncs, ph, kind), cnt in sorted(cat.items(), key=lambda x: -x[1]):
        out.append(f"  {cnt:>5}  {kern} / {coll} / {grp} {ranks} / {n} / {th} / {oncs} / {ph} / {kind}")
    return "\n".join(out)


def pick_examples(results: dict[int, dict]) -> dict:
    """One TP all-reduce in a forward (row-parallel output, Day 11's g), one in a backward (column-parallel dgrad,
    Day 11's f), and one PP exchange in the steady state, on the lowest rank that has each."""
    ex = {}
    for rank in sorted(results):
        for c in results[rank]["comms"]:
            st = " ".join(c["stack"])
            if "tp_forward" not in ex and c["collective"] == "allreduce" and c["phase"].startswith("F") \
                    and "transformer/mlp.py" in st and "mappings.py" in st:
                ex["tp_forward"] = dict(c, rank=rank)
            # Autograd's thread only has the Function.backward frame; which linear it is follows from the order
            # (GUIDE Block 4): the first all-reduce of B mb=0 on stage 0 is layer 2's linear_fc1 input gradient.
            if "tp_backward" not in ex and c["collective"] == "allreduce" and c["phase"].startswith("B") \
                    and c["launch_thread"] == "autograd":
                ex["tp_backward"] = dict(c, rank=rank)
            if "pp_p2p" not in ex and c["p2p_call"] in ("send_forward_recv_backward", "send_backward_recv_forward"):
                ex["pp_p2p"] = dict(c, rank=rank)
    return ex


def main():
    ap = argparse.ArgumentParser(description="Day 13 trace reader")
    ap.add_argument("run_dir", help="outputs/day13_megatron/run_<RID>/<tag> (holds trace_rank*.json.gz)")
    ap.add_argument("--megatron", default=None, help="Megatron-LM checkout, to resolve call lines")
    ap.add_argument("--json", default=None, help="write the full analysis here (default <run_dir>/trace_analysis.json)")
    args = ap.parse_args()
    run = Path(args.run_dir)
    src = SourceIndex(Path(args.megatron) if args.megatron else None)
    rank_of = lambda p: int(re.search(r"rank(\d+)", p.name)[1])  # noqa: E731
    stack_traces = sorted(run.glob("trace_rank*.json*"), key=rank_of)
    timing_traces = sorted(run.glob("timing_rank*.json*"), key=rank_of)
    nsys_dbs = sorted(run.glob("nsys_rank*.sqlite"), key=rank_of)
    if not stack_traces and nsys_dbs:
        return analyze_nsys(run, nsys_dbs, src, rank_of, args)
    if not stack_traces:
        raise SystemExit(f"no trace_rank*.json(.gz) in {run}")
    results = {rank_of(p): analyze_rank(p, src) for p in stack_traces}
    timing = {rank_of(p): analyze_rank(p, src) for p in timing_traces}
    print("##### Step traced WITH Python stacks: attribution (which group, phase and Megatron frame per kernel).")
    print("##### Its ms are inflated by stack recording; read order and counts here, timing below.\n")
    for rank, r in results.items():
        print(summarize(rank, r), "\n")
    if timing:
        print("##### Step traced WITHOUT stacks: order, GPU spans and idle time (still profiler-inflated, less so).\n")
        for rank, r in timing.items():
            print(summarize(rank, r).split("\nNCCL kernels")[0], "\n")
    ex = pick_examples(results)
    print("=== examples for the annotated trace (deliverable 3) ===")
    for key, c in ex.items():
        print(f"\n[{key}] rank {c['rank']}: {c['kernel']} on stream {c['stream']} "
              f"({'compute' if c['on_compute_stream'] else 'NCCL'} stream), launched from the {c['launch_thread']} "
              f"thread, phase {c['phase']}, start {c['start_ms']:.3f} ms, {c['dur_us']:.1f} us")
        print(f"  collective {c['collective']}, {c['group']} ranks {c['group_ranks']} (size {c['group_size']}), "
              f"{c['nelems']} x {c['dtype']}, dims {c['dims']}, peers {c['peers']}, p2p call {c['p2p_call']}")
        for line in c["stack"]:
            print(f"    {line}")
    out = Path(args.json) if args.json else run / "trace_analysis.json"
    out.write_text(json.dumps({"ranks": results, "timing": timing, "examples": ex}, indent=1))
    print(f"\nwrote {out}")


def analyze_nsys(run: Path, dbs: list[Path], src: SourceIndex, rank_of, args) -> None:
    """--profiler nsys runs (GUIDE C.6): the timing step from Nsight Systems, plus every thread's long calls."""
    import nsys_trace
    timing = {rank_of(p): analyze_rank(p, src, ev=nsys_trace.events(p)) for p in dbs}
    threads = {rank_of(p): nsys_trace.thread_view(p) for p in dbs}
    print("##### Timing step captured by Nsight Systems (no PyTorch profiler object in the process).")
    print("##### Order, GPU spans and idle time; collectives named by kernel (no process-group names without")
    print("##### record_param_comms, no Megatron frames without with_stack).\n")
    for rank, r in timing.items():
        print(summarize(rank, r), "\n")
    print("##### Every thread's calls of 1 ms or more inside the capture (CUDA API and OS runtime), longest first.\n")
    for rank, v in threads.items():
        print(f"=== rank {rank} ===\n{nsys_trace.summarize_threads(v)}\n")
    out = Path(args.json) if args.json else run / "trace_analysis.json"
    out.write_text(json.dumps({"ranks": {}, "timing": timing, "threads": threads, "examples": {}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
