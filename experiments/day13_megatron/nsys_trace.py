"""Day 13 - read an Nsight Systems capture (`nsys export --type sqlite`) for analyze_trace.py (GUIDE Appendix C.6).

`events(db)` turns the capture into the event shape analyze_trace.analyze_rank() reads from a PyTorch-profiler
trace, so the same order / span / idle-time code runs on both:

  CUPTI_ACTIVITY_KIND_KERNEL   -> cat "kernel"          (ts, dur in us; args.correlation, args.stream)
  CUPTI_ACTIVITY_KIND_RUNTIME  -> cat "cuda_runtime"    (runtime AND driver API calls, e.g. NCCL's cuLaunchKernelEx;
                                                         every thread of the process, with tid and correlation)
  NVTX_EVENTS (push/pop)       -> cat "user_annotation" (our "day13/step", "day13/F mb=k", "day13/B mb=k")

What nsys does not have, and the PyTorch profiler does: Python frames (with_stack) and `record_param_comms` (process
group, element count). So the nsys path reads order, spans and idle time, and names a collective by its kernel; the
Megatron-frame attribution stays with the PyTorch profiler's stack step.

`thread_view(db)` is what the PyTorch profiler cannot show at all: for every thread of the process (named, e.g.
pt_nccl_watchdg, pt_autograd_0), its long CUDA API calls and its long OS-runtime calls (pthread_mutex_lock,
pthread_cond_wait, poll, ...; nsys records only calls longer than --osrt-threshold).

Standard library only.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

NVTX_PUSHPOP, NVTX_STARTEND = 59, 60


def _tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _strings(con):
    return dict(con.execute("SELECT id, value FROM StringIds"))


def _tid(global_tid):
    return int(global_tid) & 0xFFFFFF   # nsys globalTid = (pid << 24) | tid, plus a high-bit tag


def collective_of(kernel_name: str):
    m = re.match(r"ncclDevKernel_([A-Za-z]+)", kernel_name)
    if not m:
        return None
    return {"AllReduce": "allreduce", "AllGather": "allgather", "ReduceScatter": "reduce_scatter",
            "SendRecv": "send+recv", "Broadcast": "broadcast", "Reduce": "reduce"}.get(m[1], m[1])


def events(db: Path) -> list[dict]:
    con = sqlite3.connect(str(db))
    tabs, s = _tables(con), _strings(con)
    out = []
    kcols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    name_col = "demangledName" if "demangledName" in kcols else "shortName"
    for start, end, stream, corr, nid in con.execute(
            f"SELECT start, end, streamId, correlationId, {name_col} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        name = s.get(nid, str(nid))
        args = {"correlation": corr, "stream": stream}
        coll = collective_of(name)
        if coll:
            args["Collective name"] = coll
        out.append({"ph": "X", "cat": "kernel", "name": name, "ts": start / 1e3, "dur": (end - start) / 1e3,
                    "args": args})
    for start, end, gtid, corr, nid in con.execute(
            "SELECT start, end, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        out.append({"ph": "X", "cat": "cuda_runtime", "name": s.get(nid, str(nid)), "ts": start / 1e3,
                    "dur": (end - start) / 1e3, "tid": _tid(gtid), "args": {"correlation": corr}})
    if "NVTX_EVENTS" in tabs:
        ncols = _cols(con, "NVTX_EVENTS")
        text_id = "textId" if "textId" in ncols else "NULL"
        for start, end, gtid, text, tid_ in con.execute(
                f"SELECT start, end, globalTid, text, {text_id} FROM NVTX_EVENTS "
                f"WHERE eventType IN ({NVTX_PUSHPOP}, {NVTX_STARTEND}) AND end IS NOT NULL"):
            name = text if text is not None else s.get(tid_, "")
            out.append({"ph": "X", "cat": "user_annotation", "name": name, "ts": start / 1e3,
                        "dur": (end - start) / 1e3, "tid": _tid(gtid)})
    con.close()
    return out


# Calls in which a thread only waits for work (NCCL proxies, CUDA's event handler, idle watchdogs): hidden on
# every thread except the main thread and autograd's threads, where a wait IS the stall.
IDLE = {"poll", "ppoll", "epoll_wait", "select", "pselect6", "pthread_cond_timedwait", "pthread_cond_wait",
        "nanosleep", "clock_nanosleep", "usleep", "sleep", "sem_timedwait", "sem_wait", "read", "recv", "recvmsg",
        "accept", "accept4"}


def thread_view(db: Path, min_ms: float = 1.0, top: int = 8) -> dict:
    """Per named thread: CUDA API calls and OS-runtime calls longer than `min_ms`, grouped by function
    (count, total ms, max ms), plus the longest individual calls with their start time."""
    con = sqlite3.connect(str(db))
    tabs, s = _tables(con), _strings(con)
    names = {}
    if "ThreadNames" in tabs:
        for nid, gtid in con.execute("SELECT nameId, globalTid FROM ThreadNames"):
            names[_tid(gtid)] = s.get(nid, "?")
    t_lo = con.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0] or 0
    view = defaultdict(lambda: {"cuda": defaultdict(lambda: [0, 0.0, 0.0]), "osrt": defaultdict(lambda: [0, 0.0, 0.0]),
                                "longest": []})
    srcs = [("cuda", "CUPTI_ACTIVITY_KIND_RUNTIME")] + ([("osrt", "OSRT_API")] if "OSRT_API" in tabs else [])
    for kind, table in srcs:
        for start, end, gtid, nid in con.execute(
                f"SELECT start, end, globalTid, nameId FROM {table} WHERE end - start >= ?", (int(min_ms * 1e6),)):
            tid = _tid(gtid)
            fn, dur = s.get(nid, str(nid)), (end - start) / 1e6
            main = tid == (int(gtid) >> 24) & 0xFFFFFF
            if not main:
                names.setdefault(tid, "?")
            if fn in IDLE and not main and not names.get(tid, "").startswith("pt_autograd"):
                continue
            if main:
                names[tid] = "MAIN " + names.get(tid, "python").removeprefix("MAIN ")
            agg = view[tid][kind][fn]
            agg[0] += 1
            agg[1] += dur
            agg[2] = max(agg[2], dur)
            view[tid]["longest"].append((dur, kind, fn, (start - t_lo) / 1e6))
    con.close()
    out = {}
    for tid, v in view.items():
        out[f"{names.get(tid, '?')} [{tid}]"] = {
            "cuda": {k: {"n": a[0], "total_ms": round(a[1], 2), "max_ms": round(a[2], 2)} for k, a in v["cuda"].items()},
            "osrt": {k: {"n": a[0], "total_ms": round(a[1], 2), "max_ms": round(a[2], 2)} for k, a in v["osrt"].items()},
            "longest": [{"ms": round(d, 2), "kind": k, "call": f, "at_ms": round(t, 2)}
                        for d, k, f, t in sorted(v["longest"], reverse=True)[:top]],
        }
    return out


def summarize_threads(view: dict, top_threads: int = 12) -> str:
    """Threads ordered by their longest call; one line per thread."""
    rows = sorted(view.items(), key=lambda kv: -(kv[1]["longest"][0]["ms"] if kv[1]["longest"] else 0))
    lines = []
    for name, v in rows[:top_threads]:
        calls = "; ".join(f"{c['call']} {c['ms']:.1f} ms @{c['at_ms']:.1f}" for c in v["longest"][:4])
        lines.append(f"  {name}: {calls}")
    return "\n".join(lines)
