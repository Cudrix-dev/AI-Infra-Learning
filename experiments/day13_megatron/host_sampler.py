"""Day 13 follow-up - sample what the operating system sees while stall_probe.py runs (GUIDE Appendix B).

Every --period-ms, for the main thread of each rank (pid files written by stall_probe.py --pid-dir) it records the
thread state (R running, S sleeping, D uninterruptible), its kernel wait channel, its own user/system CPU time,
voluntary / involuntary context switches and page faults. For the container it records cgroup CPU usage and
throttling, and pressure-stall (PSI) totals for cpu, memory and io. Timestamps are time.monotonic_ns(), the same
CLOCK_MONOTONIC clock as stall_probe.py's perf_counter_ns(), so the two can be joined by stall_correlate.py.

Usage: python host_sampler.py <pid-dir> <out.jsonl> [--period-ms 10] [--ranks 4]
Stops when all rank processes have exited.
"""

import argparse
import json
import os
import time
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def psi_totals(name):
    txt = read(f"/sys/fs/cgroup/{name}.pressure")
    out = {}
    for line in (txt or "").splitlines():
        kind, *fields = line.split()
        out[kind] = int(dict(f.split("=") for f in fields)["total"])   # microseconds stalled, cumulative
    return out


def cpu_stat():
    txt = read("/sys/fs/cgroup/cpu.stat") or ""
    d = dict(line.split() for line in txt.splitlines() if line)
    return {"usage_us": int(d.get("usage_usec", 0)), "throttled_us": int(d.get("throttled_usec", 0))}


def thread_sample(pid):
    """Main thread of `pid` (tid == pid): state, wchan, cpu ms, context switches, faults."""
    stat = read(f"/proc/{pid}/task/{pid}/stat")
    if stat is None:
        return None
    rest = stat[stat.rindex(")") + 2:].split()   # fields after "(comm)"
    status = read(f"/proc/{pid}/task/{pid}/status") or ""
    ctx = {k: int(v) for k, v in (line.split(":") for line in status.splitlines()
                                   if line.startswith(("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")))}
    return {"state": rest[0], "wchan": (read(f"/proc/{pid}/task/{pid}/wchan") or "?").strip() or "0",
            "minflt": int(rest[7]), "majflt": int(rest[9]),
            "utime_ms": int(rest[11]) * 1000 // CLK, "stime_ms": int(rest[12]) * 1000 // CLK,
            "vcs": ctx.get("voluntary_ctxt_switches"), "nvcs": ctx.get("nonvoluntary_ctxt_switches")}


_LAST = {}   # (pid, tid) -> cpu ticks at the previous sample


def busy_threads(pid):
    """Threads of `pid` (other than the main thread) that used CPU since the last sample or are runnable now:
    {tid: [comm, state, cpu ticks since last sample]}. Shows who runs while the main thread sleeps."""
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        if tid == str(pid):
            continue
        st = read(f"/proc/{pid}/task/{tid}/stat")
        if st is None:
            continue
        comm = st[st.index("(") + 1:st.rindex(")")]
        rest = st[st.rindex(")") + 2:].split()
        ticks = int(rest[11]) + int(rest[12])
        d = ticks - _LAST.get((pid, tid), ticks)
        _LAST[(pid, tid)] = ticks
        if d or rest[0] == "R":
            out[tid] = [comm, rest[0], d]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_dir")
    ap.add_argument("out")
    ap.add_argument("--period-ms", type=float, default=10.0)
    ap.add_argument("--ranks", type=int, default=4)
    a = ap.parse_args()
    pid_dir = Path(a.pid_dir)
    deadline = time.monotonic() + 600
    while len(list(pid_dir.glob("pid_rank*.txt"))) < a.ranks:   # wait for the job to start
        if time.monotonic() > deadline:
            raise SystemExit("no pid files")
        time.sleep(0.2)
    pids = {int(p.stem.removeprefix("pid_rank")): int(p.read_text()) for p in pid_dir.glob("pid_rank*.txt")}
    period = a.period_ms / 1e3
    with open(a.out, "w") as f:
        f.write(json.dumps({"pids": pids, "period_ms": a.period_ms, "clk_tck": CLK,
                            "wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}) + "\n")   # clock pair joins nvidia-smi wall time
        while True:
            t = time.monotonic_ns()
            ranks = {r: thread_sample(pid) for r, pid in pids.items()}
            for r, pid in pids.items():
                if ranks[r] is not None:
                    ranks[r]["others"] = busy_threads(pid)
            if all(v is None for v in ranks.values()):
                break
            f.write(json.dumps({"t": t, "cg": cpu_stat(), "psi": {n: psi_totals(n) for n in ("cpu", "memory", "io")},
                                "ranks": ranks}) + "\n")
            time.sleep(max(0.0, period - (time.monotonic_ns() - t) / 1e9))


if __name__ == "__main__":
    main()
