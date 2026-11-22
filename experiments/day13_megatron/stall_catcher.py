"""Day 13 follow-up - take all-thread native stacks of a rank in the middle of a host stall (GUIDE Appendix B).

The ranks must run stall_probe.py --pid-dir D --allow-ptrace (prctl PR_SET_PTRACER_ANY: the container has no
CAP_SYS_PTRACE and Yama ptrace_scope is 1). For each rank this script keeps one gdb attached and running the
process; gdb is told to stop on SIGUSR2 and not pass it on. Every --poll-ms it reads the rank's main-thread state
and wait channel and the state of its NCCL watchdog threads (comm "pt_nccl_watchdg"). When the main thread has
been asleep in a futex for --trigger-ms in a row while a watchdog thread is running, it sends SIGUSR2: gdb stops
the process, prints `thread apply all bt`, and detaches. One capture per rank.

Usage: python stall_catcher.py <pid-dir> <out-dir> [--ranks 0,1,2,3] [--trigger-ms 25] [--arm-after-s 40]
"""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def read(p):
    try:
        return Path(p).read_text()
    except OSError:
        return None


def state(pid, tid):
    st = read(f"/proc/{pid}/task/{tid}/stat")
    if st is None:
        return None, None
    s = st[st.rindex(")") + 2]
    return s, (read(f"/proc/{pid}/task/{tid}/wchan") or "").strip()


def watchdogs(pid):
    out = []
    for tid in os.listdir(f"/proc/{pid}/task"):
        st = read(f"/proc/{pid}/task/{tid}/stat")
        if st and "pt_nccl_watchdg" in st[st.index("("):st.rindex(")")]:
            out.append(tid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ranks", default="0,1,2,3")
    ap.add_argument("--poll-ms", type=float, default=2.0)
    ap.add_argument("--trigger-ms", type=float, default=25.0)
    ap.add_argument("--arm-after-s", type=float, default=40.0, help="ignore startup and warmup")
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pid_dir = Path(a.pid_dir)
    while len(list(pid_dir.glob("pid_rank*.txt"))) < len(ranks):
        time.sleep(0.2)
    time.sleep(1.0)
    pids = {r: int((pid_dir / f"pid_rank{r}.txt").read_text()) for r in ranks}
    gdbs = {}
    for r, pid in pids.items():
        f = open(out / f"gdb_rank{r}.txt", "w")
        gdbs[r] = (subprocess.Popen(
            ["gdb", "-p", str(pid), "-batch", "-ex", "set pagination off", "-ex", "handle all nostop noprint pass",
             "-ex", "handle SIGUSR2 stop print nopass", "-ex", "continue", "-ex", "thread apply all bt 30"],
            stdout=f, stderr=subprocess.STDOUT), f)
    t_arm = time.monotonic() + a.arm_after_s
    asleep_since = {r: None for r in ranks}
    wd = {}
    log = open(out / "catcher.log", "w")
    pending = set(ranks)
    while pending:
        t = time.monotonic()
        for r in list(pending):
            pid = pids[r]
            s, w = state(pid, pid)
            if s is None:   # rank exited
                pending.discard(r)
                continue
            if t < t_arm:
                continue
            if r not in wd:
                wd[r] = watchdogs(pid)
            if s == "S" and "futex" in w:
                asleep_since[r] = asleep_since[r] or t
            else:
                asleep_since[r] = None
                continue
            busy = [tid for tid in wd[r] if state(pid, tid)[0] == "R"]
            if busy and (t - asleep_since[r]) * 1e3 >= a.trigger_ms:
                os.kill(pid, signal.SIGUSR2)
                log.write(f"rank {r} pid {pid}: main asleep in {w} for {(t - asleep_since[r]) * 1e3:.1f} ms, "
                          f"watchdog running {busy}; SIGUSR2 at mono {time.monotonic_ns()}\n")
                log.flush()
                pending.discard(r)
        time.sleep(a.poll_ms / 1e3)
    for r, (p, f) in gdbs.items():   # captured ranks' gdb has detached; others: gdb ends with the process
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
        f.close()
    log.write("done\n")


if __name__ == "__main__":
    main()
