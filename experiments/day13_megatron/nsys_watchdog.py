"""Day 13 follow-up - the NCCL watchdog stall in an Nsight Systems capture (GUIDE Appendix C.6).

For each rank's `nsys export --type sqlite` file of a long run (run_nsys_check.sh part 2; CUDA API + OS-runtime
trace with backtraces), after the first --skip-s seconds:

 1. every long non-idle call (>= --min-ms) on any thread, with the top of its backtrace for OS-runtime calls;
 2. every long pthread_mutex_lock, and what the rank's watchdog threads did inside it (number of CUDA calls,
    time in them, longest single call);
 3. the watchdog's scans: its cudaEventQuery calls grouped into bursts (one burst per 100 ms wake-up), with the
    number of queries, the burst's duration and the median query time, normal vs slowest.

Usage: python nsys_watchdog.py <dir with nsys_rank*.sqlite> [--min-ms 15] [--skip-s 9]
Standard library only.
"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

IDLE = {"poll", "pthread_cond_timedwait", "pthread_cond_wait", "epoll_wait", "nanosleep", "sem_timedwait",
        "pthread_join", "sem_wait", "read", "recvmsg"}


def report(db: Path, min_ms: float, skip_s: float) -> str:
    con = sqlite3.connect(str(db))
    S = dict(con.execute("SELECT id, value FROM StringIds"))
    names = {g & 0xFFFFFF: S.get(n, "?") for n, g in con.execute("SELECT nameId, globalTid FROM ThreadNames")}
    t0 = con.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
    lo, thr = t0 + skip_s * 1e9, int(min_ms * 1e6)
    who = lambda g: "MAIN" if (g & 0xFFFFFF) == (g >> 24) & 0xFFFFFF else names.get(g & 0xFFFFFF, "?")  # noqa: E731
    out = [f"(times in ms from the first CUDA call; calls before {skip_s:.0f} s skipped)", "", f"1. calls >= {min_ms} ms"]
    rows = []
    for tab, cc in (("OSRT_API", "callchainId"), ("CUPTI_ACTIVITY_KIND_RUNTIME", "NULL")):
        for st, en, g, n, chain in con.execute(
                f"SELECT start, end, globalTid, nameId, {cc} FROM {tab} WHERE end - start >= ? AND start > ?", (thr, lo)):
            fn = S.get(n, "?")
            if fn in IDLE or fn.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize")):
                continue
            top = ""
            if chain is not None:
                fr = con.execute("SELECT symbol FROM OSRT_CALLCHAINS WHERE id = ? ORDER BY stackDepth LIMIT 3",
                                 (chain,)).fetchall()
                top = " < ".join(re.sub(r"\(.*", "", S.get(f[0], str(f[0])))[:48] for f in fr[1:3])
            rows.append(((st - t0) / 1e6, (en - st) / 1e6, who(g), fn.split("_v")[0], top))
    out += [f"  @{t:9.1f} {d:6.1f} ms  {w:16s} {f:26s} {top}" for t, d, w, f, top in sorted(rows)]

    wd = {t for t, n in names.items() if n == "pt_nccl_watchdg"}
    # all CUDA API calls of the watchdog threads, read once (the table has millions of rows and no index)
    wd_gtids = [g for (g,) in con.execute("SELECT DISTINCT globalTid FROM ThreadNames") if (g & 0xFFFFFF) in wd]
    marks = ",".join("?" * len(wd_gtids)) or "NULL"
    wcalls = con.execute(f"SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start > ? "
                         f"AND globalTid IN ({marks}) ORDER BY start", (lo, *wd_gtids)).fetchall()
    starts = [c[0] for c in wcalls]
    out += ["", f"2. pthread_mutex_lock >= {min_ms} ms, and the watchdog threads' CUDA calls inside it"]
    for st, en, g in con.execute(
            "SELECT o.start, o.end, o.globalTid FROM OSRT_API o JOIN StringIds s ON s.id = o.nameId "
            "WHERE s.value = 'pthread_mutex_lock' AND o.end - o.start >= ? AND o.start > ?", (thr, lo)):
        per = {}
        for s2, e2, g2, n2 in wcalls[max(0, bisect.bisect_left(starts, st - 50e6)):bisect.bisect_right(starts, en)]:
            tid = g2 & 0xFFFFFF
            if e2 > st:
                p = per.setdefault(tid, [0, 0.0, 0.0, Counter()])
                p[0] += 1
                p[1] += (min(e2, en) - max(s2, st)) / 1e6
                p[2] = max(p[2], (e2 - s2) / 1e6)
                p[3][S.get(n2, "?").split("_v")[0]] += 1
        busiest = max(per.items(), key=lambda kv: kv[1][1]) if per else None
        desc = (f"watchdog {busiest[0]}: {busiest[1][0]} calls, {busiest[1][1]:.1f} ms of the wait, longest "
                f"{busiest[1][2]:.2f} ms, {dict(busiest[1][3].most_common(2))}") if busiest else "no watchdog call"
        out.append(f"  @{(st - t0) / 1e6:9.1f} {who(g):14s} waits {(en - st) / 1e6:5.1f} ms; {desc}")

    out += ["", "3. watchdog scans (cudaEventQuery bursts), per watchdog thread with >= 100 scans"]
    for wtid in sorted(wd):
        q = [(s, e) for s, e, g, n in wcalls if (g & 0xFFFFFF) == wtid and S.get(n, "").startswith("cudaEventQuery")]
        if len(q) < 100:
            continue
        bursts, cur = [], [q[0]]
        for a, b in zip(q, q[1:]):
            if b[0] - a[1] > 20e6:
                bursts.append(cur)
                cur = [b]
            else:
                cur.append(b)
        bursts.append(cur)
        if len(bursts) < 100:
            continue
        sc = [((b[0][0] - t0) / 1e6, len(b), (b[-1][1] - b[0][0]) / 1e6,
               statistics.median((e - s) / 1e3 for s, e in b)) for b in bursts]
        out.append(f"  watchdog {wtid}: {len(sc)} scans; queries per scan median {statistics.median(x[1] for x in sc):.0f}"
                   f" (max {max(x[1] for x in sc)}); scan median {statistics.median(x[2] for x in sc):.2f} ms; "
                   f"query median {statistics.median(x[3] for x in sc):.0f} us")
        for t, n, d, m in sorted(sc, key=lambda x: -x[2])[:8]:
            if d >= min_ms:
                out.append(f"    slow scan @{t:9.1f}: {n} queries in {d:6.1f} ms, median query {m:.0f} us")
    con.close()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-ms", type=float, default=15.0)
    ap.add_argument("--skip-s", type=float, default=9.0, help="skip startup (NCCL connection setup)")
    a = ap.parse_args()
    txt = []
    for db in sorted(Path(a.dir).glob("nsys_rank*.sqlite")):
        txt.append(f"######## {db.name}\n{report(db, a.min_ms, a.skip_s)}\n")
    print("\n".join(txt))
    Path(a.dir, "watchdog_report.txt").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
