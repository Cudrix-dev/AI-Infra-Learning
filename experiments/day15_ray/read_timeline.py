"""Day 15 - a text view of a ray.timeline() trace: which worker ran which actor method when, for the last iterations.

    python experiments/day15_ray/read_timeline.py outputs/day15_ray/run_<RID>/timeline_async.json [--last 3]

The trace is Chrome-trace JSON (open it in chrome://tracing or https://ui.perfetto.dev for the full picture).
Here: complete events ('ph' == 'X') whose name contains an actor method of workers.py, grouped by process, in a
window that starts at the (last - k)-th Learner.update. Times are ms from the window start.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--last", type=int, default=3)
    a = p.parse_args()
    ev = json.load(open(a.trace))
    keep = ("sample", "update", "set_weights", "publish")
    xs = [e for e in ev if e.get("ph") == "X" and any(k in e.get("name", "") for k in keep)]
    print(f"{len(ev)} events, {len(xs)} actor-method events; names: {sorted({e['name'] for e in xs})[:12]}")
    ups = sorted((e for e in xs if "update" in e["name"] and "Learner" in e["name"]), key=lambda e: e["ts"])
    if not ups:
        ups = sorted((e for e in xs if "update" in e["name"]), key=lambda e: e["ts"])
    if not ups:
        print("no Learner.update events found")
        return
    t0 = ups[max(0, len(ups) - a.last)]["ts"]
    rows = defaultdict(list)
    for e in xs:
        if e["ts"] + e.get("dur", 0) >= t0:
            rows[(e.get("pid"), e.get("tid"))].append(e)
    for key, es in sorted(rows.items(), key=lambda kv: min(e["ts"] for e in kv[1])):
        print(f"\n{key}")
        for e in sorted(es, key=lambda e: e["ts"]):
            print(f"  {(e['ts'] - t0) / 1e3:9.2f} -> {(e['ts'] + e.get('dur', 0) - t0) / 1e3:9.2f} ms  "
                  f"{e['name']} [{e.get('cat', '')}]")
    print("\nTIMELINE: done")


if __name__ == "__main__":
    main()
