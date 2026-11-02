"""Day 04 - count the NCCL kernels in profiler traces: how many collectives does one optimizer step issue?

The Python-side counters in train_accum.py only see flat_sync and MiniDDP. DDP's Reducer calls the process group from
C++, so for DDP the GPU trace is the evidence: every NCCL collective is one `ncclDevKernel_*` (or `ncclKernel_*`)
kernel on the NCCL stream. The count per optimizer step is (kernels in the trace) / (profiled optimizer steps).

Pure Python (json only): runs on the server or on your laptop after downloading the traces.

  python experiments/day04_grad_accum/count_collectives.py outputs/day04_grad_accum/cnt_*_<rid>/trace_rank0.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

KINDS = ("AllReduce", "Broadcast", "AllGather", "ReduceScatter", "SendRecv", "Reduce")


def kind_of(name: str) -> str:
    for k in KINDS:  # AllReduce before Reduce: first match wins
        if k in name:
            return k
    return "other"


def count(trace: Path) -> dict:
    """Return {'kernels': Counter(kind -> n), 'profile_steps': int or None, 'micro_ranges': n of '## bwd' ranges}."""
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = Counter()
    bwd_ranges = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        if e.get("cat") == "kernel" and re.match(r"nccl(Dev)?Kernel", name):
            kernels[kind_of(name)] += 1
        elif e.get("cat") == "user_annotation" and name.startswith("## bwd"):  # CPU copy only (not gpu_user_annotation)
            bwd_ranges += 1
    steps = None
    meta = trace.parent / "metrics.json"
    if meta.exists():
        steps = json.loads(meta.read_text())["meta"]["args"]["profile_steps"]
    return {"kernels": kernels, "profile_steps": steps, "bwd_ranges": bwd_ranges}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--json", type=Path, default=None, help="write the table as JSON")
    args = ap.parse_args()
    labels = args.labels or [t.parent.name for t in args.traces]
    rows = []
    print(f"{'variant':28s} {'opt steps':>9s} {'backwards':>9s} {'AllReduce/step':>15s} {'Broadcast/step':>15s} "
          f"{'other/step':>11s}")
    for label, trace in zip(labels, args.traces):
        c = count(trace)
        st = c["profile_steps"] or 1
        k = c["kernels"]
        other = sum(v for kk, v in k.items() if kk not in ("AllReduce", "Broadcast"))
        row = {"variant": label, "profile_steps": st, "backwards_per_step": c["bwd_ranges"] / st,
               "allreduce_per_step": k["AllReduce"] / st, "broadcast_per_step": k["Broadcast"] / st,
               "other_per_step": other / st, "raw": dict(k)}
        rows.append(row)
        print(f"{label:28s} {st:9d} {row['backwards_per_step']:9.1f} {row['allreduce_per_step']:15.1f} "
              f"{row['broadcast_per_step']:15.1f} {row['other_per_step']:11.1f}")
    print("one ncclDevKernel_* per collective; per step = kernels in the trace / profiled optimizer steps (rank 0).")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
