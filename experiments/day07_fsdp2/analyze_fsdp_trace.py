"""Day 07 - find FSDP2's all-gathers and reduce-scatters in a torch.profiler trace, name the module each one belongs to,
and measure how much of it nothing else was hiding.

How a GPU kernel is tied to the Python code that launched it (Kineto's chrome trace):
    GPU kernel event      cat "kernel", pid = GPU, tid = CUDA stream, args.correlation = N
    CPU launch event      cat "cuda_runtime" / "cuda_driver" (cudaLaunchKernel, cuLaunchKernelEx, ...), same N, with the
                          CPU thread id and the launch time
    CPU annotation        cat "user_annotation": FSDP's record_function ranges, e.g. "FSDP::all_gather (blocks.3)",
                          "FSDP::post_backward_reduce (blocks.3)", and this harness's "## fwd j" / "## bwd j"
The label of a kernel = the innermost FSDP:: annotation (else "## ..." annotation) on the launching thread that contains
the launch time. For the NCCL kernels FSDP issues, that names the parameter group: forward all-gather, backward
all-gather (re-gather or prefetch), reduce-scatter.

Per profiled step (the CPU ProfilerStep#N ranges; a kernel belongs to the step its LAUNCH falls in):
    comm_ms          union of NCCL kernel intervals
    exposed_comm_ms  comm time during which no non-NCCL kernel ran on any stream (Day 03's definition)
    gpu_busy_ms      union of all GPU activity
    idle_ms          (first kernel start .. last kernel end) - gpu_busy: the GPU waited for the CPU (launch-bound) or
                     for nothing at all
Caveats (Day 03): an NCCL kernel's duration includes waiting for the peer rank; "overlapped" means concurrent, not free.

Pure Python + Day 03's interval helpers; works on downloaded traces.
  python experiments/day07_fsdp2/analyze_fsdp_trace.py outputs/day07_fsdp2/cnt_fsdp_block_*/trace_rank0.json --sequence
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day03_ddp_hooks"))
sys.path.insert(0, str(HERE.parent / "day04_grad_accum"))
from analyze_trace import intersect, length, union  # noqa: E402  (Day 03)
from count_collectives import kind_of  # noqa: E402  (Day 04)

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}


def is_nccl(e: dict) -> bool:
    return e.get("cat") == "kernel" and bool(re.match(r"nccl(Dev)?Kernel", e.get("name", "")))


class Annotations:
    """Innermost CPU annotation containing a time, per thread (ranges nest, so the shortest container is innermost)."""

    def __init__(self, ranges: list[dict]):
        self.by_tid = defaultdict(list)
        for e in ranges:
            self.by_tid[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"], e["name"]))
        for v in self.by_tid.values():
            v.sort()
        self.starts = {k: [a for a, _, _ in v] for k, v in self.by_tid.items()}

    def innermost(self, key, ts: float, prefix: str) -> str | None:
        v, st = self.by_tid.get(key), self.starts.get(key)
        if not v:
            return None
        best = None
        for i in range(bisect.bisect_right(st, ts) - 1, -1, -1):
            a, b, name = v[i]
            if b < ts:  # ranges before ts that ended: keep scanning, an outer range may start earlier
                continue
            if name.startswith(prefix) and (best is None or b - a < best[0]):
                best = (b - a, name)
            if ts - a > 5e6:  # 5 s back is far enough for any enclosing range in these traces
                break
        return best[1] if best else None


def analyze(path: Path) -> dict:
    ev = json.loads(path.read_text())
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    xs = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    steps = sorted((e for e in xs if e.get("cat") == "user_annotation"
                    and str(e.get("name", "")).startswith("ProfilerStep#")), key=lambda e: e["ts"])
    ann = Annotations([e for e in xs if e.get("cat") == "user_annotation"
                       and str(e.get("name", "")).startswith(("FSDP::", "## "))])
    launch = {e["args"]["correlation"]: e for e in xs if e.get("cat") in LAUNCH_CATS and "correlation" in e.get("args", {})}
    gpu = [e for e in xs if e.get("cat") in GPU_CATS]
    main_tid = (steps[0]["pid"], steps[0]["tid"]) if steps else None  # the training loop's thread

    def label_of(key, ts: float) -> tuple[str, str]:
        lab = ann.innermost(key, ts, "FSDP::")
        if lab and "(" not in lab:
            lab += " (root)"  # the root group's module FQN is "", so FSDP prints no "(name)"
        phase = ann.innermost(key, ts, "## ") or ("backward (autograd thread)" if key != main_tid else "-")
        return lab or ann.innermost(key, ts, "## ") or "?", phase

    per_step = []
    for si, st in enumerate(steps):
        a, b = st["ts"], st["ts"] + st["dur"]
        mine = []
        for e in gpu:
            le = launch.get(e.get("args", {}).get("correlation"))
            t_launch = le["ts"] if le else e["ts"]
            if a <= t_launch < b:
                mine.append((e, le))
        if not mine:
            continue
        nccl = [(e, le) for e, le in mine if is_nccl(e)]
        other = [e for e, _ in mine if not is_nccl(e)]
        u_comm = union([(e["ts"], e["ts"] + e["dur"]) for e, _ in nccl])
        u_comp = union([(e["ts"], e["ts"] + e["dur"]) for e in other])
        u_all = union([(e["ts"], e["ts"] + e["dur"]) for e, _ in mine])
        span = (u_all[-1][1] - u_all[0][0]) if u_all else 0.0
        t0 = u_all[0][0] if u_all else a
        seq = []
        for e, le in sorted(nccl, key=lambda p: p[0]["ts"]):
            label, phase = label_of((le["pid"], le["tid"]), le["ts"]) if le else ("?", "?")
            iv = [(e["ts"], e["ts"] + e["dur"])]
            seq.append({"kind": kind_of(e["name"]), "label": label, "stream": e.get("tid"),
                        "start_ms": (e["ts"] - t0) / 1e3, "dur_ms": e["dur"] / 1e3,
                        "overlap_frac": intersect(iv, u_comp) / e["dur"] if e["dur"] else 0.0, "phase": phase})
        streams = Counter(str(e.get("tid")) for e, _ in mine if not is_nccl(e))
        per_step.append({"step": si, "n_kernels": len(mine), "kinds": dict(Counter(s["kind"] for s in seq)),
                         "comm_ms": length(u_comm) / 1e3,
                         "exposed_comm_ms": (length(u_comm) - intersect(u_comm, u_comp)) / 1e3,
                         "gpu_busy_ms": length(u_all) / 1e3, "span_ms": span / 1e3,
                         "idle_ms": (span - length(u_all)) / 1e3, "compute_streams": dict(streams), "sequence": seq})
    return {"trace": str(path), "steps": per_step}


def add_min_rank(res: dict, other: dict) -> None:
    """Every rank issues the same collectives in the same order (NCCL requires it), so the i-th NCCL kernel of a step
    on rank 0 and rank 1 is the SAME collective. The rank that arrived first waits inside its kernel for the other one;
    min(duration over ranks) is the closer estimate of the transfer itself (Day 06's last-arriving-rank idea)."""
    for s0, s1 in zip(res["steps"], other["steps"]):
        if len(s0["sequence"]) != len(s1["sequence"]):
            s0["comm_min_rank_sum_ms"] = float("nan")
            continue
        for c0, c1 in zip(s0["sequence"], s1["sequence"]):
            c0["dur_min_rank_ms"] = min(c0["dur_ms"], c1["dur_ms"])
        s0["comm_min_rank_sum_ms"] = sum(c["dur_min_rank_ms"] for c in s0["sequence"])


def analyze_pair(trace_rank0: Path) -> dict:
    """analyze() rank 0's trace and attach min-over-ranks durations from rank 1's trace next to it (if present)."""
    res = analyze(trace_rank0)
    t1 = trace_rank0.with_name(trace_rank0.name.replace("rank0", "rank1"))
    if t1 != trace_rank0 and t1.exists():
        add_min_rank(res, analyze(t1))
    return res


def summary(res: dict) -> dict:
    st = res["steps"]
    if not st:
        return {}
    med = lambda k: statistics.median(s[k] for s in st)  # noqa: E731
    kinds = Counter()
    for s in st:
        kinds.update(s["kinds"])
    keys = ["comm_ms", "exposed_comm_ms", "gpu_busy_ms", "span_ms", "idle_ms"]
    if all("comm_min_rank_sum_ms" in s for s in st):
        keys.append("comm_min_rank_sum_ms")
    return {"profiled_steps": len(st), "per_step": {k: v / len(st) for k, v in kinds.items()},
            **{k: med(k) for k in keys}}


def print_sequence(res: dict, step: int = 0, max_rows: int = 200) -> None:
    s = res["steps"][step]
    print(f"\nstep {s['step']}: {len(s['sequence'])} collectives | comm {s['comm_ms']:.2f} ms (exposed "
          f"{s['exposed_comm_ms']:.2f}) | GPU busy {s['gpu_busy_ms']:.2f} of span {s['span_ms']:.2f} ms "
          f"(idle {s['idle_ms']:.2f})")
    print(f"{'#':>3s} {'kind':13s} {'launched inside':44s} {'phase':12s} {'stream':>6s} {'start ms':>9s} "
          f"{'dur ms':>7s} {'min-rank':>8s} {'overlap':>7s}")
    for i, c in enumerate(s["sequence"][:max_rows]):
        mr = c.get("dur_min_rank_ms")
        print(f"{i:3d} {c['kind']:13s} {c['label'][:44]:44s} {str(c['phase']).replace('backward (autograd thread)', 'bwd thread')[:12]:12s} "
              f"{str(c['stream']):>6s} {c['start_ms']:9.3f} {c['dur_ms']:7.3f} "
              f"{(f'{mr:8.3f}' if mr is not None else '       -')} {100 * c['overlap_frac']:6.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--sequence", action="store_true", help="print every collective of the first profiled step")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    labels = args.labels or [t.parent.name for t in args.traces]
    out = []
    for label, t in zip(labels, args.traces):
        res = analyze_pair(t)
        sm = summary(res)
        out.append({"label": label, "summary": sm, **res})
        print(f"\n=== {label}: per step {sm.get('per_step')} | comm {sm.get('comm_ms', 0):.2f} ms, exposed "
              f"{sm.get('exposed_comm_ms', 0):.2f} | busy {sm.get('gpu_busy_ms', 0):.2f} / span "
              f"{sm.get('span_ms', 0):.2f} ms (idle {sm.get('idle_ms', 0):.2f}) ===")
        if args.sequence and res["steps"]:
            print_sequence(res)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
