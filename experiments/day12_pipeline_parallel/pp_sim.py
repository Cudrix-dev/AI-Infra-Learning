"""Day 12 - CPU schedule simulator for two (or more) pipeline stages: GPipe and non-interleaved 1F1B.

No torch needed: runs on Windows or the server.

    python experiments/day12_pipeline_parallel/pp_sim.py                    # GUIDE's hand-worked example: P=2, M=4
    python experiments/day12_pipeline_parallel/pp_sim.py --check            # assert it matches the GUIDE (exit 0/1)
    python experiments/day12_pipeline_parallel/pp_sim.py --sweep 1,2,4,8,16 # fixed effective batch, vary M
    python experiments/day12_pipeline_parallel/pp_sim.py --stage-times 1:2,2:4      # optional: slow stage 1
    python experiments/day12_pipeline_parallel/pp_sim.py --schedule 1f1b --comm-style naive --send rendezvous
                                                                            # the deadlock that pairing avoids
    python experiments/day12_pipeline_parallel/pp_sim.py --show-programs    # each rank's compute + P2P program

ASSUMPTIONS (the default model; every one is a flag):
  * P stages, one per rank. Stage s runs forward F(s, m) and backward B(s, m) of microbatch m.
  * Each stage executes its own ops in a FIXED order (the schedule) and never reorders them. An op starts when the
    stage is free AND its inputs have arrived:
        F(s, m) needs the activation of F(s-1, m)      (stage 0 reads the data instead)
        B(s, m) needs F(s, m) done and the gradient of B(s+1, m)   (the last stage starts backward from its loss)
  * Times: t_F = 1 and t_B = 2 time units on every stage (backward ~ 2x forward: dX and dW matmuls). Communication
    takes --comm units (default 0). Times are exact fractions, so the ASCII grid is exact.
  * --send buffered (default): a send completes as soon as it is posted (the data waits in the receiver's queue);
    only receives wait. --send rendezvous: a send completes only when the receiver has posted the matching receive.
    Blocking point-to-point is allowed to behave like this, and it is the model in which a bad send/recv order
    deadlocks.
  * "In flight" on stage s: microbatch m from the START of F(s, m) to the END of B(s, m). During that window the
    tensors autograd saved for B(s, m) (and, on stages > 0, the received input) are resident on stage s.
    The count changes only at this stage's own ops, so the peak depends on the ORDER alone, not on the times.
  * Bubble (idle) on a stage = makespan - its busy time, where makespan = when the last stage finishes the step.
    It includes time spent waiting inside a receive. No optimizer step, no data loading, no launch overhead.

Schedules (per-stage order of compute ops):
  gpipe  F0 F1 ... F(M-1), then B0 B1 ... B(M-1)  (ascending backward order on every stage, so gradients accumulate
         in the same order as 1F1B; the original paper's figure draws them descending: same bubble, same peak)
  1f1b   Megatron's forward_backward_pipelining_without_interleaving: warmup = min(P - s - 1, M) forwards, then
         alternate (F, B) in steady state, then the remaining backwards in cooldown.
Communication programs (which sends/receives each rank posts, and which are posted TOGETHER):
  megatron (1f1b only): exactly Megatron's calls: recv_forward, send_forward, send_forward_recv_backward,
         send_backward_recv_forward, recv_backward, send_backward. A call with two ops posts both at once
         (batch_isend_irecv), so they complete together.
  naive  every send and receive is its own blocking step, placed right before/after the compute op that needs it.
         GPipe always uses this (it never needs a two-way exchange).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from pathlib import Path

SCHEDULES = ("gpipe", "1f1b")


# ---------------------------------------------------------------------------------------------------------------
# 1. The schedules: per-stage order of compute ops
# ---------------------------------------------------------------------------------------------------------------

def num_warmup(schedule: str, P: int, M: int, s: int) -> int:
    """Forwards a stage runs before its first backward."""
    if schedule == "gpipe":
        return M
    return min(P - s - 1, M)          # Megatron schedules.py: total_stages - current_stage - 1, capped at M


def op_order(schedule: str, P: int, M: int, s: int) -> list[tuple[str, int]]:
    """[("F", 0), ("F", 1), ("B", 0), ...]: the order in which stage s runs its forwards and backwards."""
    if schedule == "gpipe":
        return [("F", m) for m in range(M)] + [("B", m) for m in range(M)]
    if schedule != "1f1b":
        raise ValueError(schedule)
    w = num_warmup(schedule, P, M, s)
    ops = [("F", m) for m in range(w)]                       # warmup
    for i in range(M - w):                                   # steady state: one forward, then the OLDEST backward
        ops += [("F", w + i), ("B", i)]
    ops += [("B", m) for m in range(M - w, M)]               # cooldown
    return ops


def order_string(ops) -> str:
    return " ".join(f"{k}{m}" for k, m in ops)


def peak_in_flight_formula(schedule: str, P: int, M: int, s: int) -> int:
    return M if schedule == "gpipe" else min(P - s, M)


# ---------------------------------------------------------------------------------------------------------------
# 2. The communication programs: compute ops plus the P2P ops around them
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    kind: str                       # "F", "B" (compute) or "comm"
    m: int = -1                     # microbatch (compute steps)
    ops: tuple = ()                 # comm steps: (("send"|"recv", peer_stage, ("act"|"grad", m)), ...)
    name: str = ""                  # comm steps: Megatron's name for the call

    def label(self) -> str:
        if self.kind != "comm":
            return f"{self.kind}{self.m}"
        parts = [f"{d}({p[0][0]}{p[1]})" for d, _, p in self.ops]   # e.g. send(a1)+recv(g0)
        return "+".join(parts)


def program(schedule: str, P: int, M: int, s: int, comm_style: str = "megatron") -> list[Step]:
    first, last = s == 0, s == P - 1

    def recv_fwd(m):
        return [] if first else [("recv", s - 1, ("act", m))]

    def send_fwd(m):
        return [] if last else [("send", s + 1, ("act", m))]

    def recv_bwd(m):
        return [] if last else [("recv", s + 1, ("grad", m))]

    def send_bwd(m):
        return [] if first else [("send", s - 1, ("grad", m))]

    steps: list[Step] = []

    def comm(ops, name):
        if ops:                      # Megatron's helpers are no-ops on the first/last stage; so are ours
            steps.append(Step("comm", ops=tuple(ops), name=name))

    if schedule == "gpipe" or comm_style == "naive":
        for k, m in op_order(schedule, P, M, s):
            if k == "F":
                comm(recv_fwd(m), "recv_forward")
                steps.append(Step("F", m))
                comm(send_fwd(m), "send_forward")
            else:
                comm(recv_bwd(m), "recv_backward")
                steps.append(Step("B", m))
                comm(send_bwd(m), "send_backward")
        return steps
    if comm_style != "megatron":
        raise ValueError(comm_style)

    # 1F1B with Megatron's calls (schedules.py, forward_backward_pipelining_without_interleaving)
    w = num_warmup("1f1b", P, M, s)
    rem = M - w
    for i in range(w):                                                    # warmup forwards
        comm(recv_fwd(i), "recv_forward")
        steps.append(Step("F", i))
        comm(send_fwd(i), "send_forward")
    if rem > 0:
        comm(recv_fwd(w), "recv_forward")                                 # "need to receive first forward tensor"
    for i in range(rem):                                                  # steady 1F1B
        f, b = w + i, i
        steps.append(Step("F", f))
        comm(send_fwd(f) + recv_bwd(b), "send_forward_recv_backward")
        steps.append(Step("B", b))
        if i == rem - 1:
            comm(send_bwd(b), "send_backward")
        else:
            comm(send_bwd(b) + recv_fwd(f + 1), "send_backward_recv_forward")
    for i in range(w):                                                    # cooldown backwards
        b = rem + i
        comm(recv_bwd(b), "recv_backward")
        steps.append(Step("B", b))
        comm(send_bwd(b), "send_backward")
    return steps


def comm_sequence(steps: list[Step]) -> list[str]:
    """The P2P calls in program order, e.g. ['send_forward(a0)', 'send_forward_recv_backward(a1,g0)', ...]."""
    out = []
    for st in steps:
        if st.kind == "comm":
            out.append(f"{st.name}({','.join(p[0][0] + str(p[1]) for _, _, p in st.ops)})")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3. The engine: run every rank's program, honour data dependencies and the send model, record intervals
# ---------------------------------------------------------------------------------------------------------------

class Deadlock(Exception):
    pass


@dataclass
class Result:
    schedule: str
    P: int
    M: int
    times: list                      # [(t_F, t_B)] per stage
    comm: Fraction
    send: str
    comm_style: str
    intervals: list = field(default_factory=list)  # per stage: [(kind, m, start, end)]
    deadlock: str | None = None
    makespan: Fraction = Fraction(0)


def simulate(schedule: str, P: int, M: int, times, comm=Fraction(0), send="buffered",
             comm_style="megatron") -> Result:
    progs = [program(schedule, P, M, s, comm_style) for s in range(P)]
    res = Result(schedule, P, M, times, Fraction(comm), send, comm_style, [[] for _ in range(P)])

    # Sequence number of every op on its channel (src -> dst). Channels are FIFO: the k-th send from a to b is
    # received by the k-th receive on b from a. That is how microbatch IDs line up without being transmitted.
    seq = [dict() for _ in range(P)]                 # seq[s][(pc, op_index)] = k
    counters: dict = {}
    for s, prog in enumerate(progs):
        for pc, st in enumerate(prog):
            for j, (d, peer, _) in enumerate(st.ops):
                ch = (s, peer) if d == "send" else (peer, s)
                key = (d, ch)
                seq[s][(pc, j)] = counters.get(key, 0)
                counters[key] = counters.get(key, 0) + 1

    pc = [0] * P
    clock = [Fraction(0)] * P
    posted: list = [None] * P                        # time the current comm step was posted, or None
    op_done: list = [dict() for _ in range(P)]       # op index -> completion time, for the current comm step
    mailbox: dict = {}                               # buffered: (ch, k) -> (payload, arrival time)

    def current(s):
        return progs[s][pc[s]] if pc[s] < len(progs[s]) else None

    def check_payload(src_payload, dst_payload, ch, k):
        if src_payload != dst_payload:
            raise ValueError(f"channel {ch} message {k}: sent {src_payload}, receiver expected {dst_payload}")

    while any(pc[s] < len(progs[s]) for s in range(P)):
        progress = False
        for s in range(P):
            while (st := current(s)) is not None:
                if st.kind in ("F", "B"):
                    d = Fraction(times[s][0 if st.kind == "F" else 1])
                    res.intervals[s].append((st.kind, st.m, clock[s], clock[s] + d))
                    clock[s] += d
                    pc[s] += 1
                    progress = True
                    continue
                if posted[s] is None:                            # post the step (all its ops at once)
                    posted[s] = clock[s]
                    op_done[s] = {}
                    progress = True
                    if send == "buffered":
                        for j, (d, peer, payload) in enumerate(st.ops):
                            if d == "send":
                                k = seq[s][(pc[s], j)]
                                mailbox[((s, peer), k)] = (payload, posted[s] + res.comm)
                                op_done[s][j] = posted[s]
                for j, (d, peer, payload) in enumerate(st.ops):
                    if j in op_done[s]:
                        continue
                    k = seq[s][(pc[s], j)]
                    if send == "buffered":
                        if d == "recv" and ((peer, s), k) in mailbox:
                            src_payload, arrival = mailbox.pop(((peer, s), k))
                            check_payload(src_payload, payload, (peer, s), k)
                            op_done[s][j] = max(posted[s], arrival)
                            progress = True
                    else:                                         # rendezvous: the peer must have posted the match
                        pst = current(peer)
                        if pst is None or pst.kind != "comm" or posted[peer] is None:
                            continue
                        want = "recv" if d == "send" else "send"
                        for jj, (dd, pp, pl) in enumerate(pst.ops):
                            if dd == want and pp == s and seq[peer][(pc[peer], jj)] == k and jj not in op_done[peer]:
                                if d == "send":
                                    check_payload(payload, pl, (s, peer), k)
                                else:
                                    check_payload(pl, payload, (peer, s), k)
                                t = max(posted[s], posted[peer]) + res.comm
                                op_done[s][j] = t
                                op_done[peer][jj] = t
                                progress = True
                                break
                if len(op_done[s]) == len(st.ops):                # every op of the step completed
                    clock[s] = max([posted[s], *op_done[s].values()])
                    posted[s] = None
                    pc[s] += 1
                    progress = True
                    continue
                break                                             # blocked in this comm step
        if not progress:
            stuck = []
            for s in range(P):
                st = current(s)
                if st is not None:
                    waiting = [f"{d}({p[0][0]}{p[1]})" for j, (d, _, p) in enumerate(st.ops) if j not in op_done[s]]
                    stuck.append(f"stage {s} blocked in {st.name} waiting for {' + '.join(waiting)}")
            res.deadlock = "; ".join(stuck)
            break
    res.makespan = max(clock) if res.deadlock is None else Fraction(0)
    return res


# ---------------------------------------------------------------------------------------------------------------
# 4. Metrics and rendering
# ---------------------------------------------------------------------------------------------------------------

def stage_metrics(res: Result, s: int) -> dict:
    iv = res.intervals[s]
    busy = sum((e - b for _, _, b, e in iv), Fraction(0))
    f_start = {m: b for k, m, b, _ in iv if k == "F"}
    b_end = {m: e for k, m, _, e in iv if k == "B"}
    count = peak = 0
    for k, _, _, _ in iv:                       # the count changes only at this stage's own ops: order decides it
        if k == "F":
            count += 1
            peak = max(peak, count)
        else:
            count -= 1
    residency = {m: b_end[m] - f_start[m] for m in f_start if m in b_end}
    area = sum(residency.values(), Fraction(0))
    ms = res.makespan
    return {
        "order": order_string((k, m) for k, m, _, _ in iv),
        "busy": busy, "idle": ms - busy, "idle_fraction": (ms - busy) / ms if ms else Fraction(0),
        "peak_in_flight": peak, "mean_in_flight": area / ms if ms else Fraction(0),
        "max_residency": max(residency.values()) if residency else Fraction(0),
    }


def summary(res: Result) -> dict:
    per = [stage_metrics(res, s) for s in range(res.P)]
    total_idle = sum((p["idle"] for p in per), Fraction(0))
    uniform = len({tuple(t) for t in res.times}) == 1
    out = {
        "schedule": res.schedule, "P": res.P, "M": res.M, "times": [[str(a), str(b)] for a, b in res.times],
        "comm": str(res.comm), "send": res.send, "comm_style": res.comm_style, "deadlock": res.deadlock,
        "makespan": res.makespan, "bubble_fraction": total_idle / (res.P * res.makespan) if res.makespan else None,
        "stages": per,
    }
    if uniform and res.comm == 0:       # the closed forms assume equal stages and free communication
        tf, tb = (Fraction(x) for x in res.times[0])
        out["formula_makespan"] = (res.M + res.P - 1) * (tf + tb)
        out["formula_bubble_fraction"] = Fraction(res.P - 1, res.M + res.P - 1)
    return out


def frac_gcd(a: Fraction, b: Fraction) -> Fraction:
    if a == 0:
        return b
    if b == 0:
        return a
    return Fraction(gcd(a.numerator * b.denominator, b.numerator * a.denominator), a.denominator * b.denominator)


def grid(res: Result) -> tuple[Fraction, list[list[str]], list[list[int]]]:
    """Per stage, one label per tick ('F0', 'B2', '.') and the in-flight count during each tick."""
    tick = Fraction(0)
    for iv in res.intervals:
        for _, _, b, e in iv:
            tick = frac_gcd(frac_gcd(tick, b), e)
    tick = tick or Fraction(1)
    n = int(res.makespan / tick)
    cells, counts = [], []
    for s, iv in enumerate(res.intervals):
        row = ["."] * n
        for k, m, b, e in iv:
            for i in range(int(b / tick), int(e / tick)):
                row[i] = f"{k}{m}"
        f_start = {m: b for k, m, b, _ in iv if k == "F"}
        b_end = {m: e for k, m, _, e in iv if k == "B"}
        cnt = [sum(1 for m in f_start if f_start[m] <= i * tick < b_end.get(m, res.makespan + 1)) for i in range(n)]
        cells.append(row)
        counts.append(cnt)
    return tick, cells, counts


def render(res: Result, show_inflight: bool = True, max_cells: int = 120) -> str:
    tick, cells, counts = grid(res)
    n = len(cells[0]) if cells else 0
    if n > max_cells:
        return f"  (timeline has {n} ticks of {tick}; too wide to draw, use --json for the intervals)"
    w = max(3, max(len(c) for row in cells for c in row) + 1)
    head = "  t" if tick == 1 else f"  t (tick {tick})"

    def tlabel(i):                      # label whole time units only, so fractional ticks stay readable
        t = i * tick
        return str(t.numerator) if t.denominator == 1 else ""
    lines = [f"{head:<16}" + "".join(f"{tlabel(i):>{w}}" for i in range(n))]
    for s in range(res.P):
        lines.append(f"  stage {s:<9}" + "".join(f"{c:>{w}}" for c in cells[s]))
        if show_inflight:
            lines.append(f"    in flight    " + "".join(f"{c:>{w}}" for c in counts[s]))
    return "\n".join(lines)


def fmt(x) -> str:
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{float(x):.4g}"
    return str(x)


def report(res: Result, show_programs: bool = False) -> str:
    out = [f"\n=== {res.schedule.upper()}  P={res.P} stages, M={res.M} microbatches, (t_F, t_B) per stage "
           f"{[(fmt(Fraction(a)), fmt(Fraction(b))) for a, b in res.times]}, comm {fmt(res.comm)}, "
           f"send {res.send}, comm style {'naive' if res.schedule == 'gpipe' else res.comm_style} ==="]
    if show_programs:
        for s in range(res.P):
            prog = program(res.schedule, res.P, res.M, s, res.comm_style)
            out.append(f"  program stage {s}: " + " ".join(st.label() for st in prog))
    if res.deadlock:
        out.append(f"  DEADLOCK: {res.deadlock}")
        return "\n".join(out)
    sm = summary(res)
    out.append(render(res))
    out.append(f"  makespan {fmt(sm['makespan'])}"
               + (f" (formula (M+P-1)(t_F+t_B) = {fmt(sm['formula_makespan'])})" if "formula_makespan" in sm else "")
               + f"; bubble fraction {fmt(sm['bubble_fraction'])}"
               + (f" (formula (P-1)/(M+P-1) = {fmt(sm['formula_bubble_fraction'])})"
                  if "formula_bubble_fraction" in sm else ""))
    out.append(f"  {'stage':<6} {'order':<36} {'busy':>6} {'idle':>6} {'idle %':>7} {'peak in flight':>15} "
               f"{'mean in flight':>15} {'max residency':>14}")
    for s, p in enumerate(sm["stages"]):
        out.append(f"  {s:<6} {p['order']:<36} {fmt(p['busy']):>6} {fmt(p['idle']):>6} "
                   f"{100 * float(p['idle_fraction']):>6.1f}% {p['peak_in_flight']:>15} "
                   f"{float(p['mean_in_flight']):>15.3f} {fmt(p['max_residency']):>14}")
    return "\n".join(out)


def jsonable(x):
    if isinstance(x, Fraction):
        return float(x) if x.denominator != 1 else x.numerator
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def result_json(res: Result) -> dict:
    d = jsonable(summary(res))
    d["intervals"] = [[{"op": f"{k}{m}", "start": jsonable(b), "end": jsonable(e)} for k, m, b, e in iv]
                      for iv in res.intervals]
    d["programs"] = [[st.label() for st in program(res.schedule, res.P, res.M, s, res.comm_style)]
                     for s in range(res.P)]
    if not res.deadlock:
        tick, cells, counts = grid(res)
        d["grid"] = {"tick": jsonable(tick), "cells": cells, "in_flight": counts}
    return d


# ---------------------------------------------------------------------------------------------------------------
# 5. Sweep (fixed effective batch) and self-check
# ---------------------------------------------------------------------------------------------------------------

def sweep(P: int, Ms: list[int], TF: Fraction, TB: Fraction) -> tuple[str, list[dict]]:
    """Effective batch fixed. Microbatch time = full-batch time / M (ideal: no per-microbatch overhead).
    Activation size of one microbatch = 1/M of the full batch's activations on that stage."""
    rows, out = [], []
    hdr = (f"\n=== SWEEP: P={P}, effective batch fixed, full-batch time per stage (T_F, T_B) = ({fmt(TF)}, {fmt(TB)}); "
           f"one microbatch takes (T_F/M, T_B/M) and holds 1/M of the batch's activations ===\n"
           f"  {'M':>3} {'step time':>10} {'bubble':>8} {'idle/stage':>11} | {'GPipe peak':>10} {'act':>6} | "
           f"{'1F1B peak per stage':>20} {'act per stage':>16}")
    rows.append(hdr)
    for M in Ms:
        t = [(TF / M, TB / M)] * P
        g = summary(simulate("gpipe", P, M, t))
        f = summary(simulate("1f1b", P, M, t))
        assert g["makespan"] == f["makespan"]
        gp = [p["peak_in_flight"] for p in g["stages"]]
        fp = [p["peak_in_flight"] for p in f["stages"]]
        idle = g["stages"][0]["idle"]
        rows.append(f"  {M:>3} {float(g['makespan'] / (TF + TB)):>9.3f}x {float(g['bubble_fraction']):>8.3f} "
                    f"{float(idle / (TF + TB)):>10.3f}x | {max(gp):>10} {float(Fraction(max(gp), M)):>6.2f} | "
                    f"{str(fp):>20} {str([round(c / M, 3) for c in fp]):>16}")
        out.append({"M": M, "makespan_over_full_batch": jsonable(g["makespan"] / (TF + TB)),
                    "bubble_fraction": jsonable(g["bubble_fraction"]), "gpipe_peak": gp, "1f1b_peak": fp,
                    "gpipe_peak_activation_fraction": [c / M for c in gp],
                    "1f1b_peak_activation_fraction": [c / M for c in fp]})
    rows.append("  step time and idle are in units of one full-batch forward+backward of one stage (T_F+T_B); "
                "'act' = peak resident activations as a fraction of the full batch's activations on that stage.")
    return "\n".join(rows), out


# The GUIDE's hand-worked example (P=2, M=4, t_F=1, t_B=2, comm 0): one label per time unit, then the in-flight count.
HAND = {
    "gpipe": {
        "cells": ["F0 F1 F2 F3 . . . B0 B0 B1 B1 B2 B2 B3 B3", ". F0 F1 F2 F3 B0 B0 B1 B1 B2 B2 B3 B3 . ."],
        "in_flight": ["1 2 3 4 4 4 4 4 4 3 3 2 2 1 1", "0 1 2 3 4 4 4 3 3 2 2 1 1 0 0"],
        "peak": [4, 4], "idle": [3, 3], "mean_in_flight": [Fraction(42, 15), Fraction(30, 15)],
    },
    "1f1b": {
        "cells": ["F0 F1 . . B0 B0 F2 B1 B1 F3 B2 B2 . B3 B3", ". F0 B0 B0 F1 B1 B1 F2 B2 B2 F3 B3 B3 . ."],
        "in_flight": ["1 2 2 2 2 2 2 2 2 2 2 2 1 1 1", "0 1 1 1 1 1 1 1 1 1 1 1 1 0 0"],
        "peak": [2, 1], "idle": [3, 3], "mean_in_flight": [Fraction(26, 15), Fraction(12, 15)],
    },
}
HAND_PROGRAMS = {   # Megatron-style P2P calls per rank for 1F1B, P=2, M=4
    0: ["send_forward(a0)", "send_forward_recv_backward(a1,g0)", "send_forward_recv_backward(a2,g1)",
        "send_forward_recv_backward(a3,g2)", "recv_backward(g3)"],
    1: ["recv_forward(a0)", "send_backward_recv_forward(g0,a1)", "send_backward_recv_forward(g1,a2)",
        "send_backward_recv_forward(g2,a3)", "send_backward(g3)"],
}


def self_check() -> int:
    fails = []

    def expect(cond, what):
        print(f"  [{'ok' if cond else 'FAIL'}] {what}")
        if not cond:
            fails.append(what)

    one = [(Fraction(1), Fraction(2))] * 2
    print("Hand-worked example (P=2, M=4, t_F=1, t_B=2, comm 0, buffered sends):")
    for sch in SCHEDULES:
        res = simulate(sch, 2, 4, one)
        sm = summary(res)
        _, cells, counts = grid(res)
        exp = HAND[sch]
        expect([" ".join(r) for r in cells] == exp["cells"], f"{sch}: timeline cells equal GUIDE's grid")
        expect([" ".join(map(str, c)) for c in counts] == exp["in_flight"], f"{sch}: in-flight row equals GUIDE's")
        expect(sm["makespan"] == 15, f"{sch}: makespan 15 = (M+P-1)(t_F+t_B)")
        expect([p["idle"] for p in sm["stages"]] == exp["idle"], f"{sch}: idle per stage {exp['idle']}")
        expect(sm["bubble_fraction"] == Fraction(1, 5), f"{sch}: bubble fraction 1/5 = (P-1)/(M+P-1)")
        expect([p["peak_in_flight"] for p in sm["stages"]] == exp["peak"], f"{sch}: peak in flight {exp['peak']}")
        expect([p["mean_in_flight"] for p in sm["stages"]] == exp["mean_in_flight"],
               f"{sch}: mean in flight {[fmt(x) for x in exp['mean_in_flight']]}")
    expect({s: comm_sequence(program("1f1b", 2, 4, s)) for s in (0, 1)} == HAND_PROGRAMS,
           "1f1b: Megatron-style P2P call sequence per rank equals GUIDE's")

    print("General formulas (uniform stages, P in 2..4, M in 1..8):")
    ok_ms = ok_peak = ok_rdv = True
    for P in (2, 3, 4):
        for M in range(1, 9):
            t = [(Fraction(1), Fraction(2))] * P
            for sch in SCHEDULES:
                sm = summary(simulate(sch, P, M, t))
                ok_ms &= sm["makespan"] == (M + P - 1) * 3
                ok_peak &= [p["peak_in_flight"] for p in sm["stages"]] == \
                    [peak_in_flight_formula(sch, P, M, s) for s in range(P)]
                rv = simulate(sch, P, M, t, send="rendezvous")
                ok_rdv &= rv.deadlock is None and rv.intervals == simulate(sch, P, M, t).intervals
    expect(ok_ms, "makespan = (M+P-1)(t_F+t_B) for both schedules")
    expect(ok_peak, "peak in flight = M (GPipe) and min(P-s, M) (1F1B) on every stage")
    expect(ok_rdv, "uniform stages: rendezvous sends give the same timeline as buffered (no deadlock)")

    print("Deadlock model (rendezvous sends):")
    ok_dl = all(simulate("1f1b", 2, M, one, send="rendezvous", comm_style="naive").deadlock for M in range(2, 9))
    ok_m1 = simulate("1f1b", 2, 1, one, send="rendezvous", comm_style="naive").deadlock is None
    expect(ok_dl and ok_m1, "naive 1F1B deadlocks for every M >= 2 (P=2); M = 1 has no steady state and does not")
    expect(simulate("1f1b", 2, 4, one, comm_style="naive").deadlock is None,
           "naive 1F1B does not deadlock with buffered sends (the bug is latent)")
    print(f"\nSIMULATOR CHECK: {'PASSED' if not fails else 'FAILED (' + str(len(fails)) + ')'}")
    return 0 if not fails else 1


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------

def parse_times(spec: str, P: int, tf: str, tb: str) -> list:
    if not spec:
        return [(Fraction(tf), Fraction(tb))] * P
    parts = spec.split(",")
    if len(parts) != P:
        raise SystemExit(f"--stage-times needs {P} entries 'tF:tB', got {spec!r}")
    return [tuple(Fraction(x) for x in p.split(":")) for p in parts]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", "-P", type=int, default=2)
    ap.add_argument("--microbatches", "-M", type=int, default=4)
    ap.add_argument("--schedule", choices=("both",) + SCHEDULES, default="both")
    ap.add_argument("--tf", default="1", help="forward time per microbatch per stage (fraction ok, e.g. 1/2)")
    ap.add_argument("--tb", default="2", help="backward time per microbatch per stage")
    ap.add_argument("--stage-times", default="", help="per-stage 'tF:tB' list, e.g. 1:2,2:4 (overrides --tf/--tb)")
    ap.add_argument("--comm", default="0", help="time of one P2P transfer")
    ap.add_argument("--send", choices=("buffered", "rendezvous"), default="buffered")
    ap.add_argument("--comm-style", choices=("megatron", "naive"), default="megatron",
                    help="1F1B only: Megatron's paired calls, or every send/recv on its own")
    ap.add_argument("--sweep", default="", help="comma list of M with the effective batch fixed, e.g. 1,2,4,8,16")
    ap.add_argument("--show-programs", action="store_true")
    ap.add_argument("--check", action="store_true", help="assert the GUIDE's hand-worked numbers and formulas")
    ap.add_argument("--json", type=Path, help="write every result to this JSON file")
    args = ap.parse_args(argv)

    if args.check:
        return self_check()

    P, M = args.stages, args.microbatches
    times = parse_times(args.stage_times, P, args.tf, args.tb)
    record = {"argv": sys.argv[1:], "assumptions": "see pp_sim.py docstring", "results": []}
    if args.sweep:
        text, rows = sweep(P, [int(x) for x in args.sweep.split(",")], times[0][0], times[0][1])
        print(text)
        record["sweep"] = rows
    else:
        for sch in (SCHEDULES if args.schedule == "both" else (args.schedule,)):
            res = simulate(sch, P, M, times, Fraction(args.comm), args.send, args.comm_style)
            print(report(res, args.show_programs))
            record["results"].append(result_json(res))
        if args.schedule == "both" and all(r["deadlock"] is None for r in record["results"]):
            g, f = record["results"]
            print(f"\n  GPipe vs 1F1B: makespan {g['makespan']} vs {f['makespan']}; peak in flight per stage "
                  f"{[p['peak_in_flight'] for p in g['stages']]} vs {[p['peak_in_flight'] for p in f['stages']]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
