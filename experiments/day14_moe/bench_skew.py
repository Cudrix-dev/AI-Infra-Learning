"""Day 14 Experiment 3 - where does load skew show up when experts live on different ranks? (one MoE layer)

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/day14_moe/bench_skew.py \
        --json <file>                                   # EP=4, one process per GPU
    CUDA_VISIBLE_DEVICES=0 python experiments/day14_moe/bench_skew.py --ep 1 --json <file>     # one-GPU reference
    python experiments/day14_moe/bench_skew.py --cpu --nproc 4 --shapes tiny --steps 3        # plumbing only

One MoE layer (router + 4 experts, top-1, dropless, FP32, TF32 off), no attention: the dense part is identical
on every rank and would only add noise. The global token count is fixed per shape; each rank holds T/4 tokens.
  medium  Day 01's shapes:  d 512,  ffn 2048,  T =  4096 (16 x 256)
  large   4x tokens, 2x d:  d 1024, ffn 4096,  T = 16384           (so expert GEMMs, not launches, can dominate)
Routing patterns: balanced, hot (~70 % to expert 0), all_to_one (expert 3). Same total tokens in every pattern.

Timing protocol (README rules): `--warmup` untimed steps, then `--steps` timed steps. Each step starts after a
barrier and torch.cuda.synchronize(); CUDA events are recorded on the compute stream at every phase boundary
(forward: route, counts + device->host sync, dispatch all_to_all, experts, combine all_to_all, unpermute;
backward: grad through unpermute, combine all_to_all backward, experts backward, dispatch all_to_all backward, the
rest); the step ends with synchronize(). A rank's step time is its first-to-last event; the job's step time is
the SLOWEST rank's. An all_to_all's time includes waiting for the slowest peer to arrive. `--trace` adds ONE
profiled step after all timed steps (never mixed into the samples) and frees the profiler (del prof; gc.collect()).
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import moe_common as C
import moe_ref
from moe_model import MoELayer, RouteOpts

SHAPES = {"tiny": (64, 128, 1024), "medium": (512, 2048, 4096), "large": (1024, 4096, 16384)}
FWD = ["route", "counts", "dispatch", "experts", "combine", "unpermute"]
BWD = ["bwd_unpermute", "bwd_combine_a2a", "bwd_experts", "bwd_dispatch_a2a", "bwd_rest"]


class Marker:
    """Records (name, CUDA event, host time) at phase boundaries. CPU: host time only."""

    def __init__(self, cuda: bool):
        self.cuda, self.marks = cuda, []

    def __call__(self, name: str) -> None:
        ev = None
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        self.marks.append((name, ev, time.perf_counter()))

    def phases(self) -> tuple[dict, dict]:
        gpu, host = {}, {}
        for (_, e0, h0), (name, e1, h1) in zip(self.marks, self.marks[1:]):
            gpu[name] = e0.elapsed_time(e1) if self.cuda else (h1 - h0) * 1e3
            host[name] = (h1 - self.marks[0][2]) * 1e3          # host clock when the mark was reached
        return gpu, host


def timed_impl(marker: Marker):
    """The reference, with an AllToAllV whose BACKWARD marks its start and end (backward runs a2a #2 then #1)."""
    calls = {"n": 0}

    class TimedAllToAllV(moe_ref.AllToAllV):
        @staticmethod
        def backward(ctx, grad_out):
            calls["n"] += 1
            first = calls["n"] % 2 == 1                          # combine's backward comes first
            marker("bwd_unpermute" if first else "bwd_experts")
            out = moe_ref.AllToAllV.backward(ctx, grad_out)
            marker("bwd_combine_a2a" if first else "bwd_dispatch_a2a")
            return out

    ns = {n: getattr(moe_ref, n) for n in ("top1_route", "permute", "unpermute", "exchange_counts", "finish_grads")}
    return SimpleNamespace(AllToAllV=TimedAllToAllV, **ns)


def bench_case(shape, pattern, args, g, dev) -> dict:
    d, ffn, T = SHAPES[shape]
    ep = g.ep if g is not None else 1
    world = g.world if g is not None else 1
    rank = g.rank if g is not None else 0
    t_r = T // world
    cfg = C.MoEConfig(d_model=d, ffn=ffn, n_experts=4, aux_coeff=0.0, z_coeff=0.0)
    marker = Marker(dev.type == "cuda")
    impl = timed_impl(marker)
    layer = MoELayer(cfg, g, impl, RouteOpts(pattern=pattern, n_tokens_global=T, aux_scope="off", validate=False))
    with torch.no_grad():
        for n, p in layer.named_parameters():
            p.copy_(C.init_value(f"bench.{n}", tuple(p.shape), cfg.seed, torch.float64))
    layer = layer.to(dev)
    gen = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(t_r, d, generator=gen).to(dev).requires_grad_(True)
    gy = torch.randn(t_r, d, generator=gen).to(dev)
    token_ids = torch.arange(rank * t_r, (rank + 1) * t_r, device=dev)
    layer.mark = marker
    samples = []

    def one_step():
        marker.marks.clear()
        x.grad = None
        for p in layer.parameters():
            p.grad = None
        if g is not None:
            dist.barrier()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        marker("start")
        y = layer(x, token_ids)
        loss = (y * gy).sum()
        loss.backward()
        marker("bwd_rest" if ep > 1 else "bwd_total")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        gpu, host = marker.phases()
        step = sum(gpu.values())
        return {"gpu": gpu, "host_at": host, "step_ms": step}

    for _ in range(args.warmup):
        one_step()
    for _ in range(args.steps):
        samples.append(one_step())
    st = layer.stats
    send = st["send"] if st["send"] is not None else [t_r]
    recv = st["recv"] if st["recv"] is not None else [t_r]
    res = {"shape": shape, "pattern": pattern, "rank": rank, "d": d, "ffn": ffn, "T": T, "send_rows": send,
           "recv_rows": recv, "recv_total": sum(recv), "recv_mib": sum(recv) * d * 4 / 2 ** 20,
           "phase_ms": {k: statistics.median(s["gpu"][k] for s in samples) for k in samples[0]["gpu"]},
           "step_ms": [s["step_ms"] for s in samples],
           "host_counts_at_ms": statistics.median(s["host_at"]["counts"] for s in samples)}
    if args.trace and f"{shape}:{pattern}" in args.trace_cases.split(","):
        from torch.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with profile(activities=acts) as prof:
            one_step()
        path = args.trace / f"trace_ep{ep}_{shape}_{pattern}_rank{rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))
        res["trace"] = str(path)
        del prof
        gc.collect()          # Day 13 Appendix C: the profiler's reference cycle, freed now and not inside a step
    return res


def summarize(results: list[dict], ep: int) -> list[dict]:
    """results: one dict per rank for one (shape, pattern). The step time of the job = slowest rank, per step."""
    steps = list(zip(*[r["step_ms"] for r in results]))
    job = [max(s) for s in steps]
    slow = max(range(len(results)), key=lambda i: statistics.median(results[i]["step_ms"]))
    phases = list(results[0]["phase_ms"])
    print(f"\n--- {results[0]['shape']} (d {results[0]['d']}, ffn {results[0]['ffn']}, T {results[0]['T']}) | "
          f"{results[0]['pattern']} | EP={ep}")
    hdr = " ".join(f"{p[:11]:>11s}" for p in phases)
    print(f"rank {'recv rows':>9s} {'recv MiB':>8s} {hdr} {'step':>8s} {'host@counts':>11s}")
    for r in results:
        cols = " ".join(f"{r['phase_ms'][p]:11.3f}" for p in phases)
        print(f"{r['rank']:4d} {r['recv_total']:9d} {r['recv_mib']:8.2f} {cols} {statistics.median(r['step_ms']):8.3f} "
              f"{r['host_counts_at_ms']:11.3f}")
    spread = {p: max(r["phase_ms"][p] for r in results) - min(r["phase_ms"][p] for r in results) for p in phases}
    top = max(spread, key=spread.get)
    exp_rate = [r["phase_ms"]["experts"] / max(r["recv_total"], 1) * 1000 for r in results if r["recv_total"]]
    print(f"job step (slowest rank per step): median {statistics.median(job):.3f} ms, min {min(job):.3f}, "
          f"max {max(job):.3f} over {len(job)} steps; slowest rank {results[slow]['rank']}; largest receive "
          f"buffer {max(r['recv_total'] for r in results)} rows; phase with the largest rank-to-rank spread: "
          f"{top} ({spread[top]:.3f} ms); expert forward ms per 1000 received rows: "
          f"{[round(v, 3) for v in exp_rate]}")
    return [{"shape": results[0]["shape"], "pattern": results[0]["pattern"], "ep": ep,
             "job_step_ms_median": statistics.median(job), "job_step_ms": job, "slowest_rank": results[slow]["rank"],
             "largest_spread_phase": top, "spread_ms": spread, "per_rank": results}]


def worker(argv) -> None:
    args = parse(argv)
    C.set_exact_math(deterministic=False)          # timing: default kernels (parity is check_ep.py's job)
    g, dev = None, (torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda", 0))
    if args.ep > 1:
        dev, _ = C.init_distributed(args.cpu)
        g = C.build_groups(args.ep)
    rank0 = g is None or g.rank == 0
    if rank0:
        hw = "CPU (timings meaningless)" if dev.type == "cpu" else torch.cuda.get_device_name(dev)
        print(f"[bench_skew] EP={args.ep} on {hw}; warmup {args.warmup}, timed steps {args.steps}; "
              f"phases are medians over timed steps, CUDA events on the compute stream")
    out = []
    for shape in args.shapes.split(","):
        for pattern in args.patterns.split(","):
            res = bench_case(shape, pattern, args, g, dev)
            if g is not None:
                allr = [None] * g.world
                dist.all_gather_object(allr, res)
            else:
                allr = [res]
            if rank0:
                out += summarize(allr, args.ep)
    if rank0 and args.json:
        C.write_json(args.json, {"ep": args.ep, "env": C.environment(), "args": vars(args), "cases": out})
    if rank0:
        print("\nBENCH_SKEW: done")
    if g is not None:
        dist.barrier()
        dist.destroy_process_group()


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", type=int, default=4, choices=[1, 4])
    ap.add_argument("--shapes", default="medium,large")
    ap.add_argument("--patterns", default="balanced,hot,all_to_one")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--trace", type=Path, help="directory: one extra profiled step per --trace-cases entry")
    ap.add_argument("--trace-cases", default="large:all_to_one,large:balanced")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--json", type=Path)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = parse(argv)
    if args.nproc:
        C.spawn_local(worker, argv, args.nproc)
    else:
        worker(argv)


if __name__ == "__main__":
    main()
