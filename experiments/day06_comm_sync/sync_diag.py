"""Day 06 follow-up - is it rank arrival skew or communication? A diagnostic copy of train_zero1.py's flat loop.

Same workload as `train_zero1.py --impl flat_ar|zero1` (Day 01 model and data, Day 04 accumulation, FlatZeroAdamW,
same seeds, same LR schedule, same local device synchronizations at the phase boundaries), plus one opt-in change:
where, if anywhere, the two ranks are explicitly aligned before communicating.

  --align none            no control barrier (the original loop; local torch.cuda.synchronize() only)
  --align prev_step       the user's proposal: after iteration s-1 has completed, INCLUDING its prof.step() profiler
                          transition, synchronize the local GPU, then a CPU barrier; iteration s starts after it.
                          With the default profile schedule, the barrier before step 6 runs after the warmup->active
                          transition (end of step 5), so steps 6, 7, 8 are aligned at their start and recorded.
  --align pre_collective  after the local gradients are GPU-ready (synchronize), immediately before reduce_grads(); for
                          zero1 also after the optimizer step (synchronize), immediately before the parameter all-gather

The control barrier runs on a dedicated CPU Gloo group (dist.new_group(backend="gloo")), so it adds no NCCL kernel to
the trace and no collective to FlatZeroAdamW's counters. Both ranks call it in identical order. A barrier makes both
ranks LEAVE it at about the same time; it does not make their next launches simultaneous. The residual is measured.

--inject-delay-ms D (control only): rank 1 sleeps D ms on the CPU after its gradients are GPU-ready and before the
target reduction (before the pre-collective barrier if there is one). Never inside the timed collective.

Raw output, one file per rank: rows_rank{r}.jsonl, one JSON object per optimizer step with CLOCK_MONOTONIC
nanosecond timestamps (time.perf_counter_ns, shared by all processes on one host):
  it_begin                     top of the iteration (before any prev_step barrier)
  prev_bar_in / prev_bar_out   prev_step barrier (after the local synchronize)
  t0                           after zero_grad + synchronize: forward/backward begins
  t1                           all K backwards done + synchronize: local gradients GPU-ready
  inject_begin / inject_end    injected delay (rank 1, control runs only)
  red_bar_in / red_bar_out     pre_collective barrier before the reduction
  red_disp0 / red_disp1        reduce_grads() called / returned (host enqueue; includes the div_ launch)
  red_done                     after synchronize: the reduction (+ div_) complete on this GPU
  opt_done                     optimizer step done + synchronize
  gat_bar_in / gat_bar_out     pre_collective barrier before the all-gather (zero1)
  gat_disp0 / gat_disp1 / gat_done   the all-gather, as above (zero1; flat_ar has none)
  sched_done                   scheduler.step() + synchronize
  prof_step0 / prof_step1      prof.step() (profiled runs; includes schedule transitions and the trace export)
  it_end                       end of the iteration
plus the CPU each rank was running on at it_begin and before the reduction (/proc/self/stat).

Profiler (--profile): CPU + CUDA activities only, no shapes/stacks/memory. schedule(skip_first=5, warmup=1,
active=3): step 5 is profiler warmup, steps 6-8 are recorded, the trace is exported inside prof.step() at the end of
step 8. record_function markers name every phase with its step: "## s6 TARGET reduce_scatter", "## s6 bar_reduce", ...

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day06_comm_sync/sync_diag.py \
      --impl zero1 --align pre_collective --out outputs/day06_comm_sync/try/jobs/x
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # same as Day 01 (deterministic cuBLAS)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function, schedule  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))  # Day 02 helpers (they add Day 01's folder)
sys.path.insert(0, str(HERE.parent / "day06_zero1"))
from ddp_manual import check_in_sync  # noqa: E402
from train import PRESETS, Config, attention_ctx, build, get_batch, lr_lambda, setup_reproducibility  # noqa: E402

from zero1 import FlatZeroAdamW  # noqa: E402

SKIP_FIRST, PROF_WARMUP, PROF_ACTIVE = 5, 1, 3  # steps 0-4 off, 5 profiler warmup, 6-8 recorded
ENV_KEYS = ("NCCL_", "TORCH_NCCL_", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")


def now() -> int:
    return time.perf_counter_ns()


def current_cpu() -> int:
    """The CPU this (main) thread last ran on: field 39 of /proc/self/stat."""
    try:
        with open("/proc/self/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[36])
    except (OSError, ValueError, IndexError):
        return -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["flat_ar", "zero1"], required=True)
    p.add_argument("--align", choices=["none", "prev_step", "pre_collective"], default="none")
    p.add_argument("--preset", choices=PRESETS, default="medium")
    p.add_argument("--global-batch", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--opt-steps", type=int, default=12)
    p.add_argument("--profile", action="store_true", help="CPU+CUDA torch.profiler, steps 6-8 recorded")
    p.add_argument("--inject-delay-ms", type=float, default=0.0, help="CONTROL: rank-1 CPU delay before the reduction")
    p.add_argument("--save-tensors", action="store_true", help="save final params + gathered AdamW state (tiny)")
    p.add_argument("--check-every-step", action="store_true", help="replica check after every step (tiny only)")
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    rank, world, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 2:
        raise SystemExit("this diagnostic is written for W = 2")
    if args.profile and args.opt_steps < SKIP_FIRST + PROF_WARMUP + PROF_ACTIVE + 1:
        raise SystemExit("--profile needs --opt-steps >= 10")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout), device_id=device)
    ctrl = dist.new_group(backend="gloo")  # CPU control group: barriers here launch no GPU kernel

    def sync() -> None:
        torch.cuda.synchronize(device)

    def barrier(row: dict, key: str) -> None:
        sync()  # this rank's GPU work is done before it announces arrival
        row[key + "_in"] = now()
        dist.barrier(group=ctrl)
        row[key + "_out"] = now()

    # ---------------------------------------------------------------- the workload (as train_zero1.py, flat impls)
    cfg = Config(**PRESETS[args.preset])
    cfg.total_steps = args.opt_steps
    G, m = args.global_batch, args.micro_batch
    K = G // (world * m)
    cfg.batch_size = G
    setup_reproducibility(cfg)
    torch.manual_seed(cfg.seed + rank)
    model, _, _ = build(cfg, device)
    names = [n for n, _ in model.named_parameters()]
    params = [q for _, q in model.named_parameters()]
    z = FlatZeroAdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, shard=args.impl == "zero1")
    scheduler = torch.optim.lr_scheduler.LambdaLR(z.optim, lambda s: lr_lambda(s, cfg))
    red_name = "reduce_scatter" if args.impl == "zero1" else "all_reduce"

    prof = None
    if args.profile:
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(skip_first=SKIP_FIRST, wait=0, warmup=PROF_WARMUP, active=PROF_ACTIVE,
                                         repeat=1),
                       on_trace_ready=lambda pr: pr.export_chrome_trace(str(out / f"trace_rank{rank}.json")))
        prof.start()

    rows, in_sync_steps = [], []
    for s in range(args.opt_steps):
        row = {"step": s, "rank": rank, "it_begin": now(), "cpu_begin": current_cpu()}
        if args.align == "prev_step":
            # Iteration s-1 is complete here, including its prof.step() transition; iteration s starts after this.
            with record_function(f"## s{s} bar_prev_step"):
                barrier(row, "prev_bar")
        G_s = G
        x, y = get_batch(replace(cfg, batch_size=G_s), s, device)
        rows_r = G_s // world
        x, y = x[rank * rows_r : (rank + 1) * rows_r], y[rank * rows_r : (rank + 1) * rows_r]
        xs, ys = x.split(m), y.split(m)
        model.train()
        z.zero_grad()
        sync()
        row["t0"] = now()
        for j in range(len(xs)):
            w = xs[j].size(0) / rows_r
            with record_function(f"## s{s} fwd {j}"), attention_ctx(cfg):
                logits = model(xs[j])
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ys[j].reshape(-1))
            del logits
            with record_function(f"## s{s} bwd {j}"):
                (loss * w).backward()
            del loss
        with record_function(f"## s{s} grads_ready_sync"):
            sync()
        row["t1"] = now()
        row["cpu_pre_reduce"] = current_cpu()
        if args.inject_delay_ms and rank == 1:
            with record_function(f"## s{s} INJECTED_DELAY"):
                row["inject_begin"] = now()
                time.sleep(args.inject_delay_ms / 1e3)  # CPU only, before the barrier and before the collective
                row["inject_end"] = now()
        if args.align == "pre_collective":
            with record_function(f"## s{s} bar_reduce"):
                barrier(row, "red_bar")
        with record_function(f"## s{s} TARGET {red_name}"):
            row["red_disp0"] = now()
            z.reduce_grads()
            row["red_disp1"] = now()
            sync()
            row["red_done"] = now()
        with record_function(f"## s{s} optimizer"):
            z.step_local()
            sync()
            row["opt_done"] = now()
        if args.impl == "zero1":
            if args.align == "pre_collective":
                with record_function(f"## s{s} bar_gather"):
                    barrier(row, "gat_bar")
            with record_function(f"## s{s} TARGET all_gather"):
                row["gat_disp0"] = now()
                z.gather_params()
                row["gat_disp1"] = now()
                sync()
                row["gat_done"] = now()
        with record_function(f"## s{s} scheduler"):
            scheduler.step()
            sync()
            row["sched_done"] = now()
        if args.check_every_step:  # NCCL collectives: tiny correctness runs only, never in profiled medium runs
            in_sync_steps.append([s + 1, bool(check_in_sync(params, f"params after step {s + 1}", rank, world))])
        if prof is not None:
            row["prof_step0"] = now()
            prof.step()
            row["prof_step1"] = now()
        row["it_end"] = now()
        rows.append(row)
    if prof is not None:
        prof.stop()

    # ---------------------------------------------------------------- outside every timed region
    in_sync_final = bool(check_in_sync(params, "final params", rank, world))
    full = z.full_state()  # gathers (evidence)
    with (out / f"rows_rank{rank}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"rank": rank, "args": vars(args), "K": K, "affinity": sorted(os.sched_getaffinity(0)),
            "env": {k: v for k, v in os.environ.items() if k.startswith(ENV_KEYS)},
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "nccl_runtime": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(device), "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "counters": dict(z.collectives), "in_sync_final": in_sync_final, "in_sync_steps": in_sync_steps,
            "lr_last": z.optim.param_groups[0]["lr"], "adam_step": full["step"],
            "profile_schedule": {"skip_first": SKIP_FIRST, "warmup": PROF_WARMUP, "active": PROF_ACTIVE,
                                 "recorded_steps": list(range(SKIP_FIRST + PROF_WARMUP,
                                                              SKIP_FIRST + PROF_WARMUP + PROF_ACTIVE))}
            if args.profile else None}
    (out / f"meta_rank{rank}.json").write_text(json.dumps(meta, indent=2))
    if rank == 0 and args.save_tensors:
        torch.save({"final": {k: q.detach().cpu().clone() for k, q in zip(names, params)},
                    "state_final": {k: {"exp_avg": a, "exp_avg_sq": b}
                                    for k, a, b in zip(names, full["exp_avg"], full["exp_avg_sq"])},
                    "adam_step": full["step"]}, out / "result.pt")
    dist.barrier(group=ctrl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
