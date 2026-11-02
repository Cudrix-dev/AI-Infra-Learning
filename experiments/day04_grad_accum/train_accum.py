"""Day 04 - gradient accumulation: K microbatches per optimizer step, with and without a gradient sync per microbatch.

  --impl none        local accumulation, no communication at all. NOT training (ranks diverge): the timing floor
  --impl flat_sync   Day 02's flat all_reduce (sync_grads), called after each microbatch that should communicate
  --impl mini_ddp    Day 03's MiniDDP + no_sync() (mini_ddp_accum.py); 25 MiB buckets, 1 MiB first, rebuild
  --impl torch_ddp   DistributedDataParallel and its no_sync()

  --sync nosync           communicate only on the LAST microbatch of a window (the others run inside no_sync())
  --sync every            communicate after EVERY microbatch: correct, but K times the collectives
  --sync bwd_only_nosync  CONTROL (mini_ddp, torch_ddp): no_sync() wraps only loss.backward(), not the forward.
                          The hooks were armed by the forward, so every microbatch still communicates

Batch-size bookkeeping (the relation to write down today):
  m = --micro-batch (rows per microbatch per rank), W = world size, K = microbatches per optimizer step (derived)
  global batch G = W * K * m          per-GPU batch per optimizer step = K * m          optimizer steps = windows
  K is derived from --global-batch: K = G / (W * m). The 1-process reference uses --micro-batch G, so K = 1.

Data: optimizer step s uses Day 01's global batch get_batch(seed, s) with G_s rows (G, or --last-global-batch for the
final window). Rank r takes rows [r*G_s/W, (r+1)*G_s/W) (Day 02's contiguous slices) and splits them into
microbatches of m rows; a short final window has fewer microbatches (and possibly a smaller last one).

Loss scaling: microbatch j's mean loss is multiplied by (rows_j / rows of this rank in this window). Summed over the
window that is this rank's mean loss; DDP's 1/W average then gives the mean over the global batch. For equal
microbatches the factor is 1/K, but NOT in a short final window (that is the `fixed_k_scale` bug below).

Negative controls (--bug), each caught by a specific check in compare_accum.py:
  no_loss_scale     backward of the unscaled microbatch loss: gradients K x too large       (check A)
  zero_every_micro  zero_grad inside the microbatch loop: only the last microbatch survives (check A)
  sched_per_micro   scheduler.step() after every microbatch: LR schedule runs K x too fast (check D)
  fixed_k_scale     scale by 1/K even in the short final window                             (check E)
  modulo_last_sync  "sync when j == K-1": a short final window never syncs; ranks diverge   (check E)

Timing: the CORE of one optimizer step (all K forward/backward passes, the syncs, optimizer + scheduler) with
torch.cuda.synchronize() at its two ends only. Batch generation and zero_grad are outside. Slowest rank per step,
median over full windows after warmup.

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day04_grad_accum/train_accum.py \
      --impl torch_ddp --sync nosync --preset medium --global-batch 64 --micro-batch 8 --tag try_ddp_k4
  python experiments/day04_grad_accum/compare_accum.py --stage all          # the orchestrated version
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # same as Day 01 (deterministic cuBLAS)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function, schedule  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))  # Day 02 helpers (which add Day 01's folder themselves)
sys.path.insert(0, str(HERE))
from ddp_manual import barrier, broadcast_params, check_in_sync, log0, require_sync, run_metadata, sync_grads  # noqa: E402
from train import PRESETS, Config, attention_ctx, build, get_batch, lr_lambda, setup_reproducibility  # noqa: E402

from mini_ddp_accum import MiniDDPAccum  # noqa: E402

OUT = HERE.parents[1] / "outputs" / "day04_grad_accum"
MiB = 2**20
BUGS = ["none", "no_loss_scale", "zero_every_micro", "sched_per_micro", "fixed_k_scale", "modulo_last_sync"]


def init_dist(args) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.impl in ("mini_ddp", "torch_ddp") and world < 2:
        raise SystemExit(f"--impl {args.impl} needs a process group: launch with torchrun (>= 2 processes)")
    if args.sync == "bwd_only_nosync" and args.impl not in ("mini_ddp", "torch_ddp"):
        raise SystemExit("--sync bwd_only_nosync is a control of no_sync(): use it with mini_ddp or torch_ddp")
    if args.grad_as_view and args.impl not in ("mini_ddp", "torch_ddp"):
        raise SystemExit("--grad-as-view only applies to the bucketed wrappers")
    if not args.tag or any(c in args.tag for c in "/\\"):
        raise SystemExit("--tag must be a plain folder name")
    if (OUT / args.tag / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"{OUT / args.tag} exists: use a new --tag or --overwrite")
    if args.device == "cuda":
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(f"LOCAL_RANK {local_rank} but only {torch.cuda.device_count()} visible GPUs")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        kw = {"device_id": device} if backend == "nccl" else {}
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=args.timeout), **kw)
    return rank, world, device


def wrap(model, args, world: int, device: torch.device):
    """Return (net, no_sync, finish).

    net(x)          what forward() is called on
    no_sync()       context manager that makes the NEXT forward/backward accumulate locally (nullcontext if n/a)
    finish(comm)    called after EVERY backward; comm=True means "this microbatch must communicate"
    """
    params = list(model.parameters())
    if args.impl == "none" or world == 1:
        return model, nullcontext, lambda comm: None
    if args.impl == "flat_sync":
        # Day 02's sync_grads, unchanged. There are no hooks, so "no_sync" is simply "do not call sync_grads".
        ns = SimpleNamespace(bug="none", sync="flat", allreduce="builtin")
        broadcast_params(model)
        counter = {"collectives": 0}

        def finish(comm: bool) -> None:
            if comm:
                sync_grads(params, world, ns)
                counter["collectives"] += 1

        model.flat_counter = counter  # read back for metrics
        return model, nullcontext, finish
    if args.impl == "mini_ddp":
        net = MiniDDPAccum(model, bucket_cap_mb=args.bucket_mb if args.bucket_mb is not None else 25.0,
                           first_bucket_mb=args.first_bucket_mb, grad_as_view=args.grad_as_view,
                           rebuild=not args.no_rebuild)
        # MiniDDPAccum.finish_grad_sync is a no-op after a no_sync forward: just call it after every backward.
        return net, net.no_sync, lambda comm: net.finish_grad_sync()
    # torch_ddp: bucket_cap_mb=None keeps DDP's defaults (25 MiB, 1 MiB first bucket after the rebuild).
    # Its finalize_backward runs at the end of every synced backward by itself, so finish() has nothing to do.
    net = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, bucket_cap_mb=args.bucket_mb,
              gradient_as_bucket_view=args.grad_as_view)
    return net, net.no_sync, lambda comm: None


def run(args) -> None:
    global OUT
    if args.output_root:
        OUT = Path(args.output_root).resolve()
    rank, world, device = init_dist(args)
    cfg = Config(**PRESETS[args.preset])
    cfg.total_steps = args.opt_steps  # the LR schedule is defined in OPTIMIZER steps, not microbatches
    G, m = args.global_batch, args.micro_batch
    GL = args.last_global_batch or G
    for g in (G, GL):
        if g % world:
            raise SystemExit(f"global batch {g} not divisible by world {world}")
    if (G // world) % m:
        raise SystemExit(f"rows per rank {G // world} not divisible by --micro-batch {m}: full windows must be "
                         "K equal microbatches")
    K = G // (world * m)  # microbatches per full window
    if GL > G:
        raise SystemExit("--last-global-batch must not exceed --global-batch (it models a SHORT final window)")
    cfg.batch_size = G

    setup_reproducibility(cfg)
    torch.manual_seed(cfg.seed + rank)  # different init per rank on purpose (Day 02): the wrapper must broadcast
    model, optimizer, scheduler = build(cfg, device)
    named = dict(model.named_parameters())
    params = list(named.values())
    net, no_sync, finish = wrap(model, args, world, device)
    syncs = args.impl != "none" and world > 1  # do ranks stay identical in this mode?
    if syncs:
        require_sync(check_in_sync(params, "initial params", rank, world), args, "initial params differ")
    log0(rank, f"world={world} impl={args.impl} sync={args.sync} bug={args.bug} preset={args.preset} | "
               f"G={G} = W {world} x K {K} x m {m} (per-GPU batch per optimizer step {K * m})"
               + (f" | last window: G={GL} -> {math.ceil(GL // world / m)} microbatches" if GL != G else ""))

    warm = min(args.warmup, args.opt_steps - 1)
    prof = None
    if args.profile:
        p0 = warm + 1  # after the bucket rebuild and allocator growth
        if args.opt_steps < p0 + args.profile_steps + 1:
            raise SystemExit(f"--profile needs --opt-steps >= {p0 + args.profile_steps + 1}")
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(skip_first=p0 - 1, wait=0, warmup=1, active=args.profile_steps, repeat=1),
                       on_trace_ready=lambda p: p.export_chrome_trace(str(out / f"trace_rank{rank}.json")))
        prof.start()

    keep = not args.no_tensors
    window_losses, lr_used, n_micro, rows_t = [], [], [], []
    grads0, params1, grads_last = {}, {}, {}
    zero_to_none = not args.grad_as_view  # bucket views must be zeroed in place (Day 03)

    def sync_dev():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for s in range(args.opt_steps):
        if s == warm and device.type == "cuda":
            sync_dev()
            torch.cuda.reset_peak_memory_stats(device)
        sync_dev()
        L0 = time.perf_counter()
        G_s = GL if s == args.opt_steps - 1 else G
        x, y = get_batch(replace(cfg, batch_size=G_s), s, device)  # the GLOBAL batch of optimizer step s
        rows = G_s // world
        x, y = x[rank * rows : (rank + 1) * rows], y[rank * rows : (rank + 1) * rows]
        xs, ys = x.split(m), y.split(m)  # microbatches; a short final window has fewer (or a smaller last one)
        n = len(xs)
        net.train()
        optimizer.zero_grad(set_to_none=zero_to_none)  # ONCE per window: accumulation means not zeroing in between

        sync_dev()
        t0 = time.perf_counter()
        window_loss = torch.zeros((), device=device)
        for j in range(n):
            if args.bug == "zero_every_micro":
                optimizer.zero_grad(set_to_none=zero_to_none)  # BUG: erases microbatches 0..j-1
            # Is this the microbatch whose backward must communicate? Decide from the ACTUAL window length n.
            last = (j == K - 1) if args.bug == "modulo_last_sync" else (j == n - 1)  # BUG: j == K-1 never
            comm = last or args.sync == "every"                                     # happens in a short window
            w = xs[j].size(0) / rows  # this microbatch's share of the window (1/K for equal microbatches)
            w_bwd = {"no_loss_scale": 1.0, "fixed_k_scale": 1.0 / K}.get(args.bug, w)
            fwd_ctx = no_sync() if (not comm and args.sync == "nosync") else nullcontext()
            bwd_ctx = no_sync() if (not comm and args.sync == "bwd_only_nosync") else nullcontext()
            with fwd_ctx:  # DDP/MiniDDP decide in FORWARD whether the coming backward communicates
                with record_function(f"## fwd {j}"), attention_ctx(cfg):
                    logits = net(xs[j])
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ys[j].reshape(-1))
                del logits
                with bwd_ctx, record_function(f"## bwd {j}"):
                    (loss * w_bwd).backward()  # AccumulateGrad: p.grad += this microbatch's (scaled) gradient
            with record_function(f"## finish {j}"):
                finish(comm)
            window_loss += loss.detach() * w  # logged loss always uses the correct weights
            if args.bug == "sched_per_micro":
                scheduler.step()  # BUG: the schedule counts microbatches instead of optimizer steps
        with record_function("## optimizer"):
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()  # ONE update per window
            if args.bug != "sched_per_micro":
                scheduler.step()  # ONE schedule step per window
        sync_dev()
        t3 = time.perf_counter()

        window_losses.append(window_loss)
        lr_used.append(float(lr))
        n_micro.append(n)
        if s == 0 and keep:
            grads0 = {k: p.grad.detach().cpu().clone() for k, p in named.items()}
            params1 = {k: p.detach().cpu().clone() for k, p in named.items()}
        if s == args.opt_steps - 1 and keep:
            grads_last = {k: p.grad.detach().cpu().clone() for k, p in named.items()}
        if s == 0 and syncs:
            ok = check_in_sync([p.grad for p in params], "gradients after sync (step 0)", rank, world)
            require_sync(ok, args, "averaged gradients differ across ranks")
        if syncs and (s == 0 or (s + 1) % args.check_every == 0 or s + 1 == args.opt_steps):
            require_sync(check_in_sync(params, f"params after optimizer step {s + 1}", rank, world), args,
                         f"params differ after optimizer step {s + 1}")
        if (s + 1) % args.log_every == 0:
            log0(rank, f"opt step {s + 1:4d} | rank-0 window loss {window_loss.item():.4f} | lr {lr:.3e}")
        sync_dev()
        L1 = time.perf_counter()
        if s >= warm and n == K:  # full windows after warmup only
            rows_t.append([t3 - t0, L1 - L0])
        if prof is not None:
            prof.step()  # one profiler "step" = one optimizer step (K microbatches)
    if prof is not None:
        prof.stop()

    # final cross-rank consistency, recorded (the bugs above may legitimately break it)
    in_sync_final = check_in_sync(params, "final params", rank, world) if world > 1 else True

    # ---- global loss curve, slowest-rank timing (same statistics as Days 02/03)
    loss_t = torch.stack(window_losses).float()
    tm = torch.tensor(rows_t or [[float("nan"), float("nan")]], dtype=torch.float64, device=device)
    peak = torch.tensor([torch.cuda.max_memory_allocated(device) / MiB if device.type == "cuda" else 0.0],
                        dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= world
        dist.all_reduce(tm, op=dist.ReduceOp.MAX)  # slowest rank per (step, column)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    tm_l = tm.cpu().tolist()
    core = statistics.median(r[0] for r in tm_l) * 1e3
    tokens = G * cfg.seq_len
    timing = {"core_step_ms": core, "core_ms_per_micro": core / K, "core_tokens_per_s": tokens / (core / 1e3),
              "loop_step_ms_mean": statistics.fmean(r[1] for r in tm_l) * 1e3, "timed_steps": len(rows_t),
              "global_batch": G, "micro_batch": m, "accum": K, "world": world, "tokens_per_opt_step": tokens,
              "peak_alloc_mib_max_rank": peak.item(), "profiled": bool(args.profile)}
    log0(rank, f"core {core:.2f} ms per optimizer step ({core / K:.2f} ms per microbatch), median over "
               f"{len(rows_t)} steps, slowest rank | {timing['core_tokens_per_s']:,.0f} tok/s | peak alloc "
               f"{peak.item():.0f} MiB" + ("  [PROFILED RUN: not a benchmark]" if args.profile else ""))

    # in-process collective counters (flat, MiniDDP); DDP's are counted from the profiler trace
    counters = None
    if hasattr(model, "flat_counter"):
        counters = {"collectives": model.flat_counter["collectives"]}
    elif isinstance(net, MiniDDPAccum):
        counters = {"collectives": net.collectives, "hook_calls_armed": net.hook_calls_armed,
                    "hook_calls_skipped": net.hook_calls_skipped, "n_buckets": len(net.buckets)}
    if counters:
        counters["per_opt_step"] = {k: v / args.opt_steps for k, v in counters.items() if k != "n_buckets"}
        log0(rank, f"counters over {args.opt_steps} optimizer steps: {counters}")

    if rank == 0:
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        meta = run_metadata(args, device, world, dist.get_backend() if world > 1 else "-", len(rows_t), warm)
        meta.update(lr0=cfg.lr * lr_lambda(0, cfg), n_params=sum(p.numel() for p in params), impl=args.impl,
                    accum=K, micro_batch=m, global_batch=G, last_global_batch=GL, n_micro_per_window=n_micro,
                    lr_used=lr_used, adam_step=[float(v["step"]) for v in optimizer.state.values()][:1],
                    scheduler_last_epoch=scheduler.last_epoch, in_sync_final=bool(in_sync_final), counters=counters)
        tensors = {"grads0": grads0, "params1": params1, "grads_last": grads_last,
                   "final": {k: p.detach().cpu().clone() for k, p in named.items()}} if keep else {}
        torch.save({**tensors, "losses": loss_t.cpu().tolist(), "timing": timing, "meta": meta}, out / "result.pt")
        (out / "metrics.json").write_text(json.dumps({"timing": timing, "meta": meta}, indent=2, default=str))
        print(f"wrote {out / 'result.pt'}", flush=True)

    if world > 1:
        barrier(device)
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["none", "flat_sync", "mini_ddp", "torch_ddp"], required=True)
    p.add_argument("--sync", choices=["nosync", "every", "bwd_only_nosync"], default="nosync")
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--global-batch", type=int, default=16, help="G: rows per optimizer step, over all ranks")
    p.add_argument("--micro-batch", type=int, default=8, help="m: rows per microbatch per rank (K = G / (W*m))")
    p.add_argument("--last-global-batch", type=int, default=0, help="rows of the FINAL window (0 = G): a short "
                                                                    "final window, e.g. the end of an epoch")
    p.add_argument("--opt-steps", type=int, default=20, help="optimizer steps (= windows)")
    p.add_argument("--warmup", type=int, default=3, help="optimizer steps excluded from timing and peak memory")
    p.add_argument("--bucket-mb", type=float, default=None, help="mini_ddp default 25; torch_ddp None = defaults")
    p.add_argument("--first-bucket-mb", type=float, default=1.0, help="mini_ddp: cap of bucket 0 (DDP: 1 MiB)")
    p.add_argument("--no-rebuild", action="store_true", help="mini_ddp: keep the reverse-registration layout")
    p.add_argument("--grad-as-view", action="store_true", help="p.grad is a view of the bucket (both wrappers)")
    p.add_argument("--bug", choices=BUGS, default="none", help="negative controls (see the module docstring)")
    p.add_argument("--no-tensors", action="store_true", help="save timing/metadata only (timing runs)")
    p.add_argument("--profile", action="store_true", help="write torch.profiler chrome traces (one per rank)")
    p.add_argument("--profile-steps", type=int, default=2, help="optimizer steps in the trace")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    p.add_argument("--output-root", help="write results somewhere other than outputs/day04_grad_accum")
    p.add_argument("--tag", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
