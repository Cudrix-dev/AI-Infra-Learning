"""Day 03 - one training harness, four ways to synchronize gradients.

  --impl none        no gradient communication at all. NOT training (ranks diverge): the timing lower bound
                     "what if communication were free and no rank ever waited for another"
  --impl flat_sync   Day 02: backward finishes, THEN one flat all_reduce (synchronous, no overlap)
  --impl mini_ddp    our MiniDDP (mini_ddp.py): hooks + buckets + async all_reduce overlapped with backward
                     --bucket-mb 0 = one all_reduce per parameter, launched from its hook
  --impl torch_ddp   torch.nn.parallel.DistributedDataParallel

Same Day 01 model / data / AdamW, same strong-scaling sharding as Day 02 (rank r takes rows [r*b, (r+1)*b) of the
global batch), same evidence saved for the parity check (synced grads at step 0, params after one update, losses).

Timing: only the CORE step is timed, with torch.cuda.synchronize() at its two ends. Day 02 also synchronized between
phases; with overlap that would be wrong (a synchronize after backward also waits for the in-flight all_reduces and
silently moves communication into the "backward" number). The overhead of a mode is measured as
core(mode) - core(none) in the same repeat; where that time goes is read off the profiler (--profile).

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day03_ddp_hooks/train_ddp.py \
      --device cuda --impl mini_ddp --bucket-mb 25 --tag try_mini
  python experiments/day03_ddp_hooks/compare_ddp.py --stage all          # the orchestrated version
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
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

from mini_ddp import MiniDDP  # noqa: E402
from optimizer_graph import prepare_optimizer  # noqa: E402

OUT = HERE.parents[1] / "outputs" / "day03_ddp_hooks"
MiB = 2**20


def init_dist(args) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.impl in ("mini_ddp", "torch_ddp") and world < 2:
        raise SystemExit(f"--impl {args.impl} needs a process group: launch with torchrun (>= 2 processes)")
    if args.no_comm and args.impl not in ("mini_ddp", "torch_ddp"):
        raise SystemExit("--no-comm is an ablation of the bucketed wrappers: use it with mini_ddp or torch_ddp")
    if args.bug != "none" and args.impl != "mini_ddp":
        raise SystemExit("--bug is a negative control of MiniDDP: use it with --impl mini_ddp")
    if (args.grad_as_view or args.rebuild or args.first_bucket_mb is not None) and args.impl not in ("mini_ddp", "torch_ddp"):
        raise SystemExit("--grad-as-view / --rebuild / --first-bucket-mb only apply to the bucketed wrappers")
    if args.rebuild and args.impl == "torch_ddp":
        raise SystemExit("torch_ddp always rebuilds its buckets after the first iteration; --rebuild is for mini_ddp")
    if not args.tag or any(c in args.tag for c in "/\\"):
        raise SystemExit("--tag must be a plain folder name")
    if (OUT / args.tag / "result.pt").exists() and not args.overwrite:
        raise SystemExit(f"{OUT / args.tag / 'result.pt'} exists: use a new --tag or --overwrite")
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
    """Return (net, finish): net is what forward() is called on, finish() completes the gradient sync."""
    params = [p for p in model.parameters()]
    if args.impl == "none" or world == 1:
        return model, lambda: None
    if args.impl == "flat_sync":
        ns = SimpleNamespace(bug="none", sync="flat", allreduce="builtin")  # Day 02's sync_grads, unchanged
        broadcast_params(model)
        return model, lambda: sync_grads(params, world, ns)
    if args.impl == "mini_ddp":
        net = MiniDDP(model, bucket_cap_mb=args.bucket_mb if args.bucket_mb is not None else 25.0,
                      first_bucket_mb=args.first_bucket_mb, grad_as_view=args.grad_as_view,
                      rebuild=args.rebuild, bug=args.bug, comm=not args.no_comm)
        return net, net.finish_grad_sync
    # torch_ddp. bucket_cap_mb=None keeps DDP's defaults (25 MiB, 1 MiB first bucket after the rebuild).
    # DDP's finalize_backward runs at the end of backward by itself, so there is nothing to finish.
    net = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, bucket_cap_mb=args.bucket_mb,
              gradient_as_bucket_view=args.grad_as_view, forward_sync_buffers=bool(args.forward_sync_buffers))
    if args.no_comm:
        # Timing ablation: a comm hook that returns the bucket untouched. The Reducer still copies grads into buckets,
        # launches "reductions" in order, waits and copies back, but no all_reduce is issued. (With a comm hook the
        # Reducer also skips its 1/world scaling: see Note [DDP Communication Hook] in reducer.cpp.)
        def noop_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut
        net.register_comm_hook(None, noop_hook)
    return net, lambda: None


def run(args) -> None:
    global OUT
    if args.output_root:
        OUT = Path(args.output_root).resolve()
    rank, world, device = init_dist(args)
    cfg = Config(**PRESETS[args.preset])
    cfg.total_steps = args.steps
    cfg.batch_size = args.global_batch
    if cfg.batch_size % world:
        raise SystemExit(f"global batch {cfg.batch_size} not divisible by world {world}")
    local_bs = cfg.batch_size // world

    setup_reproducibility(cfg)
    torch.manual_seed(cfg.seed + rank)  # different init per rank on purpose (Day 02): the wrapper must broadcast
    model, optimizer, scheduler = build(cfg, device)
    named = dict(model.named_parameters())
    params = list(named.values())
    names = list(named.keys())
    net, finish = wrap(model, args, world, device)
    optimizer_update = None
    graph_bootstrap = None
    if args.optimizer_mode != "original":
        if args.grad_as_view:
            # Exercise hooks and the one-time rebuild without taking optimizer
            # updates. The last two backwards must use the same view addresses.
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            addresses = []
            for _ in range(3):
                model.zero_grad(set_to_none=False)
                x, y = get_batch(cfg, 0, device)
                x, y = x[rank * local_bs:(rank + 1) * local_bs], y[rank * local_bs:(rank + 1) * local_bs]
                with attention_ctx(cfg):
                    logits = net(x)
                    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                loss.backward()
                finish()
                torch.cuda.synchronize(device)
                addresses.append([p.grad.data_ptr() for p in params])
                del x, y, logits, loss
            if addresses[-1] != addresses[-2]:
                raise RuntimeError("bucket-view addresses have not stabilized after rebuild")
            if not all(p.grad._base is not None for p in params):
                raise RuntimeError("expected gradient tensors to remain bucket views")
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            graph_bootstrap = {"backwards": 3, "optimizer_updates": 0,
                               "last_two_addresses_equal": True, "all_grads_are_views": True}
        optimizer, scheduler, optimizer_update = prepare_optimizer(
            model, cfg, device, args.optimizer_mode, lr_lambda, preserve_grad_views=args.grad_as_view)

    syncs = args.impl != "none" and not args.no_comm  # do ranks stay identical in this mode?
    if args.impl != "none":
        require_sync(check_in_sync(params, "initial params", rank, world), args, "initial params differ")
    layout0 = net.initial_layout if isinstance(net, MiniDDP) else None
    log0(rank, f"world={world} impl={args.impl} bucket_mb={args.bucket_mb} first_bucket_mb={args.first_bucket_mb} "
               f"grad_as_view={args.grad_as_view} rebuild={args.rebuild} bug={args.bug} no_comm={args.no_comm} "
               f"preset={args.preset} "
               f"global batch {cfg.batch_size} = {world} x {local_bs}")
    if layout0:
        log0(rank, f"MiniDDP initial buckets (reverse registration order): {len(layout0)} buckets, MiB "
                   f"{[round(b['mib'], 3) for b in layout0]}")

    warm = min(args.warmup, args.steps - 1)
    prof = None
    if args.profile:
        # profile steps [p0, p0 + profile_steps); p0 > warmup so the bucket rebuild and allocator growth are done
        p0 = warm + 1
        if args.steps < p0 + args.profile_steps + 1:
            raise SystemExit(f"--profile needs --steps >= {p0 + args.profile_steps + 1}")
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        # cpu,cuda records every CPU op (~5k per medium step) and perturbs CPU submission timing; cuda records only
        # CUPTI GPU activity, so ProfilerStep/record_function ranges and CPU launch events are absent from the trace.
        acts = [ProfilerActivity.CUDA] if args.profile_activities == "cuda" else [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        prof = profile(activities=acts,
                       schedule=schedule(skip_first=p0 - 1, wait=0, warmup=1, active=args.profile_steps, repeat=1),
                       on_trace_ready=lambda p: p.export_chrome_trace(str(out / f"trace_rank{rank}.json")))
        prof.start()

    losses, grads0, params1, rows = [], {}, {}, []
    lr_used, boundaries = [], []

    def sync_dev():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for step in range(args.steps):
        if step == warm and device.type == "cuda":
            sync_dev()
            torch.cuda.reset_peak_memory_stats(device)
        sync_dev()
        L0 = time.perf_counter()
        x, y = get_batch(cfg, step, device)
        x, y = x[rank * local_bs : (rank + 1) * local_bs], y[rank * local_bs : (rank + 1) * local_bs]
        net.train()
        # With grad_as_view the grads ARE bucket memory: zero them in place. set_to_none would drop the views, and the
        # next backward would allocate fresh grads and copy them in again (correct, but the memory saving is gone).
        optimizer.zero_grad(set_to_none=not args.grad_as_view and args.optimizer_mode == "original")

        sync_dev()
        t0 = time.perf_counter()
        opt_cpu_start = None
        with record_function("## forward"), attention_ctx(cfg):
            logits = net(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        del logits
        with record_function("## backward"):
            loss.backward()  # mini_ddp / torch_ddp: all_reduces are launched from inside this call
        with record_function("## finish_grad_sync"):
            finish()
        with record_function("## optimizer"):
            opt_cpu_start = time.perf_counter()
            # Copy asynchronously for correctness evidence without a new CPU/GPU sync.
            lr = optimizer.param_groups[0]["lr"]
            lr_used.append(lr.detach().clone() if isinstance(lr, torch.Tensor) else lr)
            if optimizer_update is None:
                optimizer.step()
                scheduler.step()
            else:
                optimizer_update()
            opt_cpu_end = time.perf_counter()
        sync_dev()
        t3 = time.perf_counter()
        boundaries.append({"step": step, "core_start": t0, "optimizer_cpu_start": opt_cpu_start,
                           "optimizer_cpu_end": opt_cpu_end, "core_end": t3})

        losses.append(loss.detach())
        if step == 0:
            grads0 = {n: p.grad.detach().cpu().clone() for n, p in named.items()}
            params1 = {n: p.detach().cpu().clone() for n, p in named.items()}
            if world > 1 and syncs:
                ok = check_in_sync([p.grad for p in params], "gradients after sync (step 0)", rank, world)
                require_sync(ok, args, "averaged gradients differ across ranks")
        if syncs and (step == 0 or (step + 1) % args.check_every == 0 or step + 1 == args.steps):
            require_sync(check_in_sync(params, f"params after step {step + 1}", rank, world), args,
                         f"params differ after step {step + 1}")
        if (step + 1) % args.log_every == 0:
            log0(rank, f"step {step + 1:4d} | rank-0 local loss {loss.item():.4f}")
        sync_dev()
        L1 = time.perf_counter()
        if step >= warm:
            rows.append([t3 - t0, L1 - L0])
        if prof is not None:
            prof.step()
    if prof is not None:
        prof.stop()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / f"boundaries_rank{rank}.json").write_text(json.dumps(boundaries, indent=2))

    # ---- global loss curve, slowest-rank timing (same statistics as Day 02)
    loss_t = torch.stack(losses).float()
    tm = torch.tensor(rows, dtype=torch.float64, device=device)
    peak = torch.tensor([torch.cuda.max_memory_allocated(device) / MiB if device.type == "cuda" else 0.0],
                        dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= world
        dist.all_reduce(tm, op=dist.ReduceOp.MAX)  # slowest rank per (step, column)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    tm_l = tm.cpu().tolist()
    core = statistics.median(r[0] for r in tm_l) * 1e3
    timing = {"core_step_ms": core, "core_tokens_per_s": cfg.batch_size * cfg.seq_len / (core / 1e3),
              "loop_step_ms_mean": statistics.fmean(r[1] for r in tm_l) * 1e3, "timed_steps": len(tm_l),
              "global_batch": cfg.batch_size, "world": world, "peak_alloc_mib_max_rank": peak.item(),
              "profiled": bool(args.profile)}
    log0(rank, f"core step {core:.2f} ms median over {len(tm_l)} steps (slowest rank per step) | "
               f"{timing['core_tokens_per_s']:,.0f} tok/s | peak alloc {peak.item():.0f} MiB"
               + ("  [PROFILED RUN: not a benchmark]" if args.profile else ""))

    ready_names = None
    if isinstance(net, MiniDDP):
        ready_names = [names[i] for i in net.ready_order]

    if rank == 0:
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        meta = run_metadata(args, device, world, dist.get_backend() if world > 1 else "-", len(tm_l), warm)
        meta.update(lr0=cfg.lr * lr_lambda(0, cfg), n_params=sum(p.numel() for p in params),
                    impl=args.impl, mini_initial_layout=layout0,
                    mini_final_layout=net.layout() if isinstance(net, MiniDDP) else None,
                    mini_ready_order=ready_names)
        meta.update(optimizer_mode=args.optimizer_mode,
                    graph_bootstrap=graph_bootstrap,
                    lr_used=[v.item() if isinstance(v, torch.Tensor) else v for v in lr_used],
                    adam_steps=[v["step"].item() for v in optimizer.state.values()],
                    scheduler_last_epoch=scheduler.last_epoch,
                    persistent_grads=args.optimizer_mode != "original" or args.grad_as_view)
        tensors = {} if args.no_tensors else {  # medium: 3 x 105 MiB per run, so timing runs skip them
            "grads0": grads0, "params1": params1, "final": {n: p.detach().cpu().clone() for n, p in named.items()}}
        torch.save({**tensors, "losses": loss_t.cpu().tolist(), "timing": timing, "meta": meta}, out / "result.pt")
        (out / "metrics.json").write_text(json.dumps({"timing": timing, "meta": meta}, indent=2, default=str))
        print(f"wrote {out / 'result.pt'}", flush=True)

    if world > 1:
        barrier(device)
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["none", "flat_sync", "mini_ddp", "torch_ddp"], required=True)
    p.add_argument("--optimizer-mode", choices=["original", "ready_eager", "graph"], default="original")
    p.add_argument("--output-root", help="separate output directory for the CUDA Graph follow-up")
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--global-batch", type=int, default=16, help="strong scaling: fixed global batch (Day 02: 16)")
    p.add_argument("--bucket-mb", type=float, default=None,
                   help="bucket cap in MiB (mini_ddp default 25; 0 = one bucket per parameter). torch_ddp: None keeps "
                        "DDP's defaults, a value sets bucket_cap_mb")
    p.add_argument("--first-bucket-mb", type=float, default=None, help="mini_ddp: separate cap for bucket 0 (DDP: 1)")
    p.add_argument("--rebuild", action="store_true", help="mini_ddp: rebuild buckets in the recorded ready order "
                                                          "after the first backward (what DDP does)")
    p.add_argument("--grad-as-view", action="store_true", help="p.grad becomes a view of the bucket (mini_ddp, "
                                                               "torch_ddp gradient_as_bucket_view)")
    p.add_argument("--forward-sync-buffers", type=int, choices=[0, 1], default=1,
                   help="torch_ddp: broadcast buffers from rank 0 before every forward (DDP default 1; torch 2.14 renamed "
                        "broadcast_buffers to forward_sync_buffers). Buffers are always synced once at construction")
    p.add_argument("--bug", choices=["none", "early_launch", "no_wait"], default="none",
                   help="MiniDDP negative controls (see mini_ddp.py)")
    p.add_argument("--no-comm", action="store_true", help="TIMING ABLATION (mini_ddp, torch_ddp): hooks and copies "
                                                          "but no all_reduce (not training: ranks diverge, checks skipped)")
    p.add_argument("--no-tensors", action="store_true", help="save timing/metadata only, not the gradient/parameter "
                                                             "tensors (a medium result.pt is ~330 MB otherwise)")
    p.add_argument("--warmup", type=int, default=3, help="steps excluded from timing and peak memory")
    p.add_argument("--profile", action="store_true", help="write torch.profiler chrome traces (one per rank)")
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--profile-activities", choices=["cpu,cuda", "cuda"], default="cpu,cuda",
                   help="what --profile records; cuda omits CPU-op tracing (lower host overhead)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    p.add_argument("--tag", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
