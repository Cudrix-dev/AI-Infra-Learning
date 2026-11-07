"""Day 07 - FSDP2 harness: Day 04/06's accumulation loop with the model under DDP or FSDP2 (`fully_shard`).

  --impl single          ONE process, plain model + Day 01's AdamW: the big-batch reference (world must be 1)
  --impl ddp             DistributedDataParallel + no_sync() + AdamW on every rank: the replicated baseline
  --impl fsdp_root       fully_shard(model) only: one FSDP group holding every parameter
  --impl fsdp_block      fully_shard(each block), then fully_shard(model): n_layers + 1 groups
  --impl fsdp_block_nrf  the same, blocks with reshard_after_forward=False (no re-gather in backward)
  --impl fsdp_block_pf   fsdp_block + explicit forward prefetch (each block launches the next block's all-gather)
  --accum sync|nosync    FSDP only, K >= 2: reduce-scatter every microbatch (FSDP2 default) or only the last one
                         (set_requires_gradient_sync(False) before the others). DDP always uses no_sync (Day 04).

Batch bookkeeping, data, loss weights and the short final window are Day 04's: G = W x K x m, microbatch j weighted
by rows_j / rows_in_window, one optimizer and scheduler step per window.

Negative controls (--bug), each caught by one check in compare_fsdp.py (guide 1.7):
  no_init_bcast  skip the parameter broadcast before fully_shard (ranks are seeded differently on purpose): the
                 sharded model is a patchwork of rank 0's and rank 1's initialisations
  sync_never     set_requires_gradient_sync(False) for EVERY microbatch: no reduce-scatter ever runs, the sharded
                 .grad stays None and AdamW skips every parameter
  sum_grads      set_gradient_divide_factor(1.0): the reduced gradient is the SUM over ranks, not the mean

Per optimizer step, synchronized phase boundaries (identical for every impl):
  t0 | K x (forward, backward)  [all FSDP all-gathers / reduce-scatters, all DDP all-reduces] | t1 | optimizer | t2
     | scheduler | t3
FSDP's communication runs on its own streams INSIDE [t0, t1]; the final callback of the last backward makes the
compute stream wait for the last reduce-scatter, so t1 is after every gradient is reduced.

Memory ledger, PER RANK:
  alloc_model          after Day 01's build(): the full model on every rank (before any wrapper)
  alloc_after_wrap     after DDP / fully_shard and the optimizer rebuild (FSDP: only this rank's shards remain)
  delta_first_step     memory_allocated() across the FIRST optimizer step (AdamW allocates its moments there)
  alloc_between_micro  after microbatch 0 of the first timed window (K >= 2): what stays allocated between microbatches
  peak                 max_memory_allocated() over the steps after warmup
  timeline             one walk through microbatch 0 of the last warmup step: memory_allocated() and the peak since
                       the previous event at every block boundary (forward pre/post, backward pre/post), the loss,
                       the end of backward and the optimizer step. This is where the FSDP sawtooth shows.

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day07_fsdp2/train_fsdp.py \
      --impl fsdp_block --preset medium --global-batch 64 --micro-batch 8 --no-tensors --tag try_block
  python experiments/day07_fsdp2/compare_fsdp.py --stage all          # the orchestrated version
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # same as Day 01 (deterministic cuBLAS)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function, schedule  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))  # Day 02 helpers (which add Day 01's folder themselves)
sys.path.insert(0, str(HERE))
from ddp_manual import barrier, check_in_sync, log0, require_sync, run_metadata  # noqa: E402
from train import PRESETS, Config, attention_ctx, build, get_batch, lr_lambda, setup_reproducibility  # noqa: E402

from fsdp_setup import (  # noqa: E402
    apply_fully_shard,
    broadcast_module_state,
    optimizer_state_report,
    set_microbatch_sync,
    shard_report,
    sum_instead_of_average,
    to_full,
)

OUT = HERE.parents[1] / "outputs" / "day07_fsdp2"
MiB = 2**20
FSDP_IMPLS = {"fsdp_root": "root", "fsdp_block": "block", "fsdp_block_nrf": "block_nrf", "fsdp_block_pf": "block_pf"}
IMPLS = ["single", "ddp", *FSDP_IMPLS]
BUGS = ["none", "no_init_bcast", "sync_never", "sum_grads"]


def init_dist(args) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (args.impl == "single") != (world == 1):
        raise SystemExit("--impl single is the 1-process reference; every other impl needs torchrun with >= 2 ranks")
    if args.bug != "none" and args.impl not in FSDP_IMPLS:
        raise SystemExit("--bug controls are FSDP controls")
    if not args.tag or any(c in args.tag for c in "/\\"):
        raise SystemExit("--tag must be a plain folder name")
    if (OUT / args.tag / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"{OUT / args.tag} exists: use a new --tag or --overwrite")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout), device_id=device)
    return rank, world, device


def alloc_mib(device: torch.device, sync: bool = True) -> float:
    if sync:
        torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / MiB


class MemTimeline:
    """memory_allocated() and the peak since the previous event, at block boundaries, for ONE microbatch.

    The allocator's counters are updated on the HOST when an op is enqueued (allocation) or its tensor dies (free), so
    reading them inside hooks needs no device synchronization. Hooks run after FSDP's own hooks on the same module:
    FSDP's forward pre-hook is registered with prepend=True and its post-hook first, and its pre-backward hook is the
    first tensor hook on the output. So 'fwd_pre' sees the block already gathered, 'fwd_post' already resharded,
    'bwd_pre' already re-gathered, and 'bwd_post' (grad w.r.t. the block INPUT is ready) the block's unsharded
    gradients before FSDP's post-backward reduce-scatters and frees them."""

    def __init__(self, device: torch.device):
        self.device, self.events, self.handles, self.on = device, [], [], False

    def rec(self, what: str) -> None:
        if not self.on:
            return
        self.events.append([what, torch.cuda.memory_allocated(self.device) / MiB,
                            torch.cuda.max_memory_allocated(self.device) / MiB])
        torch.cuda.reset_peak_memory_stats(self.device)

    def attach(self, model: torch.nn.Module) -> None:
        def pre(name):
            def hook(_m, args):
                self.rec(f"fwd_pre  {name}")
                if self.on and args and torch.is_tensor(args[0]) and args[0].requires_grad:
                    args[0].register_hook(lambda g: self.rec(f"bwd_post {name}"))
            return hook

        def post(name):
            def hook(_m, _args, out):
                self.rec(f"fwd_post {name}")
                if self.on and torch.is_tensor(out) and out.requires_grad:
                    out.register_hook(lambda g: self.rec(f"bwd_pre  {name}"))
            return hook

        for i, blk in enumerate(model.blocks):
            self.handles += [blk.register_forward_pre_hook(pre(f"blocks.{i}")),
                             blk.register_forward_hook(post(f"blocks.{i}"))]
        self.handles += [model.register_forward_pre_hook(pre("model")), model.register_forward_hook(post("model"))]

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def run(args) -> None:
    global OUT
    if args.output_root:
        OUT = Path(args.output_root).resolve()
    rank, world, device = init_dist(args)
    cfg = Config(**PRESETS[args.preset])
    cfg.total_steps = args.opt_steps  # the LR schedule counts OPTIMIZER steps (Day 04)
    G, m = args.global_batch, args.micro_batch
    GL = args.last_global_batch or G
    for g in (G, GL):
        if g % world:
            raise SystemExit(f"global batch {g} not divisible by world {world}")
    if (G // world) % m:
        raise SystemExit(f"rows per rank {G // world} not divisible by --micro-batch {m}")
    K = G // (world * m)
    cfg.batch_size = G
    fsdp = args.impl in FSDP_IMPLS

    setup_reproducibility(cfg)
    torch.manual_seed(cfg.seed + rank)  # different init per rank on purpose: the wrapper (or we) must broadcast
    model, optimizer, scheduler = build(cfg, device)  # Day 01's model, AdamW(foreach=False, fused=False), LambdaLR
    alloc_model = alloc_mib(device)

    # ---------------------------------------------------------------- wrap
    net, no_sync, groups = model, nullcontext, []
    if args.impl == "ddp":
        net = DDP(model, device_ids=[device.index], bucket_cap_mb=args.bucket_mb)  # broadcasts rank 0's params
        no_sync = net.no_sync
    elif fsdp:
        # Day 01's optimizer holds references to the ORIGINAL full parameters: drop it, or they never get freed.
        del optimizer, scheduler
        if args.bug != "no_init_bcast":
            broadcast_module_state(model)  # FSDP2 shards whatever each rank holds: make it rank 0's model first
        groups = apply_fully_shard(model, FSDP_IMPLS[args.impl])
        if args.bug == "sum_grads":
            sum_instead_of_average(model)
    if fsdp or args.foreach:
        # Same hyper-parameters as Day 01's build(), over the (now sharded) parameters. For FSDP these are DTensors:
        # AdamW's elementwise ops dispatch to each rank's local shard, so the state is sharded too.
        if not fsdp:
            del optimizer, scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                      foreach=args.foreach, fused=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    names = [n for n, _ in model.named_parameters()]  # FSDP2 keeps the original names (no flat parameter)
    params = [p for _, p in model.named_parameters()]  # collected AFTER wrapping: DTensors for FSDP
    alloc_after_wrap = alloc_mib(device)

    syncs = world > 1
    replicated = [p for p in params if not isinstance(p, DTensor)]  # what R can check: FSDP has no replicas
    if syncs and replicated:
        require_sync(check_in_sync(replicated, "initial params", rank, world), args, "initial params differ")
    log0(rank, f"world={world} impl={args.impl} accum={args.accum} bug={args.bug} preset={args.preset} | G={G} = W "
               f"{world} x K {K} x m {m}" + (f" | last window G={GL}" if GL != G else "")
               + f" | {sum(p.numel() for p in params):,} parameters in {len(params)} tensors | foreach={args.foreach}"
               + (f" | FSDP groups ({len(groups)}): {groups}" if groups else ""))

    warm = min(args.warmup, args.opt_steps - 1)
    prof = None
    if args.profile:
        p0 = warm + 1
        if args.opt_steps < p0 + args.profile_steps + 1:
            raise SystemExit(f"--profile needs --opt-steps >= {p0 + args.profile_steps + 1}")
        (OUT / args.tag).mkdir(parents=True, exist_ok=True)
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(skip_first=p0 - 1, wait=0, warmup=1, active=args.profile_steps, repeat=1),
                       on_trace_ready=lambda p: p.export_chrome_trace(str(OUT / args.tag / f"trace_rank{rank}.json")))
        prof.start()

    keep = not args.no_tensors
    tl = MemTimeline(device)
    tl.attach(model)
    tl_step = warm - 1 if warm >= 1 else None  # the last warmup step: excluded from timing and from `peak`
    window_losses, lr_used, n_micro, rows_t, in_sync_steps = [], [], [], [], []
    ledger = {"alloc_model_mib": alloc_model, "alloc_after_wrap_mib": alloc_after_wrap}
    grads0, params1, grads_last = {}, {}, {}

    def sync_dev():
        torch.cuda.synchronize(device)

    def synced_grads() -> dict:
        """The averaged gradient the optimizer used, per parameter name (an all-gather per DTensor; outside t0..t3)."""
        return {n: to_full(p.grad, p) for n, p in zip(names, params)}

    for s in range(args.opt_steps):
        if s == warm:
            sync_dev()
            torch.cuda.reset_peak_memory_stats(device)
        G_s = GL if s == args.opt_steps - 1 else G
        x, y = get_batch(replace(cfg, batch_size=G_s), s, device)  # the GLOBAL batch of optimizer step s
        rows = G_s // world
        x, y = x[rank * rows : (rank + 1) * rows], y[rank * rows : (rank + 1) * rows]
        xs, ys = x.split(m), y.split(m)
        n = len(xs)
        net.train()
        optimizer.zero_grad(set_to_none=True)  # FSDP: sharded .grad -> None; no views to protect (unlike Day 06)
        snap = args.memory_snapshot and s == tl_step and rank == 0
        if snap:
            torch.cuda.memory._record_memory_history(max_entries=200_000, stacks="python")

        sync_dev()
        t0 = time.perf_counter()
        window_loss = torch.zeros((), device=device)
        for j in range(n):
            last = j == n - 1  # from the ACTUAL window length (Day 04 rule 3)
            w = xs[j].size(0) / rows  # this microbatch's share of the window
            tl.on = s == tl_step and j == 0
            if tl.on:
                sync_dev()
                torch.cuda.reset_peak_memory_stats(device)
                tl.rec("start    microbatch 0")
            if fsdp:
                set_microbatch_sync(model, last, args.accum, args.bug)
            with (nullcontext() if last else no_sync()):  # DDP decides in FORWARD (Day 04); no-op for FSDP
                with record_function(f"## fwd {j}"), attention_ctx(cfg):
                    logits = net(xs[j])
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ys[j].reshape(-1))
                del logits
                tl.rec("loss     computed")
                with record_function(f"## bwd {j}"):
                    (loss * w).backward()
            tl.rec("bwd_done (all hooks run; FSDP RS may still be in flight)")
            tl.on = False
            window_loss += loss.detach() * w
            del loss
            if s == warm and j == 0 and n > 1:
                ledger["alloc_between_micro_mib"] = alloc_mib(device)  # after microbatch 0, before microbatch 1
        sync_dev()
        t1 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        a0 = alloc_mib(device) if s == 0 else 0.0
        tl.on = s == tl_step
        tl.rec("opt_pre  (window done, grads reduced)")
        with record_function("## optimizer"):
            optimizer.step()  # FSDP: AdamW on this rank's shards only (DTensor ops on local tensors)
        sync_dev()
        t2 = time.perf_counter()
        tl.rec("opt_post")
        tl.on = False
        if s == 0:
            ledger["delta_first_step_mib"] = alloc_mib(device) - a0
        if snap:
            (OUT / args.tag).mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(OUT / args.tag / "memory_snapshot_rank0.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        scheduler.step()
        sync_dev()
        t3 = time.perf_counter()

        window_losses.append(window_loss)
        lr_used.append(float(lr))
        n_micro.append(n)
        if keep and s == 0:
            grads0 = synced_grads()
            params1 = {k: to_full(p, p) for k, p in zip(names, params)}
        if keep and s == args.opt_steps - 1:
            grads_last = synced_grads()
        if syncs and replicated and (s == 0 or (s + 1) % args.check_every == 0 or s + 1 == args.opt_steps):
            ok = check_in_sync(replicated, f"params after optimizer step {s + 1}", rank, world)
            in_sync_steps.append([s + 1, bool(ok)])
            require_sync(ok, args, f"params differ after optimizer step {s + 1}")
        if (s + 1) % args.log_every == 0:
            log0(rank, f"opt step {s + 1:4d} | rank-0 window loss {window_loss.item():.4f} | lr {lr:.3e}")
        if s >= warm and n == K:
            rows_t.append([t1 - t0, t2 - t1, t3 - t2, t3 - t0])
        if prof is not None:
            prof.step()
    if prof is not None:
        prof.stop()
    tl.detach()
    in_sync_final = check_in_sync(replicated, "final params", rank, world) if (syncs and replicated) else True

    # ---------------------------------------------------------------- per-rank report: shards, state, memory
    report = {**shard_report(params), **optimizer_state_report(optimizer), **ledger, "rank": rank,
              "peak_mib": torch.cuda.max_memory_allocated(device) / MiB, "timeline": tl.events}
    per_rank = [None] * world
    if world > 1:
        dist.all_gather_object(per_rank, report)
    else:
        per_rank = [report]
    for r in per_rank if rank == 0 else []:
        sb = r["state_bytes"]
        print(f"[rank {r['rank']}] stores {r['local_numel']:,} of {r['full_numel']:,} parameter elements "
              f"({r['dtensor_params']} DTensor / {r['plain_params']} plain tensors) | exp_avg {sb.get('exp_avg', 0) / MiB:.2f}"
              f" MiB, exp_avg_sq {sb.get('exp_avg_sq', 0) / MiB:.2f} MiB | model {r['alloc_model_mib']:.1f}, after wrap "
              f"{r['alloc_after_wrap_mib']:.1f}, delta first step {r['delta_first_step_mib']:.2f}, between microbatches "
              f"{r.get('alloc_between_micro_mib', float('nan')):.1f}, peak {r['peak_mib']:.1f} MiB", flush=True)

    # gathered state (collective for DTensors: every rank calls to_full)
    state_final = {k: {"exp_avg": to_full(optimizer.state[p]["exp_avg"], p),
                       "exp_avg_sq": to_full(optimizer.state[p]["exp_avg_sq"], p)}
                   for k, p in zip(names, params) if p in optimizer.state} if keep else {}
    adam_step = float(optimizer.state[params[0]]["step"]) if params[0] in optimizer.state else 0.0
    final = {k: to_full(p, p) for k, p in zip(names, params)} if keep else {}

    # ---------------------------------------------------------------- slowest-rank timing (Days 02-06 statistics)
    loss_t = torch.stack(window_losses).float()
    tm = torch.tensor(rows_t or [[float("nan")] * 4], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= world
        dist.all_reduce(tm, op=dist.ReduceOp.MAX)  # slowest rank per (step, column)
    med = [statistics.median(c) * 1e3 for c in zip(*tm.cpu().tolist())]
    timing = {"core_step_ms": med[3], "fwd_bwd_ms": med[0], "optim_ms": med[1], "sched_ms": med[2],
              "core_ms_per_micro": med[3] / K, "timed_steps": len(rows_t),
              "core_tokens_per_s": G * cfg.seq_len / (med[3] / 1e3), "global_batch": G, "micro_batch": m,
              "accum": K, "world": world, "profiled": bool(args.profile),
              "peak_alloc_mib_max_rank": max(r["peak_mib"] for r in per_rank)}
    log0(rank, f"core {med[3]:.2f} ms per optimizer step (fwd+bwd incl. comm {med[0]:.2f} | optimizer {med[1]:.2f} | "
               f"sched {med[2]:.2f}), median over {len(rows_t)} steps, slowest rank"
               + ("  [PROFILED RUN: not a benchmark]" if args.profile else ""))

    if rank == 0:
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        meta = run_metadata(args, device, world, dist.get_backend() if world > 1 else "-", len(rows_t), warm)
        meta.update(lr0=cfg.lr * lr_lambda(0, cfg), n_params=sum(p.numel() for p in params), n_tensors=len(params),
                    impl=args.impl, fsdp_accum=args.accum if fsdp else None, fsdp_groups=groups, accum=K,
                    micro_batch=m, global_batch=G, last_global_batch=GL, n_micro_per_window=n_micro, lr_used=lr_used,
                    adam_step=[adam_step], scheduler_last_epoch=scheduler.last_epoch, in_sync_steps=in_sync_steps,
                    in_sync_final=bool(in_sync_final), replicated_tensors=len(replicated), counters=None,
                    per_rank=per_rank)
        tensors = {"grads0": grads0, "params1": params1, "grads_last": grads_last, "state_final": state_final,
                   "final": final} if keep else {}
        torch.save({**tensors, "losses": loss_t.cpu().tolist(), "timing": timing, "meta": meta}, out / "result.pt")
        (out / "metrics.json").write_text(json.dumps({"timing": timing, "meta": meta}, indent=2, default=str))
        print(f"wrote {out / 'result.pt'}", flush=True)

    if world > 1:
        barrier(device)
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=IMPLS, required=True)
    p.add_argument("--accum", choices=["sync", "nosync"], default="sync", help="FSDP gradient sync per microbatch")
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--global-batch", type=int, default=16, help="G: rows per optimizer step, over all ranks")
    p.add_argument("--micro-batch", type=int, default=8, help="m: rows per microbatch per rank (K = G / (W*m))")
    p.add_argument("--last-global-batch", type=int, default=0, help="rows of the FINAL window (0 = G)")
    p.add_argument("--opt-steps", type=int, default=20, help="optimizer steps (= windows)")
    p.add_argument("--warmup", type=int, default=3, help="optimizer steps excluded from timing and peak memory")
    p.add_argument("--foreach", action="store_true", help="AdamW(foreach=True) instead of Day 01's per-tensor loop")
    p.add_argument("--bucket-mb", type=float, default=None, help="DDP bucket_cap_mb (None = DDP's default 25)")
    p.add_argument("--bug", choices=BUGS, default="none", help="negative controls (module docstring)")
    p.add_argument("--no-tensors", action="store_true", help="save timing/metadata only (timing runs)")
    p.add_argument("--profile", action="store_true", help="write torch.profiler chrome traces (one per rank)")
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--memory-snapshot", action="store_true",
                   help="rank 0: dump torch.cuda.memory._snapshot() of the last warmup step (view at pytorch.org/memory_viz)")
    p.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--output-root", help="write results somewhere other than outputs/day07_fsdp2")
    p.add_argument("--tag", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
