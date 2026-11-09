"""Day 08 - DeepSpeed ZeRO harness: Day 04/07's accumulation loop with the model under DDP or a DeepSpeed engine.

  --impl single          ONE process, plain model + Day 01's AdamW: the big-batch reference (world must be 1)
  --impl ddp             torch DistributedDataParallel + no_sync() + AdamW: the cross-framework baseline (Day 07's)
  --impl ds              deepspeed.initialize(model, Day 01's AdamW, config) with --zero-stage 0 | 1 | 2 | 3
  --buckets sized|default  ds only: buckets sized for this model, or DeepSpeed's defaults (ds_setup.make_ds_config)
  --overlap auto|on|off  ds only: zero_optimization.overlap_comm (auto = DeepSpeed's default: on for stage 3 only)

Batch bookkeeping, data, loss weights and the short final window are Day 04's: G = W x K x m, microbatch j weighted
by rows_j / rows_in_window, one optimizer and scheduler step per window. The DeepSpeed side of that contract is
ds_setup.ds_microbatch (YOU WRITE): scale_wrt_gas=False and set_gradient_accumulation_boundary(last).

Negative controls (--bug, ds only), each caught by one check in compare_ds.py (guide 1.7):
  double_scale   engine.backward(loss * w) with DeepSpeed's default scale_wrt_gas=True: gradients divided by K twice
  fixed_gas      DeepSpeed's canonical loop (forward, backward, engine.step() every microbatch, boundary from the
                 config's gradient_accumulation_steps): the short final window never reaches a boundary

Per optimizer step, synchronized phase boundaries (identical for every impl):
  t0 | K x (forward, backward)  [stage 2/3: bucketed reductions and stage-3 gathers; stage 0/1 and DDP: the window's
     gradient reduction in the LAST backward] | t1 | engine.step() / optimizer.step() [stage 1/2: AdamW on the fp32
     partition + all-gather of the updated params; stage 3: + all-gather of the persistent params] | t2 | scheduler | t3

Memory ledger, PER RANK (same names as Day 07 so the tables line up):
  alloc_model          after Day 01's build(): the full model on every rank
  alloc_after_wrap     after DDP / deepspeed.initialize (ZeRO: flat buffers, fp32 master partition, stage-3 partitions)
  delta_first_step     memory_allocated() across the FIRST optimizer step (AdamW allocates its moments there)
  alloc_between_micro  after microbatch 0 of the first timed window (K >= 2)
  peak                 max_memory_allocated() over the steps after warmup
  timeline             microbatch 0 of the last warmup step, at every block boundary (Day 07's MemTimeline)
  census               ds only, last warmup step: every CUDA storage named by its owner (ds_setup.tensor_census),
                       at the start of the step and right before the optimizer

  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day08_deepspeed/train_ds.py \
      --impl ds --zero-stage 2 --preset medium --global-batch 64 --micro-batch 8 --no-tensors --tag try_z2
  python experiments/day08_deepspeed/compare_ds.py --stage all          # the orchestrated version
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # same as Day 01 (deterministic cuBLAS)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function, schedule  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))  # Day 02 helpers (which add Day 01's folder themselves)
sys.path.insert(0, str(HERE.parent / "day07_fsdp2"))
sys.path.insert(0, str(HERE))
from ddp_manual import barrier, check_in_sync, log0, require_sync, run_metadata  # noqa: E402
from train import PRESETS, Config, attention_ctx, build, get_batch, lr_lambda, setup_reproducibility  # noqa: E402
from train_fsdp import MemTimeline, alloc_mib  # noqa: E402  (Day 07's block-boundary memory timeline)

from ds_setup import (  # noqa: E402
    BUGS,
    adam_step,
    base_optimizer,
    ds_microbatch,
    full_grads,
    full_params,
    full_shape,
    full_state,
    make_ds_config,
    optimizer_state_report,
    quiet_deepspeed,
    shard_report,
    tensor_census,
)

OUT = HERE.parents[1] / "outputs" / "day08_deepspeed"
MiB = 2**20
IMPLS = ["single", "ddp", "ds"]


def init_dist(args) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (args.impl == "single") != (world == 1):
        raise SystemExit("--impl single is the 1-process reference; every other impl needs torchrun with >= 2 ranks")
    if args.bug != "none" and args.impl != "ds":
        raise SystemExit("--bug controls are DeepSpeed controls")
    if not args.tag or any(c in args.tag for c in "/\\"):
        raise SystemExit("--tag must be a plain folder name")
    if (OUT / args.tag / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"{OUT / args.tag} exists: use a new --tag or --overwrite")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        # Our own process group first; deepspeed.initialize finds it and reuses it (deepspeed.comm wraps torch.distributed)
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout), device_id=device)
    return rank, world, device


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
    ds = args.impl == "ds"
    stage = args.zero_stage if ds else None

    setup_reproducibility(cfg)
    torch.manual_seed(cfg.seed + rank)  # different init per rank on purpose: the wrapper must broadcast (check C)
    model, optimizer, scheduler = build(cfg, device)  # Day 01's model, AdamW(foreach=False, fused=False), LambdaLR
    if args.foreach:
        del optimizer, scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=True,
                                      fused=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    alloc_model = alloc_mib(device)
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]

    # ---------------------------------------------------------------- wrap
    net, no_sync, engine, ds_config = model, nullcontext, None, None
    t_init = time.perf_counter()
    if args.impl == "ddp":
        net = DDP(model, device_ids=[device.index], bucket_cap_mb=args.bucket_mb)  # broadcasts rank 0's params
        no_sync = net.no_sync
    elif ds:
        import deepspeed  # imported here: the 1-process reference and DDP runs do not pay its import time

        quiet_deepspeed()
        overlap = {"auto": None, "on": True, "off": False}[args.overlap]
        ds_config = make_ds_config(stage, m, K, world, args.buckets, overlap, args.persist_threshold)
        # The engine (1) broadcasts rank 0's parameters (engine._broadcast_model), (2) flattens/partitions them for
        # ZeRO and REWRITES the client optimizer's param_groups to hold fp32 partitions (unlike FSDP2, the optimizer
        # may be built before wrapping), (3) registers gradient hooks (stage >= 1) and module hooks (stage 3).
        # The LR scheduler is NOT handed to DeepSpeed: we step it ourselves once per window, after t2 (Day 04's rule).
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
        net = engine
    init_s = time.perf_counter() - t_init
    alloc_after_wrap = alloc_mib(device)

    syncs = world > 1
    # R: parameters every rank holds in full. Stage 3 partitions all but the "persistent" (small) ones, and even those
    # are only materialised at their first fetch (before that p.data is empty), so the list is taken when checked.
    def replicated_now() -> list[torch.Tensor]:
        return [p.data for p in params if (not hasattr(p, "ds_tensor") or getattr(p, "ds_persist", False))
                and p.data.numel() == torch.Size(full_shape(p)).numel()]

    if syncs and replicated_now():
        require_sync(check_in_sync(replicated_now(), "initial params", rank, world), args, "initial params differ")
    log0(rank, f"world={world} impl={args.impl}" + (f" stage={stage} buckets={args.buckets} overlap={args.overlap}"
                                                    if ds else "")
               + f" bug={args.bug} preset={args.preset} | G={G} = W {world} x K {K} x m {m}"
               + (f" | last window G={GL}" if GL != G else "")
               + f" | {sum(torch.Size(full_shape(p)).numel() for p in params):,} parameters in {len(params)} tensors"
               + f" | foreach={args.foreach} | wrap {init_s:.1f} s")
    if ds and rank == 0:
        print("ds zero_optimization:", json.dumps(ds_config["zero_optimization"]), flush=True)

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
    ledger = {"alloc_model_mib": alloc_model, "alloc_after_wrap_mib": alloc_after_wrap, "wrap_s": init_s}
    census = {}
    grads0, params1, grads_last = {}, {}, {}

    def sync_dev():
        torch.cuda.synchronize(device)

    def loss_fn(logits, y):
        return F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

    def grads_now() -> dict:
        if ds:
            if not engine.is_gradient_accumulation_boundary():
                # Only reachable with --bug fixed_gas: DeepSpeed never reduced this window, so there is no averaged
                # gradient to read (its helpers would raise on some ranks inside a collective and hang the others).
                return {n: torch.zeros(full_shape(p)) for n, p in zip(names, params)}
            return full_grads(engine, names, params, stage)
        return {n: p.grad.detach().cpu().clone() for n, p in zip(names, params)}

    def capture(s: int) -> float:
        """Parity runs: the averaged gradients of window s, read right after its LAST backward and before any
        engine.step() (ZeRO frees them in the step). Returns the seconds it took, so the timer can exclude them."""
        nonlocal grads0, grads_last
        if not keep or s not in (0, args.opt_steps - 1):
            return 0.0
        sync_dev()
        c0 = time.perf_counter()
        g = grads_now()
        if s == 0:
            grads0 = g
        if s == args.opt_steps - 1:
            grads_last = g
        sync_dev()
        return time.perf_counter() - c0

    def params_now() -> dict:
        if ds:
            return full_params(engine, names, params, stage)
        return {n: p.detach().cpu().clone() for n, p in zip(names, params)}

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
        if not ds:
            optimizer.zero_grad(set_to_none=True)  # DeepSpeed zeroes its own gradients inside engine.step()
        snap = args.memory_snapshot and s == tl_step and rank == 0
        if snap:
            torch.cuda.memory._record_memory_history(max_entries=200_000, stacks="python")
        if ds and s == tl_step and args.census:
            census["start_of_step"] = tensor_census(engine, model, device)

        sync_dev()
        t0 = time.perf_counter()
        window_loss = torch.zeros((), device=device)
        t_cap = 0.0
        for j in range(n):
            last = j == n - 1  # from the ACTUAL window length (Day 04 rule 3)
            w = xs[j].size(0) / rows  # this microbatch's share of the window
            tl.on = s == tl_step and j == 0
            if tl.on:
                sync_dev()
                torch.cuda.reset_peak_memory_stats(device)
                tl.rec("start    microbatch 0")
            if ds:
                @contextmanager
                def phase(kind, j=j):
                    with record_function(f"## {kind} {j}"), (attention_ctx(cfg) if kind == "fwd" else nullcontext()):
                        yield
                    if kind == "fwd":
                        tl.rec("loss     computed")

                loss = ds_microbatch(engine, loss_fn, xs[j], ys[j], w, last, args.bug, phase)
                if last:
                    t_cap = capture(s)  # before any engine.step() of this window (ZeRO frees gradients in step)
                if args.bug == "fixed_gas":
                    engine.step()  # the canonical DeepSpeed loop: step() every microbatch, boundary by counter
            else:
                with (nullcontext() if last else no_sync()):  # DDP decides in FORWARD (Day 04)
                    with record_function(f"## fwd {j}"), attention_ctx(cfg):
                        logits = net(xs[j])
                    loss = loss_fn(logits, ys[j])
                    del logits
                    tl.rec("loss     computed")
                    with record_function(f"## bwd {j}"):
                        (loss * w).backward()
                loss = loss.detach()
                if last:
                    t_cap = capture(s)
            tl.rec("bwd_done (all hooks run; reductions may still be in flight)")
            tl.on = False
            window_loss += loss * w
            del loss
            if s == warm and j == 0 and n > 1:
                ledger["alloc_between_micro_mib"] = alloc_mib(device)  # after microbatch 0, before microbatch 1
        sync_dev()
        t1 = time.perf_counter() - t_cap  # evidence capture (parity runs only) is not training time
        lr = optimizer.param_groups[0]["lr"]  # the client optimizer's groups ARE the ZeRO optimizer's groups
        if ds and s == tl_step and args.census:
            census["before_optimizer"] = tensor_census(engine, model, device)
        a0 = alloc_mib(device) if s == 0 else 0.0
        tl.on = s == tl_step
        tl.rec("opt_pre  (window done, grads reduced)")
        with record_function("## optimizer"):
            if ds:
                if args.bug != "fixed_gas":
                    engine.step()  # boundary set by ds_microbatch: AdamW on the partition (+ all-gather), zero_grad
            else:
                optimizer.step()
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
            params1 = params_now()
        if syncs and replicated_now() and (s == 0 or (s + 1) % args.check_every == 0 or s + 1 == args.opt_steps):
            ok = check_in_sync(replicated_now(), f"params after optimizer step {s + 1}", rank, world)
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
    replicated = replicated_now()
    in_sync_final = check_in_sync(replicated, "final params", rank, world) if (syncs and replicated) else True

    # ---------------------------------------------------------------- per-rank report: shards, state, memory
    if ds:
        rep = {**shard_report(params, engine, stage), **optimizer_state_report(engine)}
        a_step = adam_step(engine)
    else:
        rep = {"full_numel": sum(p.numel() for p in params), "local_numel": sum(p.numel() for p in params),
               "persistent_numel": 0, "master_numel": 0,
               "state_bytes": {k: sum(st[k].numel() * 4 for st in optimizer.state.values())
                               for k in ("exp_avg", "exp_avg_sq")} if optimizer.state else {}}
        a_step = float(optimizer.state[params[0]]["step"]) if params[0] in optimizer.state else 0.0
    report = {**rep, **ledger, "rank": rank, "census": census,
              "peak_mib": torch.cuda.max_memory_allocated(device) / MiB, "timeline": tl.events}
    per_rank = [None] * world
    if world > 1:
        dist.all_gather_object(per_rank, report)
    else:
        per_rank = [report]
    for r in per_rank if rank == 0 else []:
        sb = r["state_bytes"]
        print(f"[rank {r['rank']}] keeps {r['local_numel']:,} of {r['full_numel']:,} parameter elements "
              f"({r['persistent_numel']:,} persistent) + {r['master_numel']:,} fp32 master | exp_avg "
              f"{sb.get('exp_avg', 0) / MiB:.2f} MiB, exp_avg_sq {sb.get('exp_avg_sq', 0) / MiB:.2f} MiB | model "
              f"{r['alloc_model_mib']:.1f}, after wrap {r['alloc_after_wrap_mib']:.1f}, delta first step "
              f"{r['delta_first_step_mib']:.2f}, between microbatches {r.get('alloc_between_micro_mib', float('nan')):.1f},"
              f" peak {r['peak_mib']:.1f} MiB", flush=True)
    if rank == 0 and census:
        for when, c in census.items():
            print(f"[rank 0 census, {when}] " + " | ".join(f"{k} {v:.1f}" for k, v in c.items()), flush=True)

    # gathered evidence (collectives for ZeRO: every rank calls)
    if keep:
        state_final = full_state(engine, names, params, stage) if ds else {
            n: {k: optimizer.state[p][k].detach().cpu().clone() for k in ("exp_avg", "exp_avg_sq")}
            for n, p in zip(names, params) if p in optimizer.state}
        final = params_now()
    else:
        state_final, final = {}, {}

    # ---------------------------------------------------------------- slowest-rank timing (Days 02-07 statistics)
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
        meta.update(lr0=cfg.lr * lr_lambda(0, cfg), n_params=sum(torch.Size(full_shape(p)).numel() for p in params),
                    n_tensors=len(params), impl=args.impl, zero_stage=stage, ds_config=ds_config, accum=K,
                    micro_batch=m, global_batch=G, last_global_batch=GL, n_micro_per_window=n_micro, lr_used=lr_used,
                    adam_step=[a_step], scheduler_last_epoch=scheduler.last_epoch, in_sync_steps=in_sync_steps,
                    in_sync_final=bool(in_sync_final), replicated_tensors=len(replicated),
                    ds_global_steps=getattr(engine, "global_steps", None), per_rank=per_rank)
        if ds:
            import deepspeed
            meta["deepspeed_version"] = deepspeed.__version__
        tensors = {"grads0": grads0, "params1": params1, "grads_last": grads_last, "state_final": state_final,
                   "final": final} if keep else {}
        torch.save({**tensors, "losses": loss_t.cpu().tolist(), "timing": timing, "meta": meta}, out / "result.pt")
        (out / "metrics.json").write_text(json.dumps({"timing": timing, "meta": meta}, indent=2, default=str))
        print(f"wrote {out / 'result.pt'}", flush=True)

    if world > 1:
        barrier(device)
        if engine is not None:
            engine.destroy()  # removes DeepSpeed's hooks and releases its buffers before the process group goes away
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=IMPLS, required=True)
    p.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--buckets", choices=["sized", "default"], default="sized")
    p.add_argument("--overlap", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--persist-threshold", type=int, default=None,
                   help="stage 3: stage3_param_persistence_threshold (None = DeepSpeed's 1e5)")
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
    p.add_argument("--census", action="store_true", help="ds: name every CUDA storage in the last warmup step")
    p.add_argument("--profile", action="store_true", help="write torch.profiler chrome traces (one per rank)")
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--memory-snapshot", action="store_true",
                   help="rank 0: dump torch.cuda.memory._snapshot() of the last warmup step (view at pytorch.org/memory_viz)")
    p.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    p.add_argument("--device", choices=["cuda"], default="cuda")
    p.add_argument("--output-root", help="write results somewhere other than outputs/day08_deepspeed")
    p.add_argument("--tag", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--timeout", type=int, default=180)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
