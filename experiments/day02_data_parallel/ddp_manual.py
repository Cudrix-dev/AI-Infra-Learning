"""Day 02 - Data parallel WITHOUT DDP: broadcast, shard, all_reduce, step.

Same model / data / optimizer as Day 01 (imported from day01_single_gpu/train.py). What is new:

  1. every rank starts with a DIFFERENT random init, then rank 0's params are BROADCAST to all
  2. each rank trains on its own shard of the (identical) global batch
  3. after backward(), gradients are all_reduce'd (SUM) and divided by world size
  4. every rank runs the same optimizer step on the same averaged gradient -> params stay identical

Run (from the repo root; NCCL backend, one process per GPU):
  python experiments/day02_data_parallel/ddp_manual.py --device cuda --tag ref_1gpu          # 1 GPU = reference
  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/ddp_manual.py --device cuda --tag dp2
  python experiments/day02_data_parallel/check_parity.py --device cuda                       # runs both and compares
CPU / gloo works too (--device cpu).

Rules for the collectives in this file (break one and you get a HANG, not an error):
  * every rank IN THE GROUP calls the same collectives, in the same order, with the same shapes/dtypes
    (here the group is the default group = all ranks; a sub-group involves only its own members)
  * never put a collective inside `if rank == 0:`
  * never raise on one rank only when others will go on to a collective; agree first (all_reduce a flag)
  * tensors given to a collective must live on the backend's device (CUDA for NCCL)

What is timed (see summarize_timing): per step, each phase is measured on every rank, the slowest rank is taken PER STEP,
and only then the median over steps. "core" = forward+backward+gradient sync+optimizer step. "loop" = the whole
iteration, which also includes batch generation/transfer, zero_grad, consistency checks and logging. Setup, the final
cross-rank reductions and saving are in neither. The two are different statistics (median vs mean): do not subtract them.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # same as Day 01 (matters on GPU)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day01_single_gpu"))  # reuse Day 01's model, data, config
from train import PRESETS, Config, attention_ctx, build, env_info, get_batch, lr_lambda, setup_reproducibility  # noqa: E402

from ring_allreduce import ring_all_reduce  # noqa: E402  (same folder as this script)

OUT = HERE.parents[1] / "outputs" / "day02_data_parallel"
MiB = 2**20


# ----------------------------------------------------------------------------- arguments
def validate_args(args, world: int) -> None:
    """Fail early with a clear message instead of an empty median, a modulo by zero or a meaningless experiment."""
    for name in ("steps", "check_every", "log_every", "timeout"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.global_batch < 0 or args.per_gpu_batch < 0:
        raise SystemExit("batch sizes must be positive")
    if args.global_batch and args.per_gpu_batch:
        raise SystemExit("use either --per-gpu-batch (weak scaling) or --global-batch (strong scaling), not both")
    if args.bug != "none" and world < 2:
        raise SystemExit(f"--bug {args.bug} is a negative control for data parallelism: it needs at least 2 ranks")
    if not args.tag or any(c in args.tag for c in "/\\") or args.tag in (".", ".."):
        raise SystemExit("--tag must be a plain folder name")
    if (OUT / args.tag / "result.pt").exists() and not args.overwrite:
        # All ranks evaluate the same condition before any collective, so they all exit together.
        raise SystemExit(f"{OUT / args.tag / 'result.pt'} already exists: use a new --tag (keeps evidence) or --overwrite")


# ----------------------------------------------------------------------------- process group
def init_dist(args) -> tuple[int, int, torch.device, str]:
    """torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT; plain `python` sets none (-> world 1)."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    validate_args(args, world)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but no CUDA device is available")
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(f"LOCAL_RANK {local_rank} but only {torch.cuda.device_count()} visible GPUs: "
                             "lower --nproc_per_node or fix CUDA_VISIBLE_DEVICES")
        device = torch.device("cuda", local_rank)  # one process per GPU; torchrun's LOCAL_RANK picks the GPU
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    backend = args.backend or ("nccl" if device.type == "cuda" else "gloo")
    if backend == "nccl" and device.type != "cuda":
        raise SystemExit("--backend nccl needs --device cuda")

    if world > 1:
        # env:// rendezvous: read MASTER_ADDR/PORT, rank 0 hosts a TCPStore, everyone connects and waits.
        # The timeout turns a deadlock (mismatched collectives, a broken transport) into an error, not a silent hang.
        kw = {"device_id": device} if backend == "nccl" else {}  # bind this rank to its GPU up front
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=args.timeout), **kw)
    return rank, world, device, backend


def log0(rank: int, *a) -> None:
    if rank == 0:
        print(*a, flush=True)


def barrier(device: torch.device) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


# ----------------------------------------------------------------------------- the core pieces
# ======== YOU WRITE broadcast_params AND sync_grads FIRST (Block 2 of the guide), then compare ========


def broadcast_params(model: torch.nn.Module, src: int = 0) -> None:
    """Make every rank's parameters equal to rank `src`'s: one broadcast per parameter tensor.

    Real DDP does this in its constructor too (`_sync_module_states`), and also broadcasts buffers.
    Here the only buffer is the causal mask, identical on every rank by construction (and a bool tensor).
    """
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)  # in place: rank src sends, all other ranks overwrite


def allreduce_sum_(t: torch.Tensor, algo: str) -> None:
    """In-place sum over all ranks. 'builtin' = dist.all_reduce, 'ring' = our point-to-point ring."""
    if not t.is_contiguous():  # collectives need contiguous memory
        tmp = t.contiguous()
        allreduce_sum_(tmp, algo)
        t.copy_(tmp)
        return
    if algo == "builtin":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    elif t.device.type == "cpu" or dist.get_backend() == "nccl":
        ring_all_reduce(t)  # NCCL does p2p on CUDA tensors; gloo only on CPU tensors
    else:  # gloo + CUDA tensor: stage through host memory
        host = t.detach().cpu()
        ring_all_reduce(host)
        t.copy_(host)


def sync_grads(params: list[torch.Tensor], world: int, args) -> None:
    """Average .grad over all ranks, in place. SUM first, then divide.

    Why SUM + divide rather than ReduceOp.AVG? It is explicit normalisation that does not depend on which backend or
    torch version implements AVG. (Observed 2026-10-21 on torch 2.14.1: AVG worked on both gloo and NCCL. Not tested
    on 2.13.0; hello_dist.py tells you what your install does.)
    Why divide at all? Each rank's loss is a MEAN over its local shard, so the sum over ranks is `world` times too big.
    Mean over the global batch = average of the local means ONLY when every rank's shard contributes the same number of
    loss elements (rows or valid tokens). With masks / ignored tokens, weight by valid counts, or sum losses and divide
    by a global count (Day 18).

    The per-tensor `grad is not None` assert is a LOCAL check. If it failed on only one rank, the others would wait in
    the next collective; run() therefore also does a collective consensus check at step 0.
    """
    if world == 1 or args.bug == "no_allreduce":  # bug: forget to communicate
        return
    for p in params:
        assert p.grad is not None, "a parameter got no gradient; every rank must reduce the same tensors"
    average = args.bug != "no_average"  # bug: forget the division

    if args.sync == "per_param":  # one collective per parameter tensor: many small messages
        for p in params:
            allreduce_sum_(p.grad, args.allreduce)
            if average:
                p.grad.div_(world)
    else:  # "flat": copy all grads into ONE buffer, one collective, copy back.
        # Costs an extra gradient-sized buffer and two copies, and it cannot overlap with backward (everything must
        # be ready first). Whether it is faster than per_param depends on the machine: measure, don't assume.
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        allreduce_sum_(flat, args.allreduce)
        if average:
            flat.div_(world)
        off = 0
        for p in params:
            n = p.numel()
            p.grad.copy_(flat[off : off + n].view_as(p.grad))
            off += n


# ----------------------------------------------------------------------------- consistency checks
def check_in_sync(tensors: list[torch.Tensor], what: str, rank: int, world: int) -> bool:
    """True iff every rank holds the same VALUES as rank 0 (exact element equality via torch.equal). Same answer everywhere.

    torch.equal compares element values, not raw bytes: +0.0 == -0.0 counts as equal, NaN never equals NaN.
    Stays on the tensors' own device: NCCL cannot communicate CPU tensors. The printed deviation is the maximum over
    ALL ranks (reduced together with the flag), not just this rank's.
    """
    if world == 1:
        return True
    mine = torch.cat([t.detach().reshape(-1) for t in tensors])
    ref = mine.clone()
    dist.broadcast(ref, src=0)
    differs = 0.0 if torch.equal(mine, ref) else 1.0
    worst = (mine - ref).abs().max().item()
    flag = torch.tensor([differs, worst], dtype=torch.float64, device=mine.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # now every rank knows whether ANY rank differs, and by how much
    bad = bool(flag[0].item())
    if bad and rank == 0:
        print(f"NOT IN SYNC: {what}: max over ranks of |value - rank0's value| = {flag[1].item():.3e}", flush=True)
    return not bad


def require_sync(ok: bool, args, what: str) -> None:
    """All ranks see the same `ok`, so all ranks raise together (no one is left waiting in a collective)."""
    if not ok and args.bug == "none":
        raise RuntimeError(f"ranks diverged: {what}")


def summarize_timing(rows: list[list[float]], tokens_per_step: int) -> dict:
    """rows[i] = [fwd_bwd, comm, optim, core, loop] in SECONDS for timed step i, ALREADY the max over ranks for that step.

    Medians are taken over steps, per column. Because each column is reduced separately, the phase medians need not add
    up to the core median (and no "step that never happened" is constructed by summing per-rank maxima).
    """
    if not rows:
        raise ValueError("no timed steps (increase --steps)")
    col = lambda j: [r[j] for r in rows]  # noqa: E731
    fb, comm, opt, core = (statistics.median(col(j)) * 1e3 for j in range(4))
    loop_mean = statistics.fmean(col(4)) * 1e3
    return {
        "fwd_bwd_ms": fb, "comm_ms": comm, "optim_ms": opt,
        "core_step_ms": core, "core_tokens_per_s": tokens_per_step / (core / 1e3),
        "loop_step_ms_mean": loop_mean, "loop_tokens_per_s": tokens_per_step / (loop_mean / 1e3),
        "comm_over_core": comm / core,  # ratio of two medians: an approximation of the communication share
        "timed_steps": len(rows),
    }


def run_metadata(args, device: torch.device, world: int, backend: str, n_timed: int, warm: int) -> dict:
    """Everything needed to interpret (or distrust) a result later. Plain str/int/float/None/list/dict only."""
    driver = "N/A"
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - no nvidia-smi on CPU-only machines
        pass
    gpus, uuid0 = [], None
    if device.type == "cuda":
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        uuid0 = str(getattr(torch.cuda.get_device_properties(device), "uuid", None))
    overrides = {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "TORCH_NCCL_", "OMP_", "MKL_", "CUDA_VISIBLE"))}
    return {
        "artifact_kind": "comparison artifact for check_parity / scaling; NOT a resumable training checkpoint (no optimizer state)",
        "world": world, "backend": backend, "args": vars(args), "env": env_info(device),
        "driver_version": driver, "visible_gpus": gpus, "rank0_gpu_uuid": uuid0,
        "env_overrides": overrides, "torch_num_threads": torch.get_num_threads(),
        "timing_method": ("per step: fwd_bwd, comm, optim, core(=t0..t3), loop(=whole iteration) measured on every rank with "
                          "cuda synchronize at the boundaries; max over ranks PER STEP; then median over steps "
                          "(loop: mean). core excludes batch generation/zero_grad/checks/logging; loop includes them"),
        "warmup_steps": warm, "timed_steps": n_timed,
        "condition": args.condition,
    }


# ----------------------------------------------------------------------------- main loop
def run(args) -> None:
    rank, world, device, backend = init_dist(args)
    cfg = Config(**PRESETS[args.preset])
    cfg.total_steps = args.steps  # the cosine LR schedule depends on it; ref and dp runs must agree
    # NOTE: lr_warmup_steps is 20, so a default 20-step run never leaves LR warmup: it says nothing about a finished schedule.
    if args.per_gpu_batch:
        cfg.batch_size = args.per_gpu_batch * world  # weak scaling: work per GPU fixed, global batch grows
    elif args.global_batch:
        cfg.batch_size = args.global_batch  # strong scaling: global batch fixed, shards shrink
    if cfg.batch_size < world or cfg.batch_size % world != 0:
        raise SystemExit(f"global batch {cfg.batch_size} must be a positive multiple of the world size {world}")
    local_bs = cfg.batch_size // world

    setup_reproducibility(cfg)  # seed, TF32 off, deterministic kernels (Day 01)
    # DELIBERATELY different init per rank (rank 0 keeps cfg.seed, so it equals the 1-process reference).
    # If broadcast_params did nothing, the ranks would silently train different models.
    torch.manual_seed(cfg.seed + rank)
    model, optimizer, scheduler = build(cfg, device)
    named = dict(model.named_parameters())
    params = list(named.values())

    if world > 1 and args.bug != "no_broadcast":
        broadcast_params(model, src=0)
    require_sync(check_in_sync(params, "initial params", rank, world), args, "initial params differ")

    grad_mib = sum(p.numel() * p.element_size() for p in params) / MiB
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    log0(rank, f"world={world} device={device} ({gpu}) backend={backend if world > 1 else '-'} preset={args.preset} "
               f"sync={args.sync} allreduce={args.allreduce} bug={args.bug} condition={args.condition}")
    log0(rank, f"global batch {cfg.batch_size} = {world} x {local_bs} | {len(params)} grad tensors, payload {grad_mib:.2f} MiB "
               f"per step | a ring all-reduce SENDS ~{2 * (world - 1) / world * grad_mib:.2f} MiB per rank (equal chunks)")

    losses, grads0, params1 = [], {}, {}
    rows: list[list[float]] = []  # per timed step, this rank: [fwd_bwd, comm, optim, core, loop] seconds
    warm = min(3, args.steps - 1)  # measurement warmup (Day 01): excluded from timing; steps>=1 guarantees >=1 timed step

    def sync_dev() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # GPU work is async (NCCL too): synchronize before reading a timer

    for step in range(args.steps):
        if step == warm and device.type == "cuda":
            sync_dev()
            torch.cuda.reset_peak_memory_stats(device)  # peak memory = steady state only
        sync_dev()
        L0 = time.perf_counter()  # start of the whole iteration ("loop")
        # Day 01's get_batch is a pure function of (seed, step), so every rank builds the SAME global batch and
        # takes its own slice. (DistributedSampler, Day 03, does the sharding without building the whole batch.)
        x, y = get_batch(cfg, step, device)
        x, y = x[rank * local_bs : (rank + 1) * local_bs], y[rank * local_bs : (rank + 1) * local_bs]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        sync_dev()
        t0 = time.perf_counter()  # start of the "core" step
        with attention_ctx(cfg):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))  # MEAN over local tokens
        del logits
        loss.backward()
        sync_dev()
        t1 = time.perf_counter()

        if step == 0 and world > 1:  # collective consensus: a missing grad on ONE rank must stop ALL ranks together
            missing = torch.tensor([float(any(p.grad is None for p in params))], device=device)
            dist.all_reduce(missing, op=dist.ReduceOp.MAX)
            if missing.item():
                raise RuntimeError("some rank has a parameter without a gradient")
        sync_grads(params, world, args)
        sync_dev()
        t2 = time.perf_counter()

        optimizer.step()
        scheduler.step()
        sync_dev()
        t3 = time.perf_counter()  # end of the "core" step

        losses.append(loss.detach())  # keep on device; no .item() per step (Day 01 lesson)

        if step == 0:  # evidence for check_parity.py: the synced gradient, and the params after ONE update
            grads0 = {n: p.grad.detach().cpu().clone() for n, p in named.items()}
            params1 = {n: p.detach().cpu().clone() for n, p in named.items()}
            if world > 1:
                grads_ok = check_in_sync([p.grad for p in params], "gradients after all_reduce (step 0)", rank, world)
                require_sync(grads_ok, args, "averaged gradients differ across ranks")
        if step == 0 or (step + 1) % args.check_every == 0 or step + 1 == args.steps:
            require_sync(check_in_sync(params, f"params after step {step + 1}", rank, world), args,
                         f"params differ after step {step + 1}")
        if (step + 1) % args.log_every == 0:
            log0(rank, f"step {step + 1:4d} | rank-0 local loss {loss.item():.4f}")  # .item() syncs; rare, so OK
        sync_dev()
        L1 = time.perf_counter()
        if step >= warm:
            rows.append([t1 - t0, t2 - t1, t3 - t2, t3 - t0, L1 - L0])

    # ---- global loss curve: ONE all_reduce for the whole run instead of one per step
    loss_t = torch.stack(losses).float()  # still on `device`
    if world > 1:
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= world  # mean of local means == global mean because every shard has equal size
    loss_list = loss_t.cpu().tolist()

    # ---- timing: a synchronous step is as slow as its SLOWEST rank, decided PER STEP, then the median over steps
    tm = torch.tensor(rows, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tm, op=dist.ReduceOp.MAX)  # elementwise: slowest rank for each (step, column)
    timing = summarize_timing(tm.cpu().tolist(), cfg.batch_size * cfg.seq_len)
    timing.update({"global_batch": cfg.batch_size, "world": world,
                   "peak_alloc_mib_rank0": torch.cuda.max_memory_allocated(device) / MiB if device.type == "cuda" else None})
    peak_txt = f" | peak mem {timing['peak_alloc_mib_rank0']:.0f} MiB" if device.type == "cuda" else ""
    log0(rank, f"timing over {timing['timed_steps']} steps after {warm} warmup (slowest rank per step): "
               f"core step {timing['core_step_ms']:.2f} ms median [fwd+bwd {timing['fwd_bwd_ms']:.2f} | sync {timing['comm_ms']:.2f} "
               f"| optim {timing['optim_ms']:.2f}; phase medians need not add up] | "
               f"core {timing['core_tokens_per_s']:,.0f} tok/s | loop {timing['loop_step_ms_mean']:.2f} ms mean, "
               f"{timing['loop_tokens_per_s']:,.0f} tok/s{peak_txt}")

    if rank == 0:
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        meta = run_metadata(args, device, world, backend, len(rows), warm)
        meta["lr0"] = cfg.lr * lr_lambda(0, cfg)
        meta["n_params"] = sum(p.numel() for p in params)
        torch.save(  # only tensors / numbers / strings / lists / dicts: loadable with weights_only=True (Day 01 lesson)
            {"grads0": grads0, "params1": params1,
             "final": {n: p.detach().cpu().clone() for n, p in named.items()},
             "losses": loss_list, "timing": timing, "meta": meta},
            out / "result.pt",
        )
        (out / "metrics.json").write_text(json.dumps({"timing": timing, "meta": meta}, indent=2))
        print(f"wrote {out / 'result.pt'}", flush=True)

    if world > 1:
        barrier(device)
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--tag", default="run", help="sub-folder of outputs/day02_data_parallel/ (refuses to overwrite unless --overwrite)")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing result for this tag")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--backend", choices=["gloo", "nccl"], default=None, help="default: nccl for cuda, gloo for cpu")
    p.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown",
                   help="recorded in the result: were you the only user of these GPUs? (shared = provisional numbers)")
    p.add_argument("--global-batch", type=int, default=0, help="strong scaling: fix the global batch (default: preset's 16)")
    p.add_argument("--per-gpu-batch", type=int, default=0, help="weak scaling: global batch = this x world size")
    p.add_argument("--sync", choices=["per_param", "flat"], default="per_param",
                   help="per_param: one all_reduce per tensor; flat: one all_reduce over all grads (extra buffer + copies)")
    p.add_argument("--allreduce", choices=["builtin", "ring"], default="builtin",
                   help="builtin: dist.all_reduce; ring: our point-to-point ring (ring_allreduce.py)")
    p.add_argument("--bug", choices=["none", "no_broadcast", "no_allreduce", "no_average"], default="none",
                   help="negative controls (need >= 2 ranks): inject a classic data-parallel bug; check_parity must catch it")
    p.add_argument("--check-every", type=int, default=10, help="verify cross-rank parameter equality every N steps")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120, help="seconds before a stuck collective raises")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
