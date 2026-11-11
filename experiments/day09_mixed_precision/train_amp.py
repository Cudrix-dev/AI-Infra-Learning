"""Day 09 harness: Day 01's transformer, Day 04's accumulation window, in fp32 / bf16 / fp16, checkpointing on or off.

  --precision fp32 | bf16 | fp16    fp32 = no autocast, TF32 OFF (Day 01's "highest"); bf16/fp16 = torch.autocast;
                                    fp16 also gets a GradScaler. Parameters, gradients and AdamW state stay fp32 in all
                                    three (probe_amp.py shows it).
  --ckpt off | on                   checkpoint every transformer block (amp_setup.model_forward)
  --global-batch G --micro-batch m  K = G / (W * m) microbatches per optimizer step, Day 04's loss weights
  --max-norm 1.0                    clip_grad_norm_ in every precision, so all three run the same update rule
  --bug clip_before_unscale         fp16 negative control (fp16_order.py is the focused version)
  --impl ref | mine                 amp_setup.py (reference) or my_amp.py (your YOU WRITE parts)
  --sdpa math | auto                math = Days 01-08's deterministic attention; auto = flash/efficient (optional rows)

One GPU:      python experiments/day09_mixed_precision/train_amp.py --precision bf16 --ckpt on --tag try_bf16_on
Two GPUs (the optional DDP question: in which dtype does DDP all-reduce gradients under autocast?):
  python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day09_mixed_precision/train_amp.py \
      --precision bf16 --global-batch 32 --micro-batch 16 --profile --tag try_ddp_bf16 [--comm-hook bf16]

Timing (same rules as Days 02-08): every phase boundary (forward, backward, optimizer) is a torch.cuda.synchronize(),
so per-phase times are exact and their sum is the step time ("core"). Batch generation, zero_grad and the scheduler
are outside. Median over steps >= --warmup; with two ranks the slowest rank per step. Peak memory is reset at every
phase boundary, so peaks are PER PHASE; the step peak is their max. "act held" = memory_allocated right after the
first forward + loss of a window minus at the window start: everything autograd keeps for backward (activations, the
autocast weight copies, checkpoint inputs), with the logits already deleted.
"""

from __future__ import annotations

import argparse
import hashlib
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
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.profiler import ProfilerActivity, profile, schedule  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day02_data_parallel"))  # adds Day 01's folder itself
sys.path.insert(0, str(HERE))
from ddp_manual import log0, run_metadata  # noqa: E402
from train import PRESETS, Config, build, get_batch, setup_reproducibility  # noqa: E402

from amp_setup import PRECISIONS, AmpNet, load_impl, make_scaler, sdpa_context  # noqa: E402

OUT = HERE.parents[1] / "outputs" / "day09_amp"
MiB = 2**20
PHASES = ("fwd", "bwd", "opt")


def init_dist(args) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not args.tag or any(c in args.tag for c in "/\\"):
        raise SystemExit("--tag must be a plain folder name")
    if (OUT / args.tag / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"{OUT / args.tag} exists: use a new --tag or --overwrite")
    if not torch.cuda.is_available():
        raise SystemExit("Day 09 needs a CUDA GPU (autocast bf16/fp16 and memory statistics)")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout), device_id=device)
    elif args.comm_hook != "none":
        raise SystemExit("--comm-hook needs DDP (launch with torchrun, 2 processes)")
    return rank, world, device


class StepRecorder:
    """The `on(event)` callback handed to window_step: per-phase time and memory, and evidence capture.

    On "fwd" / "bwd" / "opt" it synchronizes, adds the elapsed time to that phase, reads memory_allocated and the peak
    since the previous boundary, and resets the peak. On "pre_clip" of step 0 it can copy the gradients (the ones
    clipping is about to see: unscaled, unless the bug is on)."""

    def __init__(self, device: torch.device, params: dict, capture_grads0: bool):
        self.device, self.params, self.capture = device, params, capture_grads0
        self.peak = {p: 0.0 for p in PHASES}
        self.held, self.start_alloc, self.rows = [], [], []
        self.grads0 = None

    def _sync(self) -> None:
        torch.cuda.synchronize(self.device)

    def begin(self, step: int, steady: bool) -> None:
        self._sync()
        self.step, self.steady, self.first_fwd = step, steady, True
        self.cur = {p: 0.0 for p in PHASES}
        self.alloc0 = torch.cuda.memory_allocated(self.device)
        if steady:
            self.start_alloc.append(self.alloc0 / MiB)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.t0 = self.t = time.perf_counter()

    def on(self, event: str) -> None:
        if event in PHASES:
            self._sync()
            now = time.perf_counter()
            self.cur[event] += now - self.t
            self.t = now
            alloc = torch.cuda.memory_allocated(self.device)
            peak = torch.cuda.max_memory_allocated(self.device) / MiB
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.steady:
                self.peak[event] = max(self.peak[event], peak)
                if event == "fwd" and self.first_fwd:
                    self.held.append((alloc - self.alloc0) / MiB)
            if event == "fwd":
                self.first_fwd = False
        elif event == "pre_clip" and self.step == 0 and self.capture:
            self.grads0 = {k: p.grad.detach().float().cpu().clone() for k, p in self.params.items()}

    def end(self) -> None:
        core = self.t - self.t0
        if self.steady:
            self.rows.append([self.cur["fwd"], self.cur["bwd"], self.cur["opt"], core])


def param_sha256(params: dict) -> str:
    """SHA-256 over the raw bytes of every parameter, in name order: equal hashes = bitwise-equal models."""
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(params[k].detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def comm_summary(trace: Path, n_steps: int) -> dict:
    """Gradient collectives in a profiler trace: PyTorch's record_param_comms events (collective, dtype, elements)
    and the NCCL kernel names. Divided by the number of profiled optimizer steps."""
    ev = json.loads(trace.read_text())["traceEvents"]
    colls, raw = {}, None
    for e in ev:
        if e.get("name") != "record_param_comms" or e.get("cat") != "cpu_op":
            continue
        a = e.get("args", {})
        raw = raw or a
        name = a.get("Collective name", "?")
        if name in ("barrier", "init"):
            continue
        dtype, n = str(a.get("dtype", "?")), int(a.get("In msg nelems", 0) or 0)
        c = colls.setdefault(f"{name} {dtype}", {"calls": 0, "elements": 0, "sizes": []})
        c["calls"] += 1
        c["elements"] += n
        c["sizes"].append(n)
    kernels = {}
    for e in ev:
        if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower():
            k = e["name"].split("(")[0]
            kernels[k] = kernels.get(k, 0) + 1
    for c in colls.values():
        c["calls_per_step"] = c["calls"] / n_steps
        c["elements_per_step"] = c["elements"] / n_steps
        c["sizes"] = sorted(set(c["sizes"]))
    return {"collectives": colls, "nccl_kernels_per_step": {k: v / n_steps for k, v in kernels.items()},
            "first_record_param_comms_args": raw}


def run(args) -> None:
    global OUT
    if args.output_root:
        OUT = Path(args.output_root).resolve()
    rank, world, device = init_dist(args)
    impl = load_impl(args.impl)
    cfg = Config(**PRESETS[args.preset])
    cfg.total_steps = args.opt_steps  # the LR schedule counts OPTIMIZER steps (Day 04)
    G, m = args.global_batch, args.micro_batch
    if G % world or (G // world) % m:
        raise SystemExit(f"G={G} must split into W={world} ranks x K microbatches of m={m}")
    K = G // (world * m)
    cfg.batch_size = G
    if args.bug != "none" and args.precision != "fp16":
        raise SystemExit("--bug clip_before_unscale only exists for fp16 (there is no scale in fp32 / bf16)")

    setup_reproducibility(cfg)  # same seed on every rank: identical init (DDP also broadcasts rank 0's params)
    model, optimizer, scheduler = build(cfg, device)  # Day 01: fp32 params, AdamW(foreach=False), LambdaLR
    named = dict(model.named_parameters())
    net = AmpNet(model, args.ckpt == "on", impl.model_forward)
    no_sync = nullcontext
    if world > 1:
        net = DDP(net, device_ids=[device.index])
        no_sync = net.no_sync
        if args.comm_hook == "bf16":
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
            net.register_comm_hook(dist.group.WORLD, bf16_compress_hook)
    scaler = make_scaler(args.precision)
    log0(rank, f"precision={args.precision} ckpt={args.ckpt} sdpa={args.sdpa} impl={args.impl} bug={args.bug} | "
               f"G={G} = W {world} x K {K} x m {m} | max_norm={args.max_norm} | world={world} hook={args.comm_hook}")

    warm = min(args.warmup, args.opt_steps - 1)
    prof = None
    if args.profile:
        if args.opt_steps < warm + 1 + args.profile_steps:
            raise SystemExit(f"--profile needs --opt-steps >= {warm + 1 + args.profile_steps}")
        trace_dir = OUT / args.tag
        trace_dir.mkdir(parents=True, exist_ok=True)
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(skip_first=warm, wait=0, warmup=1, active=args.profile_steps, repeat=1),
                       on_trace_ready=lambda p: p.export_chrome_trace(str(trace_dir / f"trace_rank{rank}.json")))
        prof.start()

    rec = StepRecorder(device, named, capture_grads0=args.save_grads0)
    losses, norms, scales, skipped = [], [], [], 0
    for s in range(args.opt_steps):
        x, y = get_batch(replace(cfg, batch_size=G), s, device)  # the GLOBAL batch of optimizer step s (Day 01)
        rows = G // world
        x, y = x[rank * rows:(rank + 1) * rows], y[rank * rows:(rank + 1) * rows]
        micro = [(a, b, a.size(0) / rows) for a, b in zip(x.split(m), y.split(m))]
        net.train()
        optimizer.zero_grad(set_to_none=True)  # so "start of step" = params + optimizer state, no stale grads
        rec.begin(s, steady=s >= warm)
        with sdpa_context(args.sdpa):
            out = impl.window_step(net, optimizer, scaler, micro, precision=args.precision, max_norm=args.max_norm,
                                   no_sync=no_sync, bug=args.bug, on=rec.on)
        rec.end()
        scheduler.step()  # once per optimizer step (Day 04)
        losses.append(out["loss"])
        norms.append(out["norm"])
        scales.append(out["scale"])
        skipped += int(out["skipped"])
        if (s + 1) % args.log_every == 0:
            log0(rank, f"opt step {s + 1:4d} | loss {out['loss'].item():.4f} | norm seen by clip "
                       f"{out['norm'].item():.3e} | scale {out['scale']:.0f}")
        if prof is not None:
            prof.step()
    if prof is not None:
        prof.stop()

    # ---- slowest rank per step, then medians (Days 02-08)
    tm = torch.tensor(rec.rows or [[float("nan")] * 4], dtype=torch.float64, device=device)
    mem = torch.tensor([max(rec.peak.values()), rec.peak["fwd"], rec.peak["bwd"], rec.peak["opt"],
                        statistics.median(rec.held) if rec.held else float("nan"),
                        statistics.median(rec.start_alloc) if rec.start_alloc else float("nan"),
                        torch.cuda.max_memory_reserved(device) / MiB], dtype=torch.float64, device=device)
    loss_t = torch.stack(losses).float()
    if world > 1:
        dist.all_reduce(tm, op=dist.ReduceOp.MAX)
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t /= world
    tml = tm.cpu().tolist()
    med = [statistics.median(r[j] for r in tml) * 1e3 for j in range(4)]
    tokens = G * cfg.seq_len
    timing = {"core_step_ms": med[3], "fwd_ms": med[0], "bwd_ms": med[1], "opt_ms": med[2],
              "tokens_per_s": tokens / (med[3] / 1e3), "timed_steps": len(rec.rows), "warmup_steps": warm,
              "all_core_ms": [r[3] * 1e3 for r in tml], "profiled": bool(args.profile)}
    memory = dict(zip(["peak_mib", "peak_fwd_mib", "peak_bwd_mib", "peak_opt_mib", "act_held_mib",
                       "start_of_step_mib", "peak_reserved_mib"], mem.cpu().tolist()))
    log0(rank, f"core {med[3]:.2f} ms/step (fwd {med[0]:.2f}, bwd {med[1]:.2f}, opt {med[2]:.2f}) median of "
               f"{len(rec.rows)} | peak {memory['peak_mib']:.0f} MiB (fwd {memory['peak_fwd_mib']:.0f}, bwd "
               f"{memory['peak_bwd_mib']:.0f}) | act held {memory['act_held_mib']:.0f} MiB | start "
               f"{memory['start_of_step_mib']:.0f} MiB | skipped {skipped}"
               + ("  [PROFILED RUN: not a benchmark]" if args.profile else ""))

    if world > 1:
        dist.barrier(device_ids=[device.index])
    if rank == 0:
        out = OUT / args.tag
        out.mkdir(parents=True, exist_ok=True)
        meta = run_metadata(args, device, world, "nccl" if world > 1 else "-", len(rec.rows), warm)
        meta.update(n_params=sum(p.numel() for p in named.values()), accum=K, micro_batch=m, global_batch=G,
                    precision=args.precision, ckpt=args.ckpt, sdpa=args.sdpa, impl=args.impl, bug=args.bug,
                    comm_hook=args.comm_hook, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    param_dtypes=sorted({str(p.dtype) for p in named.values()}),
                    grad_dtypes=sorted({str(p.grad.dtype) for p in named.values() if p.grad is not None}),
                    adam_state_dtypes=sorted({f"{k}:{v.dtype}" for st in optimizer.state.values()
                                              for k, v in st.items() if torch.is_tensor(v)}))
        result = {"timing": timing, "memory": memory, "losses": loss_t.cpu().tolist(),
                  "norm_seen_by_clip": [float(n) for n in norms], "loss_scale": scales, "skipped_steps": skipped,
                  "param_sha256": param_sha256(named), "meta": meta}
        if prof is not None:
            result["comms"] = comm_summary(out / "trace_rank0.json", args.profile_steps)
            for k, c in result["comms"]["collectives"].items():
                log0(rank, f"profiled collectives per step: {k}: {c['calls_per_step']:.0f} calls, "
                           f"{c['elements_per_step']:,.0f} elements")
            log0(rank, f"NCCL kernels per step: {result['comms']['nccl_kernels_per_step']}")
        if args.save_params:
            torch.save({k: p.detach().float().cpu() for k, p in named.items()}, out / "params_final.pt")
        if args.save_grads0:
            torch.save(rec.grads0, out / "grads0.pt")
        (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {out / 'metrics.json'}", flush=True)
    if world > 1:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision", choices=PRECISIONS, required=True)
    p.add_argument("--ckpt", choices=["off", "on"], default="off")
    p.add_argument("--sdpa", choices=["math", "auto"], default="math")
    p.add_argument("--impl", choices=["ref", "mine"], default="ref")
    p.add_argument("--preset", choices=PRESETS, default="medium")
    p.add_argument("--global-batch", type=int, default=32, help="G: rows per optimizer step over all ranks")
    p.add_argument("--micro-batch", type=int, default=32, help="m: rows per microbatch per rank (K = G / (W*m))")
    p.add_argument("--opt-steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10, help="optimizer steps excluded from timing and peak memory")
    p.add_argument("--max-norm", type=float, default=1.0)
    p.add_argument("--bug", choices=["none", "clip_before_unscale"], default="none")
    p.add_argument("--comm-hook", choices=["none", "bf16"], default="none", help="DDP only: bf16_compress_hook")
    p.add_argument("--profile", action="store_true", help="chrome trace of --profile-steps steps after warmup")
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--save-params", action="store_true", help="final fp32 params (105 MiB): for the drift metric")
    p.add_argument("--save-grads0", action="store_true", help="step-0 gradients at pre_clip (105 MiB)")
    p.add_argument("--condition", choices=["exclusive", "shared", "unknown"], default="unknown")
    p.add_argument("--output-root", help="write results somewhere other than outputs/day09_amp")
    p.add_argument("--tag", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
