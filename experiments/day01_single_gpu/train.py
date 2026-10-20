"""Day 01 - Reproducible single-GPU baseline.

This one file does four jobs:
  1. Train a tiny decoder-only transformer on synthetic tokens
     (zero_grad -> forward -> backward -> optimizer.step -> scheduler.step).
  2. Save / resume a checkpoint that holds EVERYTHING needed to continue bit-exactly.
  3. Measure step time, tokens/s and peak memory correctly
     (CUDA events, measurement warmup excluded, no per-step host sync).
  4. With --memory-probe: print when each memory category (params, grads,
     optimizer state, activations) is allocated and freed.

Run from the repo root:
  python experiments/day01_single_gpu/train.py --tag full
  python experiments/day01_single_gpu/train.py --tag part --stop-at 100
  python experiments/day01_single_gpu/train.py --tag resumed --resume outputs/day01_single_gpu/part/ckpt_step100.pt
  python experiments/day01_single_gpu/check_resume.py            # runs the three above and compares them
  python experiments/day01_single_gpu/train.py --memory-probe --preset medium
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

# cuBLAS reads this when its handle is created (at the first matmul), so it must be set before
# any CUDA work happens. A fixed workspace makes cuBLAS GEMM reductions deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "day01_single_gpu"
MiB = 2**20


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    # Shapes. B: batch size, T: sequence length, V: vocab size, D: embedding dim
    batch_size: int = 16
    seq_len: int = 64
    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    # Optimisation
    lr: float = 3e-4
    weight_decay: float = 0.01
    lr_warmup_steps: int = (
        20  # ALGORITHM warmup: lr ramps up linearly over these optimizer steps
    )
    total_steps: int = 200
    # Systems
    seed: int = 42
    timing_warmup: int = (
        10  # MEASUREMENT warmup: these steps are excluded from timing stats
    )
    save_every: int = 50
    log_every: int = 20
    deterministic: bool = True


PRESETS = {
    "tiny": {},  # ~0.44M params: launch-overhead bound, good for fast correctness checks
    "medium": dict(
        batch_size=16, seq_len=256, vocab_size=8192, d_model=512, n_layers=6, n_heads=8
    ),  # ~27M params
}

# Changing any of these between "save" and "resume" means you are no longer continuing the same run.
MUST_MATCH = (
    "batch_size",
    "seq_len",
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "lr",
    "weight_decay",
    "lr_warmup_steps",
    "total_steps",
    "seed",
    "deterministic",
)


# ----------------------------------------------------------------------------- reproducibility
def setup_reproducibility(cfg: Config) -> None:
    # Seeds the CPU generator AND the generator of every CUDA device.
    # It matters here for parameter init; dropout is 0, but a real model would also draw from it.
    torch.manual_seed(cfg.seed)
    # Full-precision fp32 matmuls (no TF32), so numbers stay comparable across days and GPUs.
    torch.set_float32_matmul_precision("highest")
    if cfg.deterministic:
        # Prefer deterministic kernels; warn (instead of crash) when an op has none.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def attention_ctx(cfg: Config):
    # SDPA may pick a fused attention kernel whose backward uses atomics (run-to-run noise).
    # The MATH backend is plain matmul + softmax: slower, but bit-for-bit repeatable.
    return sdpa_kernel(SDPBackend.MATH) if cfg.deterministic else nullcontext()


# ----------------------------------------------------------------------------- data
def get_batch(
    cfg: Config, step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic arithmetic sequences: token[i] = (start + stride * i) % V.

    The model has to infer `stride` from the context, so loss falls gradually instead of
    hitting 0 in a few steps (your original +1 pattern was learnable almost instantly).

    The batch is a pure function of (seed, step): a private generator is re-seeded every step.
    Consequence: resume needs NO data-loader state, only the step counter. A real data
    loader is different - you must also checkpoint the sampler/epoch/offset.
    """
    g = torch.Generator().manual_seed(
        cfg.seed * 1_000_003 + step
    )  # CPU generator, independent of the global RNG
    starts = torch.randint(
        0, cfg.vocab_size, (cfg.batch_size, 1), generator=g
    )  # [B, 1]
    strides = torch.randint(1, 8, (cfg.batch_size, 1), generator=g)  # [B, 1]
    # Broadcasting: [B, 1] + [B, 1] * [T+1] -> [B, T+1]
    seq = (starts + strides * torch.arange(cfg.seq_len + 1)) % cfg.vocab_size
    x = seq[:, :-1].contiguous()  # input: drop the last token
    y = seq[:, 1:].contiguous()  # label: drop the first token (next-token prediction)
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------- model
class TinyTransformer(nn.Module):
    """
    tokens [B, T]
      -> token_emb + pos_emb                      [B, T, D]
      -> n_layers x pre-norm TransformerEncoderLayer with a causal mask
      -> final LayerNorm (pre-norm blocks leave the residual stream un-normalised)
      -> head: Linear D -> V                      [B, T, V] logits
    Softmax is NOT in the model: F.cross_entropy applies log_softmax internally.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=4 * cfg.d_model,
                    dropout=0.0,  # no dropout: removes one source of randomness from parity checks
                    batch_first=True,  # tensors are [B, T, D]
                    norm_first=True,  # pre-norm: LN -> attention/FFN -> residual add
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        # True = "may NOT attend". Upper triangle above the diagonal = future positions.
        mask = torch.triu(
            torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1
        )
        # persistent=False: derived from config, so it is not training state and stays out of state_dict.
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        pos = torch.arange(length, device=tokens.device)  # [T]
        h = self.token_emb(tokens) + self.pos_emb(
            pos
        )  # [B, T, D] + [T, D] broadcasts over B
        for block in self.blocks:
            h = block(h, src_mask=self.causal_mask[:length, :length])
        return self.head(self.norm(h))  # [B, T, V]


# ----------------------------------------------------------------------------- build
def lr_lambda(opt_step: int, cfg: Config) -> float:
    """LR multiplier used for optimizer step `opt_step` (0-based): linear warmup, then cosine to 10%."""
    if opt_step < cfg.lr_warmup_steps:
        return (opt_step + 1) / cfg.lr_warmup_steps
    progress = (opt_step - cfg.lr_warmup_steps) / max(
        1, cfg.total_steps - cfg.lr_warmup_steps
    )
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build(cfg: Config, device: torch.device):
    model = TinyTransformer(cfg).to(device)  # params allocated on the GPU here
    # foreach=False, fused=False selects _single_tensor_adam: the readable per-parameter loop
    # you trace in today's source reading. (The default on CUDA is foreach=True.)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        foreach=False,
        fused=False,
    )
    # A plain lambda on purpose: LambdaLR.state_dict() skips plain functions, so the checkpoint
    # holds only counters (last_epoch, _step_count) and we rebuild the function from cfg.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, cfg)
    )
    return model, optimizer, scheduler


def train_step(model, optimizer, x, y, cfg: Config) -> torch.Tensor:
    model.train()  # set the model to training mode (affects layers like dropout, batchnorm, etc.)
    optimizer.zero_grad(
        set_to_none=True
    )  # (1) p.grad = None: last step's grad memory goes back to the allocator
    with attention_ctx(cfg):
        logits = model(
            x
        )  # (2) forward: autograd saves the activations backward will need
    loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
    )  # [B*T, V] vs [B*T]
    del logits
    loss.backward()  # (3) backward: grads allocated (first touch), saved activations freed node by node
    optimizer.step()  # (4) AdamW update; the very first call also allocates exp_avg and exp_avg_sq
    # Keep the loss on the GPU. Calling .item() here would block the CPU until the GPU finishes,
    # which kills CPU/GPU overlap and distorts timing.
    return loss.detach()


# ----------------------------------------------------------------------------- checkpoint
def save_checkpoint(
    path: Path, model, optimizer, scheduler, step: int, cfg: Config, device
) -> None:
    ckpt = {
        "step": step,  # number of COMPLETED optimizer steps; the next batch to run is get_batch(step)
        "model": model.state_dict(),  # learnable weight and bias, without casual_mask because it is persistent=False
        "optimizer": optimizer.state_dict(),  # per-param exp_avg, exp_avg_sq, step + param_groups (lr, betas...)
        "scheduler": scheduler.state_dict(),  # last_epoch, _step_count, base_lrs, _last_lr
        "rng": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        },
        "config": asdict(cfg),
        # str(): torch.__version__ is a TorchVersion object (a str subclass). Pickled as-is it is a
        # custom class, which torch.load(weights_only=True) refuses to unpickle.
        "torch_version": str(torch.__version__),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(
        tmp, path
    )  # atomic rename: a crash mid-save never leaves a half-written checkpoint


def load_checkpoint(
    path: Path, model, optimizer, scheduler, cfg: Config, device, model_only=False
) -> int:
    # map_location="cpu": RNG states must stay CPU ByteTensors. model/optimizer.load_state_dict
    # copy each tensor onto the device of the matching parameter.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    saved = ckpt["config"]
    diff = {
        k: (saved[k], getattr(cfg, k))
        for k in MUST_MATCH
        if saved[k] != getattr(cfg, k)
    }
    if diff:
        raise ValueError(f"config mismatch vs checkpoint (saved, now): {diff}")

    model.load_state_dict(ckpt["model"])
    if model_only:
        # Deliberately WRONG resume (--bad-resume): fresh Adam moments, step=0, LR warmup restarts.
        print("!! bad resume: loaded model weights only")
        return ckpt["step"]
    # Order matters: the scheduler was already constructed in build() (its __init__ writes lr into
    # param_groups); optimizer.load_state_dict then restores the saved lr; scheduler counters last.
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["rng"]["cpu"])
    if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state(ckpt["rng"]["cuda"], device)
    return ckpt["step"]


# ----------------------------------------------------------------------------- environment
def env_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(
            gpu=p.name,
            gpu_mem_gib=round(p.total_memory / 2**30, 1),
            capability=f"{p.major}.{p.minor}",
        )
        try:
            v = torch.cuda.nccl.version()
            info["nccl"] = ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
        except Exception:  # NCCL is not built on Windows
            info["nccl"] = None
    return info


def count_params(model) -> tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n, nbytes / MiB


# ----------------------------------------------------------------------------- training
def train(cfg: Config, args, device: torch.device) -> None:
    out_dir = OUTPUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = device.type == "cuda"

    setup_reproducibility(cfg)
    model, optimizer, scheduler = build(cfg, device)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, cfg, device, args.bad_resume
        )
        print(f"resumed from {args.resume} at step {start_step}")

    env = env_info(device)
    n_params, param_mib = count_params(model)
    print(json.dumps(env, indent=2))
    print(
        f"params: {n_params:,} ({param_mib:.2f} MiB fp32) | tokens/step: {cfg.batch_size * cfg.seq_len:,}"
    )

    losses: dict[int, torch.Tensor] = {}
    timings: list[
        tuple[int, object, object]
    ] = []  # (step, start, end): CUDA events or perf_counter floats
    measured_from = (
        start_step + cfg.timing_warmup
    )  # first step that counts for timing/peak memory
    end_step = start_step

    for step in range(start_step, cfg.total_steps):
        if step == measured_from and is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # peak memory = steady state only
        if is_cuda:
            # Events are timestamps recorded INTO the GPU stream: no CPU stall at record time.
            t0, t1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            t0.record()
        else:
            t0 = time.perf_counter()

        x, y = get_batch(cfg, step, device)
        loss = train_step(model, optimizer, x, y, cfg)
        scheduler.step()  # after optimizer.step(): sets the lr for the NEXT step

        if is_cuda:
            t1.record()
        else:
            t1 = time.perf_counter()
        timings.append((step, t0, t1))
        losses[step] = loss
        end_step = done = step + 1

        if done % cfg.log_every == 0:
            # .item() syncs CPU and GPU; acceptable once every log_every steps, not every step.
            print(
                f"step {done:5d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
            )
        if args.stop_at and done == args.stop_at:
            save_checkpoint(
                out_dir / f"ckpt_step{done}.pt",
                model,
                optimizer,
                scheduler,
                done,
                cfg,
                device,
            )
            print(f"simulated crash: saved ckpt_step{done}.pt and stopped")
            break
        if done % cfg.save_every == 0:
            save_checkpoint(
                out_dir / f"ckpt_step{done}.pt",
                model,
                optimizer,
                scheduler,
                done,
                cfg,
                device,
            )

    if end_step == cfg.total_steps:
        save_checkpoint(
            out_dir / "final.pt", model, optimizer, scheduler, end_step, cfg, device
        )

    # ---- timing: one sync at the end, then read every event pair
    if is_cuda:
        torch.cuda.synchronize()
        step_ms = [(s, a.elapsed_time(b)) for s, a, b in timings]
    else:
        step_ms = [(s, (b - a) * 1e3) for s, a, b in timings]
    steady = [ms for s, ms in step_ms if s >= measured_from]
    tokens_per_step = cfg.batch_size * cfg.seq_len
    timing = {
        "first_step_ms": step_ms[0][1]
        if step_ms
        else None,  # cold: cuBLAS handle, allocator growth, lazy Adam state
        "steady_steps": len(steady),
        "steady_median_ms": statistics.median(steady) if steady else None,
        "steady_mean_ms": statistics.fmean(steady) if steady else None,
        "steady_p90_ms": sorted(steady)[int(0.9 * (len(steady) - 1))]
        if steady
        else None,
        "tokens_per_s": tokens_per_step * len(steady) / (sum(steady) / 1e3)
        if steady
        else None,
    }
    memory = None
    if is_cuda and steady:
        memory = {
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
            / MiB,  # live tensors
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device)
            / MiB,  # held by the caching allocator
        }

    metrics = {
        "tag": args.tag,
        "resumed_from": args.resume,
        "bad_resume": args.bad_resume,
        "start_step": start_step,
        "end_step": end_step,
        "config": asdict(cfg),
        "env": env,
        "n_params": n_params,
        "timing": timing,
        "memory": memory,
        "losses": {
            str(s): v.item() for s, v in losses.items()
        },  # many syncs, but after timing
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"timing": timing, "memory": memory}, indent=2))
    print(f"wrote {out_dir / 'metrics.json'}")


# ----------------------------------------------------------------------------- memory probe
def memory_probe(cfg: Config, device: torch.device) -> None:
    """Walk through two training steps and print memory_allocated after every phase."""
    if device.type != "cuda":
        raise SystemExit("--memory-probe needs a CUDA device")
    setup_reproducibility(cfg)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    rows: list[tuple[str, float, float, float, str]] = []
    last = [base]

    def mark(event: str, expect: str = "") -> None:
        torch.cuda.synchronize()
        now, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        rows.append(
            (
                event,
                (now - base) / MiB,
                (now - last[0]) / MiB,
                (peak - base) / MiB,
                expect,
            )
        )
        last[0] = now
        torch.cuda.reset_peak_memory_stats()  # so the "peak" column means "peak during this phase"

    model, optimizer, _ = build(cfg, device)
    n_params, P = count_params(model)
    mark("build: model.to(cuda)", f"params = {P:.2f}")
    x, y = get_batch(cfg, 0, device)
    mark("get_batch", "x, y (int64, small)")
    logits_mib = cfg.batch_size * cfg.seq_len * cfg.vocab_size * 4 / MiB
    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}

    for it in range(2):
        optimizer.zero_grad(set_to_none=True)
        mark(
            f"[it{it}] zero_grad(set_to_none=True)",
            f"grads freed: -{P:.2f}" if it else "no grads yet",
        )

        # saved_tensors_hooks: autograd calls pack() for every tensor it saves for backward.
        # Summing unique non-parameter storages = activation memory held by the graph.
        saved: dict[int, int] = {}

        def pack(t: torch.Tensor) -> torch.Tensor:
            st = t.untyped_storage()
            if st.data_ptr() not in param_storages:
                saved[st.data_ptr()] = st.nbytes()
            return t

        with (
            torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t),
            attention_ctx(cfg),
        ):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        act = sum(saved.values()) / MiB
        mark(
            f"[it{it}] forward + loss",
            f"saved activations {act:.2f} + logits {logits_mib:.2f}",
        )
        del logits  # only OUR reference: cross_entropy saved log_softmax's output, not the logits
        mark(f"[it{it}] del logits", f"-{logits_mib:.2f}")
        loss.backward()
        mark(f"[it{it}] backward", f"+grads {P:.2f}, -activations {act:.2f}")
        optimizer.step()
        mark(
            f"[it{it}] optimizer.step",
            f"+exp_avg +exp_avg_sq = {2 * P:.2f}" if it == 0 else "no new state",
        )
        del loss

    print(f"\nparams: {n_params:,} | P = {P:.2f} MiB (fp32) | config: {asdict(cfg)}")
    print(f"{'phase':36s} {'alloc':>9s} {'delta':>9s} {'peak':>9s}   expected (MiB)")
    for event, now, delta, peak, expect in rows:
        print(f"{event:36s} {now:9.2f} {delta:+9.2f} {peak:9.2f}   {expect}")
    print(
        f"\nsteady state between steps = params + optimizer state = {3 * P:.2f} MiB expected"
    )
    print(
        f"reserved by caching allocator: {torch.cuda.memory_reserved() / MiB:.2f} MiB "
        f"(nvidia-smi also adds the CUDA context, typically several hundred MiB)"
    )


# ----------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument(
        "--tag", default="run", help="sub-folder of outputs/day01_single_gpu/"
    )
    p.add_argument("--steps", type=int, help="override total_steps")
    p.add_argument("--batch-size", type=int, help="override batch_size")
    p.add_argument(
        "--stop-at",
        type=int,
        help="save a checkpoint at this step and exit (simulated crash)",
    )
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument(
        "--bad-resume",
        action="store_true",
        help="negative control: load model weights only",
    )
    p.add_argument(
        "--no-deterministic",
        action="store_true",
        help="allow fast non-deterministic kernels",
    )
    p.add_argument("--memory-probe", action="store_true")
    p.add_argument(
        "--device", default="cuda", help="'cpu' only for a quick logic smoke test"
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(**PRESETS[args.preset])
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.no_deterministic:
        cfg.deterministic = False

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "this experiment needs a CUDA GPU (use --device cpu for a logic smoke test)"
            )
        device = torch.device(args.device if ":" in args.device else "cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if args.memory_probe:
        memory_probe(cfg, device)
    else:
        train(cfg, args, device)


if __name__ == "__main__":
    main()
