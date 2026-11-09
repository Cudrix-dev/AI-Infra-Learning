"""Day 08 - the part you write: put Day 01's model under DeepSpeed ZeRO stage 0 / 1 / 2 / 3.

Day 06 wrote ZeRO-1 by hand (one flat buffer, reduce-scatter, AdamW on a slice, all-gather). Day 07 used PyTorch's
ZeRO-3 (FSDP2: one group per fully_shard call). Today the same model runs under DeepSpeed, where ONE config number,
`zero_optimization.stage`, chooses what is sharded:

    stage 0   nothing. DDP-like: full params, full grads, full AdamW state; all-reduce at the window boundary
    stage 1   AdamW state (+ DeepSpeed's fp32 "master" copy of the params) sharded; grads reduced ONCE per window
    stage 2   + gradients sharded: every microbatch's gradients are reduced in buckets DURING backward and only this
              rank's slice is kept (accumulated in an fp32 partition)
    stage 3   + parameters sharded: every submodule all-gathers its parameters just before its forward/backward and
              releases them after; gradients are reduce-scattered in buckets during backward

The engine does the rest: `deepspeed.initialize` wraps the model (hooks), replaces the param_groups of YOUR optimizer
with flat fp32 partitions, and owns backward() and step(). What you control is the config and the accumulation
contract, which is where real training bugs come from. Two parts are YOURS today:

    YOU WRITE (1)  make_ds_config: the config dict for one stage, with buckets SIZED FOR THIS MODEL (or DeepSpeed's
                   defaults, to see why defaults tuned for billion-parameter models are a memory trap here)
    YOU WRITE (2)  ds_microbatch: one microbatch inside Day 04's window, with Day 04's own loss weights and a short
                   final window. DeepSpeed's canonical loop (engine.backward divides by gradient_accumulation_steps,
                   engine.step() counts micro-steps) silently does the wrong thing for both.

Everything below the YOU WRITE parts is given: evidence helpers that reassemble full gradients / parameters / AdamW
state from the shards (collectives: every rank must call them in the same order), and a tensor census that names the
buffers DeepSpeed allocated (the memory table of today's done-when).

Rewrite the YOU WRITE bodies yourself (guide Block 2), then diff against this file.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
from torch import nn

MiB = 2**20
STAGES = (0, 1, 2, 3)
BUGS = ("none", "double_scale", "fixed_gas")

# One bucket = 25 MiB of FP32 = 6,553,600 elements: DDP's default bucket_cap_mb, so the stage 0/1/2 buckets are
# comparable with Day 03/07's DDP. DeepSpeed's own defaults are 5e8 elements (1907 MiB of FP32) PER BUFFER.
BUCKET_NUMEL = 25 * MiB // 4
# One transformer block of the medium preset = 3,152,384 elements (Day 07 guide 1.2). Stage 3 "sized": prefetch at
# most one block ahead and do not keep a block gathered just because it is reused later (FSDP2 per-block's behaviour).
BLOCK_NUMEL = 3_152_384


# ============================================================================ YOU WRITE (1): the config
def make_ds_config(stage: int, micro_batch: int, accum: int, world: int, buckets: str = "sized",
                   overlap_comm: bool | None = None, persist_threshold: int | None = None) -> dict:
    """The DeepSpeed config for one run. FP32 everywhere (precision is Day 9), no clipping, no offload.

    buckets="sized"   : buckets and stage-3 prefetch sized for a 27M-parameter model (see constants above)
    buckets="default" : leave every bucket / prefetch / reuse knob at DeepSpeed's default (the defaults trap)
    overlap_comm=None : DeepSpeed's own default (True for stage 3, False otherwise); True/False to force it
    persist_threshold : stage 3 only; None keeps DeepSpeed's 1e5. The parity stage passes 1000: every TINY-model
                        tensor has < 1e5 elements, so with the default stage 3 would partition nothing at all

    Keys worth knowing (deepspeed/runtime/zero/config.py):
      reduce_bucket_size        elements per gradient-reduction bucket. With contiguous_gradients (default True)
                                the hooks copy each gradient into a PRE-ALLOCATED buffer of exactly this many
                                elements (two buffers with overlap_comm), whatever the model size
      allgather_bucket_size     elements per all-gather of updated parameters after the stage 1/2 step
      stage3_prefetch_bucket_size   elements stage 3 may all-gather AHEAD of the module that needs them
      stage3_max_reuse_distance     a parameter reused within this many elements is NOT released (stays gathered)
      stage3_max_live_parameters    cap on gathered elements before stage 3 starts releasing
      stage3_param_persistence_threshold   parameters smaller than this are never partitioned (kept full on every
                                rank; biases and LayerNorms of the medium model). Default 1e5 unless persist_threshold
    """
    zero: dict = {"stage": stage}
    if buckets == "sized" and stage >= 1:
        zero["reduce_bucket_size"] = BUCKET_NUMEL
        zero["allgather_bucket_size"] = BUCKET_NUMEL
        if stage == 3:
            zero["stage3_prefetch_bucket_size"] = BLOCK_NUMEL
            zero["stage3_max_reuse_distance"] = BLOCK_NUMEL
            zero["stage3_max_live_parameters"] = 3 * BLOCK_NUMEL
    elif buckets != "default" and buckets != "sized":
        raise ValueError("buckets must be 'sized' or 'default'")
    if overlap_comm is not None:
        zero["overlap_comm"] = overlap_comm
    if stage == 3 and persist_threshold is not None:
        zero["stage3_param_persistence_threshold"] = persist_threshold
    return {
        # DeepSpeed checks train_batch_size == micro_batch * gradient_accumulation_steps * world. These describe a
        # FULL window; the short final window is handled by set_gradient_accumulation_boundary (YOU WRITE (2)).
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": accum,
        "train_batch_size": micro_batch * accum * world,
        "zero_optimization": zero,
        # We pass Day 01's torch.optim.AdamW (a "client" optimizer), not DeepSpeed's FusedAdam: same math as every
        # earlier day, no CUDA extension to JIT-compile. ZeRO refuses untested client optimizers unless told to.
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 0.0,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
    }


# ============================================================================ YOU WRITE (2): one microbatch
def ds_microbatch(engine, loss_fn, x: torch.Tensor, y: torch.Tensor, weight: float, last: bool,
                  bug: str = "none", phase=None) -> torch.Tensor:
    """Forward + backward of ONE microbatch through the engine. Returns the UNWEIGHTED loss (for logging).

    Day 04's contract: microbatch j is weighted by rows_j / rows_in_window, the gradient of the window is the sum of
    the weighted microbatch gradients, and the window may be SHORT (the last one has 2 microbatches, not 4).

    DeepSpeed's defaults assume a fixed window of `gradient_accumulation_steps` microbatches:
      * engine.backward(loss) divides the loss by gradient_accumulation_steps (scale_wrt_gas=True). We already weight
        by rows: pass scale_wrt_gas=False, or the gradient is divided by K twice (--bug double_scale).
      * the boundary (reduce for stage 0/1, optimizer step) is (micro_steps + 1) % gas == 0. A short final window never
        reaches it. set_gradient_accumulation_boundary(last) BEFORE each forward overrides the counter (--bug fixed_gas
        keeps the counter: the last window's update never happens).
    The caller calls engine.step() once after the last microbatch (with the override, step() acts on the boundary).
    `phase("fwd")` / `phase("bwd")` are the harness's context managers (profiler ranges, attention backend, memory
    timeline); they do not change the math.
    """
    phase = phase or (lambda _kind: nullcontext())
    if bug != "fixed_gas":
        engine.set_gradient_accumulation_boundary(last)  # stage 0/1 reduce in the epilogue of THIS backward if last
    with phase("fwd"):
        loss = loss_fn(engine(x), y)
    with phase("bwd"):
        if bug == "double_scale":
            engine.backward(loss * weight)  # BUG: DeepSpeed ALSO divides by gradient_accumulation_steps
        else:
            engine.backward(loss * weight, scale_wrt_gas=False)
    return loss.detach()


# ============================================================================ given: optimizer / evidence helpers
def quiet_deepspeed() -> None:
    """DeepSpeed logs a lot at INFO on every rank; keep warnings."""
    from deepspeed.utils import logger
    logger.setLevel(logging.WARNING)


def base_optimizer(engine) -> torch.optim.Optimizer:
    """The torch AdamW that actually runs. Stage 0: the client optimizer itself. Stage >= 1: wrapped by DeepSpeed's
    ZeRO optimizer (`.optimizer`), with param_groups rewritten to hold ONE flat fp32 partition per group."""
    opt = engine.optimizer
    return getattr(opt, "optimizer", opt)


def adam_step(engine) -> float:
    st = base_optimizer(engine).state
    return float(next(iter(st.values()))["step"]) if st else 0.0


def full_grads(engine, names, params, stage: int) -> dict:
    """The AVERAGED gradient each parameter received, full shape, on CPU. Call between the boundary backward and
    engine.step() (afterwards ZeRO frees / zeroes it). COLLECTIVE for stage >= 1: every rank calls, same order.

    stage 0: p.grad was all-reduced in place by the boundary backward's epilogue.
    stage 1/2: safe_get_full_grad -> the param's fragment of this rank's averaged fp32 gradient partition, placed in a
               zero buffer and all-reduced so every rank gets the full tensor.
    stage 3: the same through the stage-3 optimizer (its grad partition buffer)."""
    from deepspeed.utils import safe_get_full_grad
    out = {}
    for n, p in zip(names, params):
        try:
            g = p.grad if stage == 0 else safe_get_full_grad(p)
        except ValueError:  # DeepSpeed: "only available immediately after backward and before engine step"
            g = None  # e.g. --bug fixed_gas: no boundary was reached, so there is no reduced gradient to read
        out[n] = torch.zeros(full_shape(p), dtype=torch.float32) if g is None else g.detach().float().cpu().clone()
    return out


def full_params(engine, names, params, stage: int) -> dict:
    """Full parameters on CPU (stage >= 1: DeepSpeed's fp32 MASTER copy, gathered; collective)."""
    from deepspeed.utils import safe_get_full_fp32_param
    return {n: (p.detach() if stage == 0 else safe_get_full_fp32_param(p)).float().cpu().clone()
            for n, p in zip(names, params)}


def full_state(engine, names, params, stage: int) -> dict:
    """AdamW moments per parameter name, full shape (collective for stage >= 1)."""
    from deepspeed.utils import safe_get_full_optimizer_state
    out = {}
    base = base_optimizer(engine)
    for n, p in zip(names, params):
        if stage == 0:
            if p not in base.state:
                continue
            st = {k: base.state[p][k] for k in ("exp_avg", "exp_avg_sq")}
        else:
            st = {k: safe_get_full_optimizer_state(p, k) for k in ("exp_avg", "exp_avg_sq")}
            if any(v is None for v in st.values()):
                continue
        out[n] = {k: v.detach().float().cpu().clone() for k, v in st.items()}
    return out


def full_shape(p: torch.Tensor) -> tuple:
    """Stage 3 replaces p.data by an EMPTY tensor between uses; the original shape is kept in p.ds_shape."""
    return tuple(getattr(p, "ds_shape", p.shape))


def shard_report(params, engine, stage: int) -> dict:
    """Elements of model parameters THIS rank keeps resident between steps, and the fp32 master elements.
    Stage 3: p.ds_tensor is the rank's partition (p.data is empty unless gathered); persistent params (ds_persist) are
    ALSO kept full in p.data on every rank. Stage 1/2: the full model params (views of one flat buffer)."""
    full = local = persistent = 0
    for p in params:
        n = torch.Size(full_shape(p)).numel()
        full += n
        if hasattr(p, "ds_tensor"):
            local += p.ds_tensor.numel()
            if getattr(p, "ds_persist", False):
                persistent += n
                local += n
        else:
            local += p.numel()
    master = 0
    if stage >= 1:
        zero = engine.optimizer
        flat = getattr(zero, "fp32_partitioned_groups_flat", None) or getattr(zero, "single_partition_of_fp32_groups", [])
        master = sum(t.numel() for t in flat)
    return {"full_numel": full, "local_numel": local, "persistent_numel": persistent, "master_numel": master}


def optimizer_state_report(engine) -> dict:
    by_key = {}
    for st in base_optimizer(engine).state.values():
        for k, v in st.items():
            if torch.is_tensor(v) and v.is_cuda:
                by_key[k] = by_key.get(k, 0) + v.numel() * v.element_size()
    return {"state_bytes": by_key}


# ============================================================================ given: tensor census
def tensor_census(engine, model: nn.Module, device: torch.device) -> dict:
    """Name every CUDA storage the training state holds, in MiB, each storage counted ONCE (first name wins).

    Walk order (so shared storage gets the most meaningful name): model parameters (p.data; stage 1/2 params are views
    of DeepSpeed's flat buffer), stage-3 partitions (p.ds_tensor), model buffers, the torch optimizer's state, then
    every instance attribute of DeepSpeed's ZeRO optimizer (depth-limited: lists, dicts, plain objects). The result
    also reports memory_allocated() and the part the walk did not find (cuBLAS workspaces, allocator rounding, and
    anything DeepSpeed keeps outside the walked objects). Names are DeepSpeed's own attribute names, so they track
    the installed version; read them next to the source (guide Block 3)."""
    seen: dict[int, str] = {}
    sizes: dict[str, int] = {}

    def add(t, name):
        if not (torch.is_tensor(t) and t.is_cuda and t.device == device):
            return
        s = t.untyped_storage()
        key = s.data_ptr()
        if key == 0 or key in seen:
            return
        seen[key] = name
        sizes[name] = sizes.get(name, 0) + s.nbytes()

    for p in model.parameters():
        add(p.data, "model params (p.data)")
        if p.grad is not None:
            add(p.grad, "model grads (p.grad)")
        if hasattr(p, "ds_tensor"):
            add(p.ds_tensor, "stage3 param partitions (p.ds_tensor)")
    for b in model.buffers():
        add(b, "model buffers (causal mask)")
    base = base_optimizer(engine)
    for g in base.param_groups:
        for t in g["params"]:
            add(t, "optimizer params (fp32 master partition)")
            if t.grad is not None:
                add(t.grad, "optimizer grads (fp32 grad partition)")
    for st in base.state.values():
        for k, v in st.items():
            add(v, f"optimizer state {k}")

    visited: set[int] = set()

    def walk(obj, name, depth):
        if depth > 4 or id(obj) in visited or isinstance(obj, (nn.Module, torch.optim.Optimizer)):
            return
        visited.add(id(obj))
        if torch.is_tensor(obj):
            add(obj, name)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v, name, depth + 1)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, name, depth + 1)
        elif type(obj).__module__.startswith("deepspeed") and hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                walk(v, name if depth > 0 else f"zero.{k}", depth + 1)

    if engine.optimizer is not base:  # stage >= 1
        for k, v in vars(engine.optimizer).items():
            walk(v, f"zero.{k}", 1)
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device)
    found = sum(sizes.values())
    out = {k: v / MiB for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])}
    out["(sum of named storages)"] = found / MiB
    out["(memory_allocated)"] = alloc / MiB
    out["(not found by the walk)"] = (alloc - found) / MiB
    return out
