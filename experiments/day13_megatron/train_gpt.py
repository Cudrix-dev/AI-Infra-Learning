"""Day 13 - a small GPT trained by Megatron Core with TP and PP, one profiled steady step.

    # on the instance, from the project root; one process per GPU
    export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM
    torchrun --standalone --nproc_per_node=4 experiments/day13_megatron/train_gpt.py --tp 2 --pp 2 --run-id try1
    torchrun --standalone --nproc_per_node=2 experiments/day13_megatron/train_gpt.py --tp 2 --pp 1 --run-id try1
    torchrun --standalone --nproc_per_node=1 experiments/day13_megatron/train_gpt.py --tp 1 --pp 1 --run-id try1

This is the hand-written equivalent of `pretrain_gpt.py` + `megatron/training/training.py`, reduced to the five
pieces every Megatron Core training job has (GUIDE Block 1, Section 2):

  1. initialize_distributed   torch.distributed + parallel_state.initialize_model_parallel(tp, pp)  -> the groups
  2. model_provider           TransformerConfig + GPTModel(pre_process, post_process) for THIS rank's stage and shard
  3. data iterator            yields one microbatch per next(); every rank consumes the same sequence
  4. forward_step_func        (data_iterator, model) -> (output_tensor, loss_func); Megatron calls it once per microbatch
  5. get_forward_backward_func() -> 1F1B (PP > 1) or no_pipelining (PP = 1); it runs ALL microbatches of the step,
                              the P2P sends/receives, and every backward. Then: grad reductions, optimizer.step().

Where your Day 11/12 code sits:
  - Day 11's column -> row parallel MLP is `mlp.linear_fc1` (ColumnParallelLinear) -> `mlp.linear_fc2`
    (RowParallelLinear) in every TransformerLayer; attention has the same pair (`linear_qkv` -> `linear_proj`).
    The forward all-reduce after each row-parallel linear is your `g`, the backward all-reduce of the column-parallel
    input gradient is your `f`.
  - Day 12's run_1f1b is `forward_backward_pipelining_without_interleaving`; its P2P helpers are P2PCommunicator.

Deliberately simple: Megatron's local layer spec (no Transformer Engine, no apex), FP32 with TF32 off, dropout 0,
untied embeddings, DP = 1 (so no DDP wrapper and no data-parallel gradient sync), Day 01's medium shapes
(d 512, 8 heads, 6 layers, vocab 8192, sequence 256) and 16 sequences per step as 8 microbatches x 2.

Same full model in every layout: `canonical_init` draws every FULL parameter from a generator seeded by its GLOBAL
name and then keeps this rank's TP shard, so TP=1/PP=1, TP=2/PP=1, TP=1/PP=2 and TP=2/PP=2 start from identical
weights and should report the same losses up to FP32 reduction-order rounding (compare_runs.py checks that).

Outputs (per run id and layout tag, e.g. tp2pp2):
  outputs/day13_megatron/run_<RID>/<tag>/rank<r>.json   groups, shapes, losses, step times, memory, parameters
  outputs/day13_megatron/run_<RID>/<tag>/summary.json   rank 0: the group check and the last stage's losses
  outputs/day13_megatron/run_<RID>/<tag>/trace_rank<r>.json.gz   the profiled step (torch.profiler, with stacks)
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import gzip
import json
import os
import re
import shutil
import sys
import time
import zlib
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rank_mapper  # noqa: E402  (pure Python, no torch)

# Day 01 medium shapes.
HIDDEN, HEADS, LAYERS, FFN, VOCAB, SEQ = 512, 8, 6, 2048, 8192, 256

# Microbatch counters, reset every step. Megatron calls forward_step_func and backward_step once per microbatch in
# the schedule's order; on each stage microbatches run forward in ascending order and backward FIFO (Day 12 §2.3),
# so the k-th call is microbatch k. The labels end up as profiler ranges "day13/F mb=k" and "day13/B mb=k".
COUNT = {"F": 0, "B": 0}
PROFILER = {"tool": "torch"}   # set from --profiler in main()


def rng(name: str):
    """A named range: a PyTorch-profiler range by default, an NVTX range with --profiler nsys (no torch profiler
    object exists then; nsys records NVTX from outside Python)."""
    return torch.cuda.nvtx.range(name) if PROFILER["tool"] == "nsys" else record_function(name)


def parse_args():
    ap = argparse.ArgumentParser(description="Day 13: Megatron Core GPT, TP x PP, one profiled step")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--sp", action="store_true", help="sequence parallelism (needs --tp > 1)")
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--profile-step", type=int, default=2,
                    help="step traced WITH Python stacks (attribution: which Megatron frame launched each kernel); -1 = none")
    ap.add_argument("--timing-step", type=int, default=3,
                    help="step traced WITHOUT stacks (order and idle time: with_stack inflates host time ~3x); -1 = none")
    ap.add_argument("--profiler", choices=["torch", "nsys"], default="torch",
                    help="nsys: no PyTorch profiler; NVTX ranges, and the timing step bracketed by cudaProfilerStart/"
                         "Stop for `nsys profile --capture-range=cudaProfilerApi` (run_nsys_check.sh, GUIDE C.6). "
                         "The stack step is then not traced.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mapper", choices=["ref", "mine"], default="ref", help="which rank_mapper to compare against")
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref",
                    help="forward_step_func/loss_func: the reference here, or yours in my_step.py")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="outputs/day13_megatron")
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------- 1. distributed
def initialize_distributed(tp: int, pp: int):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Megatron's TP layers rely on kernels running in launch order (the async dgrad all-reduce must start before the
    # weight-gradient GEMM it overlaps with); one hardware queue per device guarantees that. It must be set before
    # the CUDA context exists, i.e. in the launching shell. GUIDE Block 1, Section 5.
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
        raise SystemExit("Set CUDA_DEVICE_MAX_CONNECTIONS=1 in the launching shell (see GUIDE Section 5).")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world = dist.get_world_size()
    assert world == tp * pp, f"DP must be 1 today: world {world} != tp {tp} x pp {pp}"
    # The only call that creates groups. Default order "tp-cp-ep-dp-pp": TP varies fastest (GUIDE Section 1).
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
                                             create_gloo_process_groups=False)
    # Megatron's TP RNG tracker: identical seeds inside a TP group for replicated work, different ones for sharded
    # work. Attention forks it even with dropout 0, so it must exist.
    model_parallel_cuda_manual_seed(1234)
    return parallel_state


def collect_groups(ps) -> dict:
    """The groups NCCL actually built, as global ranks, read back through the parallel_state getters."""
    def ranks(group):
        return None if group is None else dist.get_process_group_ranks(group)
    return {
        "tp": ranks(ps.get_tensor_model_parallel_group()),
        "pp": ranks(ps.get_pipeline_model_parallel_group()),
        "dp": ranks(ps.get_data_parallel_group()),
        "mp": ranks(ps.get_model_parallel_group()),
        "embedding": ranks(ps.get_embedding_group(check_initialized=False)),
        "position_embedding": ranks(ps.get_position_embedding_group(check_initialized=False)),
    }


# ----------------------------------------------------------------------------------------------- 2. the model
class SequenceParallelTorchLayerNorm(torch.nn.LayerNorm):
    """torch LayerNorm that marks its parameters for the TP gradient all-reduce, used ONLY with --sp.

    Without apex or Transformer Engine the local spec falls back to WrappedTorchNorm, which refuses sequence
    parallelism (transformer/torch_norm.py:50, found in the pre-test). The math is not the problem: LayerNorm works
    per token, so a rank holding 128 of the 256 positions normalizes them correctly. The problem is the weight and
    bias gradients: each rank sees only its tokens, so the gradients are partial sums that must be all-reduced over
    TP. apex's FusedLayerNorm flags them with `sequence_parallel=True` (fusions/fused_layer_norm.py:119-120) and
    finalize_model_grads reduces every flagged parameter. This class does the same flagging, nothing else.
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-5, **_unused):
        super().__init__(hidden_size, eps=eps)
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)
        setattr(self.bias, "sequence_parallel", config.sequence_parallel)


def layer_spec(args, config, ps):
    """Megatron's local spec; with --sp, its three LayerNorms (input, pre-MLP, final) use the class above."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer import transformer_block

    spec = get_gpt_layer_local_spec()
    if not args.sp:
        return spec
    spec.submodules.input_layernorm = SequenceParallelTorchLayerNorm
    spec.submodules.pre_mlp_layernorm = SequenceParallelTorchLayerNorm
    n = transformer_block.get_num_layers_to_build(config, vp_stage=None, pp_rank=ps.get_pipeline_model_parallel_rank())
    return transformer_block.TransformerBlockSubmodules(layer_specs=[spec] * n,
                                                        layer_norm=SequenceParallelTorchLayerNorm)


def model_provider(args, ps):
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=LAYERS, hidden_size=HIDDEN, num_attention_heads=HEADS, ffn_hidden_size=FFN,
        tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp, sequence_parallel=args.sp,
        params_dtype=torch.float32, pipeline_dtype=torch.float32,       # pipeline_dtype: dtype of the P2P buffers
        hidden_dropout=0.0, attention_dropout=0.0,
        perform_initialization=False,           # canonical_init below writes every parameter
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,     # needs apex's fused wgrad kernel
        bias_activation_fusion=False, masked_softmax_fusion=False, bias_dropout_fusion=False,
        persist_layer_norm=False, apply_rope_fusion=False,
    )
    # pre_process: this stage owns the embedding; post_process: this stage owns the output layer and the loss.
    # With PP = 2: stage 0 = embedding + layers 0-2, stage 1 = layers 3-5 + final norm + output layer.
    model = GPTModel(
        config=config, transformer_layer_spec=layer_spec(args, config, ps), vocab_size=VOCAB,
        max_sequence_length=SEQ, pre_process=ps.is_pipeline_first_stage(), post_process=ps.is_pipeline_last_stage(),
        parallel_output=True,                   # logits stay vocab-sharded; the loss is vocab-parallel cross-entropy
        share_embeddings_and_output_weights=False,  # untied: no embedding-group all-reduce (GUIDE Section 1.4)
        position_embedding_type="learned_absolute",
    )
    return model.cuda(), config


def global_param_name(model, name: str) -> str:
    """decoder.layers.<local i>.x -> decoder.layers.<global i>.x (stage 1 holds local layers 0-2 = global 3-5)."""
    m = re.match(r"decoder\.layers\.(\d+)\.(.*)", name)
    if not m:
        return name
    layer = model.decoder.layers[int(m.group(1))]
    return f"decoder.layers.{layer.layer_number - 1}.{m.group(2)}"   # layer_number is global and 1-based


@torch.no_grad()
def canonical_init(model, tp: int, tp_rank: int) -> list[dict]:
    """Draw each FULL parameter from a generator seeded by its global name, keep this rank's TP shard.

    Megatron marks a sharded parameter with tensor_model_parallel=True, partition_dim and partition_stride
    (tensor_parallel/layers.py, set_tensor_model_parallel_attributes). Contiguous chunks along partition_dim are
    exactly Megatron's shards: e.g. linear_qkv's rows are grouped per attention head ([q k v] per head), so TP rank 0
    gets heads 0-3 with their q, k and v.
    """
    record = []
    for name, p in model.named_parameters():
        gname = global_param_name(model, name)
        sharded = bool(getattr(p, "tensor_model_parallel", False))
        full_shape = list(p.shape)
        dim = None
        if sharded:
            dim = p.partition_dim
            assert getattr(p, "partition_stride", 1) == 1, (name, p.partition_stride)
            full_shape[dim] *= tp
        g = torch.Generator().manual_seed(zlib.crc32(gname.encode()))
        if "layernorm" in gname:
            full = torch.ones(full_shape) if gname.endswith("weight") else torch.zeros(full_shape)
        else:
            full = torch.randn(full_shape, generator=g) * 0.02
        if sharded:
            full = full.chunk(tp, dim=dim)[tp_rank]
        p.copy_(full)
        record.append({"name": name, "global": gname, "shape": list(p.shape), "full_shape": full_shape,
                       "tp_sharded": sharded, "partition_dim": dim,
                       "sequence_parallel": bool(getattr(p, "sequence_parallel", False))})
    return record


# --------------------------------------------------------------------------------------------------- 3. data
def make_global_batch(args) -> list[dict]:
    """One fixed global batch of random tokens, reused every step (so the loss visibly falls). Every rank builds the
    same tensors from the same seed: TP ranks must see the same tokens, and PP stages only use what they need
    (stage 0 the tokens, the last stage the labels)."""
    g = torch.Generator().manual_seed(args.seed)
    n = args.microbatches * args.micro_batch_size
    toks = torch.randint(0, VOCAB, (n, SEQ + 1), generator=g)
    pos = torch.arange(SEQ).unsqueeze(0).expand(args.micro_batch_size, SEQ).contiguous()
    b = args.micro_batch_size
    return [{"tokens": toks[i * b:(i + 1) * b, :SEQ].contiguous(), "labels": toks[i * b:(i + 1) * b, 1:].contiguous(),
             "position_ids": pos} for i in range(args.microbatches)]


# ------------------------------------------------------------------------------------------- 4. forward_step_func
def ref_loss_func(output_tensor: torch.Tensor):
    """Last stage only. output_tensor = per-token cross-entropy [b, s] (GPTModel computes it when labels are given).
    Megatron then divides by num_microbatches (schedules.py forward_step_calc_loss), Day 04's 1/M weight."""
    loss = output_tensor.float().mean()
    # clone(): Megatron scales the returned loss IN PLACE (`output_tensor /= num_microbatches`), and detach() alone
    # shares storage, so the logged value would silently become loss / M (found in the pre-test: 1.14 vs 9.12).
    return loss, {"lm loss": loss.detach().clone()}


def ref_forward_step_func(data_iterator, model):
    # Every stage takes one microbatch per call, even if it only needs part of it: each rank iterates its own copy,
    # and the k-th call on every stage must be microbatch k (the last stage needs microbatch k's labels).
    mb = next(data_iterator)
    dev = torch.cuda.current_device()
    # attention_mask=None: the local spec's causal softmax builds the causal mask itself
    # (fusions/fused_softmax.py, `attn_mask_type == AttnMaskType.causal and mask is None`).
    # On a non-first stage the tokens are ignored: the model reads the received activation that the schedule
    # handed over with set_input_tensor (schedules.py forward_step).
    out = model(mb["tokens"].to(dev), mb["position_ids"].to(dev), None, labels=mb["labels"].to(dev))
    return out, ref_loss_func


def traced(forward_step_func):
    """Wrap a forward_step_func so its model forward shows up as the profiler range "day13/F mb=k". The returned
    loss_func runs later (Megatron calls it after the forward), outside the range."""
    def forward_step(data_iterator, model):
        k = COUNT["F"]
        COUNT["F"] += 1
        with rng(f"day13/F mb={k}"):
            return forward_step_func(data_iterator, model)
    return forward_step


def load_step_functions(impl: str):
    """ref -> the functions above; mine -> my_step.py. Exits with code 3 while a TODO remains (before any
    process group exists, so no rank hangs)."""
    if impl == "ref":
        return ref_forward_step_func
    import inspect
    import my_step
    todo = [f.__name__ for f in (my_step.loss_func, my_step.forward_step_func)
            if "raise NotImplementedError" in inspect.getsource(f)]
    if todo:
        print(f"INCOMPLETE (TODOs left): {', '.join(todo)} in my_step.py")
        sys.exit(3)
    return my_step.forward_step_func


def install_backward_ranges():
    """Wrap Megatron's backward_step in a profiler range. The schedules look `backward_step` up as a module global
    at call time (schedules.py: `backward_func = backward_step`), so replacing the module attribute is enough.
    This changes no numerics: the wrapper only adds a named range around the original call."""
    from megatron.core.pipeline_parallel import schedules
    original = schedules.backward_step

    def backward_step(*a, **kw):
        k = COUNT["B"]
        COUNT["B"] += 1
        with rng(f"day13/B mb={k}"):
            return original(*a, **kw)

    schedules.backward_step = backward_step


@torch.no_grad()
def grad_record(model, params: list[dict]) -> dict:
    """Step 0, before optimizer.step(): sum of squares (FP64) of this rank's gradient of each parameter, keyed by
    GLOBAL name. compare_runs.py rebuilds each full parameter's gradient norm by adding the TP shards' sums (and
    taking replicated parameters from TP rank 0) and compares layouts. The scalar loss is too coarse for that: it is
    a mean over 4096 tokens, and per-element FP32 differences average out below its last bit."""
    out = {}
    for (name, p), meta in zip(model.named_parameters(), params):
        assert meta["name"] == name
        g = p.grad
        out[meta["global"]] = {"sumsq": float(g.double().pow(2).sum()) if g is not None else None,
                               "tp_sharded": meta["tp_sharded"]}
    return out


def allreduce_sequence_parallel_grads(model, tp_group):
    """With SP, LayerNorm weights/biases and row-parallel biases see only this rank's sequence shard, so their
    gradients are partial sums: all-reduce them over TP (one coalesced call). Megatron does the same in
    distributed/finalize_model_grads.py `_allreduce_non_tensor_model_parallel_grads`; we call it by hand because we
    skip the DDP wrapper at DP = 1."""
    grads = [p.grad for p in model.parameters() if getattr(p, "sequence_parallel", False) and p.grad is not None]
    if not grads:
        return 0
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, group=tp_group)
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)
    return len(grads)


# ------------------------------------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    PROFILER["tool"] = args.profiler
    assert not args.sp or args.tp > 1, "--sp needs --tp > 1"
    # FP32 with TF32 off everywhere, so layouts are comparable to FP32 rounding.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    step_fn = traced(load_step_functions(args.impl))

    ps = initialize_distributed(args.tp, args.pp)
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    rank, world = dist.get_rank(), dist.get_world_size()
    tag = f"tp{args.tp}pp{args.pp}" + ("_sp" if args.sp else "") + ("_mine" if args.impl == "mine" else "")
    out_dir = Path(args.out) / f"run_{args.run_id}" / tag
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=False)   # refuse to overwrite a previous run
    dist.barrier()

    # ---- the group check: what Megatron built vs your mapper -------------------------------------------------
    tp_rank, pp_rank = ps.get_tensor_model_parallel_rank(), ps.get_pipeline_model_parallel_rank()
    dp_rank = ps.get_data_parallel_rank()
    mine = collect_groups(ps)
    everyone = [None] * world
    dist.all_gather_object(everyone, {"rank": rank, "coords": [tp_rank, dp_rank, pp_rank], "groups": mine,
                                     "device": torch.cuda.get_device_name(), "cuda_visible":
                                     os.environ.get("CUDA_VISIBLE_DEVICES")})
    group_check = None
    if rank == 0:
        built = {k: sorted({tuple(e["groups"][k]) for e in everyone if e["groups"][k] is not None})
                 for k in rank_mapper.GROUP_KINDS}
        built = {k: [list(g) for g in v] for k, v in built.items()}
        fn_c, fn_g = ((rank_mapper.ref_coords, rank_mapper.ref_groups) if args.mapper == "ref"
                      else (rank_mapper.my_coords, rank_mapper.my_groups))
        try:
            expected = rank_mapper.canon(fn_g(args.tp, 1, args.pp))
            coords_ok = all(tuple(e["coords"]) == fn_c(e["rank"], args.tp, 1, args.pp) for e in everyone)
        except NotImplementedError as e:
            print(f"INCOMPLETE (TODOs left): {e}. Use --mapper ref, or finish rank_mapper.py.")
            expected, coords_ok = None, False
        group_check = {"built": built, "expected": expected, "coords_ok": coords_ok,
                       "passed": coords_ok and built == expected, "mapper": args.mapper}
        print(f"[groups] layout DP=1 TP={args.tp} PP={args.pp}, order tp-cp-ep-dp-pp")
        print("rank  (tp, dp, pp)  TP          PP          DP     MP              embedding   pos-emb   device")
        for e in everyone:
            g = e["groups"]
            print(f"{e['rank']:>4}  {str(tuple(e['coords'])):<12} {str(g['tp']):<11} {str(g['pp']):<11} "
                  f"{str(g['dp']):<6} {str(g['mp']):<15} {str(g['embedding']):<11} {str(g['position_embedding']):<9}"
                  f" {e['device']} (visible {e['cuda_visible']})")
        print(f"[groups] built by Megatron == {args.mapper} mapper: "
              f"{'PASSED' if group_check['passed'] else 'FAILED'}")

    # ---- model, data, schedule -------------------------------------------------------------------------------
    model, config = model_provider(args, ps)
    params = canonical_init(model, args.tp, tp_rank)
    n_params = sum(p.numel() for p in model.parameters())
    install_backward_ranges()
    fb_func = get_forward_backward_func()      # 1F1B for PP > 1, no_pipelining for PP = 1
    p2p_shapes = schedules.get_tensor_shapes(seq_length=SEQ, micro_batch_size=args.micro_batch_size,
                                             decoder_seq_length=SEQ, config=config,
                                             tp_group=ps.get_tensor_model_parallel_group(),
                                             cp_group=ps.get_context_parallel_group())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch = make_global_batch(args)
    tp_group = ps.get_tensor_model_parallel_group()
    is_last = ps.is_pipeline_last_stage()
    if rank == 0:
        print(f"[model] {LAYERS} layers d={HIDDEN} heads={HEADS} ffn={FFN} vocab={VOCAB} seq={SEQ}; "
              f"global batch {args.microbatches} x {args.micro_batch_size}; SP {'on' if args.sp else 'off'}; "
              f"schedule {fb_func.__name__}; P2P tensor shapes {p2p_shapes}")
    print(f"[rank {rank}] stage {pp_rank} (pre_process={model.pre_process}, post_process={model.post_process}), "
          f"tp_rank {tp_rank}: {len(params)} tensors, {n_params / 1e6:.3f} M parameters", flush=True)

    steps, grad_sumsq = [], None
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        COUNT.update(F=0, B=0)
        optimizer.zero_grad(set_to_none=True)
        nsys = args.profiler == "nsys"
        stacks = step == args.profile_step and not nsys
        profiling = stacks or step == args.timing_step
        prof = (profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=stacks,
                        with_stack=stacks) if profiling and not nsys else contextlib.nullcontext())
        torch.cuda.synchronize()
        capture_start_ms = None
        if profiling and nsys:
            # cudaProfilerStart: nsys starts recording here. Turning tracing on takes a different time on each rank,
            # so it must come BEFORE the barrier; after it, the late ranks start the step late and the others wait
            # for them at the first P2P (GUIDE C.6).
            c0 = time.perf_counter()
            torch.cuda.profiler.start()
            capture_start_ms = (time.perf_counter() - c0) * 1e3
        dist.barrier()
        t0 = time.perf_counter()
        with prof:
            with rng(f"day13/step {step}"):
                losses = fb_func(forward_step_func=step_fn, data_iterator=iter(batch), model=model,
                                 num_microbatches=args.microbatches, seq_length=SEQ,
                                 micro_batch_size=args.micro_batch_size, decoder_seq_length=SEQ, forward_only=False)
                n_sp = allreduce_sequence_parallel_grads(model, tp_group) if args.sp else 0
                if step == 0:
                    grad_sumsq = grad_record(model, params)
                optimizer.step()
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if profiling and nsys:
            torch.cuda.profiler.stop()    # outside the timed region; nsys writes its report when the process exits
        loss = float(torch.stack([d["lm loss"] for d in losses]).mean()) if is_last and losses else None
        kind = "stack" if stacks else (("nsys" if nsys else "timing") if profiling else None)
        steps.append({"step": step, "ms": dt, "loss": loss, "profiled": kind, "sp_grads_reduced": n_sp,
                      "forward_calls": COUNT["F"], "backward_calls": COUNT["B"],
                      **({"capture_start_ms": capture_start_ms} if capture_start_ms is not None else {})})
        if profiling and not nsys:
            raw = out_dir / (f"trace_rank{rank}.json" if stacks else f"timing_rank{rank}.json")
            prof.export_chrome_trace(str(raw))
            with raw.open("rb") as src, gzip.open(str(raw) + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            raw.unlink()
            # Free the profiler NOW, outside the timed step. torch.profiler.profile holds bound methods of itself
            # (self.action_map), a reference cycle, so dropping the name does not free it; the cyclic GC does, at
            # its next run - which used to land inside F1 of the next step as a ~50 ms stall on every rank
            # (GUIDE Appendix B, stall_probe.py). gc.collect() here moves that cost between steps.
            del prof
            gc.collect()
        if is_last and tp_rank == 0:
            print(f"[step {step}] loss {loss:.7f}  {dt:8.1f} ms{f'  (profiled: {kind})' if kind else ''}", flush=True)

    record = {"rank": rank, "world": world, "tp": args.tp, "pp": args.pp, "sp": args.sp,
              "coords": {"tp": tp_rank, "dp": dp_rank, "pp": pp_rank}, "groups": mine,
              "pre_process": model.pre_process, "post_process": model.post_process,
              "p2p_tensor_shapes": [list(s) for s in p2p_shapes], "schedule": fb_func.__name__,
              "microbatches": args.microbatches, "micro_batch_size": args.micro_batch_size, "seq": SEQ,
              "steps": steps, "n_params": n_params, "params": params, "grad_sumsq_step0": grad_sumsq,
              "cuda_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
              "env": {"CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
                      "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version()))}}
    (out_dir / f"rank{rank}.json").write_text(json.dumps(record, indent=1))
    gathered = [None] * world
    dist.all_gather_object(gathered, {"rank": rank, "losses": [s["loss"] for s in steps], "is_last": is_last})
    if rank == 0:
        last = [g for g in gathered if g["is_last"]]
        tp_equal = all(g["losses"] == last[0]["losses"] for g in last)
        summary = {"tag": tag, "tp": args.tp, "pp": args.pp, "sp": args.sp, "impl": args.impl,
                   "group_check": group_check, "step_ms": [s["ms"] for s in steps], "profiled":
                   [s["profiled"] for s in steps],
                   "losses": last[0]["losses"], "last_stage_tp_ranks_equal": tp_equal,
                   "p2p_tensor_shapes": [list(s) for s in p2p_shapes], "schedule": fb_func.__name__}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        ok = group_check["passed"] and tp_equal
        print(f"RUN {tag}: groups {'ok' if group_check['passed'] else 'FAIL'}, last-stage TP ranks report the same "
              f"loss {'ok' if tp_equal else 'FAIL'} -> {'PASSED' if ok else 'FAILED'}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
