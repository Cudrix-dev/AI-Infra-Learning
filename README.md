# AI Infra Learning

A 20-day study of training and RL infrastructure (Oct 2026). Each day builds one distributed-training mechanism from scratch, then uses the real framework, and the last week wires everything into an end-to-end RL pipeline: rollout → train → weight sync.

Three skills, practised every day:

- **Implementation:** a minimal working version of the mechanism, without high-level frameworks.
- **Source literacy:** trace one call chain to find where communication, parameter updates and scheduling happen.
- **Analysis:** explain why something is fast or slow, backed by correctness checks and profiler data.

Two workloads only, so results stay comparable: a **small transformer on synthetic tokens** (DDP, FSDP, ZeRO, TP, PP, Triton) and an **existing PPO/SAC** implementation (parallel rollout, weight sync, CUDA Graphs, RL throughput).

## Plan

### Days 1–5: single-GPU baseline to DDP internals

When parameters, gradients, optimizer state and activations are allocated and freed, and when gradient communication fires during backward.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 1 | Reproducible single-GPU baseline | Training loop, checkpoint save/resume, timing and peak memory | Resume matches uninterrupted training |
| 2 | Data parallel without DDP | Broadcast params, `all_reduce` grads; ring all-reduce from `send/recv` | N GPUs match 1 GPU on the same global batch |
| 3 | DDP + autograd gradient hooks | Mini bucketed DDP with `register_post_accumulate_grad_hook` | Per-param vs own buckets vs DDP in the profiler |
| 4 | Gradient accumulation | Microbatches with and without `no_sync()` | Matches one equivalent large-batch step |
| 5 | CUDA Graph pitfalls | Eager vs `torch.compile` vs manual graph capture | Parity over several steps; capture vs replay explained |

### Days 6–10: sharding, precision and kernels

Which memory category each technique cuts, what it costs in communication, and when a hand-written kernel beats PyTorch.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 6 | ZeRO-1 from scratch | AdamW state sharded across ranks | Matches unsharded AdamW; state per rank logged |
| 7 | FSDP2 | `fully_shard` on the whole model vs per block | DDP vs FSDP2 memory and time; `all_gather` / `reduce_scatter` found |
| 8 | DeepSpeed ZeRO in practice | Same model under ZeRO stages 1, 2, 3 | Memory table per stage; why a higher stage isn't faster |
| 9 | Mixed precision + activation checkpointing | `autocast`, grad clipping, checkpointing | Drift, memory and time for FP32 vs BF16/FP16 |
| 10 | Triton fused softmax | Row-wise fused softmax (stretch: RMSNorm) | Correct across row lengths; GB/s vs eager and `torch.compile` |

### Days 11–15: model parallelism, MoE and RL sampling

Split a model by tensor, sequence, pipeline and expert dimensions, and decouple rollout workers from the learner.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 11 | Tensor parallel MLP | Column- then row-parallel linear with custom autograd Functions | Outputs and grads match 1 GPU |
| 12 | GPipe + 1F1B | 2-stage pipeline; CPU schedule simulator | Bubble and peak in-flight activations for GPipe vs 1F1B |
| 13 | Process groups + sequence parallel | Rank → (DP, TP, PP) mapping; DP=2 × TP=2 on 4 GPUs | Correct groups; TP vs TP+SP communication drawn |
| 14 | MoE + expert parallelism | Top-1 MoE layer (stretch: `all_to_all` dispatch) | Load balance under uniform vs skewed inputs |
| 15 | Ray + RLlib | 2 CPU rollout actors + 1 GPU learner, tagged with `policy_version` | Time breakdown per phase; staleness explained |

### Days 16–20: LLM RL pipeline, capstone and report

Move weights from a sharded trainer to an inference engine, understand rollout scheduling, trace one verl update, and finish with a distributed PPO backend.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 16 | Training → rollout weight sync | Gather FSDP2 shards, load into a separate inference process | Same greedy output before and after sync; each phase timed |
| 17 | vLLM as the rollout backend | CPU continuous-batching scheduler; serve a small model in vLLM | Tokens/s vs concurrency; TTFT measured |
| 18 | verl + LLM RL semantics | Token-level PPO loss with `response_mask` | Padding changes neither loss nor grads; one verl update traced |
| 19 | Capstone: distributed PPO backend | Single-GPU and DDP backends for my PPO | 1-GPU vs 2-GPU update parity on a frozen batch |
| 20 | Performance report | Profile, fix one bottleneck, A/B test | Report + 5-minute talk |

## How each day works

| Block | Time | What I do |
| --- | --- | --- |
| Design | 30 min | Sketch the data flow and communication diagram |
| Code | 2 h | Write the core implementation from scratch |
| Read | 1 h | Trace one call chain in the framework source |
| Run + summarize | 30–60 min | Run experiments, write `dayNN/notes.md`, commit |

Every reported number follows these rules: compile/warmup/steady state reported separately; synchronize before reading a timer; the slowest rank sets the step time; weak and strong scaling kept apart; RL reports always include rollout, update, weight sync and end-to-end time; negative results count if they are diagnosed.

## Repository layout

```
dayNN/                  GUIDE.md (walkthrough and checklist) + notes.md (300–600 word write-up) for each day
experiments/dayNN_*/    Code for each day, with its own README
common/                 Code shared between days
outputs/                Logs, checkpoints and measurements from runs (git-ignored)
```

## Setup

An NVIDIA GPU whose driver supports CUDA 13.0. Single-GPU days also run on native Windows; multi-GPU days need Linux (or WSL2) for NCCL, with 2 GPUs (4 for Day 13).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy
```

Or install everything, including the optional packages, from the requirements file:

```bash
pip install -r requirements.txt
```

Check that PyTorch sees the GPUs:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Run every script from the repository root. Each experiment's README lists its commands and flags.

## Hardware used

| Machine | GPUs | Used for |
| --- | --- | --- |
| Laptop | 1 × RTX 5060 Laptop (8 GB), Windows 11 | Day 01 baseline; editing and git |
| `4090-local` | 2 × RTX 4090 (48 GB), Linux, exclusive | Day 01 on one GPU; 2-GPU correctness (GPU 1 is on a PCIe 4.0 x1 slot) |
| `server` | 2 × RTX 6000D, Linux, shared | 2-GPU runs; needs `NCCL_P2P_DISABLE=1` |
| `5090-server2` | 8 × RTX 5090 (32 GB), Linux, shared | Day 01 on one GPU; 4- and 8-GPU scaling |
