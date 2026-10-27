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
| 13 | Process groups + sequence parallel | Rank → (DP, TP, PP) mapping; practise DP=2 or TP=2 on the dual A100, simulate DP=2 × TP=2 with 4 logical ranks on CPU | Correct groups; TP vs TP+SP communication drawn; distinguish simulated mappings from GPU measurements |
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

The current experiment environment is **one Linux Vast.ai instance with 2 × NVIDIA A100-SXM4-40GB**, connected by NV12. Use one process per GPU for all GPU experiments, with matching 1-GPU and 2-GPU baselines. Local Windows is used for editing and git. Larger rank layouts may be studied with a CPU simulator; the plan does not assume four or eight physical GPUs.

The accepted image already includes Python 3.12, PyTorch 2.14.1+cu130, CUDA build 13.0 and NCCL 2.30.7; the host driver is 595.71.05. **Activate the existing environment before installing anything:**

```bash
cd /workspace/AI-Infra-Learning
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
```

The project `.venv` is a symlink to the template's `/venv/main`. Non-interactive SSH commands may need `/venv/main/bin/python` explicitly. On a fresh environment only, install project dependencies from the requirements file (`torch>=2.13`); do not reinstall the host GPU driver:

```bash
python -m pip install -r requirements.txt
```

Check that PyTorch sees the GPUs:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Run every script from the project root. Each experiment's README lists its commands and flags. The current remote project was uploaded as a minimal experiment copy, not a full git checkout; do not assume `git pull` is available there until a checkout is created.

## Current experiment hardware (accepted 2026-10-27)

| Machine | Configuration | Used for |
| --- | --- | --- |
| Vast.ai dual A100 | 2 × A100-SXM4-40GB, NV12; default NCCL `P2P/CUMEM/read`; EPYC 7532 host, about 30.72 CPU-equivalents of quota and 129 GB offered RAM | All current and upcoming GPU experiments: 1 → 2 GPU training, communication, profiling and correctness checks |
| Local Windows workspace | Editor and git | Write/review code and notes; retain downloaded results |

The rented GPUs are allocated exclusively, but the entire host CPU/resources are not reserved. Keep `--condition shared` for the existing measurements and repeat timing experiments against their own baselines. P2P is enabled by default; disabling it is only a command-local comparison experiment.

Day 02 acceptance: ring self-test and parity/negative controls passed; 256 MiB builtin all-reduce took about 1.75 ms (153 GB/s), strong scaling reached about 1.40×, and weak throughput scaling about 1.94×. See [Day 02 notes](day02/notes.md) and the [acceptance report](outputs/vast_acceptance_20261027/report.zh-CN.md) for settings and limits. Historical Day 01 timings retain their original machine provenance and are not A100 baselines.

SSH for this rental: `ssh -p 26936 root@23.127.144.217`; project `/workspace/AI-Infra-Learning`. The endpoint can change after recreating the rental. `/workspace` currently uses container storage, not a persistent volume: download needed results before recycle/destroy. An idle running instance is still billed.
