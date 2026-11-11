# AI Infra Learning

A 20-day study of training and RL infrastructure (Oct 2026). Each day builds one distributed-training mechanism from scratch, then uses the real framework, and the last week wires everything into an end-to-end RL pipeline: rollout → train → weight sync.

Three skills, practised every day:

- **Implementation:** a minimal working version of the mechanism, without high-level frameworks.
- **Source literacy:** trace one call chain to find where communication, parameter updates and scheduling happen.
- **Analysis:** explain why something is fast or slow, backed by correctness checks and profiler data.

Two workloads only, so results stay comparable: a **small transformer on synthetic tokens** (DDP, FSDP, ZeRO, TP, PP, Triton) and an **existing PPO/SAC** implementation (parallel rollout, weight sync, CUDA Graphs, RL throughput).

## Progress (updated 2026-11-11)

Day numbers and directories are fixed; when an objective is covered by earlier work, the day is closed with a record rather than renumbered.

| Day | Status | Evidence |
| --- | --- | --- |
| 1 | Complete | [notes](day01/notes.md) |
| 2 | Complete | [notes](day02/notes.md) |
| 3 | Complete, plus an optimizer CUDA Graph / bucket-view / profiler-interference follow-up | [notes](day03/notes.md), [CUDA_GRAPH.md](day03/CUDA_GRAPH.md) |
| 4 | Complete: parity, five negative controls, collective counts, timing and slow link measured (run `a100_20261102T042757Z`) | [notes](day04/notes.md) |
| 5 | Closed without a separate exercise: the objective was met by the Day 03 follow-up; `torch.compile`, dynamic inputs, RNG, fwd/bwd and distributed capture deferred to Days 10, 17, 20 | [closure record](day05/notes.md) |
| 6 | Complete: own run `a100_20261102T124935Z` (bitwise parity with DDP + AdamW, state per rank logged) and arrival-skew follow-up `a100_sync_20261103T120618Z`; independent reimplementation and unaided final explanation remain open practice items | [guide](day06/GUIDE.md), [measured notes](day06/notes.md), [comm sync](day06/COMM_SYNC.md) |
| 7 | Personal run `a100_manual_20261104_01` passed: FSDP2 parity, collective counts, memory and timing on NVLink and with P2P disabled. Independent reimplementation and unaided explanation remain practice items. | [guide](day07/GUIDE.md), [measured notes](day07/notes.md) |
| 8 | Complete. Personal core run `a100_core_20261108T090011Z` passed parity, count and memory. | [guide](day08/GUIDE.md), [measured notes](day08/notes.md) |
| 9 | Complete. Personal full run `manual_full_20261109T065636Z` passed every check: dtype probe, the fp16-order control detected, checkpointing bitwise equal, the matrix, and the optional DDP stage. Notes are filled with measured results and a post-run debrief. Writing `my_amp.py` (`window_step`, `model_forward`) and passing both `--impl mine` checks is deferred as an open practice item. | [guide](day09/GUIDE.md), [measured notes](day09/notes.md) |

The day freed by Day 5 is schedule slack for Days 7–8; it is not refilled with new exercises.

**Plan change 2026-11-08.**

- **Day 8 (Option A).** The full ZeRO 0–3 analysis overlapped Days 06 (hand-written ZeRO-1) and 07 (FSDP2). Day 8 now requires only three observations from the completed core run: the bucket collective and reduction frequency, the ZeRO-3 vs ZeRO-2 start-of-step memory, and one accumulation pitfall shown by a negative control. Predictions, the reimplementation, the source walkthrough, and the timing and slow-link analysis are optional; the runner supports `--core` to execute the required stages alone.
- **Day 13 (Megatron Core).** A short four-GPU Megatron Core run (`DP=1, TP=2, PP=2`) replaces the simulation-only part of Day 13 (full brief below the Days 11–15 table). It displaces:
  - the four-logical-rank CPU simulation of `DP=2 × TP=2`, which becomes the fallback, re-targeted to `TP=2 × PP=2`;
  - the own-code `DP=2` or `TP=2` practice on the dual A100;
  - the generic framework-reading hour, which becomes Megatron Core's `parallel_state`, tensor-parallel layers and 1F1B schedule.

  Sequence parallelism stays a drawing. Days 11 and 12 each gain a 10-minute pointer into the matching Megatron Core source. The plan stays at 20 days.

## Plan

### Days 1–5: single-GPU baseline to DDP internals

When parameters, gradients, optimizer state and activations are allocated and freed, and when gradient communication fires during backward.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 1 | Reproducible single-GPU baseline | Training loop, checkpoint save/resume, timing and peak memory | Resume matches uninterrupted training |
| 2 | Data parallel without DDP | Broadcast params, `all_reduce` grads; ring all-reduce from `send/recv` | N GPUs match 1 GPU on the same global batch |
| 3 | DDP + autograd gradient hooks | Mini bucketed DDP with `register_post_accumulate_grad_hook` | Per-param vs own buckets vs DDP in the profiler |
| 4 | Gradient accumulation | Microbatches with and without `no_sync()` | Matches one equivalent large-batch step |
| 5 | CUDA Graph pitfalls | Originally eager vs `torch.compile` vs manual capture; covered by Day 3's optimizer-graph follow-up, remaining topics deferred ([record](day05/notes.md)) | Met by Day 3: bitwise eager vs replay parity over 24 steps; capture vs replay explained |

### Days 6–10: sharding, precision and kernels

Which memory category each technique cuts, what it costs in communication, and when a hand-written kernel beats PyTorch.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 6 | ZeRO-1 from scratch | AdamW state sharded across ranks | Matches unsharded AdamW; state per rank logged |
| 7 | FSDP2 | `fully_shard` on the whole model vs per block | DDP vs FSDP2 memory and time; `all_gather` / `reduce_scatter` found |
| 8 | DeepSpeed ZeRO in practice | Same model under ZeRO stages 0–3 with the given runner (full analysis optional since 2026-11-08) | Three observations from my run: (1) the NCCL collective behind ZeRO-1/2 gradient buckets and their reduction frequency at K = 4; (2) why ZeRO-3 starts a step above ZeRO-2 in FP32, from the memory table and census; (3) one accumulation pitfall, shown by a negative control |
| 9 | Mixed precision + activation checkpointing | `autocast`, grad clipping, checkpointing | Drift, memory and time for FP32 vs BF16/FP16 |
| 10 | Triton fused softmax | Row-wise fused softmax (stretch: RMSNorm; optional: `torch.compile` `reduce-overhead`, i.e. CUDA Graphs, where launch time dominates) | Correct across row lengths; GB/s vs eager and `torch.compile` |

### Days 11–15: model parallelism, MoE and RL sampling

Split a model by tensor, sequence, pipeline and expert dimensions, check the hand-written versions against Megatron Core, and decouple rollout workers from the learner.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 11 | Tensor parallel MLP | Column- then row-parallel linear with custom autograd Functions; pointer: compare with Megatron Core's `ColumnParallelLinear` / `RowParallelLinear` and `tensor_parallel/mappings.py` (prepares Day 13) | Outputs and grads match 1 GPU |
| 12 | GPipe + 1F1B | 2-stage pipeline; CPU schedule simulator; pointer: compare with Megatron Core's `forward_backward_pipelining_without_interleaving` and `p2p_communication` (prepares Day 13) | Bubble and peak in-flight activations for GPipe vs 1F1B |
| 13 | Process groups in Megatron Core | Draw the rank groups for `DP=1, TP=2, PP=2`, then run a small GPT in Megatron Core on a rented 4-GPU instance and capture a short trace; optional `TP=2, PP=1` 2-GPU baseline with the same model, global batch and precision; TP vs TP+SP drawn. Fallback if no 4-GPU host: CPU rank mapper for `TP=2 × PP=2` + 2-GPU `TP=2` and `PP=2` runs (see the Day 13 brief below) | Megatron's TP/PP/DP groups equal my drawn groups (checked by script); annotated trace with one TP all-reduce inside a stage and one PP send/recv between stages, each tied to a Megatron Core source location. Fallback criteria are separate and labelled simulated |
| 14 | MoE + expert parallelism | Top-1 MoE layer (stretch: `all_to_all` dispatch) | Load balance under uniform vs skewed inputs |
| 15 | Ray + RLlib | 2 CPU rollout actors + 1 GPU learner, tagged with `policy_version` | Time breakdown per phase; staleness explained |

### Day 13 brief (for writing the Day 13 guide and code)

No Day 13 guide or code exists yet. A first draft was removed on 2026-11-08, before I had done Days 11–12, and will be rewritten on the day. Write the guide for someone who has just built TP and 1F1B by hand and is still new to both: explain each group and each transfer from the Day 11–12 code before naming the Megatron function.

**Goal.** See the hand-written mechanisms of Day 11 (column-, then row-parallel linear) and Day 12 (1F1B over two stages) inside one production job, Megatron Core at `DP=1, TP=2, PP=2` on four GPUs, and show both kinds of communication in a trace.

**Primary deliverables (four GPUs):**

1. **Run configuration:** host, `nvidia-smi topo -m`, the GPU order used, torch/NCCL versions, the Megatron-LM tag and commit, model shape, global batch, precision.
2. **Rank groups drawn before launch, then verified.** Draw the TP, PP, DP, model-parallel and embedding groups for ranks 0–3 by hand, plus a small hand-written rank → (TP, DP, PP) mapper. Then check that the groups Megatron actually created (`parallel_state` getters → `torch.distributed.get_process_group_ranks`) are the same. Megatron's default order is `tp-cp-ep-dp-pp`, TP fastest: TP `[0,1] [2,3]`, PP `[0,2] [1,3]`.
3. **An annotated trace of one steady step** showing:
   - one **TP collective inside a pipeline stage** (an all-reduce from a row-parallel forward or a column-parallel backward);
   - one **PP point-to-point transfer between stages** (activation or activation gradient).

   For each: rank, NCCL kernel, phase, process-group ranks and size, and one Megatron Core source location (file:line in the pinned checkout), with a short explanation.
4. **The 1F1B order** of forward/backward microbatches on both stages, read from the trace, and the idle time it implies.

**Optional:** `DP=1, TP=2, PP=1` on two GPUs as a trace baseline, with the same model, global batch and precision. It shows what PP adds to the schedule: point-to-point kernels, warmup/cooldown order, idle time at the ends, and fewer TP all-reduces per rank.

**Constraints for the guide and code:**

- Pin one Megatron-LM release that works with the image's torch 2.14.1+cu130 / NCCL 2.30.7 without reinstalling torch, and **verify it before relying on it**. The removed draft named `core_v0.13.0` (used from a source clone via `PYTHONPATH`), but it was never tested.
- Pre-test on the dual A100 first (`TP=2, PP=1` and `TP=1, PP=2`), before the four-GPU rental is paid for. Those runs double as the fallback's GPU part.
- Include a short topology check for the rental: the TP pairs must sit on the fastest links (all-reduce per layer), and PP pairs can take slower ones. Use the same template image as Days 02–08. Budget about 2 h of rental.
- Keep things simple and readable:
  - Megatron's local layer spec (no Transformer Engine or apex), FP32 with TF32 off, dropout 0, untied embeddings;
  - Day 01's medium shapes (d 512, 8 heads, 6 layers, vocab 8192, sequence 256, 16 sequences per step as 8 microbatches × 2);
  - DP = 1, so no data-parallel gradient sync is needed;
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- Useful trace sources: PyTorch's `record_param_comms` events carry the collective name and the process-group ranks; Python stacks (`with_stack=True`) give the Megatron frame; backward TP all-reduces are launched from autograd's thread.
- Mark simulated four-rank results and measured GPU results separately everywhere.

**Displaced work:**

- The four-logical-rank CPU simulation of `DP=2 × TP=2` becomes the fallback, re-targeted to `TP=2 × PP=2`.
- The own-code `DP=2` / `TP=2` practice on the dual A100 is dropped (Days 11–12 cover the hand-written code).
- The read hour becomes Megatron Core's `parallel_state` (`initialize_model_parallel`, `RankGenerator`), `tensor_parallel/layers.py` + `mappings.py`, and `pipeline_parallel/schedules.py` + `p2p_communication.py`.
- TP vs TP+SP stays a drawing.

**Fallback (no four-GPU host, or Megatron cannot be set up in time), with its own completion criteria:**

- (a) The hand-written mapper for `TP=2 × PP=2` matches Megatron's `RankGenerator` on CPU, labelled simulated.
- (b) A two-GPU `TP=2, PP=1` trace with one TP all-reduce tied to its source.
- (c) A two-GPU `TP=1, PP=2` trace with one send/recv tied to its source, plus the 1F1B order.
- (d) A plain statement that TP and PP were never measured together in one job.

If Megatron will not run at all, read the source at the pinned tag and annotate the Day 11–12 traces with the matching Megatron functions, labelled "source reading, no Megatron run".

### Days 16–20: LLM RL pipeline, capstone and report

Move weights from a sharded trainer to an inference engine, understand rollout scheduling, trace one verl update, and finish with a distributed PPO backend.

| Day | Topic | Build | Done when |
| --- | --- | --- | --- |
| 16 | Training → rollout weight sync | Gather FSDP2 shards, load into a separate inference process | Same greedy output before and after sync; each phase timed |
| 17 | vLLM as the rollout backend | CPU continuous-batching scheduler; serve a small model in vLLM; read how decode CUDA Graphs handle changing batch sizes (deferred from Day 5) | Tokens/s vs concurrency; TTFT measured |
| 18 | verl + LLM RL semantics | Token-level PPO loss with `response_mask` | Padding changes neither loss nor grads; one verl update traced |
| 19 | Capstone: distributed PPO backend | Single-GPU and DDP backends for my PPO | 1-GPU vs 2-GPU update parity on a frozen batch |
| 20 | Performance report | Profile, fix one bottleneck, A/B test (forward/backward or distributed graph capture only if the bottleneck is launch-bound) | Report + 5-minute talk |

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

The current experiment environment is **one Linux Vast.ai instance with 2 × NVIDIA A100-SXM4-40GB**, connected by NV12. Use one process per GPU for all GPU experiments, with matching 1-GPU and 2-GPU baselines. Local Windows is used for editing and git. One exception: Day 13 rents a **4-GPU instance for a few hours** for the Megatron Core `TP=2 × PP=2` run (topology check in the Day 13 brief under the plan). Every other larger rank layout is studied with a CPU simulator and labelled simulated; the plan does not assume eight physical GPUs.

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

SSH: the endpoint changes with each rental (recent ones: `23.127.144.217:26936`, `ssh6.vast.ai:20163`); take the current host and port from the Vast.ai instance page. Project `/workspace/AI-Infra-Learning`. Rentals also differ in CPU/NUMA placement (the Day 4 pre-test host had the GPUs on different NUMA nodes, the Day 4 run host on the same one), and P2P-disabled sync costs differed by up to ~5× between rentals (~15 to ~75 ms per 105 MiB): record `nvidia-smi topo -m` with every run and compare timings only within a host. `/workspace` currently uses container storage, not a persistent volume: download needed results before recycle/destroy. An idle running instance is still billed.
