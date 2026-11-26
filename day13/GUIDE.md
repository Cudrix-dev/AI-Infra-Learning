# Day 13: Megatron Core at DP=1, TP=2, PP=2: process groups, one GPT step, one traced 1F1B (four-A100 guide)

> **Run status, 2026-11-21.** Manual `ref` (five layouts) and `mine` (three layouts) both passed on four A100-SXM4-80GB GPUs. The learner's rank mapper matched Megatron's RankGenerator across nine layouts and the groups created in all three mine jobs. Every layout reproduced the single-GPU loss for all five steps. The largest step-0 full-parameter gradient-norm difference was 1.80e-7 relative. The effect of SP on PP message shape and the 1F1B order were reproduced. [Manual results](#appendix-b-manual-results-measured) · [Host stall: found and fixed, 2026-11-22](#appendix-c-the-host-stall-found-and-fixed-assistant-run-2026-11-22-measured) · [Pre-test results](#appendix-a-pre-test-results-pretest3-assistant-run-2026-11-16-pre-test) · [Completed notes](notes.md)

## Required to close Day 13 (about 5 h including the run)

| # | You can explain | Evidence | Where |
| --- | --- | --- | --- |
| 1 | **The rank groups** of `DP=1, TP=2, PP=2`: why TP is `[0,1] [2,3]` and PP is `[0,2] [1,3]`, what the model-parallel, embedding and position-embedding groups are, and what untied embeddings change | your `rank_mapper.py` passes; `[groups] built by Megatron == mine mapper: PASSED` in your run | §1, Block 2 Stage 1 |
| 2 | **How one Megatron GPT training step is wired**: model provider, `forward_step_func`, data iterator, `get_forward_backward_func`, loss ÷ microbatches, and where your Day 11 layers and Day 12 1F1B sit | your `my_step.py` passes with the same losses as the reference | §2, Block 2 Stage 2 |
| 3 | **Sequence parallelism**: the choice for the main run (off), why, and which collectives and P2P shapes change when it is on | `compare.log` counts and shapes for `tp2pp2` vs `tp2pp2_sp` | §4 |
| 4 | **An annotated trace**: one TP collective inside a stage and one PP send/recv between stages (rank, NCCL kernel, phase, group ranks and size, Megatron file:line), and the 1F1B order and idle time read from it | `analyze_tp2pp2.log`, the trace in Perfetto, notes §5–§6 | §6, Block 4 |
| 5 | **The topology check**: TP pairs on the fastest links, and why `CUDA_DEVICE_MAX_CONNECTIONS=1` | `topology.log`, `environment.log` | §5 |

These five rows cover the README brief's four primary deliverables: run configuration, groups drawn and verified, an annotated trace, and the 1F1B order. The optional `TP=2, PP=1` baseline runs on two of the GPUs as a smoke test.

**Completion record**

- [x] Record the guided predictions and corrections for P1–P10, and check the Day 11 and Day 12 predictions against the manual results (notes §1).
- [x] Complete `rank_mapper.py` (TODO 1–2) and `my_step.py` (TODO 3–4), and pass them in a run with `--impl mine --mapper mine`.
- [x] Run, download and verify both manual sessions.
- [x] Confirm the manual-run Vast instance (61775446) stopped. The console check is recorded in [notes §2](notes.md#2-run-record).
- [x] Fill notes §2–§8 from your logs, including the annotated trace table (§5) and the 1F1B order with idle time (§6).
- [x] Review Block 3 Stops 1–5 and fill the mapping table in notes §9.

**Optional (not needed to close the day):** Block 3 Stop 6 (`pretrain_gpt.py` / `training.py`), the tied-embedding experiment in §1.4, and the `CUDA_DEVICE_MAX_CONNECTIONS` A/B in §5.

**Scope on purpose.** One four-GPU host, DP = 1, two pipeline stages with three transformer layers each, FP32 with TF32 off, the local layer spec (no Transformer Engine or apex), dropout 0, untied embeddings, 8 microbatches × 2 sequences × 256 tokens, 5 optimizer steps. No interleaved schedule, no data parallelism, no distributed optimizer, no checkpointing. Timing is recorded but it is **not a benchmark**: at this size the step is host-bound (§6.4).

---

## What is new today (and what you already have)

You already have Day 11's column → row parallel MLP (`f` = identity forward / all-reduce backward, `g` = all-reduce forward / identity backward), and Day 12's two-stage 1F1B (`send_forward_recv_backward`, FIFO pop, warmup $P - s - 1$). Both ran on two GPUs, each alone.

New today:

- **Two kinds of groups at once.** Every rank is in a TP group *and* a PP group. Its all-reduces go to one group and its send/recv to the other. Who is in which group is decided once, by a rank-numbering rule, before the model exists.
- **A production training step.** On Days 11–12 you wrote the loop. Today Megatron owns the loop over microbatches, the P2P exchanges and every backward call. You hand it two functions and a model.
- **A real model around your layers.** Embedding (vocab-parallel), attention (another column → row pair), LayerNorms, an output layer and a vocab-parallel cross-entropy. Each adds collectives you did not write.
- **Reading a trace you did not instrument.** Kernels from two threads and three CUDA streams on each rank. Today you tie each kernel back to a group and a Megatron source line.

---

## Read this first: the host and the setup

The manual host was Vast.ai container 61775446 with 4 × A100-SXM4-80GB and driver 595.91.07. All six GPU pairs reported NV12. GPU 0–1 had CPU affinity 0–63 (NUMA 0), and GPU 2–3 had affinity 64–127 (NUMA 1). The runner chose pair 0,1 and four-GPU order 0,1,2,3. [Manual topology](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/topology.log)

The pre-test host was Vast.ai container 60863550 with 4 × A100-SXM4-80GB. Every pair was **NV12**, all four GPUs were on NUMA node 1, the driver was 580.178.04, with torch 2.14.1+cu130, NCCL 2.30.7 and Python 3.12.15. It cost about \$5.42/h and was used for about 38 minutes. A rental with the same template image works the same way. Your rental may have a different topology, which is why Stage 0 checks it first.

From your Windows project root, after the instance is up (`<host>`, `<port>` from the Vast console):

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/AI-Infra-Learning/experiments && ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv'
```

```bash
scp -P <port> README.md requirements.txt root@<host>:/workspace/AI-Infra-Learning/
```

```bash
scp -P <port> -r day13 root@<host>:/workspace/AI-Infra-Learning/
```

```bash
scp -P <port> -r experiments/day13_megatron root@<host>:/workspace/AI-Infra-Learning/experiments/
```

On the instance:

```bash
cd /workspace/AI-Infra-Learning && source .venv/bin/activate
nvidia-smi topo -m
bash experiments/day13_megatron/setup_megatron.sh        # clones core_v0.19.2 to /workspace/Megatron-LM, installs einops
```

Expected: `Megatron-LM core_v0.19.2 (4b4acac9a1d28ea6829c8d4f566d75698a21249d) ready at /workspace/Megatron-LM`. Megatron is used from the source checkout on `PYTHONPATH`. It is not pip-installed, and Transformer Engine and apex are not needed. The only missing dependency in the image is `einops`. Megatron prints warnings about missing TE/apex and a `GPTModel IS DEPRECATED` notice (it points to `HybridModel`). Both are harmless here.

If you copied the files from Windows and a `.sh` script fails with `$'\r': command not found`, run `sed -i 's/\r$//' experiments/day13_megatron/*.sh`.

---

## Files

All in `experiments/day13_megatron/` (its [README](../experiments/day13_megatron/README.md) lists outputs and deviations):

| File | What it is | Read it when |
| --- | --- | --- |
| `rank_mapper.py` | rank → (TP, DP, PP) and the groups; reference + **YOU WRITE** `my_coords`, `my_groups` | §1, Stage 1 |
| `my_step.py` | **YOU WRITE** `loss_func`, `forward_step_func` | §2, Stage 2 |
| `train_gpt.py` | the Megatron job, the five pieces of §2 in order, with the group check and two profiled steps | §2, before Stage 2 |
| `analyze_trace.py` | ties every NCCL kernel to group, phase, thread, stream and Megatron frame. It also reads the 1F1B order and idle time | §6, Block 4 |
| `compare_runs.py` | layouts vs the 1-GPU job: losses, gradient norms, P2P shapes, NCCL counts, order | Block 4 |
| `topo_check.py` | link of every GPU pair, fastest pair, 4-GPU order | §5, Stage 0 |
| `find_symbols.py` | file:line of the Block 3 stops in your checkout | Block 3 |
| `setup_megatron.sh`, `run_server_once.sh`, `download_results.ps1` | setup, everything once (archived), download + verify on Windows | Block 4 |

---

## Block 1: Concepts (about 2.5 h). Predict first, then open the answer

Write each prediction in notes §1 before you open the `<details>`.

### §1 How Megatron Core builds the process groups

#### §1.1 One numbering rule: `tp-cp-ep-dp-pp`, leftmost fastest

Megatron gives every rank a coordinate for each kind of parallelism and numbers the ranks so that the **leftmost** kind in the order string changes fastest. The default order is `tp-cp-ep-dp-pp` (`initialize_model_parallel`'s `order` argument, `parallel_state.py:617`). Today CP (context parallel) and EP (expert parallel) are 1, so only TP, DP and PP matter. A global rank is a mixed-radix number whose lowest digit is the TP rank:

$$
\text{rank} = r_\text{tp} + \text{TP} \cdot (r_\text{dp} + \text{DP} \cdot r_\text{pp})
$$

where $r_\text{tp}$, $r_\text{dp}$ and $r_\text{pp}$ are the rank's TP, DP and PP coordinates.

It is the same rule as a 3-D array laid out in memory with TP as the contiguous axis. `generate_masked_orthogonal_rank_groups` (`parallel_state.py:269`) states it in its docstring, then solves the reverse problem: list the ranks that share some coordinates and vary the others.

**P1.** For `DP=1, TP=2, PP=2`, write each rank's `(tp, dp, pp)`. Then do `TP=2, DP=2, PP=2` (8 ranks): which ranks are on pipeline stage 1, and which are TP rank 1?

<details><summary>Answer P1</summary>

- DP=1, TP=2, PP=2: rank $0 \mapsto (0,0,0)$, rank $1 \mapsto (1,0,0)$, rank $2 \mapsto (0,0,1)$, rank $3 \mapsto (1,0,1)$.
- TP=2, DP=2, PP=2: rank $r \mapsto (r \bmod 2,\ \lfloor r/2 \rfloor \bmod 2,\ \lfloor r/4 \rfloor)$. Stage 1 is ranks 4, 5, 6, 7, and TP rank 1 is ranks 1, 3, 5, 7.

`rank_mapper.py --layouts all` prints these and four more layouts. In the pre-test they matched Megatron's `RankGenerator` (`mapper.log`).
</details>

#### §1.2 A group = the ranks that agree on every other coordinate

| Kind | Ranks that share | `DP=1, TP=2, PP=2` | Used for |
| --- | --- | --- | --- |
| TP | dp, pp | `[0,1] [2,3]` | Day 11's all-reduces inside a stage (and SP's all-gather / reduce-scatter) |
| PP | tp, dp | `[0,2] [1,3]` | Day 12's send/recv between stages |
| DP | tp, pp | `[0] [1] [2] [3]` | gradient all-reduce. Size 1 today, so nothing runs |
| MP (model parallel, `tp-pp`) | dp | `[0,1,2,3]` | one full model replica. Used for things every shard of a replica must agree on (e.g. grad-norm reduction) |
| embedding | first + last rank of each PP group | `[0,2] [1,3]` | tied input/output embedding: weight copy at init and gradient all-reduce |
| position embedding | first rank of each PP group | `[0] [1]` | learned positions (stage 0 only) |

**P2.** Rank 2 asks each getter for its group: `get_tensor_model_parallel_group`, `get_pipeline_model_parallel_group`, `get_data_parallel_group`, `get_model_parallel_group`, `get_embedding_group`, `get_position_embedding_group`. What ranks come back from each? And why are ranks 0 and 2 (not 0 and 1) the pair that exchanges activations?

<details><summary>Answer P2</summary>

TP `[2, 3]`, PP `[0, 2]`, DP `[2]`, MP `[0, 1, 2, 3]`, embedding `[0, 2]`, and **`None`** for the position-embedding group. Rank 2 is in no position-embedding group, so the getter has nothing to return (`run_tp2pp2.log`, last column `pos-emb`: `None` on ranks 2 and 3).

Activations travel from stage 0 to stage 1 between ranks with the **same TP rank**. Rank 0 holds TP shard 0 of layers 0–2, and rank 2 holds TP shard 0 of layers 3–5. With SP off, both TP ranks of stage 0 hold the same full activation after the row-parallel all-reduce, so each sends its own identical copy to its partner. That is your Day 12 prediction. §4 shows why the pairing matters once SP is on.
</details>

#### §1.3 How the groups are created

`initialize_model_parallel` (`parallel_state.py:601`) builds `RankGenerator(tp, ep=1, dp, pp, cp, order)` (`:859`). For each kind it loops over `get_ranks(kind)` (`RankGenerator.get_ranks` `:534`), calls `create_group(ranks, ...)` for **every** group, and keeps only the one containing this rank. Examples are model-parallel `:1147`, TP `:1164`, PP `:1240`, and embedding `:1269` from each PP group. The getters (`get_tensor_model_parallel_group` `:1704`, `get_pipeline_model_parallel_group` `:1749`, ...) return the kept one.

**P3.** Rank 0 is not in TP group `[2, 3]`. Why does it still call `create_group([2, 3])`?

<details><summary>Answer P3</summary>

`create_group` calls `torch.distributed.new_group` (`parallel_state.py:259`). PyTorch requires every process of the default group to call `new_group` for every new group, in the same order, even processes that will not be members. The calls must match across ranks to agree on group identities. If rank 0 skipped `[2, 3]`, its later `new_group` calls would be out of step with ranks 2 and 3, and group creation would hang or pair the wrong groups. NCCL communicators themselves are created lazily, on a group's first collective, and only on member ranks.
</details>

#### §1.4 The embedding group, and what untied embeddings change

GPT's input embedding ($\text{vocab} \times h$, stage 0) and output layer ($\text{vocab} \times h$, last stage) can share one weight ("tied"). When the two stages are different ranks, each holds a copy, and the copies must stay equal. That is why the embedding group pairs the **first and last** rank of each PP group: `[0, 2]` and `[1, 3]`, one per TP shard. With vocab-parallel layers, rank 0's rows 0–4095 must match rank 2's rows 0–4095.

**P4.** Today's embeddings are **untied** (`share_embeddings_and_output_weights=False`). (a) Does the embedding group still exist? (b) Which collectives over `[0, 2]` run (i) at initialization and (ii) per step, tied vs untied? (c) What else changes?

<details><summary>Answer P4</summary>

(a) Yes. `initialize_model_parallel` always builds it from the PP groups (`:1269`), and the pre-test printed `embedding [0, 2]` on ranks 0 and 2. The group exists. Whether anything uses it depends on the model.

(b) **Tied:** (i) at init the last stage zero-fills its output weight, and an all-reduce over the embedding group copies stage 0's values into it (`setup_embeddings_and_output_layer`, `language_module.py:202`, all-reduce at `:308`). (ii) Every step, `finalize_model_grads` (`distributed/finalize_model_grads.py:563`) calls `_allreduce_word_embedding_grads` (`:164`, call at `:659`). The call sums the two copies' gradients, $\text{vocab}/\text{TP} \times h$ = 4096 × 512 FP32 = 8 MiB per rank here. **Untied:** neither runs. `_get_shared_word_embedding_weight` returns `None` when the weights are not shared (`:146`), so the embedding all-reduce has nothing to reduce. The pre-test trace has **no** collective over `[0, 2]` except the PP send/recv.

(c) Untied embeddings add parameters: the last stage holds its own `output_layer.weight` (4096 × 512 per TP rank). Each of them gets its own optimizer state. Our script never calls `finalize_model_grads` at all (DP = 1, untied, no DDP wrapper). A production run calls it every step, and it is a no-op for the embedding here. *Optional experiment:* set `share_embeddings_and_output_weights=True` in `train_gpt.py`, call `finalize_model_grads` after the schedule, and look for the 8 MiB all-reduce over `[0, 2]` in the trace (not pre-tested).
</details>

#### §1.5 Checking the groups

You write the mapper in Block 2, Stage 1. `rank_mapper.py --megatron PATH` compares it with Megatron's `RankGenerator` without creating a process group (pure Python, labelled **[simulated]** for layouts with more ranks than the host has). Then `train_gpt.py` compares it inside each real job with the groups NCCL actually built, read back through the getters and `torch.distributed.get_process_group_ranks`.

### §2 How a real Megatron GPT training step is wired

#### §2.1 Five pieces

Every Megatron Core training script has the same five pieces. `train_gpt.py` has them in this order. Megatron's own `pretrain_gpt.py` + `megatron/training/training.py` add argument parsing, a real dataset, the distributed optimizer, checkpoints and logging around the same five pieces (Block 3 Stop 6).

| # | Piece | `train_gpt.py` | What it decides |
| --- | --- | --- | --- |
| 1 | initialize distributed | `initialize_distributed`: `init_process_group`, `initialize_model_parallel(tp, pp)`, `model_parallel_cuda_manual_seed` | the groups (§1) |
| 2 | model provider | `model_provider`: `TransformerConfig` + `GPTModel(pre_process=is_pipeline_first_stage(), post_process=is_pipeline_last_stage())` | which layers and which shard this rank builds |
| 3 | data iterator | `make_global_batch` → `iter(batch)` (8 dicts of `[2, 256]` tokens/labels) | what each `next()` returns |
| 4 | `forward_step_func(data_iterator, model) → (output_tensor, loss_func)` | `ref_forward_step_func` / your `my_step.forward_step_func` | one microbatch's forward and how its loss is computed |
| 5 | `get_forward_backward_func()` → schedule → `optimizer.step()` | `fb_func(forward_step_func=..., data_iterator=..., model=..., num_microbatches=8, seq_length=256, micro_batch_size=2, ...)` | the order of all F/B, every P2P, every backward |

On rank 2, `GPTModel` builds layers 3–5 (`get_num_layers_to_build`, `transformer_block.py:70`), the final LayerNorm and the output layer, but no embedding. `layer_number` is still global (4–6), which is how `canonical_init` gives every layout the same weights. In `GPTModel.forward` (`gpt_model.py:519`) the embedding runs only when `pre_process` (`:345`). A non-last stage returns the hidden state right after the decoder (`:683`). The output layer (`:762`) and the per-token loss (`:794`) run only on the last stage.

**P5.** For `DP=1, TP=2, PP=2`, M = 8: how many times per step does each rank call `forward_step_func`, `loss_func` and `backward_step`? On which ranks is `loss_func` called? What does `fb_func` return on rank 0 and on rank 2? Should stage 1 call `next(data_iterator)` even though it ignores the tokens?

<details><summary>Answer P5</summary>

Every rank calls `forward_step_func` 8 times and `backward_step` 8 times (`rank<r>.json`: `forward_calls 8, backward_calls 8` every step). `loss_func` runs only where `post_process` is true: ranks 2 and 3, 8 times each. `fb_func` returns a list with one loss dict per microbatch on the last stage, and an empty list on ranks 0 and 1 (`forward_step_calc_loss`, `schedules.py:290`, appends only on the last stage). That is why `train_gpt.py` reads the loss from the last stage.

**Yes, every stage takes one microbatch per call.** Each rank iterates its own copy of the same 8 microbatches. The k-th call on rank 2 must return microbatch k's labels. In 1F1B the k-th forward on every stage is microbatch k (Day 12 §2.3), so taking exactly one per call keeps them aligned. Production code skips loading data on middle stages entirely (`get_batch` in `pretrain_gpt.py:97` returns `None`s there), which is the same rule with less I/O.
</details>

#### §2.2 Loss scaling by the number of microbatches

Your `loss_func` returns the mean cross-entropy over the microbatch's 512 tokens. Megatron then divides it by `num_microbatches` before backpropagating (`forward_step_calc_loss`, `output_tensor /= num_microbatches`, `schedules.py:335`). Summed over 8 microbatches, that is the mean over all 4096 tokens: Day 04's $n_m / N$ weight, Day 12's `mb_loss`. With the 3-tuple form `(loss_sum, num_tokens, log)`, Megatron divides by the token count instead (`:321–327`, path not used today).

**P6.** (a) What scalar does `torch.autograd.backward` start from on rank 2 for microbatch k? (b) You return `(loss, {"lm loss": loss.detach()})`. Step 0's true mean loss is 9.124. What does your log show, and why?

<details><summary>Answer P6</summary>

(a) $\text{mean}_k / 8$: the microbatch's token mean divided by 8 (on both TP ranks of stage 1. Vocab-parallel cross-entropy gives both TP ranks the same per-token loss).

(b) **1.1405**, which is 9.124 / 8. `detach()` returns a tensor that **shares storage** with `loss`, and Megatron's `/=` is **in place**, so your logged value is divided too. This happened in the assistant's first development run (`1.1405004`, see Appendix A.4). `clone()` fixes it (`ref_loss_func`). The gradients were right all along. Only the log was wrong.
</details>

#### §2.3 Where your Day 11 and Day 12 code sit

| Your code | Inside the Megatron step | Where |
| --- | --- | --- |
| Day 11 `ColumnParallelLinear` (`f` = all-reduce of `dx` in backward) | `linear_qkv` and `linear_fc1` of every layer, and `output_layer` | `tensor_parallel/layers.py:869`, all-reduce `:639` (async), wait `:750` |
| Day 11 `RowParallelLinear` (`g` = all-reduce in forward) | `linear_proj` and `linear_fc2` of every layer | `layers.py:1249`, `reduce_from_tensor_model_parallel_region` `:1480` → `mappings.py:35` |
| (not in Day 11) vocab-parallel embedding / cross-entropy | first-stage embedding, last-stage loss | `layers.py:230` (all-reduce `:377`), `cross_entropy.py:119` (3 all-reduces of $b \times s$ values) |
| Day 12 `run_1f1b` | `forward_backward_pipelining_without_interleaving` | `pipeline_parallel/schedules.py:2120` |
| Day 12 `P2P.send_forward_recv_backward` | `P2PCommunicator.send_forward_recv_backward` → `_communicate` → `batch_isend_irecv` | `p2p_communication.py:527`, `:275`, `:49` |
| Day 12 `received.requires_grad_()` + `forward_step` | `set_input_tensor` (the received activation replaces the embedding output on stage 1) | `schedules.py:501` → `transformer_block.py:443` |
| Day 04 / Day 12 loss weight | `output_tensor /= num_microbatches` | `schedules.py:335` |

#### §2.4 What happens after the schedule

`training.py`'s `train_step` (`megatron/training/training.py:2336`) calls the schedule (`:2407`). Inside it, after the last backward, `finalize_model_grads` runs (`config.finalize_model_grads_func`, set at `training.py:3550`). It finishes DP gradient reductions, all-reduces the tied embedding's gradient (§1.4), and with SP all-reduces the LayerNorm gradients over TP (§4). Then `optimizer.step()` runs (`:2474`). Today DP = 1 and the embeddings are untied, so the only one of these we need is the SP LayerNorm reduction. `train_gpt.py` does it by hand (`allreduce_sequence_parallel_grads`) and uses plain `torch.optim.Adam`.

### §3 TP inside a pipeline stage: which collectives, and how many

Each transformer layer is two Day 11 blocks: attention (`linear_qkv` column → `linear_proj` row) and MLP (`linear_fc1` column → `linear_fc2` row). With SP off, per microbatch and per layer that is **2 forward all-reduces** (outputs of `linear_proj` and `linear_fc2`, your `g`). It is also **2 backward all-reduces** (input gradients of `linear_qkv` and `linear_fc1`, your `f`), each `[s, b, h]` = `[256, 2, 512]` FP32 = 1 MiB.

**P7.** Count the TP NCCL kernels per rank in one step (M = 8) for `TP=2, PP=2`, stage 0 and stage 1 separately, split into forward and backward. The first stage has the vocab-parallel embedding. The last stage has the output layer (column-parallel) and the vocab-parallel cross-entropy, which all-reduces three `[b, s]` tensors per microbatch: max logit, target logit and sum of exponentials. Then do `TP=2, PP=1`. How many P2P kernels does each stage launch?

<details><summary>Answer P7</summary>

| Layout, rank | Forward all-reduces (1 MiB) | Forward all-reduces (512 values) | Backward all-reduces (1 MiB) | P2P SendRecv kernels |
| --- | ---: | ---: | ---: | ---: |
| `tp2pp2` stage 0 (ranks 0, 1) | 8 × (2·3 + 1 embedding) = **56** | 0 | 8 × 2·3 = **48** | **9** |
| `tp2pp2` stage 1 (ranks 2, 3) | 8 × 2·3 = **48** | 8 × 3 = **24** | 8 × (2·3 + 1 output layer) = **56** | **9** |
| `tp2pp1` (ranks 0, 1) | 8 × (2·6 + 1) = **104** | **24** | 8 × (2·6 + 1) = **104** | 0 |

Each stage of a two-stage 1F1B launches $1 + (M - 1) + 1 = 9$ P2P kernels. Stage 0 has `send_forward` (warmup), 7 × `send_forward_recv_backward` and `recv_backward` (cooldown). Stage 1 has `recv_forward`, 7 × `send_backward_recv_forward` and `send_backward`. A batched send+recv is **one** `ncclDevKernel_SendRecv`. The pre-test and both manual runs reproduced these counts for their SP-off layouts. [Manual mine counts](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log)

Splitting into stages **divides** the per-rank all-reduce count (232 → 104 / 128), because each rank holds half the layers. It does not change the total for the model. TP inside a stage costs the same per layer as in Day 11.

**Check your Day 11 prediction** (Day 11 notes §7): it said 2 + 2 per layer per microbatch on every stage, `[s, b, h]`, forward from `layers.py:1480` and backward from `layers.py:637–639`, waiting at `:749–750`. All confirmed. One correction for today: the size is **FP32**, $s \cdot b \cdot h \cdot 4\ \text{B}$ = 1 MiB, not BF16. Your note left out the embedding, the output layer and cross-entropy.
</details>

### §4 Sequence parallelism: off for the main run, on as a comparison

**What SP is, in one sentence.** Outside the TP region (LayerNorm, residual add, dropout), the activations are normally **replicated**: both TP ranks hold the same `[256, 2, 512]`. Sequence parallelism (SP) **splits them along the sequence** instead: rank 0 keeps tokens 0–127 and rank 1 keeps tokens 128–255. Inside the TP region nothing changes; it is still exactly Day 11. Only the collectives at the two borders of the TP region change. Your Day 11 notes §8.1 sketched this. The four figures below show it step by step.

#### §4.1 Three collectives and their mirrors

Read this figure first. It shows what each collective does to one `[256, 2, 512]` activation on two ranks, and the two facts the rest of §4 uses:

- an all-reduce is a reduce-scatter followed by an all-gather;
- every forward collective has a fixed "mirror" in backward.

![The three TP collectives on two ranks: ALL-REDUCE turns two partial sums P0, P1 into the full sum on both ranks ([256,2,512] to [256,2,512]); REDUCE-SCATTER sums and keeps only each rank's tokens ([256,2,512] to [128,2,512]); ALL-GATHER concatenates each rank's tokens ([128,2,512] to [256,2,512]). Rule card: all-reduce = reduce-scatter then all-gather; forward to backward mirrors: identity f to all-reduce, all-reduce g to identity, all-gather to reduce-scatter, reduce-scatter to all-gather, split to all-gather](figures/d13_sp_collectives.svg)

#### §4.2 One block forward: SP off vs SP on

Each transformer layer has two blocks (attention, MLP) with the same shape of communication. The figure shows the MLP with every tensor shape and matmul. Compare the two panels column by column. The green TP region in the middle is **identical**. Only the ENTER and LEAVE bars change.

![Forward of one MLP block at TP=2. SP off: x [256,2,512] replicated, LayerNorm, f = identity (no communication), fc1 column-parallel h_r = x·A1_r (512×512)·(512×1024) to [256,2,1024], GELU, fc2 row-parallel z_r = a_r·A2_r to [256,2,512] partial, ALL-REDUCE (layers.py:1480, mappings.py:35), y [256,2,512]. SP on: x shard [128,2,512] (tokens 0–127 on rank 0, 128–255 on rank 1), LayerNorm on own tokens, ALL-GATHER to [256,2,512] (layers.py:571), the same TP region, REDUCE-SCATTER to [128,2,512] (layers.py:1476), y shard [128,2,512]](figures/d13_sp_forward.svg)

**The rule to remember:**

| Where | Why a collective is needed | SP off | SP on |
| --- | --- | --- | --- |
| **Enter** the TP region (input of `linear_qkv`, `linear_fc1`, `output_layer`) | a column-parallel linear needs **every token** of its input | nothing (`f` = identity): the input is already full on both ranks | **all-gather** along the sequence, `[128,2,512] → [256,2,512]` (`layers.py:571`) |
| **Leave** the TP region (output of `linear_proj`, `linear_fc2`) | a row-parallel linear produces **partial sums** | **all-reduce** (`g`): every rank gets every token's sum (`layers.py:1480`) | **reduce-scatter**: each rank gets the sum for its own tokens, `[256,2,512] → [128,2,512]` (`layers.py:1476`) |

Two more consequences, both shown in the figures for P8:

- LayerNorm γ, β and the row-parallel biases now see only half the tokens on each rank, so their gradients must be all-reduced over TP once per step (`finalize_model_grads.py:452`).
- The pipeline sends the sharded activation: `get_tensor_shapes` divides the sequence by TP (`schedules.py:2114`).

#### §4.3 The choice for the main run: SP off

1. It is exactly Day 11's layers: one all-reduce per row or column layer, so every TP kernel in the trace maps to your `f` or `g`.
2. SP's benefit is activation memory: LayerNorm/residual activations ÷ TP. At 256 tokens × 2 sequences that is a few MiB. Nothing here needs it.
3. In this setup, SP needs a deviation. Without apex or TE, the local spec falls back to torch's LayerNorm, which asserts `sequence parallel not supported by torch LayerNorm` (`transformer/torch_norm.py:50`). The pre-test hit this assertion (`pretest1`). `train_gpt.py --sp` replaces it with `SequenceParallelTorchLayerNorm`, which only flags the parameters, as apex's `FusedLayerNorm` does (`fusions/fused_layer_norm.py:119–120`). That is fine for an optional comparison, but it should not be in the main run.

**P8.** Use the two facts of §4.1 and the forward figure of §4.2.

- (a) **Backward of one block with SP on:** which collectives run, in what order, and with which shapes? *Hint:* apply the mirror rule to the two forward bars. Then ask what fc1's weight gradient $dA_{1,r} = x^\top \cdot dh_r$ needs, given that only the `[128, 2, 512]` shard of `x` was saved.
- (b) **Per rank per step** at `TP=2, PP=2`, M = 8: count each collective in forward and backward on stage 0 and on stage 1. Remember the embedding on stage 0, and the final LayerNorm, output layer and cross-entropy on stage 1. Also give the P2P shape. Does the loss change?

<details><summary>Answer P8 (pre-test, rank 0 = stage 0 and rank 2 = stage 1; <code>analyze_tp2pp2_sp.log</code>, <code>compare.log</code>)</summary>

**(a) Backward of one block.** The mirror of the forward reduce-scatter is all-gather #1. The mirror of the forward all-gather is a reduce-scatter. Then comes one extra all-gather, #2: the column-parallel layer's weight gradient needs the full-sequence `x`, and only the shard was saved, so the backward gathers it again. That re-gather is SP's price for storing half the activation.

![Backward of the MLP block with SP on: dy shard [128,2,512], ALL-GATHER #1 (mirror of the forward reduce-scatter, mappings.py:377 to :118, sync) to dz [256,2,512], fc2 backward local da_r = dz·A2_rᵀ to [256,2,1024], GELU backward, fc1 input grad dx_r = dh_r·A1_rᵀ to [256,2,512] partial, REDUCE-SCATTER (layers.py:650, async; SP off: ALL-REDUCE layers.py:639) to [128,2,512], LayerNorm backward. Weight-gradient path: saved x shard [128,2,512], ALL-GATHER #2 (layers.py:617, async) to x_full [256,2,512], dA1_r = x_fullᵀ·dh_r [512,1024]. Once per step: ALL-REDUCE of LayerNorm and row-bias gradients, 9216 values on stage 0, 10240 on stage 1](figures/d13_sp_backward.svg)

**(b) The whole step.** The figure follows one microbatch through both stages. The table counts per microbatch, then multiplies by M = 8.

![One microbatch through both stages with SP on: stage 0 word embedding (vocab rows split, partial) to ALL-REDUCE [256,2,512] (layers.py:377), add position and split to [128,2,512] with no communication (backward: ALL-GATHER), 3 layers with 2 AG + 2 RS forward and 4 AG + 2 RS backward per layer, P2P [128,2,512] rank 0 to 2 and rank 1 to 3; stage 1: 3 layers, final LayerNorm, ALL-GATHER to [256,2,512] (layers.py:571), output layer column-parallel to [256,2,4096], cross-entropy with 3 ALL-REDUCEs of 512 values](figures/d13_sp_pipeline.svg)

| Per rank, per microbatch | Stage 0 | Stage 1 |
| --- | --- | --- |
| F all-gather | 3 layers × 2 = **6** | 6 + 1 (output layer) = **7** |
| F reduce-scatter | 3 × 2 = **6** | **6** |
| F all-reduce | **1** (embedding, 262144) | **3** (cross-entropy, 512 each) |
| B all-gather | 6 mirrors + 6 re-gathers + 1 (embedding split) = **13** | 6 mirrors + 7 re-gathers = **13** |
| B reduce-scatter | **6** | 6 + 1 (output layer) = **7** |
| Once per step, after the schedule | 1 all-reduce, 9216 values | 1 all-reduce, 10240 values |

Times 8 microbatches gives exactly what the trace counted:

| Collective (elements) | Stage 0, SP off | Stage 0, SP on | Stage 1, SP off | Stage 1, SP on | Where (Megatron frame in the trace) |
| --- | ---: | ---: | ---: | ---: | --- |
| F all-reduce (262144) | 56 | **8** (embedding only) | 48 | 0 | `mappings.py:22 _reduce` |
| F reduce-scatter (in 262144, out 131072) | — | **48** | — | **48** | row-parallel outputs, `layers.py:1476` → `mappings.py:159` |
| F all-gather (in 131072, out 262144) | — | **48** | — | **56** (+ output layer) | column-parallel inputs, `layers.py:571` (frame `:528 forward`) |
| F all-reduce (512, cross-entropy) | — | — | 24 | 24 | `cross_entropy.py:120` |
| B all-reduce (262144) | 48 | 0 | 56 | 0 | `layers.py:639` |
| B reduce-scatter (262144) | — | **48** | — | **56** | column dgrad, `layers.py:650` (frame `:581 backward`), async |
| B all-gather (131072) | — | **104** = 48 + 56 | — | **104** = 56 + 48 | re-gathers for wgrad `layers.py:617` (frame `:581`, async): 48 / 56; mirrors `mappings.py:118` (sync): 48 + 8 embedding / 48 |
| all-reduce after the schedule | — | **1** (9216 = 3 layers × (2 LN × 1024 + 2 row biases × 512)) | — | **1** (10240, + final LN) | LayerNorm/bias grads, `allreduce_sequence_parallel_grads` |
| P2P SendRecv | 9 × `[256, 2, 512]` | 9 × **`[128, 2, 512]`** | 9 × `[256, 2, 512]` | 9 × **`[128, 2, 512]`** | `schedules.py:2114` |

Every 1 MiB all-reduce became a reduce-scatter plus an all-gather of the same total size. A ring all-reduce *is* a reduce-scatter followed by an all-gather, so the bytes on the wire stay the same. The extra traffic is all-gather #2, the backward re-gather of each column-parallel input for its weight gradient: 48–56 extra all-gathers per stage. In return, every P2P transfer halves (0.5 MiB), and the activations between blocks halve.

The losses were **bitwise equal** with SP off and on, and to 1 GPU. Both manual runs reproduced the P8 counts when checked against the derivation. The SP rows in `compare.log` report the observed counts with the automatic count check skipped. In mine, rank 0 peak CUDA allocation was 209.29 MiB with SP off and 196.76 MiB with SP on. The step-0 gradient norms differed from 1 GPU by at most 1.80e-7 relative (SP off: 1.76e-7). **Your Day 12 prediction is confirmed:** each stage-0 rank sends its own `[128, 2, 512]` shard to the stage-1 rank with the same TP index. `get_tensor_shapes` divides by TP (`schedules.py:2114`), which was the open item in your note. [Manual SP comparison](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log)
</details>

### §5 Topology: TP on the fastest links, and `CUDA_DEVICE_MAX_CONNECTIONS=1`

**Where each kind of traffic goes.** Per step at M = 8, stage 0's TP group moves 104 all-reduces of 1 MiB on the critical path of every layer. Stage 0's PP pair moves 16 transfers of 1 MiB (8 activations out, 8 gradients in), which run between layers on a separate stream. So the TP pairs need the fastest links, and PP tolerates slower ones. Ranks map to GPUs through `CUDA_VISIBLE_DEVICES`: global rank r runs on its r-th entry. With Megatron's order, ranks (0, 1) and (2, 3) are the TP pairs, so you choose `CUDA_VISIBLE_DEVICES=a,b,c,d` with (a, b) and (c, d) on the best links.

**P9.** A host shows `GPU0–GPU1 NV12`, `GPU2–GPU3 NV12`, and `SYS` between the two pairs. (a) Which `CUDA_VISIBLE_DEVICES` order for `TP=2, PP=2`, and which links do the PP transfers take? (b) What goes wrong with `0,2,1,3`? (c) The pre-test host showed NV12 between all pairs. What does the choice matter there?

<details><summary>Answer P9</summary>

(a) `0,1,2,3`. The TP pairs are GPU0–1 and GPU2–3 (NV12). The PP transfers go GPU0→GPU2 and GPU1→GPU3 over `SYS` (PCIe + the CPU interconnect), which is acceptable for 16 MiB per step in 1 MiB pieces. (b) Ranks 0 and 1 would sit on GPU0 and GPU2, so every TP all-reduce crosses `SYS`. That is 104–128 per rank per step on the critical path, against 16 P2P transfers that would have been cheap on NVLink. (c) Nothing. `topo_check.py` prints `every pair has the same link (e.g. an NVSwitch board): any order is equally good here` and keeps `0,1,2,3`. The link is 12 × 25 GB/s per direction (`environment.log`, `nvidia-smi nvlink -s`). All-pair NV12 on four GPUs is consistent with four GPUs of an NVSwitch board. Not verified.
</details>

**Why `CUDA_DEVICE_MAX_CONNECTIONS=1`.** The GPU accepts work through several hardware queues (connections, 8 by default). Kernels launched on different streams can sit in different queues, and the GPU may start a kernel launched *later* before one launched *earlier*. Megatron's column-parallel backward depends on launch order. It computes the input gradient, **launches the async all-reduce** of it on the NCCL stream (`layers.py:639`), then launches the weight-gradient GEMM on the compute stream, and waits only at the end (`:750`). The comment at `:640–641` says it relies on `CUDA_DEVICE_MAX_CONNECTIONS=1` so the all-reduce is scheduled before the weight-gradient GEMM. With one queue, kernels dispatch in host launch order. The all-reduce gets its SMs first and genuinely overlaps the GEMM instead of being stuck behind it. The same comment appears for the SP all-gather (`:621`) and reduce-scatter (`:653`). `train_gpt.py` refuses to start without it, and the runner sets it before any CUDA context exists. It must be in the environment when the process starts. Setting it after `torch.cuda` has initialized has no effect.

*Optional A/B (not pre-tested):* comment out the check, run `tp2pp2` without the variable, and compare where the backward all-reduce kernels (stream 43 in the pre-test) start relative to the weight-gradient GEMMs. At this size the effect is likely hidden by host overhead (§6.4).

### §6 Reading a profiler trace

`train_gpt.py` traces two steps on every rank. Step 2 is traced **with** Python stacks (`trace_rank<r>.json.gz`): it is for attribution, and its times are inflated about 3.5× (510 ms vs 141–158 ms unprofiled). Step 3 is traced **without** stacks (`timing_rank<r>.json.gz`) and is used for the order and idle time. It is still about 2× inflated (318 ms).

#### §6.1 How a kernel is tied to a group and a source line

```text
GPU kernel ncclDevKernel_AllReduce_Sum_f32_RING_LL  (cat kernel, args.correlation = 1234)
  └─ launched by cuLaunchKernelEx / cudaLaunchKernel on CPU thread T  (cat cuda_driver / cuda_runtime, same correlation)
       └─ inside record_param_comms on thread T  (args: Collective name, Process Group Description and Ranks,
       │                                           In msg nelems, dtype, Input Dims, Dst/Src Rank for P2P)
       └─ inside python_function frames on thread T  ("megatron/core/tensor_parallel/mappings.py(22): _reduce" ...)
       └─ at a time inside "day13/F mb=k" or "day13/B mb=k" on the MAIN thread  (the phase)
```

Three details matter:

- **Frame names carry the def line, not the executing line.** For a decorated function, it is the first decorator's line (Python's `co_firstlineno`): `layers.py(581): backward` is `@staticmethod` at 581, and `def backward` is at 583. `analyze_trace.py --megatron PATH` finds the call line by searching the parent's body (`calls all_reduce at ~line 35`). That is a text search, not profiler data.
- **Backward kernels come from autograd's thread.** PyTorch runs CUDA backward on a per-device worker thread. That thread's Python stack starts at the `autograd.Function.backward`, so you see `layers.py:581 backward` and nothing above it. Which linear it is follows from the order: within one microbatch's backward, the layers run last to first, MLP before attention.
- **NCCL kernels launch through the driver API** (`cuLaunchKernelEx`), so look for their launch under `cuda_driver`, not `cuda_runtime`. The first version of the analyzer missed this.

**P10.** For a TP forward all-reduce, a TP backward all-reduce and a PP exchange on rank 0: predict the launching thread, the CUDA stream (compute stream or a separate NCCL stream), and the kernel name. What does a long `ncclDevKernel_SendRecv` mean?

<details><summary>Answer P10 (pre-test, rank 0, `analyze_tp2pp2.log` examples)</summary>

| | Thread | Stream | Kernel | Why |
| --- | --- | --- | --- | --- |
| TP forward (`g`) | main | **7 = the compute stream** | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | synchronous `all_reduce` runs on the current stream (`mappings.py:35`), so later compute waits for it |
| TP backward (`f`) | **autograd** | 43, an NCCL stream | same kernel | `async_op=True` (`layers.py:639`), overlapping the weight-gradient GEMM |
| PP exchange | main | 47, another NCCL stream | `ncclDevKernel_SendRecv` (one kernel for send + recv) | `batch_isend_irecv` (`p2p_communication.py:49`) |

A SendRecv kernel starts when posted and finishes when the peer's matching operation has arrived and the data has moved. Its duration is mostly **waiting**. The first steady exchange on rank 0 (`send_forward_recv_backward`, a₁ out and g₀ in) took 33.5 ms in the pre-test and 27.218 ms in the manual mine stack trace. Rank 2 must finish F0 and B0 before it sends g₀. The interval includes waiting for the matching operations and data transfer. [Manual trace](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/analyze_tp2pp2.log)
</details>

#### §6.2 The annotated TP collective and PP transfer (pre-test, rank 0)

`analyze_tp2pp2.log` → "examples" (shortened):

```text
[tp_forward] rank 0: ncclDevKernel_AllReduce_Sum_f32_RING_LL on stream 7 (compute stream), main thread, phase F mb=0
  allreduce, TENSOR_MODEL_PARALLEL_GROUP ranks [0, 1] (size 2), 262144 x Float, dims [[256, 2, 512]]
    megatron/core/tensor_parallel/mappings.py:22 _reduce  (calls all_reduce at ~line 35)
    megatron/core/tensor_parallel/mappings.py:229 forward                     (_ReduceFromModelParallelRegion)
    megatron/core/tensor_parallel/mappings.py:498 reduce_from_tensor_model_parallel_region
    megatron/core/tensor_parallel/layers.py:1430 forward  (calls reduce_from_tensor_model_parallel_region at ~line 1480)
    megatron/core/transformer/mlp.py:258 forward                              (so: linear_fc2 of layer 0)
    megatron/core/transformer/transformer_layer.py:816 _forward_mlp
[pp_p2p] rank 0: ncclDevKernel_SendRecv on stream 47 (NCCL stream), main thread, between F1 and B0, 33532.6 us
  send+recv, PIPELINE_MODEL_PARALLEL_GROUP ranks [0, 2] (size 2), 262144 x Float, dims [[256, 2, 512], [256, 2, 512]], peers [2, 2]
    megatron/core/pipeline_parallel/p2p_communication.py:17 _batched_p2p_ops  (calls batch_isend_irecv at ~line 49)
    megatron/core/pipeline_parallel/p2p_communication.py:275 _communicate
    megatron/core/pipeline_parallel/p2p_communication.py:526 send_forward_recv_backward     (decorator line; def :527)
    megatron/core/pipeline_parallel/schedules.py:2120 forward_backward_pipelining_without_interleaving  (calls ... ~line 2382)
[tp_backward] rank 0: same AllReduce kernel on stream 43 (NCCL stream), autograd thread, phase B mb=0
  allreduce, TENSOR_MODEL_PARALLEL_GROUP ranks [0, 1], 262144 x Float
    megatron/core/tensor_parallel/layers.py:581 backward       (the first all-reduce of B0 on stage 0 = layer 2's linear_fc1 dx)
```

Open the same trace in Perfetto (<https://ui.perfetto.dev>, drag in `trace_rank0.json.gz`). Search for `record_param_comms`, click one, and read its args. Then follow the flow arrow to the kernel. Do this for one kernel of each kind before you trust the analyzer.

#### §6.3 Reading the 1F1B order

The order comes from the main thread's `day13/F mb=k` and `day13/B mb=k` ranges, which wrap Megatron's own calls (`traced(...)` and the wrapped `schedules.backward_step`). Pre-test, every layout and every rank:

```text
stage 0 (ranks 0, 1):  F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7     warmup 1 = P - s - 1, cooldown 1
stage 1 (ranks 2, 3):  F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7     no warmup
```

This is Day 12's 1F1B at P = 2, M = 8. Both TP ranks of a stage run the same order. They must: their all-reduces pair tensors by call order.

#### §6.4 Idle time: what the trace shows, and what it cannot

**Idealized [derived].** With t_F = 1, t_B = 2 and free communication, 1F1B at P = 2, M = 8 takes $(M + P - 1)(t_F + t_B) = 27$ units. Each stage idles 3, a bubble of $(P - 1)/(M + P - 1) = 1/9$, or 11.1 %. Stage 1 idles before F0 (fill) and after B7. Stage 0 idles while the first gradient makes its round trip and before its last backward (drain).

**Measured, stack step, `tp2pp2` [pre-test, profiled].** The same pattern appears, at profiler-inflated scale:

| What | Rank 0 (stage 0) | Rank 2 (stage 1) |
| --- | --- | --- |
| fill | — | first kernel of F0 at 14.2 ms. Stage 0's F0 GPU span is 0.7–10.9 ms |
| first round trip | 39.0 ms idle after F1, 33.5 ms of it overlapping the SendRecv that waits for g₀ | — |
| drain | 11.1 ms idle after B6, waiting for g₇ | B7 ends at 254.4 ms. Rank 0's B7 ends at 267.2 ms |
| compute kernels in the window | 22.9 ms of 267 ms | 32.4 ms of 254 ms |

**The trace locates several pipeline waits.** Compute kernels fill about 10% of each rank's pre-test window. The remaining time includes compute-stream NCCL kernels and idle spans, with the idle spans split by overlapping communication and a residual other/CPU bucket. That residual cannot be attributed to a specific host cause from these traces. The small model and profiler overhead limit GPU utilization. The two unprofiled step samples per layout are reported in A.1 and B.1. Fill, first round trip and drain are visible, while the fraction of the step spent in pipeline dependencies is **[not isolated]**. Two more pre-test observations:

- **Stage imbalance:** stage 1 has about 45 % more compute than stage 0 in this pre-test (`tp1pp2` timing step: 47.3 vs 32.5 ms of kernels; the ratio varies from run to run at this size). It has the same three layers plus the 512 → 8192 output layer and the cross-entropy. That is the Day 12 §4 "slow stage" in a real model.
- **One-off host stalls:** the timing step had a 40–50 ms gap on one stage, inside a forward, on both of its TP ranks at once. With the synchronous TP all-reduce on the compute stream, a stall on one rank stretches its partner's span too. Do not read it as a pipeline bubble: it has no SendRecv overlap. **Cause found and fixed in [Appendix C](#appendix-c-the-host-stall-found-and-fixed-assistant-run-2026-11-22-measured):** Python's cyclic GC was freeing the previous step's profiler inside F1. `train_gpt.py` now frees it between steps, which shortens the timing step from about 310 to about 200 ms. A second, rarer stall behind the NCCL watchdog's mutex is explained there too.

---

## Block 2: Code (about 1.5 h)

You write two small files. Everything else is provided. While a TODO remains, the script prints `INCOMPLETE (TODOs left): ...` and exits with code 3 before any communication.

### Stage 0 (15 min): setup, topology, symbols

On the instance (after the setup in "Read this first"):

```bash
D13=experiments/day13_megatron
python $D13/topo_check.py                                   # links, fastest pair, 4-GPU order
python $D13/find_symbols.py /workspace/Megatron-LM          # Block 3's table, "0 not found"
python $D13/rank_mapper.py --megatron /workspace/Megatron-LM --layouts all   # reference: PASSED
```

`topo_check.py` and `find_symbols.py` also run on Windows: `--file` takes a saved `topo -m`, and the second needs a local `megatron/core` checkout (Block 3).

### Stage 1 (25 min): **YOU WRITE** `rank_mapper.py` TODO 1–2

`my_coords(rank, tp, dp, pp)` → `(tp_rank, dp_rank, pp_rank)` (§1.1), and `my_groups(tp, dp, pp)` → every group kind (§1.2). They run without torch, on Windows too:

```bash
python experiments/day13_megatron/rank_mapper.py --impl mine --layouts all
```

**Check:** your table for ranks 0–3 matches §1.2 and the last line reads `RANK MAPPER (mine, 9 layouts, vs closed form): PASSED`. Then on the instance add `--megatron /workspace/Megatron-LM` (`... and Megatron's RankGenerator): PASSED`). Inside a job, `--mapper mine` makes `train_gpt.py` compare the groups NCCL built against yours.

### Stage 2 (30 min): **YOU WRITE** `my_step.py` TODO 3–4

`loss_func(output_tensor)` and `forward_step_func(data_iterator, model)`. Their docstrings give the contract. §2.1–§2.2 and P5–P6 give the reasons. Test on one GPU first, then on four:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/workspace/Megatron-LM OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 $D13/train_gpt.py --tp 1 --pp 1 --impl mine --run-id try1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 $D13/train_gpt.py --tp 2 --pp 2 --impl mine --mapper mine --run-id try1
```

**Check:** both print `[step 0] loss 9.1240034 ... [step 4] loss 8.3576717` (the pre-test's values, identical in every layout) and end with `RUN ...: ... PASSED`. A first loss near 1.14 means P6(b). The `_mine` suffix keeps your output folders apart from the reference's.

### Stage 3 (5 min + 4.5 min run): everything once

```bash
RID=manual_$(date -u +%Y%m%dT%H%M%SZ)
bash $D13/run_server_once.sh "${RID}_ref"                              # reference step functions and mapper
bash $D13/run_server_once.sh "${RID}_mine" --impl mine --mapper mine --quick   # yours: tp1pp1, tp2pp2, tp2pp2_sp
```

### Stage 4 (30 min): annotate the trace

Fill notes §5 from `analyze_tp2pp2.log` and the trace itself (Perfetto). Pick one TP collective inside a stage and one PP exchange. For each, give rank, kernel, stream and thread, phase, group ranks and size, elements and shape, and the Megatron file:line, plus one sentence on why it is there. Then fill §6 from the order and idle lines.

### Pitfalls (symptom → cause)

| Symptom | Cause |
| --- | --- |
| `Set CUDA_DEVICE_MAX_CONNECTIONS=1 in the launching shell` | the variable is missing in this shell (the runner sets it, single commands need the `export`) |
| `KeyError: 'LOCAL_RANK'` | `train_gpt.py` started with `python` instead of `torchrun` |
| `ModuleNotFoundError: megatron` / `einops` | `PYTHONPATH` not set to the checkout / run `setup_megatron.sh` |
| `AssertionError: sequence parallel not supported by torch LayerNorm` | `--sp` with the stock local spec. `train_gpt.py` handles it, so if you see this, `layer_spec` was bypassed (§4) |
| `[step 0] loss 1.14...` | logged value shares storage with the loss, P6(b) |
| loss correct on the last stage but the run hangs at the first backward | a stage skipped `next(data_iterator)` or called it twice, and the stages disagree on microbatch count. Or one rank raised: read the **first** traceback |
| `[groups] ... FAILED` with `--mapper mine` | your group lists differ. Compare with `rank_mapper.py --impl mine` on CPU first |
| `mkdir ... exists` (FileExistsError) | the run id was used before. Output folders are never overwritten |

---

## Block 3: Read Megatron Core (about 1 h; Stops 1–5 required, Stop 6 optional)

**Pinned version: Megatron-LM `core_v0.19.2`, commit `4b4acac9a1d28ea6829c8d4f566d75698a21249d`.** Every line number below was produced by `find_symbols.py` on that commit (pre-test `symbols.log`, `0 not found`), or read in the same checkout. On Windows, check out only `megatron/core`, because some test paths exceed the path limit:

```bash
git clone --no-checkout --depth 1 --branch core_v0.19.2 https://github.com/NVIDIA/Megatron-LM.git
```

```bash
cd Megatron-LM && git restore --source=HEAD -- megatron/core
```

```bash
python ../experiments/day13_megatron/find_symbols.py .
```

Paths are relative to `megatron/core/`.

### Stop 1 (15 min): the groups. `parallel_state.py`

- `generate_masked_orthogonal_rank_groups` `:269`: read the docstring's equation (1) and the `dp_group` example. It is §1.1.
- `RankGenerator` `:465`, `get_ranks(token)` `:534`. The token `'tp-pp'` asks for the ranks that vary in TP and PP together: the model-parallel group.
- `default_embedding_ranks` `:560` (first and last of a PP group), `default_position_embedding_ranks` `:569` (first only).
- `initialize_model_parallel` `:601`: default `order` `:617`, the generator `:859`, model-parallel groups `:1147`, TP `:1164`, PP `:1240`, embedding `:1269`. Each loop creates every group and keeps its own (P3).
- Getters: `get_model_parallel_group` `:1697`, `get_tensor_model_parallel_group` `:1704`, `get_pipeline_model_parallel_group` `:1749`, `get_data_parallel_group` `:1758`, `get_embedding_group` `:1863`, `get_position_embedding_group` `:1870`.

*Q: `get_ranks('tp-pp')` produced `[[0, 1, 2, 3]]` here. For `TP=2, DP=2, PP=2`, what does it produce, and why is that "one model replica"?*

### Stop 2 (20 min): TP inside a layer. `tensor_parallel/layers.py`, `mappings.py`, `cross_entropy.py`

- `ColumnParallelLinear` `:869`: `allreduce_dgrad` `:1066` (TP > 1 and SP off). Forward `copy_to_tensor_model_parallel_region` `:1147` is your `f` (identity forward).
- `LinearWithGradAccumulationAndAsyncCommunication` `:525`, `backward` (decorator `:581`, def `:583`): `grad_input = grad_output.matmul(weight)`, then the **async all-reduce** `:639`, then the weight gradient, then `handle.wait()` `:750`. With SP: the forward all-gather `:571`, the backward re-gather of the saved input `:617` (waited at `:630`), and the reduce-scatter `:650` (waited at `:733`). §4's figures place each one. Also read the three `CUDA_DEVICE_MAX_CONNECTIONS` comments (`:621`, `:640`, `:653`).
- `RowParallelLinear` `:1249`, forward `:1430`: `reduce_from_tensor_model_parallel_region` `:1480` (your `g`), or `reduce_scatter_to_sequence_parallel_region` `:1476` with SP.
- `mappings.py`: `_reduce` `:22` (`all_reduce` at `:35`), `_gather_along_first_dim` `:118`, `_reduce_scatter_along_first_dim` `:159`, `reduce_from_tensor_model_parallel_region` `:498`, `gather_from_sequence_parallel_region` `:522`, `reduce_scatter_to_sequence_parallel_region` `:536`.
- `VocabParallelEmbedding` `:230`: each rank looks up only its vocab rows (others give 0). The layer has all-reduce (`:377`) and reduce-scatter (`:372`) paths. In this experiment the word embedding uses all-reduce in both runs. With SP on, the embedding output is split locally along the sequence after adding positions. Backward all-gathers the output gradient so the vocabulary-row owner sees the gradient for every occurrence (§4).
- `cross_entropy.py`: `_VocabParallelCrossEntropy` `:119`, first all-reduce `:130`. Find the other two and say what each reduces.

*Q: why does the embedding need a forward all-reduce but no backward collective?*

### Stop 3 (10 min): one model across stages. `models/gpt/gpt_model.py`, `transformer/transformer_block.py`

- `GPTModel` `:51`, `forward` `:519`: embedding only with `pre_process` `:345`, the decoder `:583`, return early when not `post_process` `:683`, output layer `:762`, loss `:794` → `compute_language_model_loss` (`models/common/language_module/language_module.py:155`, vocab-parallel cross-entropy `:192`).
- `TransformerBlock.set_input_tensor` `:443`: where the schedule's received activation enters stage 1 (`schedules.py:501`).
- `get_num_layers_to_build` `transformer_block.py:70`: 6 layers / 2 stages.

### Stop 4 (10 min): the schedule you already know. `pipeline_parallel/`

Day 12 Block 3 covered it. Today, check the pieces that differ:

- `get_tensor_shapes` `schedules.py:2088`: `(seq, micro_batch, hidden)`, $\div\ \text{TP}$ with SP `:2114`.
- `forward_backward_pipelining_without_interleaving` `:2120`: `P2PCommunicator(pp_group=get_pipeline_model_parallel_group())`. Its neighbours are **global** ranks from the PP group, so rank 0's `next_rank` is 2. Warmup `:2243`, steady `send_forward_recv_backward` `:2382`, `send_backward_recv_forward` `:2412`.
- `forward_step_calc_loss` `:290`, `/= num_microbatches` `:335`.
- `p2p_communication.py`: `P2PCommunicator` `:140`, `_communicate` `:275`, `_batched_p2p_ops` `:17` → `batch_isend_irecv` `:49`.

### Stop 5 (10 min): after the schedule. `distributed/finalize_model_grads.py`

- `finalize_model_grads` `:563`: DP sync, then `_allreduce_non_tensor_model_parallel_grads` (call `:647`, def `:416`, SP condition `:452`), then `_allreduce_word_embedding_grads` (call `:659`, def `:164`).
- `_get_shared_word_embedding_weight` `:132`: `if model_module.share_embeddings_and_output_weights` `:146`. That condition is §1.4's "untied → nothing to reduce".
- `language_module.py` `setup_embeddings_and_output_layer` `:202`, all-reduce of the weight `:308` (tied only).

### Stop 6 (optional, 10 min): the production loop. Repository root (full clone on the instance)

- `pretrain_gpt.py`: `get_batch` `:97`, `loss_func` `:209`, `forward_step` `:282`, `train_valid_test_datasets_provider` `:444`, `get_embedding_ranks` `:480`.
- `megatron/training/training.py`: `get_model` `:1706`, `setup_model_and_optimizer` `:2017`, `train_step` `:2336` → `forward_backward_func(...)` `:2407` → `optimizer.step()` `:2474`. `config.finalize_model_grads_func = finalize_model_grads` `:3550`, `get_forward_backward_func(` `:3614`.

These lines were read on the instance's full checkout (`grep -n`), not produced by `find_symbols.py`, which covers `megatron/core` only.

---

## Block 4: Run + summarize (30 min)

### Commands (the manual run)

```bash
cd /workspace/AI-Infra-Learning && source .venv/bin/activate
bash experiments/day13_megatron/setup_megatron.sh
RID=manual_$(date -u +%Y%m%dT%H%M%SZ)
bash experiments/day13_megatron/run_server_once.sh "${RID}_ref"                                  # ~4.5 min
bash experiments/day13_megatron/run_server_once.sh "${RID}_mine" --impl mine --mapper mine --quick  # ~2 min
```

Stages, in order (each logged to `outputs/day13_megatron/session_<RID>/<stage>.log`):

1. `environment`: versions, GPUs, P2P matrix, `topo -m`, `nvlink -s`, the Megatron tag and commit, and a project snapshot.
2. `topology`, `symbols`, `mapper`.
3. Jobs: `run_tp1pp1` (1 GPU), `run_tp2pp1` and `run_tp1pp2` on the fastest pair, `run_tp2pp2` and `run_tp2pp2_sp` on the chosen 4-GPU order.
4. `analyze_<layout>`, `compare`, `verify_core`.

The archive is `outputs/day13_exports/day13_<RID>.tar.gz` (about 28 MB) with a `SHA256SUMS` manifest. **The runner never stops the instance.**

**Expected:** `RANK MAPPER (...): PASSED`, five `RUN ...: ... PASSED` in ref and three in mine with `--quick`, `COMPARE run_<RID>: PASSED`, and `OUTPUT CHECKS PASSED: ...`. Both manual sessions ended with exit code 0.

### Download to Windows, then stop the instance

```powershell
experiments/day13_megatron/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>_ref
```

```powershell
experiments/day13_megatron/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>_mine
```

Each call checks the archive's SHA-256, extracts it to `outputs/day13_downloads/<RID>/`, and re-hashes every manifest entry. Look for `Manifest OK` and `exit_code=0`, then stop the instance. From inside: `vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` prints `stopping instance <id>.`, and SSH then refuses. Confirm in the Vast console that the instance shows *stopped*: an idle running instance is still billed.

### How to read the results

1. **`mapper.log` and each `run_*.log`:** the group table and `[groups] built by Megatron == ... mapper: PASSED`. Your notes §3 table comes from the 4-GPU one.
2. **`compare.log`, top table:** losses per step (equal in every layout in the pre-test and both manual sessions), the P2P shape and the unprofiled step times. **Gradient norms:** PP alone is bitwise equal to 1 GPU (`77 norms bitwise equal`). Any TP layout differs in the 7th digit, because the reduction order changes. Day 11 made the same point for one MLP.
3. **`compare.log`, counts:** each rank's `(collective, phase, elements) → count` against §3's derived counts, and its F/B order against 1F1B.
4. **`analyze_tp2pp2.log`:** the stack step's per-rank catalogue (every NCCL kernel grouped by kind, thread, stream, phase and Megatron frame), the three examples (§6.2), then the timing step's order, spans and idle split. The idle split measures overlap with a SendRecv, overlap with other-stream NCCL, and the residual other/CPU. To identify a pipeline dependency, check which activation or gradient the following work needs.
5. **`run_*.log`:** `[step k] loss ... ms` per step. Steps 2 and 3 are profiled, and steps 0–1 include warm-up. Only steps 1 and 4 are unprofiled steady steps, and the step time is host-bound (§6.4).

### What to write in `notes.md`

§1 predictions (yours) against the answers, plus checks of your Day 11/12 predictions; §2 run record; §3 the groups; §4 the step wiring in your words; §5 the annotated trace; §6 the 1F1B order and idle time; §7 SP on vs off; §8 topology; §9 the Megatron mapping; §10 explanations; §11 not covered. Label values **[measured]**, **[pre-test]**, **[simulated]**, **[derived]**, and causes the data does not isolate **[not isolated]**.

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. A colleague launches DP=1 TP=2 PP=2 with <code>CUDA_VISIBLE_DEVICES=0,2,1,3</code> on a host where only 0–1 and 2–3 are NVLink pairs. What breaks, what does not, and what do you change?</summary>

Nothing breaks functionally: groups, losses and order are unchanged. TP ranks 0 and 1 now sit on GPU0 and GPU2, so all 104–128 TP all-reduces per rank per step cross the slow link on the critical path, while the 16 P2P transfers use NVLink. Change the order to `0,1,2,3` so TP pairs are NVLink pairs (§5).
</details>

<details><summary>2. Why is rank 2's activation partner rank 0 and not rank 1, and what would go wrong with SP on if the pairing were crossed?</summary>

PP groups are ranks with the same TP rank (and DP rank), so `[0, 2]` and `[1, 3]`. With SP on, rank 0 holds positions 0–127 and rank 1 holds 128–255, and the receiver with the same TP index expects exactly that shard. Crossing them would feed stage 1 the wrong half of the sequence: same shape, no error, wrong numbers.
</details>

<details><summary>3. Your loss log shows 1.14 while the model is learning normally. Why, and where is the line in Megatron that does it?</summary>

The logged tensor shares storage with the returned loss, and `forward_step_calc_loss` divides it by `num_microbatches` in place (`schedules.py:335`): 9.12 / 8. Clone before logging. Gradients are unaffected.
</details>

<details><summary>4. You see <code>ncclDevKernel_AllReduce_Sum_f32_RING_LL</code> on two different streams of rank 0. Which is which, and which one can overlap compute?</summary>

On the compute stream: the forward all-reduces (synchronous, `g`, launched by the main thread). On an NCCL stream, launched from autograd's thread: the backward all-reduce of a column-parallel input gradient (`async_op=True`, `layers.py:639`, `f`). The backward one overlaps the weight-gradient GEMM, which is the reason for `CUDA_DEVICE_MAX_CONNECTIONS=1`.
</details>

<details><summary>5. Untied → tied embeddings at DP=1 TP=2 PP=2: which collectives appear, over which ranks, how large?</summary>

At init, one all-reduce over `[0, 2]` and one over `[1, 3]` copies stage 0's word embedding into the last stage's output weight. Each step, `finalize_model_grads` all-reduces the word-embedding gradient over the same groups: $4096 \times 512$ FP32 = 8 MiB per rank. The groups themselves exist in both cases.
</details>

<details><summary>6. TP=2 PP=2 took 141–158 ms per step and one GPU took 159–171 ms. Is Megatron's parallelism broken?</summary>

No. Each rank's compute kernels dropped to 23–33 ms (vs 84 ms on one GPU in the profiled timing step), but the step is host-bound: Python and launch overhead per microbatch did not shrink, and TP adds collective launches. Scaling shows only once kernels dominate: bigger hidden size, longer sequences, or larger microbatches (§6.4).
</details>

**Next (Day 14):** MoE + expert parallelism. The `ep` in `tp-cp-ep-dp-pp` becomes non-trivial.

---

## Appendix A: pre-test results (`pretest3`, assistant run, 2026-11-16) [pre-test]

`bash experiments/day13_megatron/run_server_once.sh pretest3`, 2026-11-16 02:05:57–02:08:10 UTC (2 min 13 s). These are **pre-test** numbers for comparison, not your measurements.

**Host.** Vast.ai container 60863550 (hostname `afbdc7f1edfe`), 4 × **A100-SXM4-80GB**, NV12 between every pair (12 × 25 GB/s), all GPUs on NUMA node 1 (CPU affinity 64–127, 192–255), P2P access everywhere, driver 580.178.04. Python 3.12.15, torch 2.14.1+cu130, CUDA build 13.0, NCCL 2.30.7, Megatron-LM `core_v0.19.2` (`4b4acac9`), einops 0.8.2, `CUDA_DEVICE_MAX_CONNECTIONS=1`. GPU choice: pair `0,1`, 4-GPU order `0,1,2,3`. Exit code 0. `OUTPUT CHECKS PASSED: mapper vs RankGenerator, 5 Megatron jobs (tp1pp1, tp1pp2, tp2pp1, tp2pp2, tp2pp2_sp): groups == mapper, losses and step-0 gradient norms == 1-GPU reference, P2P shapes, NCCL counts, 1F1B order`. Archive SHA-256 `a46101dcb68cf02c0841cc8d1d9d5df2a34ff908504de4d05bb1823ff6b98389`, 87 manifest files re-verified on Windows.

Logs: [`outputs/day13_downloads/pretest3/session_pretest3/`](../outputs/day13_downloads/pretest3/session_pretest3/) ([environment](../outputs/day13_downloads/pretest3/session_pretest3/environment.log), [topology](../outputs/day13_downloads/pretest3/session_pretest3/topology.log), [symbols](../outputs/day13_downloads/pretest3/session_pretest3/symbols.log), [mapper](../outputs/day13_downloads/pretest3/session_pretest3/mapper.log), [run tp2pp2](../outputs/day13_downloads/pretest3/session_pretest3/run_tp2pp2.log), [analyze tp2pp2](../outputs/day13_downloads/pretest3/session_pretest3/analyze_tp2pp2.log), [analyze tp2pp2_sp](../outputs/day13_downloads/pretest3/session_pretest3/analyze_tp2pp2_sp.log), [compare](../outputs/day13_downloads/pretest3/session_pretest3/compare.log)). Per-layout JSON and traces are in [`run_pretest3/`](../outputs/day13_downloads/pretest3/run_pretest3/).

### A.1 Layouts: groups, losses, gradients, shapes, step time

| Layout | GPUs | Groups = mapper | Loss step 0 … 4 | Step-0 grad norms vs 1 GPU (77 params) | P2P shape | Params / rank | Unprofiled step ms (steps 1, 4) |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| tp1pp1 | 0 | yes | 9.1240034, 8.9108639, 8.7221317, 8.5237522, 8.3576717 | reference | — | 27.44 M | 158.7, 171.2 |
| tp1pp2 | 0,1 | yes | identical | **77 bitwise equal** | [256, 2, 512] | 13.78 M / 13.65 M | 121.6, 118.5 |
| tp2pp1 | 0,1 | yes | identical | 0 bitwise, worst 1.76e-7 | (not used) | 13.79 M | 183.2, 193.5 |
| tp2pp2 | 0,1,2,3 | yes | identical | 0 bitwise, worst 1.76e-7 | [256, 2, 512] | 6.96 M / 6.83 M | 141.4, 158.3 |
| tp2pp2_sp | 0,1,2,3 | yes | identical | 0 bitwise, worst 1.80e-7 | **[128, 2, 512]** | 6.96 M / 6.83 M | 175.0, 183.4 |

The worst gradient-norm difference was always `decoder.layers.1.input_layernorm.bias`. "Identical" means equal float32 bits. **The loss is a coarse check:** it is a mean over 4096 tokens, and TP's per-element rounding differences average out below its last bit. The gradient norms show them. Unprofiled step times come from two steps per layout on a host-bound workload: they are not a benchmark. CUDA peak allocated on rank 0: 607 MiB (tp1pp1), 329 (tp2pp1), 373 (tp1pp2), 209 (tp2pp2), 197 (tp2pp2_sp).

### A.2 NCCL kernels per rank in one step (stack step; all equal to §3's derived counts)

| Layout | Rank (stage) | F all-reduce 262144 | F all-reduce 512 | B all-reduce 262144 | P2P |
| --- | --- | ---: | ---: | ---: | ---: |
| tp2pp1 | 0, 1 | 104 | 24 | 104 | 0 |
| tp1pp2 | 0 (0), 1 (1) | — | — | — | 9, 9 |
| tp2pp2 | 0, 1 (stage 0) | 56 | 0 | 48 | 9 |
| tp2pp2 | 2, 3 (stage 1) | 48 | 24 | 56 | 9 |

SP on: see P8's table. 1F1B order on every rank as in §6.3.

### A.3 Profiled-step breakdown (timing step, `timing_rank<r>`, profiler-inflated)

| Layout, rank | Window ms | Compute kernels | NCCL on compute stream | Idle: waiting for P2P | Idle: during other-stream NCCL | Idle: other/CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp1pp1, 0 | 189.8 | 84.4 | 0 | 0 | 0 | 105.4 |
| tp2pp1, 0 / 1 | 236.5 / 237.6 | 59.8 / 50.8 | 25.8 / 62.4 | 0 / 0 | 26.6 / 54.4 | 124.2 / 70.0 |
| tp1pp2, 0 / 1 | 214.0 / 203.3 | 32.5 / 47.3 | 0 / 0 | 55.7 / 36.1 | 0 / 0 | 125.8 / 120.0 |
| tp2pp2, 0 / 1 / 2 / 3 | 273.4 / 274.6 / 258.7 / 259.1 | 23.0 / 23.4 / 32.7 / 27.6 | 3.4 / 34.9 / 15.5 / 26.3 | 67.2 / 65.5 / 37.7 / 36.3 | 14.3 / 34.0 / 23.8 / 62.5 | 165.6 / 116.8 / 148.9 / 106.4 |

"NCCL on compute stream" is the forward all-reduces, and it includes waiting for the TP partner, which is why ranks of one pair differ. A 40–50 ms host stall inside one forward (F1 on stage 0, F6 on stage 1) is part of "other/CPU" in `tp2pp2` (§6.4). It was isolated later: the GC freeing the previous step's profiler (Appendix C.2).

### A.4 Development findings that changed the code

- **Logged loss ÷ M.** The first dev run printed `[step 0] loss 1.1405004` for a true 9.124: the logged tensor was `loss.detach()`, and Megatron's in-place `/= num_microbatches` divided it too. Fixed with `clone()`. This became P6(b).
- **SP vs torch LayerNorm.** `pretest1` (kept in `outputs/day13_downloads/pretest1/`, exit code 1) passed the mapper and four jobs, then `tp2pp2_sp` failed on all ranks with `AssertionError: sequence parallel not supported by torch LayerNorm` (`transformer/torch_norm.py:50`). Fixed with `SequenceParallelTorchLayerNorm` (§4).
- **Bitwise-equal losses** in `pretest2` across TP layouts prompted the step-0 gradient-norm check: the loss hid TP's rounding (A.1).
- **Analyzer:** NCCL kernels launch through `cuLaunchKernelEx` (`cuda_driver`), and backward GPU ranges are empty because autograd's thread launches those kernels. Both are handled in `analyze_trace.py`. `with_stack` inflated the step about 3.5×, which led to the separate timing step.

### A.5 `pretest3_mineplumbing`

The `--impl mine --mapper mine` path was checked end to end. A *temporary* copy of the project (`/workspace/mineplumb`, deleted afterwards) had the four TODOs filled with the reference code. `run_server_once.sh pretest3_mineplumbing --impl mine --mapper mine --quick` passed the mapper (9 layouts vs `RankGenerator`) and `tp1pp1_mine`, `tp2pp2_mine`, `tp2pp2_sp_mine` with the same losses and gradient-norm differences as the reference. Archive SHA-256 verified, 67 manifest files. With the stubs as shipped, `train_gpt.py --impl mine` and `rank_mapper.py --impl mine` both print `INCOMPLETE (TODOs left)` and exit with code 3 (checked on the instance).

### A.6 What the pre-test supports and what it does not

- **Supported [pre-test]:** Megatron Core `core_v0.19.2` trains a GPT with the local spec at `TP=2 × PP=2` on torch 2.14.1+cu130 without TE or apex (SP needs the LayerNorm stand-in). The groups equal the closed-form mapper and `RankGenerator`. Every layout reproduces the 1-GPU losses bit for bit, and the step-0 gradient norms within 1.8e-7. NCCL counts per rank equal the counts derived from the model, and the order is 1F1B. The TP forward/backward all-reduces and the PP SendRecv are tied to their groups, threads, streams and Megatron frames.
- **Not measured:** the pipeline bubble as a fraction of the step (host-bound), the effect of `CUDA_DEVICE_MAX_CONNECTIONS`, the effect of GPU order (all links equal on this host), tied embeddings, BF16, DP > 1, the interleaved schedule, NCCL bandwidth.

## Appendix B: Manual results [measured]

Both sessions ran on 2026-11-21 on Vast.ai container 61775446. The ref session ran five layouts (2 min 14 s), and the mine session ran three (1 min 24 s). Both exited 0. The archive hashes and all extracted manifest entries were rechecked locally (96 ref files and 76 mine files). The Vast console stop confirmation is recorded in [notes §2](notes.md#2-run-record). [Ref output check](../outputs/day13_downloads/manual_20261121T082746Z_ref/session_manual_20261121T082746Z_ref/verify_core.log) · [Mine output check](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/verify_core.log) · [Run record](notes.md#2-run-record)

### B.1 Correctness and layout checks

Across all eight jobs, the five losses were **9.1240034, 8.9108639, 8.7221317, 8.5237522, 8.3576717**. In the three common layouts, ref and mine had identical recorded step-0 gradient sums of squares for each parameter on every corresponding rank. The discrepancy column below instead compares each layout's reconstructed full-parameter gradient norms against its own single-GPU baseline. [Ref comparison](../outputs/day13_downloads/manual_20261121T082746Z_ref/session_manual_20261121T082746Z_ref/compare.log) · [Mine comparison](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log)

| Layout | Worst step-0 grad-norm relative difference vs 1 GPU | PP buffer | Ref step 1 / 4 (ms) | Mine step 1 / 4 (ms) |
| --- | ---: | --- | ---: | ---: |
| tp1pp1 | baseline | no PP | 163.3 / 161.4 | 162.0 / 178.6 |
| tp1pp2 | 0 (77 bitwise-equal norms) | [256,2,512] | 124.8 / 142.6 | — |
| tp2pp1 | 1.76e-7 | no PP | 199.6 / 207.7 | — |
| tp2pp2 | 1.76e-7 | [256,2,512] | 145.9 / 163.7 | 166.1 / 160.9 |
| tp2pp2_sp | 1.80e-7 | [128,2,512] | 171.8 / 178.0 | 171.9 / 173.0 |

The two unprofiled steady samples per job vary across runs. They document this experiment's timing. The SP comparison reproduces the expected change in activation shape and collectives. For example, mine has 104 backward all-gathers per rank at either stage and one post-schedule gradient all-reduce of 9216 elements on stage 0 or 10240 on stage 1. Both SP settings have nine PP kernels per rank. [Mine comparison](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log) · [Detailed SP counts](notes.md#7-sequence-parallelism-on-vs-off)

### B.2 Mine profiler examples and idle time

| Rank 0, tp2pp2_mine stack trace | Phase | Group | Thread / stream | Duration |
| --- | --- | --- | --- | ---: |
| MLP row-parallel output all-reduce | F0 | TP [0,1] | main / 7 | 49.9 µs |
| Column-parallel input-gradient all-reduce | B0 | TP [0,1] | autograd / 43 | 49.6 µs |
| Batched send activation 1, receive gradient 0 | between F1 and B0 | PP [0,2] | main / 47 | 27.218 ms |

The mine trace shows 32.39 ms of compute-stream idle after rank 0's F1 (27.22 ms overlapping SendRecv) and 11.19 ms after B6 (10.28 ms overlapping P2P). Rank 2's first F0 kernel starts at 9.32 ms in its own trace window. The two stage orders match §6.3 in both SP settings. Rank 0 ran compute kernels for 24.45 ms out of its 270.06 ms stack-trace window. Its compute-stream idle fraction includes communication overlaps and residual gaps. An experimental estimate of the idealized 1/9 bubble would require isolating the time due to pipeline dependencies. [Mine annotated trace](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/analyze_tp2pp2.log) · [Full idle breakdown](notes.md#6-1f1b-order-and-idle-time)

## Appendix C: The host stall, found and fixed (assistant run, 2026-11-22) [measured]

§6.4 and A.3 left one thing open: a 40–50 ms gap inside one forward of the timing step, on both TP ranks of a stage at once, with no SendRecv overlap. This appendix reproduces it on a new instance and follows it to its cause. There turned out to be **two** causes:

| | Cause 1: the freed profiler | Cause 2: the NCCL watchdog |
| --- | --- | --- |
| Where it shows | the first steps after a profiled step (Day 13's timing step, F1) | anywhere, about once per 300–900 steps, three steps in a row |
| Size | 50–95 ms per stall | ~80–100 ms per host call, steps of 180–199 ms vs a median of 87 ms |
| Cause | Python's cyclic GC frees the previous step's `torch.profiler.profile` objects in the middle of a forward | the NCCL watchdog thread holds the process group's work-list mutex for its whole scan of ~250–300 outstanding collectives. For a few hundred ms, the median `cudaEventQuery` takes ~244 µs instead of 2–3 µs, so one scan takes 50–70 ms, and the thread issuing the next collective waits for the mutex that long (C.6) |
| Fix | `del prof; gc.collect()` right after the export, outside the timed step (now in `train_gpt.py`) | removing the lock from the path (`TORCH_NCCL_BLOCKING_WAIT=1`) removes it, but costs +67 % per step, so it is **not adopted**. The cost is 1.4–2.1 % of wall time |
| Status | fixed and verified, 3 of 3 runs | mechanism found down to the driver call, with gdb (C.3) and with Nsight Systems (C.6). Why `cudaEventQuery` slows down 100× is **[not isolated]** |

Container 61910115 (Vast.ai; 4× A100-SXM4-40GB, NV12, driver 610.57.04), torch 2.14.1+cu130, NCCL 2.30.7, Python 3.12, Megatron Core `core_v0.19.2`, same model and `tp2pp2` layout as the main run. The instance ran 04:28–05:17 UTC and was stopped from inside (`vastai stop instance`). The archive is in `outputs/day13_downloads/stall_fix/`. Its SHA-256 was checked, and all 311 files were re-hashed against `day13_stall/SHA256SUMS`; the one remaining entry is the manifest's hash of itself.

### C.1 Method: make the stall repeatable, then measure every layer

A five-step job with one stall is too little data. So the first tool is a probe that runs the same job for hundreds of steps and timestamps everything on the host:

- **`stall_probe.py`** imports `train_gpt.py` (same groups, model, init and batch). It wraps `forward_step`, `schedules.backward_step` and the six P2P methods, and records every call's start and end with `time.perf_counter_ns()`. It also hooks `gc.callbacks`, so every Python GC run (its generation, duration and number of objects freed) is on the same clock. It can repeat `train_gpt.py`'s profiler pattern, switch GC to manual, pin ranks to their GPU's NUMA CPUs, and record the caching allocator's counters each step. A *stall* is a host F/B call longer than that rank's median + 10 ms.
- **`host_sampler.py`** runs beside it. Every 5 ms, for each rank's main thread, it records the state (R running, S sleeping, D disk), the kernel wait channel, its own CPU time, context switches and page faults. It also records which of the rank's other threads used CPU. For the container it records cgroup CPU use and throttling, and the pressure-stall totals (PSI) for CPU, memory and IO. `nvidia-smi -lms 50` logs SM clocks, power and throttle reasons. Everything is on CLOCK_MONOTONIC, which is shared by all processes on a host.
- **`stall_correlate.py`** joins the two: for every stall, and for 200 random normal calls as a baseline, what did the thread, the container and the GPU do during the call?
- **`stall_catcher.py`** takes native stacks *during* a stall. It keeps gdb attached to each rank. When a main thread has slept in a futex for 25 ms while a watchdog thread runs, it sends SIGUSR2, and gdb stops the process and prints every thread's stack. The container has no `CAP_SYS_PTRACE` and Yama `ptrace_scope` is 1, so the probe opts in with `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` (`--allow-ptrace`).

There is no `perf` and no eBPF in an unprivileged container. `/proc`, `gc.callbacks` and gdb are what is left, and they were enough.

### C.2 Cause 1: the profiler is freed by the GC, inside the next forward

**Reproduction** (`run_stall_matrix.sh`, 40 steps with `train_gpt.py`'s pattern, stack step then timing step, three times; default GC; `prof` dropped by rebinding the name, as `train_gpt.py` did). Gen-0 collections that free 115 or 230 objects (one or two profilers' worth) take 51–95 ms, plus one of 9 ms. They land inside forwards on every rank, and each rank at a different step (14, 17, 22, 23, 28, 31, 34). Over the 29 unprofiled steps: median 106.9 ms, p90 173.8 ms, max 203.6 ms, and five steps more than 10 ms over the median.

**Why a gen-0 collection takes 50+ ms.** `del prof` does not free the profiler. `torch.profiler.profile.__init__` (`torch/profiler/profiler.py:1024`) stores `self.action_map` (`:1088`), a dict whose values are bound methods of the object itself (`self.start_trace`, `self.stop_trace`, `self._trace_ready`, …). That is a reference cycle: the reference count never reaches zero, and only the cyclic GC can free it. The probe checks this with a `weakref`: after `del prof` the object is still alive on every rank. The GC runs when the count of new container objects passes 700 (Python 3.12 thresholds 700/10/10). That point falls at a random place in the next step, and freeing the profiler's event buffers is what costs the 50–95 ms. Each rank allocates differently, so the GC fires at different moments on different ranks. With synchronous TP collectives and P2P, one rank's stall becomes waiting on its partner and on the other stage (notes §6).

**Two fixes, both measured (same 40-step pattern):**

| Case | Slow unprofiled steps | Max unprofiled step | Where the GC cost went |
| --- | ---: | ---: | --- |
| default GC, profiler dropped (reproduction) | 5 | 203.6 ms | inside F/B calls, 50–95 ms each |
| A: `del prof; gc.collect()` right after the export | 0 | 108.9 ms | 141–427 ms between steps, outside any timed region |
| B: `gc.disable(); gc.freeze()`, `gc.collect()` every 10 steps (Megatron's `--manual-gc`) | 0 | 110.2 ms | 67–97 ms every 10 steps, between steps |
| B in steady state, 300 unprofiled steps | — | — | median 85.8 ms vs 89.1 ms with default GC: manual GC costs nothing |

Megatron's own training loop has option B for exactly this reason: `--manual-gc` disables the GC and collects at a fixed step interval, with the comment "align the timing of garbage collection across ranks" (`megatron/training/training.py:3586–3593`, collection at `:3222–3224`). Every rank then pays the cost at the same point, between steps, instead of at random points inside the pipeline.

**The fix in `train_gpt.py`** is option A, because the job only profiles two steps: `del prof` and `gc.collect()` after each trace export. Verified with `run_stall_fix_check.sh` (`tp2pp2`, three runs of the old file and three of the fixed one, interleaved):

| Run | Timing step (old → fixed) | Longest idle gap in the timing step, ranks 0–3 (old → fixed) |
| --- | --- | --- |
| 1 | 305.5 → 198.2 ms | 63.8 / 64.5 / 59.9 / 60.7 → 11.0 / 11.0 / 9.0 / 8.2 ms |
| 2 | 319.8 → 198.5 ms | 66.3 / 65.9 / 62.2 / 60.3 → 10.7 / 10.8 / 8.4 / 8.3 ms |
| 3 | 319.4 → 201.8 ms | 67.5 / 68.0 / 60.9 / 66.0 → 11.6 / 11.5 / 9.4 / 8.8 ms |

All six jobs pass their checks with the same losses. The stack step (~460 ms) and the unprofiled steps do not change. The 40–50 ms stall in §6.4 and notes §6 was this.

### C.3 Cause 2: a rare stall behind the NCCL watchdog's mutex

With the profiler out of the way, unprofiled runs still show one episode each: three consecutive steps of 180–199 ms, while the median is 85–89 ms.

| Run (unprofiled, default settings unless noted) | Steps | Median | Max | Episode (steps > median + 50 ms) | Time above the median |
| --- | ---: | ---: | ---: | --- | ---: |
| base | 295 | 89.1 ms | 198.4 ms | 132–134 | 1.64 % |
| GC disabled (manual, every 10 steps) | 295 | 85.8 ms | 193.7 ms | 46–48 | 2.04 % |
| ranks pinned to their GPU's NUMA CPUs | 295 | 85.0 ms | 199.3 ms | 252–254 | 1.71 % |
| sampled, `s1` | 595 | 86.8 ms | 181.0 ms | 519–521 | 1.41 % |
| sampled, `s2` | 895 | 87.0 ms | 188.5 ms | 257–259 | 2.13 % |
| `TORCH_NCCL_BLOCKING_WAIT=1` | 2995 | **145.1 ms** | 191.5 ms | none | 0.83 % |

The slow calls are mostly on ranks 2 and 3 (stage 1) at the same moment, sometimes on ranks 0 and 1. With blocking wait, the slowest steps (191.5 ms at step 16, 188.9 ms at step 375) are single steps about 45 ms over that run's median. None of them is a three-step episode.

**Ruled out, with the sampler:**

- **Python GC.** No GC run overlaps any of these calls, and the episode stays with the GC disabled.
- **CPU placement.** Pinning each rank to its GPU's NUMA node does not remove it.
- **CPU starvation.** cgroup `throttled_usec` is 0. PSI cpu adds at most 68 µs inside any long-stall window, and there are no involuntary context switches. 128 CPUs serve 4 ranks.
- **Memory.** No major faults, and PSI memory and IO are 0. The caching allocator made no new `cudaMalloc`/`cudaFree` after warmup.
- **The GPU.** SM clock 1410 MHz with no throttle reason around every stall.

**What the sampler shows instead.** During a stall the rank's main thread is asleep (state S in 100 % of samples) in `futex_wait_queue_me`, a user-space lock or condition variable. In normal calls it sleeps 16–55 % of the time. Meanwhile one of the rank's **`pt_nccl_watchdg`** threads is running in 7–9 consecutive 5 ms samples. In normal calls that thread barely shows.

**What gdb shows (rank 3, the first capture, 26 ms into the stall):**

```text
main thread         pthread_cond_wait <- autograd ReadyQueue::pop <- Engine::execute <- THPEngine_run_backward   (backward runs on the device thread)
pt_autograd_3       pthread_mutex_lock  <- c10d::ProcessGroupNCCL::workEnqueue  <- ProcessGroupNCCL::allreduce   (a TP all-reduce in backward)
pt_nccl_watchdg     (libcuda, 6 frames) <- cuEventQuery <- cudaEventQuery <- WorkNCCL::finishedGPUExecutionInternal
                                        <- ProcessGroupNCCL::Watchdog::runLoop
all other threads   idle in poll / pthread_cond_(timed)wait (20 NCCL process groups, each with a watchdog, a heartbeat and proxy threads)
```

**The source** (`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`, torch 2.14.1, commit `5c48869`) explains the chain:

- **The lock.** Every collective ends in `workEnqueue`, which takes `workMetaListMutex_` to append the work to the watchdog's list (`:3583`).
- **The scan.** The watchdog wakes every 100 ms (`kWatchdogThreadSleepMillis`, `:493`). It takes the *same* mutex (`:2273`) and keeps it for the whole scan of the list. The scan checks each work with `isCompleted()`, which is one `cudaEventQuery` per outstanding collective (`:2441–2463`). The code's own comment at the end of the loop expects the scan to be "slowed down (but not hung) by cuda api contention".
- **The result.** When the scan takes ~80 ms, the next collective's `workEnqueue` waits ~80 ms. So does the host call that issued it, and, through the synchronous all-reduce and the P2P sync, its TP partner and the other stage. Why the scan is slow was first read from this snapshot as "one `cudaEventQuery` takes ~80 ms". The Nsight Systems trace in C.6 corrects that: the scan contains hundreds of queries, and the median query time in the slow scans is about 100× higher than usual. A single gdb snapshot shows where a thread is, not how long it has been there.

**The causal test.** With `TORCH_NCCL_BLOCKING_WAIT=1`, `workEnqueue` skips the list (`if (!terminateProcessGroup_.load() && !blockingWait_)`, `:3582`). The watchdog then has nothing to scan, and nothing can hold the mutex. In 2995 steps there was no episode; before, there was one in every run of 295–895 steps. The mechanism is confirmed. But blocking wait makes every `work.wait()` block the CPU until the collective has finished on the GPU, so the host can no longer run ahead. The backward host call goes from 4.5 to 9–11 ms, and the median step from 87 to **145 ms**. That is far worse than the 1.4–2.1 % the episodes cost. **Decision: keep the default.** At this size the episode is cheaper than any fix found here.

**What is not isolated.** Why the median CUDA event query time rises from 2–3 µs to ~244 µs for a few hundred ms, on both TP ranks of a stage at the same moment (C.6). libcuda is stripped, and the watchdog is in state R inside the driver the whole time. The constant 244 µs and the two processes slowing together point at something shared in the driver or kernel, not at PyTorch. Next steps, if this mattered at scale:

- a newer PyTorch, in case the watchdog no longer holds the mutex across event queries;
- a CUDA/driver upgrade;
- fewer NCCL process groups (Megatron creates 20 per rank here, each with its own watchdog);
- a report to PyTorch, with `stall_catcher.py`'s stacks attached.

### C.4 What to take from this

- **A symptom is not a cause.** Both stalls looked the same in the profiler trace: a long forward with no SendRecv overlap. Only following the waiting chain thread by thread told them apart. Cause 1 was the main thread *running* Python's GC. Cause 2 was the main thread *sleeping* on a lock that another thread held.
- **Moving a cost is a valid fix when it moves it off the critical path.** This answers the question asked in the notes review: a `synchronize()` before 1F1B only moves the stall, because it stays inside the step and every rank still waits. `gc.collect()` between steps (or Megatron's `--manual-gc`) also only moves the cost, but to a point where no rank is waiting for another. That is the difference.
- **Measure a fix's steady-state cost before adopting it.** Blocking wait removed the rare stall and slowed every step by 67 %.
- **In a synchronous job, every rank's hiccup is everyone's.** The 1F1B schedule turns one rank's 80 ms into P2P waiting on the other stage (notes §6). The bigger the job, the more often some rank hiccups, which is why production loops align their GC and keep the host off the critical path.

### C.5 Files and commands

| File | What it does |
| --- | --- |
| `stall_probe.py` | the long run with host timestamps, GC hooks, profiler pattern, `--gc manual`, `--pin-numa`, `--pid-dir`, `--allow-ptrace` |
| `run_stall_matrix.sh` | the six cases of C.2 and the first rows of C.3 (~6 min) |
| `host_sampler.py`, `stall_correlate.py`, `run_sampled.sh` | the sampled runs of C.3 (`CATCH=1` adds `stall_catcher.py`) |
| `stall_catcher.py` | gdb stacks in the middle of a stall |
| `run_stall_fix_check.sh` | old vs fixed `train_gpt.py`, N runs each (needs `train_gpt_before.py`, a copy of the pre-fix file) |
| `train_gpt.py --profiler nsys`, `nsys_trace.py`, `run_nsys_check.sh` | C.6: the timing step under Nsight Systems, read by `analyze_trace.py`; then `stall_probe.py` under nsys |
| `nsys_watchdog.py` | C.6: long calls, mutex waits and watchdog scans in a long nsys capture |

```bash
bash experiments/day13_megatron/run_stall_matrix.sh m1
bash experiments/day13_megatron/run_sampled.sh s1 600
CATCH=1 bash experiments/day13_megatron/run_sampled.sh c1 1500
TORCH_NCCL_BLOCKING_WAIT=1 bash experiments/day13_megatron/run_sampled.sh bw1 3000
bash experiments/day13_megatron/run_stall_fix_check.sh fx1 3
bash experiments/day13_megatron/run_nsys_check.sh n1 3 1200
```

### C.6 Nsight Systems instead of the PyTorch profiler (assistant run, 2026-11-22) [measured]

Asked after C.2: can `nsys` replace the PyTorch profiler, so that no profiler object is left for the GC? Yes. nsys records from outside Python, through its own injected library, so the training process never creates a `torch.profiler.profile`. As a bonus, it records every CUDA API call and every long OS-runtime call (with a backtrace) on **every thread**. That is exactly the view C.3 had to build from `/proc` samples and gdb. Same instance as C.1–C.5 (restarted), Nsight Systems 2025.3.1. The instance ran 05:40–06:33 UTC and was stopped from inside. The archive's SHA-256 matched, and all 187 manifest entries were re-hashed locally.

**How the job is traced** (`run_nsys_check.sh`):

- **Code changes in `train_gpt.py --profiler nsys`.** No PyTorch profiler is created. The `day13/step`, `day13/F mb=k` and `day13/B mb=k` ranges become NVTX ranges (`torch.cuda.nvtx.range`). Only the timing step is recorded: `torch.cuda.profiler.start()` / `stop()` bracket it, which nsys sees as a capture range. The stack step is not traced in this mode.
- **Launch.** Each rank runs under its own nsys:

  ```bash
  torchrun --no-python ... nsys profile -t cuda,nvtx,osrt --osrt-threshold=100000 --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop -o run/nsys/nsys_rank%q{RANK} python -u train_gpt.py ...
  ```

  This container has `perf_event_paranoid` = 4, so CPU sampling and context-switch tracing are off. CUDA, NVTX and OS-runtime tracing, including OS-runtime backtraces, need no privileges.
- **Reading the capture.** `nsys export --type sqlite`, then `nsys_trace.py` turns the kernel, CUDA-API and NVTX tables into the events `analyze_trace.py` already reads. The same order, span and idle-time code therefore runs on both tools. It also prints every thread's calls ≥ 1 ms.
- **What nsys does not give here:** process-group names and element counts (PyTorch's `record_param_comms`), and Megatron frames (`with_stack`). Collectives are named by their kernel. The attribution step stays with the PyTorch profiler.

**One pitfall, found by the first smoke run.** `cudaProfilerStart` takes **548–741 ms**, and the time differs from rank to rank. In the first version it ran *after* the step's barrier. Each rank then started the step when its own capture was ready. Rank 0 sat idle for 129.6 ms after F1 (128.8 ms of it overlapping the SendRecv that waits for stage 1). The four ranks reported timing steps of 249.8, 330.6, 116.2 and 221.8 ms. Moving `start()` before the barrier fixed it. A profiler's start-up cost is per rank, so it has to finish before the ranks synchronize.

**PyTorch profiler vs nsys, same job, three runs each, interleaved** (both with the C.2 fix):

| | PyTorch profiler, timing step | nsys, timing step |
| --- | --- | --- |
| Timing step, rank 0 | 198.3 / 201.3 / 201.1 ms | **126.6 / 126.2 / 124.7 ms** |
| GPU window, stage 0 / stage 1 | 143–145 / 135–137 ms | 123–125 / 116–118 ms |
| Longest idle gap, stage 0 / stage 1 | 11.3–12.2 / 7.8–9.3 ms | 9.2–10.1 / 6.8–7.2 ms |
| Unprofiled steps in the same jobs | 93–115 ms | 96–117 ms |
| 1F1B order | identical in all six jobs | |

The PyTorch profiler, even without stacks, adds about 90 ms to a ~110 ms step; nsys adds about 15 ms. nsys's start-up shows instead in step 0 (1422–1445 vs 1296–1321 ms) and in `cudaProfilerStart`, both outside the timing step.

**Cause 2 under nsys.** `stall_probe.py`, 1200 steps, whole run under `nsys -t cuda,osrt --osrt-threshold=10000000`. Tracing every CUDA call has a cost: median step 120.5 ms instead of ~87 ms (+38 %), and 240–270 MB per rank (600–670 MB as sqlite). `nsys_watchdog.py` then reads three things from the capture:

1. **The chain, with backtraces and no ptrace.** The main thread and the autograd thread wait 16–71 ms in `pthread_mutex_lock` ← `ProcessGroupNCCL::workEnqueue` ← `allreduce`. This happens three times about one step apart, on both stage-1 ranks, twice in the run: at about 34.6–35.0 s and 98.0–98.5 s of each rank's own timeline.
2. **What the watchdog does meanwhile.** It is not stuck in one call. Inside each wait, the TP group's watchdog makes 109–483 CUDA calls back to back: `cudaEventQuery` and the `cudaThreadExchangeStreamCaptureMode` of PyTorch's capture-mode guard. Together they fill 99 % of the wait, and the longest single call is 1.9 ms.
3. **Why the scan is slow.** Rank 2's TP watchdog scanned 1436 times. Normally a scan is 318 queries in 1.44 ms, about **3 µs per query**. The six slow scans have 138–294 queries in 34–72 ms, with a median query time of **243–244 µs** in each slow scan. The list is no longer than usual. Median query time is about 100× higher. The C.3 reading of the gdb snapshot ("one `cudaEventQuery` takes ~80 ms") was wrong in its detail. A snapshot shows *where* a thread is; only a trace shows *how long* each call took.

nsys also shows a second, rarer pattern. Two threads of one process block at the same moment for 30–190 ms:

| Rank | Watchdog `cudaEventQuery` | Other thread at the same moment |
| --- | --- | --- |
| 3 | 145.9 ms | autograd thread in `cudaEventCreateWithFlags`, 188 ms |
| 3 | 104.1 ms | main thread in a 112 ms `mmap` (Python's allocator asking for a new arena) |
| 1 | 29.8 ms | main thread in an 87 ms `mmap` (Python's arena) |
| 1 | 32.5 ms | autograd thread in `cudaEventCreateWithFlags`, 51 ms |

A small anonymous `mmap` taking 100 ms means something held the process's address-space lock. Nothing traced here holds it. **[not isolated]**

**What this adds to C.3.** Cause 2 is a driver-level slowdown: for a few hundred ms, `cudaEventQuery` becomes ~100× slower, on both TP ranks of a stage at once. It becomes a host stall only because PyTorch 2.14's watchdog holds the work-list mutex while it makes those calls. The driver-level reason stays **[not isolated]**.

**When to use which.**

| Tool | Use it for | Limits |
| --- | --- | --- |
| nsys, capture range | timing a step | no Python objects, low overhead |
| nsys, whole run | rare stalls: every thread, every call, with backtraces | +38 % step time, ~250 MB per rank per 1200 steps |
| PyTorch profiler, `with_stack` | which Megatron line and which process group a kernel belongs to | its own objects must be freed outside the step (C.2) |

The archive is in `outputs/day13_downloads/nsys/`. It holds the six jobs, both smoke runs, the probe's report and rank 3's `.nsys-rep`; open that file in the Nsight Systems GUI and look at 34.5–35.1 s.
