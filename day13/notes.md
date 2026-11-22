# Day 13: Megatron Core at DP=1, TP=2, PP=2 (notes)

> **Status: manual runs verified, 2026-11-21.** Both runs were on Vast.ai container 61775446 (4 × A100-SXM4-80GB). The pre-test columns preserve the earlier run on container 60863550. Prediction notes summarize the guided discussion. [GUIDE](GUIDE.md) · [Reference comparison](../outputs/day13_downloads/manual_20261121T082746Z_ref/session_manual_20261121T082746Z_ref/compare.log) · [Mine comparison](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log)

Labels: **[measured]** your run · **[pre-test]** the assistant's run · **[simulated]** computed without GPUs · **[derived]** worked out from the model or the schedule · **[not isolated]** a cause the data does not separate · **[understanding]** your own reasoning before the answer.

## 1. Predictions

### 1.1 Reasoning from the guided discussion

These entries summarize answers and corrections from the discussion. They are a review record.

| # | Question | Reasoning from the discussion [understanding] | Answer (GUIDE) | Matched? |
| --- | --- | --- | --- | --- |
| P1 | `(tp, dp, pp)` of ranks 0–3 at `DP=1 TP=2 PP=2`; stage-1 ranks and TP-rank-1 ranks at `TP=2 DP=2 PP=2` | TP increments first. With DP=1, the carry from rank 1 goes straight to PP. With DP=2, stage 1 is 4–7, TP index 1 is 1/3/5/7, and DP index 1 is 2/3/6/7. | `(0,0,0) (1,0,0) (0,0,1) (1,0,1)`; stage 1 = 4–7; TP rank 1 = 1, 3, 5, 7 | Yes |
| P2 | What each of the six getters returns on rank 2; why 0↔2 exchange activations | From rank 2, vary one coordinate for TP/PP/DP, fix DP for MP, and use PP endpoints for embeddings. Rank 2 has no position-embedding group. | TP `[2,3]`, PP `[0,2]`, DP `[2]`, MP `[0..3]`, embedding `[0,2]`, position embedding `None`; same TP rank | Yes |
| P3 | Why rank 0 calls `create_group([2,3])` | Worked through the shared creation order after the prompt. All ranks enter the same calls, and each rank retains the groups it belongs to. | `new_group` must be called by every rank in the same order | Understood after explanation |
| P4 | Embedding group with untied embeddings: exists? collectives at init / per step, tied vs untied | The PP layout creates the group regardless of weight sharing. Tied weights need init and gradient synchronization. Untied weights have separate Adam state. | exists; tied: init all-reduce (`language_module.py:308`) + 8 MiB grad all-reduce per step; untied: none (`finalize_model_grads.py:146`) | Yes after distinguishing stored weight [4096,512] from its transpose |
| P5 | Calls per rank per step of `forward_step_func`, `loss_func`, `backward_step`; return of `fb_func`; `next()` on stage 1? | Each rank runs eight forwards and backwards. The last stage runs loss; rank 0 backpropagates the activation gradient from rank 2. The iterator and return values were worked through together. | 8 / 8 on every rank; `loss_func` 8 on ranks 2, 3 only; list of 8 dicts on the last stage, `[]` on stage 0; yes, one per call | Yes |
| P6 | Backward seed scalar; what the log shows with `loss.detach()` | Use mean_k/8 for equal microbatches. detach() shares storage, so the log would be 1.1405. clone() preserves the original value. | `mean_k / 8`; 1.1405 (in-place `/=` at `schedules.py:335`) | Yes after confirming the PyTorch method |
| P7 | TP NCCL kernels per rank per step (F / B) for `tp2pp2` stage 0, stage 1, and `tp2pp1`; P2P kernels per stage | Two forward and two backward reductions per block → 48 each per stage before the head/tail extras. Stage extras are per microbatch. Warmup send + seven batched exchanges + final recv = nine PP kernels. | stage 0: 56 F + 48 B; stage 1: 48 F + 24 small + 56 B; tp2pp1: 104 + 24 + 104; 9 SendRecv per stage | Yes after converting per-microbatch extras to per-step counts |
| P8 | (a) SP on, backward of one block: collectives, order, shapes; (b) counts per rank per step on both stages, P2P shape, loss change | Worked through sequence shards, forward gather/reduce-scatter, backward gradient gather/reduce-scatter plus the saved-input re-gather, and the LayerNorm parameter sync. | all-reduce → reduce-scatter + all-gather; +48–56 backward re-gathers; LN/bias all-reduce 9216 / 10240; P2P `[128,2,512]`; loss unchanged | Yes after correcting the P2P shape and distinguishing dX from parameter gradients |
| P9 | `CUDA_VISIBLE_DEVICES` order on a host with two NVLink pairs joined by `SYS`; what `0,2,1,3` breaks; what it matters on an all-NV12 host | Place GPU 0/1 and 2/3 in the TP groups, leaving GPU 0/2 and 1/3 for PP on the hypothetical split topology. | `0,1,2,3`; TP all-reduces cross `SYS`; nothing | Yes |
| P10 | Thread, stream and kernel of a TP forward, a TP backward and a PP exchange; meaning of a long SendRecv | Asynchronous backward f all-reduces dX while dW can proceed. The first steady SendRecv includes waiting for the peer to produce the earlier activation gradient. | main/compute stream; autograd/NCCL stream (async); main/another NCCL stream, `ncclDevKernel_SendRecv`; waiting for the peer | Yes after interpreting the trace |

### 1.2 Checking the Day 11 and Day 12 predictions for Day 13

| Source | Prediction (summarized from the source note) | Run evidence | Matched? (mine) |
| --- | --- | --- | --- |
| Day 11 notes §7 | per layer per microbatch, 2 forward + 2 backward all-reduces of `[s, b, h]` on every stage, over the stage's TP group; forward from `layers.py:1480`, backward async from `layers.py:637–639`, wait `:749–750` | [measured] Mine: stage 0 56 F / 48 B, stage 1 48 F / 56 B plus 24 small CE reductions. Tensors are [256,2,512] FP32 (1 MiB). The MLP forward stack reaches layers.py:1480 and the backward stack is on the autograd thread, stream 43. [Trace](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/analyze_tp2pp2.log) | Yes for block counts, shape and call path. Use FP32 for this experiment. |
| Day 12 notes §6 | groups TP `[0,1] [2,3]`, PP `[0,2] [1,3]`; 1F1B per PP group; SP off: each stage-0 rank sends a full `[256,2,512]` copy to its partner; SP on: `[128,2,512]` shards (to verify in `get_tensor_shapes`) | [measured] The groups and both stages' 1F1B orders agree. Each PP message changes from [256,2,512] to [128,2,512] with SP, with nine kernels per rank in both runs. [Comparison](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log) | Yes |

## 2. Run record

| Item | Pre-test `pretest3` [pre-test] | Reference rerun [measured] (mine) | My implementation [measured] (mine) |
| --- | --- | --- | --- |
| Run ID | `pretest3` | `manual_20261121T082746Z_ref` | `manual_20261121T082746Z_mine` |
| Time (UTC) | 2026-11-16 02:05:57–02:08:10 (2 min 13 s) | 2026-11-21 08:27:46–08:30:00 (2 min 14 s) | 2026-11-21 08:30:01–08:31:25 (1 min 24 s) |
| Host | container 60863550, 4 × A100-SXM4-80GB, all pairs NV12, NUMA 1 | Container 61775446, 4 × A100-SXM4-80GB; all pairs NV12; NUMA 0 and 1 | Same container and GPU layout |
| Software | torch 2.14.1+cu130, NCCL 2.30.7, Python 3.12.15, driver 580.178.04, Megatron `core_v0.19.2` (`4b4acac9`) | torch 2.14.1+cu130, NCCL 2.30.7, Python 3.12.15, driver 595.91.07, Megatron core_v0.19.2 (4b4acac9) | Same runtime and commit |
| GPU choice | pair `0,1`; 4-GPU order `0,1,2,3` | Pair 0,1; four-GPU order 0,1,2,3 | Pair 0,1; four-GPU order 0,1,2,3 |
| Configuration | `--impl ref --mapper ref`; tp1pp1, tp2pp1, tp1pp2, tp2pp2, tp2pp2_sp | `--impl ref --mapper ref`; tp1pp1, tp2pp1, tp1pp2, tp2pp2, tp2pp2_sp | `--impl mine --mapper mine --quick`; tp1pp1, tp2pp2, tp2pp2_sp |
| Verdict | exit 0; mapper, 5 jobs, compare and output checks PASSED | Exit 0; mapper, 5 jobs, compare and output checks passed | Exit 0; mapper, 3 jobs, compare and output checks passed |
| Archive SHA-256 | `a46101dcb68cf02c0841cc8d1d9d5df2a34ff908504de4d05bb1823ff6b98389` | `85fdbdb7a30678e1c867fcf5835d318cdbdc570494f2b59a38dd8d6a0619e189` | `f3cb5f9e58ef60434184c9796306c6be5b9f28196f2a6a6abe873c81bf9c0f19` |
| Manifest | 87 files verified on Windows | 96 extracted files rehashed and verified locally | 76 extracted files rehashed and verified locally |
| Instance stopped | 02:11 UTC, from inside (`stopping instance 60863550.`) | Stopped after both runs; confirmed in the Vast console (2026-11-21) | Same instance, same stop |

## 3. The process groups

| Kind | Megatron getter | Pre-test ranks 0 / 1 / 2 / 3 [pre-test] | Mine mapper agrees? (mine) | One sentence: why this group (mine) |
| --- | --- | --- | --- | --- |
| TP | `get_tensor_model_parallel_group` | `[0,1] [0,1] [2,3] [2,3]` | Yes [measured] | Fix DP and PP. These ranks hold different slices of the same stage layers. |
| PP | `get_pipeline_model_parallel_group` | `[0,2] [1,3] [0,2] [1,3]` | Yes [measured] | Fix TP and DP. These ranks send activations and activation gradients between stages. |
| DP | `get_data_parallel_group` | `[0] [1] [2] [3]` | Yes [measured] | Fix TP and PP. This group would synchronize replicas of the same shard. DP=1 gives singleton groups. |
| MP | `get_model_parallel_group` | `[0,1,2,3]` on every rank | Yes [measured] | Fix DP. Include every stage and every TP shard of one model replica. |
| embedding | `get_embedding_group` | `[0,2] [1,3] [0,2] [1,3]` | Yes [measured] | Pair the first and last ranks of each PP group for a shared embedding shard when weights are tied. |
| position embedding | `get_position_embedding_group` | `[0] [1] None None` | Yes [measured] | Use only the PP first rank, where learned position embeddings live. |

Mapper vs `RankGenerator`, 9 layouts (5 beyond 4 ranks are **[simulated]**): pre-test PASSED. Mine PASSED, including the 9-layout check against the closed form and RankGenerator. [Mapper](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/mapper.log) · [Actual four-GPU groups](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/run_tp2pp2.log).

The input embedding and output head are independent parameters with their own gradients and Adam state. The embedding group exists, and the tied-weight initialization and gradient collectives are skipped in this configuration.

## 4. The training step, in my words

| Piece | Where in `train_gpt.py` | Production equivalent | What it decides (mine) |
| --- | --- | --- | --- |
| initialize distributed | `initialize_distributed` | `megatron/training/initialize.py` via `pretrain` | Binds the local CUDA device and creates the groups for this rank. |
| model provider | `model_provider` | `pretrain_gpt.py` model provider / `training.py:1706 get_model` | Builds three decoder layers per stage in the four-GPU job, with the embedding on stage 0 and the output head and loss on stage 1. |
| data iterator | `make_global_batch` → `iter` | `train_valid_test_datasets_provider` `pretrain_gpt.py:444`, `get_batch` `:97` | Provides one tokens/positions/labels dictionary per forward. Stage 1 advances its iterator to align labels with the activation it received. |
| `forward_step_func` + `loss_func` | `my_step.py` | `pretrain_gpt.py:282`, `:209` | Calls GPT with the next microbatch on the current CUDA device. The last stage reduces per-token cross-entropy to a mean and returns a separate copy for logging. |
| schedule + optimizer | `get_forward_backward_func()` → Adam | `training.py:2407`, `:2474` | Schedules eight forwards and backwards per rank, including PP exchanges. After accumulation and any SP parameter sync, Adam updates the local parameters once. |

Megatron divides the microbatch mean by 8 in `pipeline_parallel/schedules.py:335` before backward. The means then contribute equally to the whole-step gradient. The in-place division would also change a log tensor made with detach() alone; my_step.py uses detach().clone().

## 5. Annotated trace: one TP collective and one PP exchange

| Field | TP collective [pre-test] | TP collective (mine) | PP exchange [pre-test] | PP exchange (mine) |
| --- | --- | --- | --- | --- |
| Rank | 0 | 0 | 0 | 0 |
| Kernel | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | `ncclDevKernel_SendRecv` | `ncclDevKernel_SendRecv` |
| Stream / thread | 7 (compute) / main | 7 (compute) / main | 47 (NCCL) / main | 47 (NCCL) / main |
| Phase | F mb=0 | F mb=0 | between F1 and B0 (first steady exchange) | Between F1 and B0 (first steady exchange) |
| Group, ranks, size | `TENSOR_MODEL_PARALLEL_GROUP` `[0,1]`, 2 | TP [0,1], size 2 | `PIPELINE_MODEL_PARALLEL_GROUP` `[0,2]`, 2 | PP [0,2], size 2 |
| Elements, shape, dtype | 262144, `[256,2,512]`, FP32 | 262144 FP32, [256,2,512] | 262144 each way, `[256,2,512]` ×2, peers 2/2 | 262144 FP32 each way, [256,2,512], peer 2 |
| Duration | 830 µs (stack step, profiler-inflated) | 49.9 µs (stack step; includes waiting for rank 1) | 33.5 ms (mostly waiting for rank 2) | 27.218 ms (stack step, includes peer readiness and transfer) |
| Megatron file:line | `mappings.py:22 _reduce` (all_reduce ~`:35`) ← `:229` ← `:498` ← `layers.py:1430` (call ~`:1480`) ← `mlp.py:258` ← `transformer_layer.py:816` | mappings.py:22 _reduce (call ~35) ← layers.py:1430 (call ~1480) ← mlp.py:258 | `p2p_communication.py:17` (`batch_isend_irecv` ~`:49`) ← `:275 _communicate` ← `:526 send_forward_recv_backward` ← `schedules.py:2120` (call ~`:2382`) | p2p_communication.py:17 _batched_p2p_ops (call ~49) ← :275 ← :526 ← schedules.py:2120 (call ~2382) |
| Why it is there (one sentence) | MLP linear_fc2 sums both TP ranks' partial outputs for microbatch 0. | MLP linear_fc2 sums both TP ranks' partial outputs for microbatch 0. | Rank 0 sends microbatch 1's activation and receives microbatch 0's boundary gradient from rank 2 so that it can start B0. | Rank 0 sends microbatch 1's activation and receives microbatch 0's boundary gradient from rank 2 so that it can start B0. |

Backward TP all-reduce for comparison [pre-test]: rank 0, stream 43, autograd thread, B mb=0, 1.42 ms, `layers.py:581 backward` (layer 2 `linear_fc1` input gradient, by order). Mine: rank 0, stream 43, autograd worker, B mb=0, 49.6 µs, layers.py:581, TP [0,1]. Frame lines identify function entries; call-site lines are resolved from source.

A collective kernel's duration is measured from when this rank's kernel starts until every member has arrived and the data has moved. So it includes waiting for the partner as well as the transfer. The pre-test's 830 µs and my 49.9 µs are for the same 1 MiB all-reduce in the same position (F mb=0), so the difference is mostly partner skew, not link speed. The same holds for the 27.2 ms and 33.5 ms SendRecv. None of these durations is a bandwidth measurement **[not isolated]**. [Manual trace examples](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/analyze_tp2pp2.log).

## 6. 1F1B order and idle time

| Rank | Order [pre-test] | Order (mine) |
| --- | --- | --- |
| 0, 1 (stage 0) | `F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7` | `F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7` |
| 2, 3 (stage 1) | `F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7` | `F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7` |

| Idle feature | Idealized [derived] | Stack step, tp2pp2 [pre-test] | Mine |
| --- | --- | --- | --- |
| Bubble fraction | `(P−1)/(M+P−1)` = 1/9 ≈ 11 % | not measurable: compute ≈ 10 % of the window, host-bound | [not isolated] Stack-step compute kernels take 24.45 of 270.06 ms on rank 0 (9.1%). Its 228.11 ms idle includes several kinds of gaps. |
| Fill (stage 1 waits for a₀) | 1 forward | stage 1's first F0 kernel at 14.2 ms | Rank 2 first F0 kernel at 9.32 ms in its trace window; 8.02 ms initial P2P overlap. |
| First round trip (stage 0 waits for g₀) | 2 units after F1 | 39.0 ms idle after F1, 33.5 ms overlapping the SendRecv | Rank 0 idle 32.39 ms after F1; 27.22 ms overlaps first steady SendRecv. |
| Drain (stage 0 waits for g₇) | before B7 | 11.1 ms idle after B6 | Rank 0 idle 11.19 ms after B6; 10.28 ms overlaps P2P. |
| Stage imbalance | 0 (equal stages) | stage 1 has about 45 % more compute (output layer + CE); `tp1pp2` 47.3 vs 32.5 ms | Reference PP-only timing trace: 47.32 ms compute on stage 1 vs 37.77 ms on stage 0 (about 25% more). |
| Host stalls | — | 40–50 ms inside F1 (stage 0) / F6 (stage 1) in the timing step; cause found later (GUIDE Appendix C) | Timing step: rank 0 has a 39.70 ms gap with zero P2P overlap inside F1 (span 8.70–55.03 ms); rank 2 has a 53.22 ms gap with zero overlap inside F5 (span 148.11–208.7 ms). The TP partners show the same gaps (rank 1: 38.43 ms; rank 3: 47.02 ms). The cause was isolated later: the cyclic GC freeing the previous step's profiler, now fixed in `train_gpt.py` ([GUIDE Appendix C](GUIDE.md#appendix-c-the-host-stall-found-and-fixed-assistant-run-2026-11-22-measured), assistant run 2026-11-22). Their effect on the other stage is traced below. |

Timing step breakdown, tp2pp2 [pre-test], ms (window / compute / NCCL on compute stream / waiting for P2P / other-stream NCCL / other): r0 273.4 / 23.0 / 3.4 / 67.2 / 14.3 / 165.6; r1 274.6 / 23.4 / 34.9 / 65.5 / 34.0 / 116.8; r2 258.7 / 32.7 / 15.5 / 37.7 / 23.8 / 148.9; r3 259.1 / 27.6 / 26.3 / 36.3 / 62.5 / 106.4.

Mine [measured], ms (window / compute / NCCL on compute stream / P2P overlap / other-stream NCCL overlap / other): r0 284.16 / 24.33 / 16.57 / 72.40 / 36.36 / 134.51; r1 285.60 / 23.05 / 20.78 / 74.18 / 18.29 / 149.31; r2 270.33 / 27.53 / 20.12 / 40.85 / 30.53 / 151.31; r3 269.60 / 27.18 / 21.64 / 39.05 / 38.00 / 143.72. These buckets use temporal overlap. Check the data dependencies before attributing an idle span to communication.

**A host stall on one stage appears as P2P waiting on the other stage [measured; the stall's cause is in GUIDE Appendix C].** The timing step shows this in both directions:

| Stall (no P2P overlap) | Data the other stage needs next | Resulting wait on the other stage |
| --- | --- | --- |
| Stage 0, inside F1: rank 0 39.70 ms, rank 1 38.43 ms | activation a₁, sent only after F1 ends | rank 2 idle 29.94 ms after B0 (29.56 ms overlapping P2P); rank 3 29.23 ms (28.49 ms) |
| Stage 1, inside F5: rank 2 53.22 ms, rank 3 47.02 ms | gradient g₅, sent only after B5, which follows F5 | rank 0 idle 53.96 ms after F6 (53.45 ms overlapping P2P); rank 1 52.86 ms (52.41 ms) |

So most of the "P2P overlap" bucket here is waiting caused by the other stage's host stall. It is not the 1F1B bubble and not transfer time. That is why the bubble cannot be read off as the P2P wait in this run. Both TP ranks of a stage stall together, because the synchronous forward all-reduce on the compute stream ties each rank to its partner. [Manual timing trace](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/analyze_tp2pp2.log) · [PP-only reference](../outputs/day13_downloads/manual_20261121T082746Z_ref/session_manual_20261121T082746Z_ref/analyze_tp1pp2.log).

## 7. Sequence parallelism: on vs off

The main trace uses SP off so that its two TP all-reduce paths map directly to Day 11's f and g. The SP run tests the activation-memory and communication trade-off.

| Per rank per step | Stage 0 off [pre-test] | Stage 0 on [pre-test] | Stage 1 off [pre-test] | Stage 1 on [pre-test] | Mine [measured], stage 0 / stage 1 unless stated |
| --- | ---: | ---: | ---: | ---: | --- |
| F all-reduce 262144 | 56 | 8 | 48 | 0 | Off 56 / 48 · on 8 / 0 |
| F reduce-scatter (out 131072) | — | 48 | — | 48 | On 48 / 48 |
| F all-gather (out 262144) | — | 48 | — | 56 | On 48 / 56 |
| F all-reduce 512 (CE) | — | — | 24 | 24 | Off and on 0 / 24 |
| B all-reduce 262144 | 48 | 0 | 56 | 0 | Off 48 / 56 · on 0 / 0 |
| B reduce-scatter | — | 48 | — | 56 | On 48 / 56 |
| B all-gather | — | 104 | — | 104 | On 104 / 104 |
| LN/bias grad all-reduce | — | 1 × 9216 | — | 1 × 10240 | On 1 × 9216 / 1 × 10240 |
| P2P shape | `[256,2,512]` | `[128,2,512]` | `[256,2,512]` | `[128,2,512]` | Off [256,2,512] · on [128,2,512]; nine kernels per rank in both |
| Loss steps 0–4 | bitwise equal to 1 GPU | bitwise equal | — | — | Off and on exactly match single GPU and ref: 9.1240034 → 8.9108639 → 8.7221317 → 8.5237522 → 8.3576717 |
| Worst step-0 grad-norm rel diff vs 1 GPU | 1.76e-7 | 1.80e-7 | — | — | Off 1.76e-7 · on 1.80e-7 (77 parameters) |
| CUDA peak MiB rank 0 | 209 | 197 | — | — | Off 209.29 · on 196.76 (12.52 MiB less, computed before rounding) |
| Unprofiled step ms (steps 1, 4) | 141.4, 158.3 | 175.0, 183.4 | — | — | Off 166.1, 160.9 · on 171.9, 173.0 |

The row-parallel reverse gathers output gradients. The column-parallel backward also gathers saved input activations for the local weight gradient (48 such re-gathers on stage 0, 56 on stage 1). Stage 0 additionally gathers the embedding output gradient once per microbatch so that a token's gradient reaches its vocabulary-row owner. The manual SP counts (including 104 total backward gathers per rank on either stage) were compared with the §4 derivation; compare.log shows the automatic count check is skipped for SP. [Manual comparison](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/compare.log).

## 8. Topology

| Item | Pre-test [pre-test] | Mine |
| --- | --- | --- |
| `topo -m` links | NV12 between all 4 GPUs | All six GPU pairs NV12. CUDA P2P access reported available for every pair. |
| NVLink per link | 12 × 25 GB/s | 12 links per GPU; nvidia-smi reports 25 GB/s per link. |
| NUMA / CPU affinity | node 1, 64–127, 192–255 | GPU 0/1: NUMA 0, CPUs 0–63. GPU 2/3: NUMA 1, CPUs 64–127. |
| Fastest pair / 4-GPU order | `0,1` / `0,1,2,3` (all equal) | 0,1 / 0,1,2,3. Both TP and PP pairs use NV12. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 1 (required by `train_gpt.py`) | 1 in both run environments |

TP reductions lie inside every block and are frequent. PP exchanges occur at stage boundaries. On this host all six links use NV12, so the topology check kept the natural order. [Topology](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/topology.log) · [Environment](../outputs/day13_downloads/manual_20261121T082746Z_mine/session_manual_20261121T082746Z_mine/environment.log).

The column-parallel backward submits the async dX reduction before the independent dW GEMM, then waits. The launch-order setting helps the earlier communication start in time to overlap the GEMM (layers.py:640). The manual run used one setting.

## 9. My code ↔ Megatron Core (`core_v0.19.2`, commit `4b4acac9`)

| My code | Megatron Core | Location | Same / different (my words) |
| --- | --- | --- | --- |
| `rank_mapper.my_coords` | `generate_masked_orthogonal_rank_groups` | `parallel_state.py:269` | My mapper decodes rank with division and modulo; Core uses the same mixed-radix coordinates and strides for group enumeration. |
| `rank_mapper.my_groups` | `RankGenerator.get_ranks` | `parallel_state.py:534` | My buckets key on fixed coordinates. Core uses the mask to enumerate coordinates that may vary. |
| embedding ranks | `default_embedding_ranks` | `parallel_state.py:560` | Both take the first and last PP rank, with a single rank for PP=1. |
| `my_step.loss_func` | `pretrain_gpt.loss_func`; `forward_step_calc_loss` | `pretrain_gpt.py:209`; `schedules.py:290`, `/=` `:335` | I average the token losses and copy the logging value. The schedule applies the microbatch scaling. |
| `my_step.forward_step_func` | `pretrain_gpt.forward_step` | `pretrain_gpt.py:282` | Both adapt batch data and call the model; the exercise uses a synthetic iterator and returns a loss callback. |
| Day 11 column-parallel `f` | `LinearWithGradAccumulationAndAsyncCommunication.backward` | `tensor_parallel/layers.py:583`, all-reduce `:639`, wait `:750` | The backward sums partial input gradients. Core launches asynchronously, computes dW, then waits for dX. |
| Day 11 row-parallel `g` | `RowParallelLinear.forward` → `reduce_from_tensor_model_parallel_region` | `layers.py:1430`, `:1480`; `mappings.py:35` | Both sum the forward partial outputs without a matching backward reduction in the SP-off path. |
| Day 12 `run_1f1b` | `forward_backward_pipelining_without_interleaving` | `schedules.py:2120` | The same warmup, steady and drain pattern is implemented with one PP lane per TP coordinate. |
| Day 12 `send_forward_recv_backward` | `P2PCommunicator.send_forward_recv_backward` | `p2p_communication.py:527` → `:275` → `:49` | Core uses batched P2P for the newer activation and older boundary gradient in the same exchange. |
| Day 12 `requires_grad_()` on received input | `set_input_tensor` | `schedules.py:501` → `transformer_block.py:443` | The schedule receives the boundary activation, passes it via set_input_tensor and uses its gradient to continue backward upstream. |
| SP LayerNorm grad all-reduce (`allreduce_sequence_parallel_grads`) | `_allreduce_non_tensor_model_parallel_grads` | `distributed/finalize_model_grads.py:416`, condition `:452` | The exercise flattens the marked parameter gradients and sums them over TP once before Adam; Core handles this in gradient finalization. |

## 10. Explanations (mine)

1. The five losses agree exactly across layouts. Step-0 full-parameter gradient norms show TP rounding at about 1.8e-7 relative, while PP alone matches all 77 norms exactly. The shared ref/mine layouts also have identical per-rank recorded gradient sums of squares.
2. On the mine run, unprofiled steps 1 and 4 were 162.0 / 178.6 ms on one GPU and 166.1 / 160.9 ms on four GPUs with SP off. The result varies by step. Small GPU workloads, host overhead and communication limit what these two samples can establish about speedup.
3. The first steady SendRecv took 27.218 ms in mine and 33.5 ms in the pre-test. It exchanges activation 1 for gradient 0. The interval includes readiness of the peer as well as transport, so it does not estimate link bandwidth.
4. To expose the pipeline bubble, use larger per-stage compute with balanced stages, hold the number of microbatches fixed, and repeat the timing trace without Python stacks. Attribute only dependency-backed gaps to pipeline idle.

## 11. Not covered today

- The pipeline bubble as a measured fraction of the step (host-bound at this size) **[not isolated]**.
- The effect of `CUDA_DEVICE_MAX_CONNECTIONS` (no A/B run).
- GPU order effects with heterogeneous GPU links (both hosts reported all-pair NV12).
- Tied embeddings, BF16, DP > 1, the interleaved schedule, the distributed optimizer, NCCL bandwidth.
