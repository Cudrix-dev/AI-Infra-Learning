# Day 14: a top-1 MoE layer with expert parallelism, EP=4 (notes)

> **GPU reruns completed on 2026-11-27.** Both full sessions passed. The four-GPU `mine` run matched the one-GPU reference in every tested routing case, with a worst relative error of 6.13e-7. Both archives and both terminal logs are available locally. [GUIDE rerun summary](GUIDE.md#appendix-c-gpu-reruns-2026-11-27)

**Labels.** [measured] comes from the archived runs. [derived] is arithmetic from the stated assumptions or data. [simulated] covers the EP=8 count model. [understanding] records reasoning from the walkthrough, including corrections made after discussion. [not isolated] marks questions these runs leave open.

## 1. Predictions and corrections [understanding]

| Question | Reasoning and result |
| --- | --- |
| P1 · routing weight | `idx=[0,2,0]`, `w=[.6103,.8700,.25]`. The first token's weight gradient across logits is `[.2378,−.1370,−.0504,−.0504]`. A competing logit affects the selected probability through the denominator. The gradients sum to zero because shifting all logits equally leaves softmax unchanged. Selecting one logit and then taking softmax gives constant weight 1 and zero task gradient through the gate. |
| P2 · capacity | At cf 1 / 1.25 / 2, capacity is 4 / 5 / 8, drop indices are `[7,10,13,14]` / `[10,13]` / none, and padding gives 16 / 20 / 32 computed rows. At cf 1.25, only 14 of the 20 rows are real. When sources share a fixed total budget, pooling their slots can retain tokens that local quotas would discard. |
| P3 · permutation | `order=[1,3,5,0,4,2,6,7]` and `counts=[3,0,2,3]`. `order[k]` is the original row of sorted row `k`. Write back with `y[order[k]] = y_sorted[k]`. Applying the same gather twice can pass for identity or pairwise swaps, so include a cycle of length ≥3 in the test. |
| P4 · dispatch | Rank 1 sends `[86,42,0,0]` and receives `[0,42,32,0]` (74 rows). Rank 2 receives zero and still sends 128. Combine swaps the split lists and backward reverses each payload exchange. Counts are exchanged once, for 3 forward and 2 backward all-to-alls per layer. |
| P5 · memory | EP=4 balanced versus all-to-one gives 2 vs 8 MiB for receive and 8 vs 32 MiB for the expert intermediate. EP=8 yields 1 and 4 MiB when balanced, respectively. At fixed global batch, the hot rank's all-to-one peak stays 8 and 32 MiB. |
| P6 · gradient groups | EP=4 expert gradients already contain contributions from all four sources, so divide by DP=4. With EP=2, sum only matching expert replicas over EDP, still dividing by 4. Mixing different experts corrupts direction. Omitting the division gives 4× on nonzero gradients. |
| P7 · skew | Large balanced/all-to-one expert work is about 12/48 ms [derived]. Remote dispatch is 48 MiB divided by 200 GB/s, about 0.25 ms. The idle ranks wait in forward combine and dispatch backward. Expect a large ratio of roughly 3–4× and a smaller medium ratio. |
| P8 · auxiliary loss | With counts `[7,5,3,1]` and `P=[.4,.3,.2,.1]`, `L_aux/α=1.25` and the derivative over `P` is `[1.75,1.25,.75,.25]`. Moving 0.01 from expert 0 to 3 with the counts fixed lowers the normalized loss by 0.015. Uniform `P` yields loss 1 even when hard counts are uneven. One-hot `f=P` gives 4. AutoScaler yields the same gradient as adding the term explicitly in the local test. |
| P9–P12 · optional | P9: local minus global is a sum of cross-rank covariances. Identical per-rank counts are sufficient for equality. P10: at world 8, EP groups are `[0,1,2,3]` and `[4,5,6,7]`, with EDP pairs `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]` (TP=PP=1). P11: grouped GEMM batches expert projections, and finer-grained routing can increase dispatched rows. P12: stable ordering helps reproducibility; different reduction orders can yield small floating-point differences. |

## 2. Run record [measured]

| Item | Reference session | Mine session |
| --- | --- | --- |
| Run ID | `day14_20261127T023509Z_ref` | `day14_20261127T023509Z_mine` |
| UTC time on 2026-11-27 | 02:35:09–02:37:47 (2 min 38 s) | 02:37:48–02:40:28 (2 min 40 s) |
| Command | `run_server_once.sh <ref-run-id> --impl ref` (full) | `run_server_once.sh <mine-run-id> --impl mine` (full) |
| Exit and markers | 0, `CORE_SUCCESS.txt` and `SUCCESS.txt` present | 0, `CORE_SUCCESS.txt` and `SUCCESS.txt` present |
| Archive SHA-256 prefix/suffix and manifest | `4af6e6fd…4c5b042e` verified, 65/65 | `d2ddfdda…25096363` verified, 68/68 |
| Terminal and core logs | [Terminal log](../outputs/day14_downloads/day14_20261127T023509Z_ref/day14_20261127T023509Z_ref.log) · [Core verification](../outputs/day14_downloads/day14_20261127T023509Z_ref/session_day14_20261127T023509Z_ref/verify_core.log) | [Terminal log](../outputs/day14_downloads/day14_20261127T023509Z_mine/day14_20261127T023509Z_mine.log) · [Core verification](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/verify_core.log) |

Both sessions used container `62787249` with 4 × NVIDIA A100-SXM4-40GB and driver 595.71.05. All GPU pairs had NV12 and P2P access. GPU 0–1 were on NUMA 0, GPU 2–3 on NUMA 1. The environment was `/venv/main`, Python 3.12.16, PyTorch 2.14.1+cu130, CUDA build 13.0 and NCCL 2.30.7. The server had no Megatron checkout at the configured path, so the symbols stage was skipped. The downloaded artifacts do not record the server's stop status.

In both sessions, optional stage exits were EP=2 parity **0**, local aux **1** (expected), aux training **0** and traces **0**. The runner uses `mine` for local checks, an additional baseline, EP=4 dispatch and EP=4 parity in the mine session. Its route statistics, negative controls, timing and optional stages use `ref`.

[Environment and topology](../outputs/day14_downloads/day14_20261127T023509Z_ref/session_day14_20261127T023509Z_ref/environment.log) · [Mine stage commands](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/commands.txt) · [Optional stage statuses](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/optional_status.txt)

## 3. Experiment 1: routing and capacity on one GPU

Both sessions reported the same layer-0 results for the medium preset (4096 tokens). Capacity and buffer sizes come from routing counts. The receive buffers here are calculated from the counts and the FP32 row width.

| Routing | Tokens per expert | Max/mean | CV | Load entropy | Idle | Router entropy | Largest EP=4 receive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `learned` | `[921, 909, 1008, 1258]` | 1.229 | 0.137 | 0.993 | 0 | 0.810 | 1258 rows / 2.46 MiB |
| `learned+bias` | `[2900, 330, 410, 456]` | 2.832 | 1.059 | 0.665 | 0 | 0.719 | 2900 rows / 5.66 MiB |
| `balanced` | `[1024, 1024, 1024, 1024]` | 1.000 | 0.000 | 1.000 | 0 | — | 1024 rows / 2.00 MiB |
| `hot` | `[2867, 412, 409, 408]` | 2.800 | 1.039 | 0.678 | 0 | — | 2867 rows / 5.60 MiB |
| `ragged` | `[2219, 597, 0, 1280]` | 2.167 | 0.806 | 0.704 | 1 | — | 2219 rows / 4.33 MiB |
| `all_to_one` | `[0, 0, 0, 4096]` | 4.000 | 1.732 | 0.000 | 3 | — | 4096 rows / 8.00 MiB |

| Routing | cf | C global / per source | Dropped global / per source | Same dropped set? | Global padded / real kept |
| --- | --- | --- | --- | --- | --- |
| `learned` | 1 | 1024 / 256 | 234 (5.7%) / 255 (6.2%) | no | 4096 / 3862 |
| `learned` | 1.25 | 1280 / 320 | 0 (0.0%) / 6 (0.1%) | no | 5120 / 4096 |
| `learned` | 2 | 2048 / 512 | 0 (0.0%) / 0 (0.0%) | yes | 8192 / 4096 |
| `learned+bias` | 1 | 1024 / 256 | 1876 (45.8%) / 1876 (45.8%) | no | 4096 / 2220 |
| `learned+bias` | 1.25 | 1280 / 320 | 1620 (39.6%) / 1620 (39.6%) | no | 5120 / 2476 |
| `learned+bias` | 2 | 2048 / 512 | 852 (20.8%) / 852 (20.8%) | no | 8192 / 3244 |
| `balanced` | 1 | 1024 / 256 | 0 (0.0%) / 0 (0.0%) | yes | 4096 / 4096 |
| `balanced` | 1.25 | 1280 / 320 | 0 (0.0%) / 0 (0.0%) | yes | 5120 / 4096 |
| `balanced` | 2 | 2048 / 512 | 0 (0.0%) / 0 (0.0%) | yes | 8192 / 4096 |
| `hot` | 1 | 1024 / 256 | 1843 (45.0%) / 1843 (45.0%) | no | 4096 / 2253 |
| `hot` | 1.25 | 1280 / 320 | 1587 (38.7%) / 1587 (38.7%) | no | 5120 / 2509 |
| `hot` | 2 | 2048 / 512 | 819 (20.0%) / 819 (20.0%) | no | 8192 / 3277 |
| `ragged` | 1 | 1024 / 256 | 1451 (35.4%) / 2304 (56.2%) | no | 4096 / 2645 |
| `ragged` | 1.25 | 1280 / 320 | 939 (22.9%) / 1920 (46.9%) | no | 5120 / 3157 |
| `ragged` | 2 | 2048 / 512 | 171 (4.2%) / 939 (22.9%) | no | 8192 / 3925 |
| `all_to_one` | 1 | 1024 / 256 | 3072 (75.0%) / 3072 (75.0%) | no | 4096 / 1024 |
| `all_to_one` | 1.25 | 1280 / 320 | 2816 (68.8%) / 2816 (68.8%) | no | 5120 / 1280 |
| `all_to_one` | 2 | 2048 / 512 | 2048 (50.0%) / 2048 (50.0%) | no | 8192 / 2048 |

Adding +1.5 to expert 0's router bias moved 1,979 additional tokens to it, raising the max/mean from 1.229 to 2.832. At cf 1.0, learned routing dropped **21 additional tokens** under per-source quotas. For ragged routing the excess was **853 tokens**. On learned routing, position and probs each dropped 234, with only 43 in common. Even when two capacity schemes drop the same number, the affected tokens can differ.

At EP=8 [simulated], the largest receive had 512 rows for balanced, 2867 for hot and 4096 for all-to-one, or 1.0×, 5.6× and 8.0× the balanced share.

[Routing and capacity log](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/route_stats.log)

## 4. Experiment 2: EP=4 parity and negative controls

The ragged test forces each source to send 128 token rows while the experts receive `[278,74,0,160]`. The figures follow the same traffic measured in both reruns.

![Forward pass of one top-1 MoE layer at EP=4 with ragged routing, small preset. Four source ranks on the left each hold 128 tokens and send them, grouped by expert, through the dispatch all_to_all (input splits = the rank's row of the send matrix, output splits = its column). Band widths are proportional to rows: rank 0 sends 128 rows to expert 0; rank 1 sends 86 to expert 0 and 42 to expert 1; rank 2 sends 32 to expert 1 and 96 to expert 3; rank 3 sends 64 to expert 0 and 64 to expert 3. In the middle, expert 0 on rank 0 runs 278 rows, expert 1 runs 74, expert 2 receives 0 rows and runs on an empty [0, d] tensor, and expert 3 runs 160. The combine all_to_all, with the splits swapped, returns each rank's 128 rows on the right, where they are unpermuted, scaled by the combine weight w and added to the residual h. A footer notes that one all_to_all of the counts plus one .tolist() gives every rank its row and column before the dispatch.](assets/d14_ep_forward.svg)

![Backward pass of the same layer, flowing right to left over the same bands. Each source rank starts from dy, multiplies by w and permutes. The combine backward all_to_all uses input splits = row and output splits = column. Each expert computes the gradients of its input rows, and its weight gradient dW_j sums over all its rows from every source: 278 rows from ranks 0, 1 and 3 for expert 0, 74 from ranks 1 and 2 for expert 1, exactly zero (not None) for expert 2, and 160 from ranks 2 and 3 for expert 3. The dispatch backward all_to_all (input splits = column, output splits = row) returns the input-row gradients, which are scattered back to token order, together with the router gradient through w and the aux loss. No count exchange and no host sync happen in backward. A bottom panel shows finish_grads. Dense parameters are all-reduced with SUM over DP [0,1,2,3] and then divided by 4, which makes them identical on all ranks; forgetting the division would make them 4 times too large (derived; this is not one of the controls). Expert parameters are divided by DP with no all-reduce, because the bands already brought every source's rows to the expert's rank. At EP=2 × EDP=2 they are divided by 4 and then all-reduced over the EDP group [j, j+2]. The controls expert_sum_over_ep (mixes different experts, ratio about 0.25) and expert_no_dp_scale (ratio 4.0000) break this rule.](assets/d14_ep_backward.svg)

Both figures are generated by [`assets/make_figures.py`](assets/make_figures.py), the same script that draws the GUIDE's Block 1 flowchart.

### 4.1 One-GPU baselines

| Pattern | CE [measured] | Min top-1 score margin |
| --- | --- | --- |
| `learned` | 6.7032547 | 4.56e-04 |
| `balanced` | 6.7257767 | 4.32e-04 (forced route) |
| `hot` | 6.7142153 | 9.99e-05 (forced route) |
| `ragged` | 6.7276926 | 2.14e-04 (forced route) |
| `all_to_one` | 6.7239490 | 5.26e-04 (forced route) |
| `presorted` | 6.7331486 | 1.16e-05 (forced route) |

All three baselines reported the same **tensor digest** `8fcdb6b6d9b532f3` (shown here as the log prefix). The four-GPU mine run used the reference baseline from its own session. Forced expert assignments ignore the score margin.

[Reference baseline](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/baseline.log) · [Mine baseline](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/baseline_mine.log)

### 4.2 Four-GPU NCCL dispatch and parity

Both runs passed all **8 dispatch checks** across ragged and all-to-one: count exchange, labelled rows in forward and backward, and zero-size splits. Routing and counts were exact, logits were bitwise equal in every case, and the relative error tolerance for FP32 was 1e-5.

| Case | Ref worst relative error | Mine worst relative error | Verdicts |
| --- | --- | --- | --- |
| `learned` | `grad.router.L1` 4.33e-07 · 3/18 bitwise | `grad.router.L1` 4.33e-07 · 3/18 bitwise | PASS / PASS |
| `balanced` | `dx` 4.44e-07 · 2/18 bitwise | `dx` 4.44e-07 · 2/18 bitwise | PASS / PASS |
| `ragged` | `grad.router.L1` 4.86e-07 · 4/18 bitwise | `grad.router.L1` 4.86e-07 · 4/18 bitwise | PASS / PASS |
| `all_to_one` | `grad.router.L0` 5.93e-07 · 8/18 bitwise | `grad.router.L0` 6.13e-07 · 8/18 bitwise | PASS / PASS |

Per rank per step for the two-layer parity model, both runs recorded **6 forward all-to-alls, 4 backward all-to-alls, and 2 forward all-reduces**. The finish phase logged **1** dense all-reduce for the reference bucket and **26** for the per-parameter mine implementation. The finish count is diagnostic. These parity results leave the performance impact of per-parameter reduction unmeasured.

For layer 0 in ragged routing, the send matrix was `[[128,0,0,0],[86,42,0,0],[0,32,0,96],[64,0,0,64]]` in both runs. Rank 2 still sent 32 rows to rank 1 and 96 to rank 3. Its expert received zero rows, and both layer-0 and layer-1 expert-2 gradients were exact zeros.

[Ref parity JSON](../outputs/day14_downloads/day14_20261127T023509Z_ref/session_day14_20261127T023509Z_ref/ep4_parity.json) · [Mine parity JSON](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/ep4_parity.json) · [Mine dispatch log](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/ep4_dispatch.log)

### 4.3 Negative controls [measured with ref]

| Control | Observed in both sessions | Explanation |
| --- | --- | --- |
| `wrong_inverse`, ragged | 14 checks failed: logits, CE, dx, dense, both routers, layer-1 aux and z losses, and experts 0, 1 and 3 in both layers | Misplaced expert outputs corrupt token alignment. Layer 0 routing happened before the bad writeback. |
| `wrong_inverse@presorted` | Zero failures | Identity order makes gathering twice look correct. |
| `expert_sum_over_ep` | All 8 expert gradient checks failed. Nonzero reference gradients had ratios 0.250–0.269. | The sum mixes different parameters. It can even introduce nonzero entries for the idle expert. |
| `expert_no_dp_scale` | Six checks failed (experts 0, 1 and 3 in both layers), ratio 4.0000 and rel 3.000 | Scaling a zero expert gradient by four leaves it zero. |
| `recv_equals_send` | `SplitMetadataError` on all four ranks before the token collective | Rank 0 claims to send 0 to rank 1, which incorrectly expects 86. Peer-by-peer agreement matters even when total sizes match. |

### 4.4 Optional parity, scope and auxiliary training [measured with ref]

- **EP=2 × EDP=2:** all four cases passed in both sessions, worst rel **6.72e-7** (`grad.router.L0`, all-to-one).
- **Local aux vs global baseline:** learned / balanced / ragged gave **FAIL / PASS / FAIL**, as expected. Learned aux ratios were **1.0051 / 1.0243**, ragged **1.0196 / 1.0363** (layers 0 / 1). Expert gradient mismatches were confined to layer 0 because a deeper auxiliary loss backpropagates through its input into earlier layers.
- **Aux training:** 200 steps, one seed, expert-0 initial bias +2. Initial layer-0 max/mean was 3.461 for all settings. Both sessions produced the same endpoints.

| Aux coefficient | Max/mean at step 100 | Max/mean at step 200 | CE at step 200 |
| --- | --- | --- | --- |
| 0.1 | 1.664 | 1.641 | 0.6040 |
| 0.01 | 2.930 | 2.953 | 0.6508 |
| 0.0 | 3.273 | 3.328 | 0.6644 |

[Controls](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/ep4_controls.log) · [EP=2 parity](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/ep2_parity.log) · [Local auxiliary loss](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/aux_local.log) · [Auxiliary training](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/aux_train.json)

## 5. Experiment 3: skew at EP=4

Both sessions timed the **reference implementation** after 5 warmups over 20 steps. A job step is the maximum rank time in that step. The table reports the median and range of those maxima. Ratios compare against the same session's balanced median.

### 5.1 Job step [measured, ms]

| Shape | Routing | Peak recv rows | Ref EP=1 | Ref EP=4 median (range) | Ref ratio | Mine session EP=1 | Mine session EP=4 median (range) | Mine session ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medium | balanced | 1024 | 5.781 | 4.097 (3.026–4.256) | 1.00× | 5.795 | 4.201 (3.087–4.514) | 1.00× |
| medium | hot | 2867 | 4.727 | 4.277 (4.226–6.048) | 1.04× | 4.932 | 4.701 (4.264–6.078) | 1.12× |
| medium | all_to_one | 4096 | 4.152 | 5.282 (4.834–5.401) | 1.29× | 4.152 | 4.432 (4.381–11.472) | 1.05× |
| large | balanced | 4096 | 49.749 | 14.477 (13.600–15.390) | 1.00× | 49.745 | 13.890 (13.584–14.220) | 1.00× |
| large | hot | 11471 | 49.663 | 34.507 (34.479–34.759) | 2.38× | 49.666 | 34.593 (34.565–34.799) | 2.49× |
| large | all_to_one | 16384 | 47.751 | 47.455 (47.430–48.782) | 3.28× | 47.747 | 47.443 (47.410–47.736) | 3.42× |

The large all-to-one penalty reproduced at **3.28× and 3.42×**, while the medium penalty was **1.29× and 1.05×**. Medium hot exceeded all-to-one in the second session. Its balanced median was also higher than in the historical pre-test, and individual medium steps varied appreciably. The benchmark does not isolate the source of this timing variation.

### 5.2 Per-rank timings, large all-to-one [measured, ms]

These are the phase medians from the mine session's reference benchmark.

| Rank | Recv rows | Experts | Combine | Bwd experts | Bwd dispatch | Dispatch | Counts incl. sync | Step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.052 | 15.344 | 0.038 | 30.577 | 0.168 | 0.415 | 47.434 |
| 1 | 0 | 0.045 | 15.389 | 0.037 | 30.580 | 0.165 | 0.397 | 47.442 |
| 2 | 0 | 0.043 | 15.412 | 0.038 | 30.584 | 0.163 | 0.383 | 47.441 |
| 3 | 16384 | 14.928 | 0.160 | 30.104 | 0.161 | 0.336 | 0.382 | 47.269 |

Rank 3 spent **14.928 + 30.104 = 45.032 ms** in expert forward and backward, versus **4.036 + 7.773 = 11.809 ms** when balanced. This is close to P7's 48 vs 12 ms. On ranks 0–2, combine took **15.34–15.41 ms** and dispatch backward took **30.58 ms**, consistent with waiting for rank 3's expert outputs and input gradients. In the large hot case, rank 0 processed 11,471 rows, yet rank 3 had the highest median step time. Report skew as the job-step ratio versus balanced.

Rank 3's full receive buffer was **64 MiB**, of which **48 MiB** came from the other GPUs [derived]. The dispatch phase was **0.336 ms** in the mine session, compared with the **0.25 ms** transfer estimate. The balanced and all-to-one large EP=4 ratios against their own one-GPU medians imply about **3.58×** speedup and **1.01×** speedup, respectively, in that session.

Expert forward time per 1000 received rows ranged **0.91–1.06 ms** for large and **0.27–0.62 ms** for medium across both sessions. The large shape was closer to linear in row count. The experiment does not separate launch overhead from occupancy or other effects for the medium shape. Each session saved eight optional GPU traces separately from the timed runs.

[Ref-session EP=4 timing](../outputs/day14_downloads/day14_20261127T023509Z_ref/session_day14_20261127T023509Z_ref/bench_ep4.log) · [Mine-session EP=4 timing](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/bench_ep4.log) · [Mine-session EP=1 timing](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/bench_ep1.log)


## 6. Day 14 code ↔ Megatron Core [understanding]

The source anchors below are from the GUIDE's `core_v0.19.2` walkthrough (commit `4b4acac9`). The new server skipped the symbol scan. [GUIDE Block 3](GUIDE.md#block-3-read-megatron-core-1-h-stops-16-required-stop-7-optional)

| Day 14 path | Megatron Core anchor | Connection |
| --- | --- | --- |
| `top1_route` | `moe_utils.py:840`, `:853` | Day 14 takes the selected probability from a softmax over all experts. Megatron's pre-softmax flag enables that path. Its default softmax path selects top-k first, so the selected weight is constantly 1 at k=1. |
| Global auxiliary loss and AutoScaler | `router.py:422`, `:520`, `:632`; `moe_utils.py:304` | Day 14 uses dense-DP counts, while default Megatron aux uses TP×CP counts and has a separate global variant. AutoScaler attaches the auxiliary gradient. Per-token-loss mode adjusts its scale for token normalization later in the step. |
| `capacity_keep_mask` | `moe_utils.py:256`, `:990`, `:1002` | Day 14's capacity experiment evaluates both scopes. Megatron computes source-local capacity before dispatch and can select the highest-probability tokens per expert column. |
| `exchange_counts`, then `.tolist()` | `token_dispatcher.py:557`, `:565`, `:580`, `:590`, `:948` | Day 14 exchanges each destination's slice with all-to-all. Megatron gathers the ranks' count vectors, computes the local splits, starts the host copy early, and waits at the chosen sync point. |
| `permute` and `unpermute` | `token_dispatcher.py:665`, `:884`; `moe_utils.py:563` | Group by expert on the way out and write the result back to the original row. Top-k requires summing multiple expert contributions into each token row. |
| `AllToAllV` | `token_dispatcher.py:698`, `:711`, `:857`; `mappings.py:470` | Dispatch and combine swap the split lists, as does each call's backward. Day 14 keeps the weight at the source. Megatron sends it to the expert so weighting can be fused with expert work. |
| `sort_chunks` | `token_dispatcher.py:775`; `moe_utils.py:607` | With multiple local experts, turn source-major receive chunks into expert-major chunks, then reverse that grouping before returning outputs. |
| `finish_grads` | `distributed_data_parallel.py:208`, `:213`, `:624` | Dense gradients use the dense DP group. Expert gradients use EDP when there are replicas. The dispatch backward path has already delivered the token-level contributions from the EP sources to the expert. |
| `layout_ranks` | `parallel_state.py:880`, `:887`, `:1357`, `:1457` | EP groups distribute distinct experts. EDP groups link copies of the same expert. Megatron has a separate rank generator for the expert layout. |

## 7. Explanations from the walkthrough [understanding]

| Prompt | Explanation |
| --- | --- |
| Q1 · Why can +1.5 move so many tokens? | Every token compares expert 0 with its best competitor. Many margins lie within the bias increment, so one constant can flip many choices. The small-preset minimum margin quantifies the closest decision. It does not describe the medium-preset margin distribution. |
| Q2 · Position vs probs | Position protects earlier tokens. Probs protects high scores and can evict an earlier token when a later one scores higher. The dropless parity tests avoid these batch-dependent keep decisions. |
| Q3 · Unpermute | Initialize `n_tokens` rows, scatter by original row index, and leave dropped rows at zero. An out-of-place `index_copy` expresses a differentiable write into the output. Checking a non-involutive permutation catches the repeated-gather bug. |
| Q4 · Count exchange | All-gather retains every source–destination count, useful for inspecting global traffic. At EP=64 with one expert per rank, each rank materializes 4096 integers with all-gather versus 64 received integers with the Day 14 exchange [derived]. |
| Q5 · Padding tradeoff | At EP=4 and cf 1.25, a padded receive has 1280 rows per rank for either routing pattern, or 2.5 MiB for d=512 FP32. Balanced routing pads 256 extra expert rows per rank (25%). Padding trades potentially severe memory spikes for fixed overhead and possible overflow drops. |
| Q6 · Count gradients | The hard expert counts come through argmax and integer counting. The count all-reduce runs in forward. Each source can backpropagate through its own softmax probabilities using the shared load fractions. |
| Q7 · Similar rank step times under skew | The idle ranks arrive at a dependent collective earlier and wait for the busy expert. Compare the job-step median with balanced routing of the same shape to report the skew penalty. |
| Q8 · Soft counts | Setting `f=P` gives `E Σ P_i²` up to the coefficient, minimized at 1 for uniform mean probabilities. Uniform aggregate scores alone can still produce skewed hard choices, so actual assignment counts track the load we want to balance. |
| Stop 1 · Softmax order | In top-1, full softmax lets competitors influence the selected weight and its task gradient. Post-selection softmax yields 1, leaving enabled auxiliary and z-loss terms as router gradient paths. |
| Stop 2 · Auxiliary normalization | A later division by 100 would shrink a pre-attached 0.2 term to 0.002, so attach 20 to recover 0.2 after division. The Megatron per-token path uses the local token count times TP×CP group size. Default aux counts are TP×CP scoped. |
| Stop 3 · Local capacity | Megatron drops before dispatch using source-local candidate lists. In the measured learned pattern at cf 1.0 this costs 21 extra dropped tokens compared with a global pool. |
| Stop 4 · Host sync | The GPU can group rows using expert IDs without a CPU split list. It needs host-visible send and receive lengths to enqueue the uneven token collective. Megatron starts the copy on a side stream and can enqueue local work before waiting. |
| Stop 5 · Weight placement | Moving one scalar weight per routed assignment to the expert adds one forward exchange and one backward exchange for its gradient. It can fuse weighting with expert activations before the final projection. Top-k combine can accumulate already weighted contributions. |
| Stop 6 · Gradient groups | First decide who holds the same parameter, then normalize for the global loss. Only EDP replicas share an expert gradient reduction. In the four-source exercise the divisor is four, including when only two EDP ranks communicate. |

## 8. Limits of the reruns

- The optional EP=2 check and every timed stage used `ref`. Four-GPU NCCL correctness for `mine` is established for EP=4. Its EP=2 end-to-end parity and runtime costs, including the 26 dense gradient collectives, remain unmeasured.
- The benchmark timed one MoE layer on one four-GPU NVLink host. Two sessions show some timing variation, especially for medium cases. The cause, scaling across hosts, and full-model training throughput remain [not isolated].
- Megatron was studied through the GUIDE's code anchors. These reruns exercised the Day 14 harness, including the local-scope approximation. Grouped GEMM, communication overlap, shared and fine-grained experts, and Sinkhorn routing were reading topics.
- EP=8 is [simulated]. Multi-node all-to-all was outside this run. Top-2 had a CPU gradient check and no multi-GPU test. Router BF16 and FP8 numerics were outside the FP32 run.
- Auxiliary training used one seed and 200 steps on a toy task. It leaves longer-term routing dynamics, capacity tuning and generalization open.
