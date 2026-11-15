# Day 12: two-stage pipeline parallelism, GPipe and non-interleaved 1F1B

Both of my dual-A100 runs on 2026-11-15 passed. The reference rerun detected all three planted bugs, and my `my_pp.py` passed the boundary check and all eight GPipe/1F1B cases. Both archives and their internal manifests were verified after download. The earlier assistant runs remain in [GUIDE Appendix A](GUIDE.md#appendix-a-pre-test-results-pretest1-assistant-run-2026-11-14-pre-test).

Labels, as on Days 10–11: **[measured]** values from my run, **[pre-test]** values from the assistant's run, **[simulated]** values from `pp_sim.py` under its stated assumptions, **[derived]** arithmetic, **[understanding]** my reasoning, **[not isolated]** a cause the run does not separate.

Simulator assumptions for every [simulated] value unless a row says otherwise: P = 2 stages, M = 4 microbatches, t_F = 1 and t_B = 2 on every stage, communication 0, buffered sends, each stage runs its ops in a fixed order. "In flight" on a stage = from the start of F(m) to the end of B(m) there.

## 1. Predictions

This recap records my initial answers and the corrections we made during the walkthrough. The full numerical sweep is labeled [derived].

| # | Question | My prediction and reasoning [understanding] | Answer (GUIDE) | Matched? |
| --- | --- | --- | --- | --- |
| P1 | When can stage 1 start F0? | At `t=1`. Stage 1 waits for stage 0 to finish the first forward pass and send activation `a₀`. | t = 1, after stage 0's F0 produces `a_0` | Yes |
| P2 | GPipe grid; when does stage 0 start B0? | I initially counted only one slot for stage 1's `B0` and got `t=6`. It starts at `t=5` and takes two slots. Stage 0 receives `g₀` at `t=7` and starts its own `B0` immediately. | t = 7 (stage 1's B0 ends then) | After correction |
| P3 | 1F1B orders and grid; what does stage 0 do at t = 2–4? | Stage 0 finishes `F1` and waits for `g₀` during `[2,4)`. My first grid omitted the second slot of stage 1's `B0`. The corrected exchanges are `a₁/g₀` at `t=4`, `a₂/g₁` at `t=7`, and `a₃/g₂` at `t=10`. At `t=13`, stage 1 sends back only `g₃`. | `F0 F1 B0 F2 B1 F3 B2 B3` / `F0 B0 F1 B1 F2 B2 F3 B3`; waits for `g_0` | After correction |
| P4 | Idle cells and peak in flight per stage | Each GPU is idle for 3 slots under either schedule, so the bubble is `6/30=20%`. In GPipe, both GPUs finish all four forwards before starting backward, giving peaks of `[4,4]`. In 1F1B, stage 0 holds batches `{0,1}` when `F1` starts and `{1,2}` when `F2` starts. Stage 1 finishes each batch's backward before starting the next batch, giving peaks of `[2,1]` across the two stages. | 3 and 3 idle in both; peaks GPipe [4, 4], 1F1B [2, 1] | Yes |
| P5 | Scalar demo: what crosses, `w0.grad`, and the two mistakes | The forward pass sends `a₀=3, a₁=6`, and backward sends `g₀=12, g₁=24`. Applying the chain rule gives `w₀.grad=12×1+24×2=60` and `w₁.grad=6×3+12×6=90`. Swapping the batch pairings gives `w₀.grad=48`. If the receiver omits `requires_grad`, it can still compute its own weight gradient, while the received tensor's `.grad` is `None`. | `a` = 3, 6; `g` = 12, 24; 60 / 90; swapped pairing 48; no `requires_grad` → `None` | Yes |
| P6 | Naive send-then-recv in 1F1B steady state | Under rendezvous semantics, rank 0 blocks while sending `a₁` and rank 1 blocks while sending `g₀`. Each waits for the other to post a receive. In GPipe, rank 0 sends and rank 1 receives during forward. The direction reverses during backward, so each phase has a matching receive for every send. | both ranks block in their send; GPipe has no two-way exchange | Directions correct initially. GPipe conclusion corrected in discussion. |
| P7 | Peak in flight for general P, M; depends on times? | With four stages, rank 1 does two warmup forwards and reaches a peak of 3 when it starts `F2`. In general, GPipe peaks at `M` and 1F1B peaks at `min(P−s,M)`. For two GPUs, I initially used `s=1` for the first stage. Rank 0 corresponds to `s=0` with zero-based indexing. Execution times affect how long batches stay in flight. The operation order determines the peak count. | GPipe M; 1F1B min(P − s, M); order alone | After correcting the stage index |
| P8 | M = 1 → 8 at a fixed effective batch | At `M=8`, GPipe stage 0 holds all 8 microbatches, equivalent to the entire batch's activations. 1F1B holds two microbatches of one sequence each, or `1/4` of the full batch. In the simulator, each GPU computes for 24 slots and idles for 3, giving a bubble of `1/9`. With correct normalization, the mathematical loss and gradients are independent of how the batch is split. Summing the mean losses from equal-sized microbatches directly multiplies the loss and gradients by `M`. | bubble (P−1)/(M+P−1); GPipe activations ≈ whole batch; 1F1B min(P−s, M)/M of it; gradients unchanged up to rounding | Yes. Full sweep below. |

Two GPUs with a fixed batch of 8 sequences [derived]

| M | Sequences per microbatch | Simulated bubble | GPipe peak in flight (rank 0 / 1) | 1F1B peak in flight (rank 0 / 1) | Rank 0 peak activation fraction of full batch (GPipe / 1F1B) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 1/2 | 1 / 1 | 1 / 1 | 1 / 1 |
| 2 | 4 | 1/3 | 2 / 2 | 2 / 1 | 1 / 1 |
| 4 | 2 | 1/5 | 4 / 4 | 2 / 1 | 1 / 1/2 |
| 8 | 1 | 1/9 | 8 / 8 | 2 / 1 | 1 / 1/4 |

## 2. Run record

| Item | Reference rerun [measured] | My implementation [measured] |
| --- | --- | --- |
| Run ID | `manual_ref_20261115T113822Z` | `manual_mine_20261115T114621Z` |
| Time (UTC) | 2026-11-15 11:38:27–11:38:58 (31 s) | 2026-11-15 11:46:21–11:46:54 (33 s) |
| Configuration | `--impl ref`, GPipe and 1F1B at M = 1, 2, 4, 8 | `--impl mine`, same matrix |
| Verdict | Exit 0, boundary and all 8 cases passed, all 3 negative controls detected | Exit 0, boundary and all 8 cases passed |
| Archive SHA-256 | `ef0f56d80a0d318432943e2392c1dfb8547d98cb23de2b0b2eea08c69935ecf8` | `7f76eeb9e50429ec4f33be8a178f5c4819107ea48696c298e02049bfb2416385` |
| Internal manifest | 45 files verified | 45 files verified |

Both runs used Vast.ai container `60779373` with two A100-SXM4-80GB GPUs connected over NV12. Both GPUs were on NUMA node 0 and reported bidirectional P2P. Driver 580.178.04, Python 3.12.15, torch 2.14.1+cu130 (CUDA build 13.0), NCCL 2.30.7. The independently generated single-GPU baselines reported the same `problem_sha256` and bitwise repeatability at every M. The archived `mine` source is byte-identical to the local `my_pp.py`.

[Reference session](../outputs/day12_downloads/manual_ref_20261115T113822Z/session_manual_ref_20261115T113822Z/) · [Reference parity JSON](../outputs/day12_downloads/manual_ref_20261115T113822Z/parity_ref_manual_ref_20261115T113822Z.json) · [My session](../outputs/day12_downloads/manual_mine_20261115T114621Z/session_manual_mine_20261115T114621Z/) · [My parity JSON](../outputs/day12_downloads/manual_mine_20261115T114621Z/parity_mine_manual_mine_20261115T114621Z.json)

## 3. Simulated vs measured schedule (M = 4)

| Schedule | Rank / stage | Order [simulated and measured in both runs] | Idle units [simulated] | In-flight trace [measured in both runs] | Peak [simulated and measured] |
| --- | --- | --- | ---: | --- | ---: |
| GPipe | 0 | `F0 F1 F2 F3 B0 B1 B2 B3` | 3 | `1 2 3 4 3 2 1 0` | 4 |
| GPipe | 1 | `F0 F1 F2 F3 B0 B1 B2 B3` | 3 | `1 2 3 4 3 2 1 0` | 4 |
| 1F1B | 0 | `F0 F1 B0 F2 B1 F3 B2 B3` | 3 | `1 2 1 2 1 2 1 0` | 2 |
| 1F1B | 1 | `F0 B0 F1 B1 F2 B2 F3 B3` | 3 | `1 0 1 0 1 0 1 0` | 1 |

The trace has one entry per operation. The count rises at the start of each forward and falls at the end of each backward.

**P2P calls, 1F1B, M = 4** [measured in both runs, equal to the simulator's Megatron-style program]

- rank 0: `send_forward(a0) send_forward_recv_backward(a1,g0) send_forward_recv_backward(a2,g1) send_forward_recv_backward(a3,g2) recv_backward(g3)`
- rank 1: `recv_forward(a0) send_backward_recv_forward(g0,a1) send_backward_recv_forward(g1,a2) send_backward_recv_forward(g2,a3) send_backward(g3)`

In the M = 1 boundary test, both ranks ran `F0 B0`. Rank 1's received input was a leaf with `requires_grad=True`, and it sent back an FP32 gradient of shape `[256, 8, 512]`. Rank 0 had no input gradient to send.

**GPU timestamps, 1F1B, M = 4, rank 0** [measured, `mine`]. In ms, `F0 0.030–0.654, F1 0.711–1.169, B0 2.444–3.101, F2 3.104–3.516, B1 4.062–4.722, F3 4.726–5.138, B2 5.663–6.319, B3 7.261–8.303`. The gap before `B0` is about 1.28 ms, and the cooldown gap before `B3` is about 0.94 ms [derived]. Both line up with places where the simulated rank 0 waits for a downstream gradient. Smaller gaps of about 0.55 ms before `B1` and 0.53 ms before `B2` [derived] have no simulated counterpart. With equal stage times, stage 1's `F + B` takes as long as stage 0's `B + F`, so the gradient should arrive just as rank 0 finishes its forward. Here, rank 1 took about 0.53 ms per forward and 1.05 ms per backward, while rank 0 took about 0.41 ms for `F2`/`F3` and 0.66 ms for `B0`–`B2` [derived]. Stage 1 is therefore the slower stage, the GUIDE §4 case. Why rank 1 is slower is [not isolated]. The timestamps show between-compute gaps. The experiment did not isolate P2P latency or measure a GPU bubble percentage. [My event log](../outputs/day12_downloads/manual_mine_20261115T114621Z/events_manual_mine_20261115T114621Z/mine_1f1b_M4.jsonl)

## 4. Parity and the M sweep (effective batch 8 × 256 tokens fixed)

The checker requires `rel ≤ 1e-5` for `y`, loss and all 16 parameter gradients against both single-GPU references, plus matching order, peak and labels. The two runs reported the same values below. All logged P2P call sequences matched the simulator as well.

| Schedule | M | Bubble [simulated] | Peak in flight [measured] | Worst rel vs full batch [measured] | Bitwise vs accumulation, same M [measured] | CUDA peak MiB r0 / r1 [measured] |
| --- | ---: | ---: | --- | --- | --- | --- |
| GPipe | 1 | 0.500 | [1, 1] | 0 | all | 120.3 / 120.3 (first case) |
| GPipe | 2 | 0.333 | [2, 2] | 5.49e-7 | all | 90.0 / 92.0 |
| GPipe | 4 | 0.200 | [4, 4] | 5.49e-7 | all | 81.0 / 84.0 |
| GPipe | 8 | 0.111 | [8, 8] | 6.98e-7 (`y`) | all but loss | 78.5 / 82.0 |
| 1F1B | 1 | 0.500 | [1, 1] | 0 | all | 104.0 / 104.0 |
| 1F1B | 2 | 0.333 | [2, 1] | 5.49e-7 | all | 90.0 / 56.0 |
| 1F1B | 4 | 0.200 | [2, 1] | 5.49e-7 | all | 45.0 / 28.0 |
| 1F1B | 8 | 0.111 | [2, 1] | 6.98e-7 (`y`) | all but loss | 24.5 / 16.0 |

CUDA peak here means `max_memory_allocated() − memory_allocated()` at the schedule start, per rank. The parameters, preallocated gradient buffers, inputs, targets and output buffer are already included in the starting value. NCCL's own buffers are outside the PyTorch allocator's measurement.

- Within each run, the checker reported bitwise agreement between GPipe and 1F1B for `y` and all 16 parameter gradients at every M.
- Against single-GPU accumulation at the same M, `y` and all 16 gradients were also bitwise equal in all cases. Only the reported scalar loss at M = 8 differed, by one FP32 spacing (absolute `4.77e-7`, relative `8.30e-8`). The checker uses `torch.stack(...).sum()` and the baseline uses sequential `+=`. This final detached sum is separate from the per-microbatch losses used for backward [derived from the code].
- At M = 4, switching to 1F1B reduced the incremental peak by about 36 MiB on rank 0 (44%) and 56 MiB on rank 1 (67%) [derived]. Rank 1 alternates `F_m → B_m`, so its peak falls from four active batches to one.
- GPipe M = 1 read 16.25 MiB more per rank than 1F1B M = 1 with the same order and calls. GPipe was the first case in both processes [not isolated].
- On rank 0, subtracting the estimate of saved activations (`count × 9b MiB`, with b sequences per microbatch) leaves about 18.0, 9.0 and 6.5 MiB at M = 2, 4 and 8 for either schedule [derived]. The components of that remainder were not isolated.

## 5. Negative controls (1F1B, M = 4)

| Control | Expected from the discussion [understanding] | Reference rerun [measured] |
| --- | --- | --- |
| `detached_input` | Rank 1 can compute its own parameter gradients. The planted control sends zeros in place of the missing activation gradient. | All 8 rank-0 parameter gradients failed, ratio 0.0. |
| `lifo_backward` | Pairing a gradient with another batch's saved graph can silently corrupt rank-0 gradients. Its final bias sums the arriving gradients. | 7 rank-0 gradients plus `order` and `labels` failed. Rank 1 sent `[0,1,2,3]`, while rank 0 requested `[1,2,3,0]`. `blk1.fc2.bias` passed. |
| `mean_per_microbatch` | Summing four per-microbatch means without a factor of 1/4 scales the loss and gradients by four. | Loss and all 16 gradients failed, ratio 4.0. |

All three produced the expected failures and the reference run exited 0. For `blk1.fc2.bias`, the local derivative of the output with respect to the bias is 1. Its gradient is the sum of the incoming `g_m` over tokens and does not use any saved activation. Permuting which saved microbatch receives each `g_m` therefore leaves the total bias gradient unchanged mathematically. The other seven rank-0 gradients multiply `g_m` by saved activations, or pass it back through the saved weights and activation mask, so the wrong pairing changes them.

## 6. My code ↔ Megatron Core (`core_v0.19.2`, commit `4b4acac9`)

Locations are verified for that commit (`find_symbols.py`). The last column reflects the walkthrough and my completed local implementation.

| My code | Megatron Core | Location | Same / different (my words) |
| --- | --- | --- | --- |
| `baseline_1gpu.py` accumulation loop | `forward_backward_no_pipelining` | `pipeline_parallel/schedules.py:696`, loop `:788` | At PP=1, the full model runs forward and backward one microbatch at a time and accumulates parameter gradients. This matches the single-GPU baseline and requires no activation or gradient transfers between stages. |
| schedule choice | `get_forward_backward_func` | `schedules.py:48`, `:156–162` | The usual selector maps PP=1 to the baseline and PP=2 with `vp_size=None` to my `run_1f1b`. Enabling virtual stages selects the interleaved schedule, where each physical rank can hold multiple model chunks. My `run_gpipe` serves as an experimental comparison and has no branch in this selector. |
| `run_1f1b` warmup count (TODO 3a) | `num_warmup_microbatches = total_stages - current_stage - 1` | `schedules.py:2243–2245` | The number of warmup forwards is the number of downstream stages, capped at `M`. With two GPUs, rank 0 runs one warmup forward and rank 1 enters steady state directly. |
| warmup loop, saving `(mb, input, output)` | warmup loop, `input_tensors.append` | `schedules.py:2299–2334` | Both implementations receive the input, run the forward pass, send the output, and save the tensors needed for backward. I store the microbatch ID, input, and output together in a `deque`. Megatron stores inputs and outputs separately. |
| first steady `recv_forward` (TODO 3b) | "Before running 1F1B..." | `schedules.py:2336–2342` | If any steady-state iterations remain, receive the input for microbatch `warmup` before the loop. Later inputs arrive through the paired exchange at the end of the preceding iteration. |
| steady loop (TODO 3c) | `send_forward_recv_backward`, `pop(0)`, `backward_func`, `send_backward_recv_forward` | `schedules.py:2382`, `:2393`, `:2402`, `:2412` | Each iteration sends a new activation, receives the gradient for the oldest saved microbatch, and runs backward for that microbatch. My `popleft()` corresponds to Megatron's `pop(0)`. When another forward remains, send the gradient and receive the next input together. In the final iteration, send only the gradient. |
| cooldown (TODO 3d) | cooldown loop | `schedules.py:2416–2440` | After all forwards have been scheduled, receive each remaining gradient, run backward, and send the input gradient in FIFO order until the queue is empty. With two GPUs and `M=1`, rank 0 runs `B0` here. |
| `forward_step`, received input (TODO 1a) | `forward_step`, `set_input_tensor` | `schedules.py:397`, `:501` | Every stage after the first uses the received activation for its forward pass. I pass it directly to `st.module(x)`. Megatron hands it to the model through `set_input_tensor`. |
| `mb_loss` (n_m / N) | `output_tensor /= num_microbatches` | `schedules.py:335` | The experiment divides the sum of squared errors for each microbatch by the number of elements in the full batch. For equal-sized microbatches, this is equivalent to taking each microbatch's mean loss and dividing it by the number of microbatches. |
| `backward_step` (TODO 1d–1e) | `backward_step`: `retain_grad`, `autograd.backward(..., grad_tensors=...)`, `x.grad` | `schedules.py:532`, `:554`, `:574`, `:584` | The last stage starts backward from the scalar loss. The other stages supply the received output gradient. Each stage with an upstream neighbor returns its input's `.grad`. The experiment receives a leaf tensor, and Megatron uses `retain_grad()` to also capture gradients on non-leaf inputs. |
| (not done) | `deallocate_output_tensor`, `custom_backward` | `schedules.py:166`, `:196`, `:199` | The experiment keeps the original output for standard autograd. After sending an output, Megatron can replace its `.data` with a one-element tensor, preserve the computation history, and use custom backward. |
| `P2P` helpers | `P2PCommunicator` and its six methods | `pipeline_parallel/p2p_communication.py:140`, `:424–560` | Both implementations have four one-way helpers and two paired helpers. The first and last stages skip operations on the side with no neighbor. Megatron gets each neighbor's global rank from the PP process group. |
| `received.requires_grad_()` | receive buffers with `requires_grad=True` | `p2p_communication.py:333` | Both implementations enable local gradient computation on the received activation. I enable gradient tracking before the forward pass. Megatron enables it when creating the receive buffer. |
| `batch_isend_irecv` in `P2P._communicate` | `_batched_p2p_ops` (default `batch_p2p_comm=True`) | `p2p_communication.py:17`, `:49`, `:374`; `model_parallel_config.py:390` | Post all sends and receives for the exchange before waiting. This gives the sends for `a₁` and `g₀` matching receives. |
| (simulator `--comm-style naive` fix 2) | `_p2p_ops`, even/odd order | `p2p_communication.py:55`, `:79` | The simulator's naive path reproduces the deadlock when both ranks send first. Megatron's unbatched path uses opposite send and receive orders on even and odd ranks to match the operations. |
| `mb_shape = (seq, batch/M, d)` | `get_tensor_shapes` → `(seq_length, micro_batch_size, hidden_size)` | `schedules.py:2088` | Both implementations use the sequence length, sequences per microbatch, and hidden size to determine the communication shape. The Day 12 example at `M=4` is `[256,2,512]`. Megatron can also adjust the sequence dimension according to its configuration. |

For Day 13 [understanding, a prediction to check there]: at `DP=1, TP=2, PP=2`, which ranks exchange activations, which Megatron call issues the steady-state exchange, and what shape I expect for Day 13's model:

> Drafted with the assistant on 2026-11-15, from Day 11 §8.1 and this day's mapping. Not yet checked.
>
> - **Groups.** With Megatron's default order `tp-cp-ep-dp-pp`, TP varies fastest. TP groups are `[0,1]` (stage 0) and `[2,3]` (stage 1). PP groups are `[0,2]` and `[1,3]`. Each TP rank exchanges activations only with the rank of the same TP index in the other stage: rank 0 ↔ rank 2, rank 1 ↔ rank 3. These are two independent FIFO channels. Gradients come back 2 → 0 and 3 → 1.
> - **Stages.** Six layers split 3 + 3. Stage 0 also holds the embedding. Stage 1 also holds the final LayerNorm, the output layer and the loss. Embeddings are untied, so no gradient sync between the first and last stages.
> - **Steady-state call.** Ranks 0 and 1 call `send_forward_recv_backward`. Ranks 2 and 3 call `send_backward_recv_forward`. Each is one `batch_isend_irecv` inside its own PP group, as in my `run_1f1b`. Both TP ranks of a stage must run the same 1F1B order. Their TP collectives pair tensors by call order, so a rank on a different microbatch would add the wrong tensors or hang.
> - **Shape, sequence parallelism off.** After the row-parallel all-reduce, both TP ranks hold the same full output. Rank 0 and rank 1 each send an identical `[256, 2, 512]` copy, 1 MiB in FP32. Inside a stage, the TP collectives are the all-reduces of my Day 11 prediction.
> - **Shape, sequence parallelism on.** The row-parallel output is reduce-scattered along the sequence. Rank 0 holds positions 0–127 and rank 1 holds positions 128–255, each with all 512 features. Each sends its own `[128, 2, 512]` shard, 0.5 MiB in FP32. The shards line up because rank 2 is TP rank 0 of stage 1 and expects positions 0–127. Stage 1 starts with a per-token LayerNorm, then all-gathers the sequence at its first column-parallel linear. Inside a stage, the all-reduces become reduce-scatters and all-gathers.
> - **Where to check.** `get_tensor_shapes` (`schedules.py:2088`) should divide the sequence length by the TP size when `sequence_parallel` is on. This was not confirmed in the source. In the trace, the send/recv kernels run over `[0,2]` or `[1,3]`, and the TP collectives over `[0,1]` or `[2,3]`.

## 7. Explanations (my own words)

1. **Ownership and what crosses.** Which blocks and data each stage holds; the two tensors per microbatch that cross; what never crosses.
   > Rank 0 holds blocks 0–1 and the inputs. Rank 1 holds blocks 2–3 and the targets. For each microbatch, the forward transfer carries only the activation `a_m`, and the backward transfer carries only `g_m = ∂L/∂a_m`. Parameters, parameter gradients, targets, and loss stay on their respective ranks.
2. **Autograd across the boundary.** Why the receiver makes the activation a leaf that requires grad; why the sender needs `grad_tensors`; what each stage keeps per microbatch in flight.
   > Rank 0 keeps the original output with its `grad_fn`. Rank 1 enables gradient tracking on the received leaf tensor and keeps it along with the microbatch loss. After running backward from the loss, rank 1 sends the leaf's `.grad` back. Rank 0 supplies that gradient as `grad_tensors` for the original output and continues backward through its own graph. The leaf needs `requires_grad`: without it, autograd treats the received tensor as a constant and stores no ∂L/∂received, so `.grad` stays `None` and there is nothing to send back. The sender needs `grad_tensors` because its output is a `[seq, b, d]` tensor, not a scalar. Backward from a non-scalar needs the upstream vector ∂L/∂output, and here that vector exists only on rank 1.
3. **Matching microbatches.** FIFO channel + same backward order + FIFO local queue; what `lifo_backward` broke and why nothing crashed.
   > Both ranks send and receive gradients in the same microbatch order, and the schedule takes the oldest saved forward record from the queue. Taking the newest record still gives a tensor of the right shape, so computation continues with the gradient applied to the wrong graph. The negative control exposes this through the microbatch labels and rank 0 parameter gradients.
4. **Deadlock.** Why 1F1B's steady state needs paired calls and GPipe does not; the two fixes in Megatron.
   > In steady state, rank 0 sends a new activation while rank 1 sends an earlier gradient. If both ranks start with a blocking send, each waits for the other to post a receive. Posting the paired sends and receives before waiting resolves this, as does using opposite orders on even and odd ranks. GPipe communicates in one direction at a time during its forward and backward phases.
5. **Bubble vs residency.** Why both schedules have the same idealized bubble, why 1F1B holds at most P − s microbatches, and when that matters (M > P).
   > With equally fast stages and zero communication time, both schedules have a bubble fraction of `(P−1)/(M+P−1)`. Each stage still does `M` forwards and `M` backwards. Under either order, a stage must wait while the pipeline fills before its first forward, and wait again while the last gradients drain back. Each stage idles `(P−1)(t_F+t_B)` in total. 1F1B only reorders work between those waits, so it cannot shorten them. 1F1B completes backwards earlier, so the peak per stage falls from `M` to `min(P−s,M)`. Stage `s` runs `P−s−1` warmup forwards, then never starts a new forward without finishing a backward. The saving appears only when `M > P−s`. At `M=2`, rank 0 has `P−s = 2 = M`, and both schedules measured 90.0 MiB. Rank 1 has `P−s = 1`, and its peak fell from 92.0 to 56.0 MiB. At `M=4`, rank 1 went from four microbatches in flight to one, and its measured incremental memory peak fell from 84 to 28 MiB.
6. **Changing M at a fixed effective batch.** Bubble, peak activation bytes for each schedule, numerics, and what the ideal model leaves out.
   > At a fixed total batch size, increasing `M` reduces the simulated bubble. GPipe still retains roughly one full batch of activations at peak. Each 1F1B stage retains about `min(P−s,M)/M` of a batch at peak. Normalizing the loss by the number of elements in the full batch preserves the mathematical result across different splits. Measured on rank 0 for M = 2, 4, 8, GPipe stayed near one full batch at 90.0, 81.0 and 78.5 MiB. 1F1B roughly halved each time M doubled, at 90.0, 45.0 and 24.5 MiB. Numerically, `y` and all 16 gradients were bitwise equal to single-GPU accumulation at the same M. Against the full batch they differed by at most 6.98e-7 relative, from summing in a different order. The ideal model leaves out communication time, per-microbatch launch overhead, the lower efficiency of smaller matmuls, and unequal stage times. Those costs grow with `M` and can cancel the smaller bubble.
7. **(Optional) A slow stage.** What sets the makespan, why the peak counts do not change, what changes instead.
   >

## 8. Not covered or not isolated

- The bubble on GPUs is only simulated. The GPU timestamps show order and gaps, but this small model is launch-bound and they are not a benchmark.
- P2P latency and bandwidth were not measured. The effect of communication time is simulated only (GUIDE §4).
- The 16.25 MiB extra on the first case of a process [not isolated].
- Why rank 1's forward and backward take longer than rank 0's, which leaves the ~0.5 ms waits before `B1` and `B2` [not isolated].
- How the per-M CUDA remainder splits between backward temporaries and received buffers [not isolated].
- No BF16, no P > 2, no interleaved schedule, no `deallocate_output_tensor`, no TP inside a stage (Day 13).

## Completion

- [x] P1–P8 initial reasoning and corrections recorded as a discussion recap (§1).
- [x] `my_pp.py` passed the dual-GPU boundary check and GPipe/1F1B at M = 1, 2, 4, 8 with `--impl mine`.
- [x] Reference run with controls and `--impl mine` run downloaded and verified. §2–§5 filled from those runs.
- [x] Block 3 Stops 1–5 reviewed in the walkthrough and §6 mapping filled.
- [x] §7 explanations recorded from the walkthrough.
- [ ] Optional: GUIDE §4 (slow stage, communication time).
