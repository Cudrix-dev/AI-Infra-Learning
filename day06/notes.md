# Day 06 — ZeRO-1 from scratch

> **Measured: my manual run `a100_20261102T124935Z` (2026-11-02).** The tables below use this run, replacing the assistant's earlier `pretest1` figures. Session: about 20:49:35–21:00:35 Singapore time (12:49:35–13:00:35 UTC), exit code 0; `OUTPUT CHECKS PASSED: 52 runs, 12 traces, 3 summary tables, 4 detected controls`.
>
> Setup: **Vast.ai, 2 × A100-SXM4-40GB** (`ssh4.vast.ai:17591`), NV12, both GPUs on NUMA node 1 (CPU affinity 32–63, 96–127); Python 3.12.15, PyTorch 2.14.1+cu130, CUDA build 13.0, NCCL 2.30.7, `OMP_NUM_THREADS=1`, `condition=shared`.
>
> Local evidence: `C:/Users/qly24/Downloads/day06_a100_20261102T124935Z/results/`; archive and `.sha256` alongside it. Archive SHA256 `82a19058bcec33877bafa1c85769d77a9d7e79fc1cdf8f8b078f51513bb2622c` and all 179 manifest files verified. The snapshot's `zero1.py`, `train_zero1.py`, `compare_zero1.py` and runner match the current local experiment code. Derived summary: [`analysis.json`](../outputs/day06_downloads/a100_20261102T124935Z/analysis.json). Historical pre-test retained separately at `outputs/day06_downloads/pretest1/`.
>
> Code: [`experiments/day06_zero1/`](../experiments/day06_zero1/README.md) · Guide: [`GUIDE.md`](GUIDE.md)

**Status (2026-11-04): complete.** The done-when criterion is met by measured evidence; independent practice (an unaided reimplementation and explanation) remains an open item, not a blocker. The manual run verifies parity, state/memory, collective counts and timings; the communication follow-up is recorded in §8.7. The guided discussion has covered Blocks 1–4 and the core source path, including local optimizer ownership, hyperparameter synchronization, parameter broadcasts, state consolidation/indexing, and the overlap branch. This records assisted understanding, not an independently authored reimplementation or an unaided final explanation.

Evidence labels:

- **[measured]** from manual run `a100_20261102T124935Z`; source logs and summary JSONs in the evidence directory above.
- **[understanding]** mechanism or interpretation; not necessarily isolated by this run.
- **[not tested]** outside this experiment's coverage.

## 1. What systems problem did I solve today?

Data parallelism keeps an identical copy of AdamW's two moment tensors on every rank: 2P = 209.4 MiB per rank for the medium model in FP32, the largest single item of model state. ZeRO-1 keeps each element's moments only on the rank that owns that element. Today's `FlatZeroAdamW` flattens all parameters into one buffer, gives each rank one contiguous half, reduce-scatters the locally accumulated window gradient so each rank receives the averaged gradient of its half, runs AdamW on that half only, and all-gathers the updated parameters. The same harness runs DDP + AdamW, the matched unsharded flat control (`flat_ar`), PyTorch's `ZeroRedundancyOptimizer`, and a variant that reduce-scatters every microbatch.

[measured] ZeRO-1 is **bitwise identical** to DDP + AdamW over 6 optimizer steps with accumulation and a short final window: step-0 and last-window gradients, parameters, and the gathered `exp_avg` / `exp_avg_sq`. Each rank holds exactly half of the state (52.34 + 52.34 MiB instead of 104.69 + 104.69 MiB), and the matched flat control's first-step allocation falls from 209.38 to 104.69 MiB. DDP's measured first-step allocation is 210.38 MiB, including a 1.00 MiB excess over its state tensors.

## 2. The mechanism in my own words [understanding]

1. **Only the state is sharded.** Parameters stay full on every rank (forward and backward need them); the full local gradient exists during backward. AdamW's moments are read only by the optimizer, once per step, and only for the elements a rank updates.
2. **AdamW is elementwise**, so a slice may cut through a weight matrix: each element's update needs its own param, grad, `exp_avg`, `exp_avg_sq` and shared scalars. Global norms (clipping), per-tensor statistics (LAMB, Adafactor) or whole matrices (Shampoo, Muon) would need extra collectives or whole-tensor ownership.
3. **The all-reduce splits in two.** Reduce-scatter (each rank gets the summed gradient of its slice) → divide by W → AdamW on the slice → all-gather (everyone gets the updated slices). Same bytes as one all-reduce; one extra collective after the optimizer.
4. **Flat buffers make it cheap.** Parameters are views of `flat_param`, gradients views of `flat_grad`, the optimizer's parameter is `nn.Parameter(flat_param[lo:hi])` (shared storage), and both collectives run in place. The price: `zero_grad` must zero in place; `set_to_none` silently detaches the views.
5. **The accumulation boundary.** Accumulate the full gradient locally for K microbatches and reduce-scatter once (P of gradient memory between microbatches, 1 reduce-scatter per window), or reduce-scatter every microbatch into a P/W accumulator (P/W between microbatches, K reduce-scatters). One optimizer step and one all-gather per window either way; parameters are full for the whole window.

## 3. Call-chain reference (pytorch@v2.14.1) — assisted walkthrough completed 2026-11-04

| Item | Answer |
| --- | --- |
| Entry | `ZeroRedundancyOptimizer.step()` (`zero_redundancy_optimizer.py:1124`) after DDP's backward |
| Partition | `_partition_parameters` (`:651`): sort by `numel` descending, each tensor to the least-loaded rank |
| Local step | `_local_step` (`:1047`): `_sync_param_groups` (LR in), `self.optim.step()` over owned tensors, LR out |
| Parameter sync | `_sync_params` (`:813`) → `_broadcast_params_from_rank` (`:760`): `broadcast(param.data, src=owner, async_op=True)` per parameter, or per rank bucket with `parameters_as_bucket_view` |
| Gradient reduction | none in ZeRO: DDP's bucket all-reduce of the full gradient |
| State | local `AdamW` over owned tensors only (`_init_local_optimizer`, `:1516`); `consolidate_state_dict` (`:513`) moves each rank's state dict to one rank with `_broadcast_object` |
| AdamW | `_single_tensor_adam` (`adam.py:347`): `mul_`, `lerp_`, `mul_().addcmul_()`, `sqrt()/bc2`, `addcdiv_`, all elementwise |
| Overlap variant | `hook_with_zero_step` (`ddp_zero_hook.py:183`): owned step + broadcast inside DDP's bucket hook (not run) |

## 4. My minimal version vs the framework

| Aspect | `FlatZeroAdamW` (`zero1`) | `ZeroRedundancyOptimizer` |
| --- | --- | --- |
| Partition | flat, padded, W equal contiguous slices; may split tensors | whole tensors, greedy by size; tiny 221,184 / 216,704, medium 13,762,560 / 13,680,640 elements |
| Gradient reduction | reduce-scatter of the flat window sum, in place | DDP's all-reduce of the full gradient |
| Local optimizer | one `AdamW` over one slice tensor (≈10 kernels) | one `AdamW` over the owned tensors (per-tensor loop) |
| Parameter sync | one in-place all-gather | one broadcast per parameter (78 medium), or per rank (bucket view) |
| Bytes sent per rank, W = 2 | P/2 + P/2 = P | P (all-reduce) + P/2 (broadcasts) = 1.5 P |
| Correct for | elementwise optimizers only | any optimizer whose update is per tensor |
| Checkpoint | `full_state()`: an all-gather of the slices (evidence only) | `consolidate_state_dict(to)` + `state_dict()` |

## 5. Evidence it is correct [measured, `parity.log`]

Tiny preset, 6 optimizer steps of G = 32 = 2 ranks × K 4 × m 4, last window G = 16 (windows `[4, 4, 4, 4, 4, 2]`). Reference: 1 process, one batch of G rows per step, Day 01's AdamW. A–C Day 02 tolerances, D exact, E A's tolerance; R replicas bitwise after every step; S gathered state vs the DDP + AdamW run (≤ 1e-4 × max).

| Variant | A grad | B params-1 | C loss-0 | D | E last window | R | S vs DDP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ddp`, `torch_zero`, `torch_zero_bucket`, `flat_ar`, `zero1` | 2.794e-9 | 2.608e-7 | 0 | equal (6) | 8.324e-9 | yes | 0 (bitwise) |
| `zero1_rs_every` | 2.794e-9 | 2.608e-7 | 0 | equal (6) | 8.382e-9 | yes | `exp_avg` 4.7e-10, `exp_avg_sq` 7.1e-14 |

Bitwise against `ddp` (step-0 gradient, first-update parameters, last-window gradient, final parameters, gathered state): `torch_zero`, `torch_zero_bucket`, `flat_ar`, `zero1` all **identical**; `zero1_rs_every` not (final parameters 3.643e-7 apart), as predicted in Guide §1.6. Against the 1-process reference the state drifts to 1.7e-4 (`exp_avg`) and 1.2e-5 (`exp_avg_sq`) relative after 6 steps, the same for every variant including DDP.

| Negative control | Fails | Passes | Verdict |
| --- | --- | --- | --- |
| `no_reduce_scatter` (`zero1`) | A (6.9e-3), B, E (ratio 0.630), S | C, D, **R** | detected by A |
| `no_allgather` (`zero1`) | B (1.52e-5, barely above 1e-5), E, **R** (diverged after step 1), S | A, C, D | detected by R |
| `grad_set_to_none` (`zero1`) | A (gradient 0), B, E (ratio 0.000), S | C, D, **R** | detected by A |
| `rs_overwrite` (`zero1_rs_every`) | A (9.8e-3), B, E (ratio 0.650), S | C, D, **R** | detected by A |

Three of four ZeRO bugs keep the replicas bitwise identical: the all-gather distributes each owner's wrong slice. In Day 02's DDP, a missing reduction made replicas diverge; here only the gradient and state checks catch it.

## 6. Optimizer state per rank and memory [measured]

### 6.1 State per rank

| Run | Rank | Elements owned | `exp_avg` + `exp_avg_sq` | Allocation across step 1 | `step` |
| --- | --- | --- | --- | --- | --- |
| tiny `ddp` | 0, 1 | 437,888 (30 tensors) | 3.34 MiB | 3.34 MiB | CPU |
| tiny `torch_zero` | 0 / 1 | 221,184 (6) / 216,704 (24) | 1.69 / 1.65 MiB | 1.69 / 1.65 MiB | CPU |
| tiny `zero1` | 0, 1 | 218,944 each | 1.67 MiB | 1.67 MiB | CPU |
| medium `ddp` | 0, 1 | 27,443,200 (78) | 209.38 MiB | 210.38 MiB | CPU |
| medium `torch_zero` | 0 / 1 | 13,762,560 (14) / 13,680,640 (64) | 105.00 / 104.38 MiB | 106.00 / 105.38 MiB | CPU |
| medium `flat_ar` | 0, 1 | 27,443,200 | 209.38 MiB | 209.38 MiB | CPU |
| medium `zero1` | 0, 1 | 13,721,600 each | 104.69 MiB | 104.69 MiB | CPU |

The two measurements agree exactly for the flat runs. For the per-tensor runs (`ddp`, `torch_zero`), the allocation exceeds the state tensors by 1.00 MiB. This is consistent with allocator block rounding [understanding], but this run does not independently isolate that excess; the earlier assistant-only allocator probe is not evidence from this manual run. State tensor bytes and allocated-byte deltas are reported separately.

`torch_zero` balances **elements** (13.76 M vs 13.68 M) but not **tensors** (14 vs 64): greedy by size puts the two 4.2 M-element matrices (embedding, head) and the large linear weights first.

### 6.2 What stays allocated (medium, K = 4)

| Variant | After wrapping | Between microbatches | − `zero1` | Predicted (Guide §1.4) | Peak |
| --- | --- | --- | --- | --- | --- |
| `ddp` | 210.4 | 595.1 | +216.7 | +2P = +209.4 | 1227.3 |
| `torch_zero` | 210.4 | 490.7 / 490.1 | +112.3 | +P + 0.3 = +105.0 | 1122.9 |
| `flat_ar` | 209.4 | 483.1 | +104.7 | +P = +104.7 | 1115.3 |
| `zero1` | 209.4 | 378.4 | 0 | 3P = 314.1 absolute | 1010.6 |
| `zero1_rs_every` | 157.1 | 326.0 | −52.3 | −P/2 = −52.3 | 958.3 |

MiB. Wrapping and between-microbatch memory use repeat 0's per-rank records (maximum over ranks, except where both values are shown); peak is the median over 3 repeat peaks. State sharding saves exactly P = 104.6875 MiB against `flat_ar`, and `rs_every` saves another P − P/W = 52.34375 MiB between microbatches. Peak differences equal these differences: peak minus between-microbatch allocation is 632.25 MiB for all five rows at m = 8, consistent with one microbatch's transient working memory.

The three flat rows have the same 64.315 MiB residual above the §1.4 ledger. cuBLAS workspaces are a possible explanation [understanding], not isolated here. DDP-based rows have another 7.3125 MiB above the ledger after accounting for `torch_zero`'s uneven state ownership; 1.00 MiB appears at wrapping and another 1.00 MiB across the first optimizer step. The remaining excess was not isolated.

**ZeRO-1 cut the peak by 9.4% here** (1115.3 → 1010.6 MiB against the matched flat control, a 104.7 MiB reduction). Against DDP the drop is 1227.3 → 1010.6 MiB; about 112.0 MiB of that is already present in `ddp` → `flat_ar` (removing bucket duplication plus other layout overhead). Activations dominate at m = 8; the saving is a fixed P per rank, so its share grows with model size and shrinks with batch size.

## 7. Collectives per optimizer step [measured, `count.log`; medium, K = 4, rank-0 trace]

| | AllReduce | ReduceScatter | AllGather | Broadcast |
| --- | --- | --- | --- | --- |
| `ddp` | 4 | 0 | 0 | 1 |
| `torch_zero` | 4 | 0 | 0 | **79** |
| `torch_zero_bucket` | 4 | 0 | 0 | 3 |
| `flat_ar` | 1 | 0 | 0 | 0 |
| `zero1` | 0 | 1 | 1 | 0 |
| `zero1_rs_every` | 0 | 4 | 1 | 0 |

All as predicted: DDP's 4 bucket all-reduces + 1 buffer broadcast per window (Day 04); `torch_zero` adds one broadcast per parameter tensor (78), the bucket view one per rank (2); `zero1` one reduce-scatter + one all-gather; `rs_every` K reduce-scatters.

## 8. What the timing data supports and its limits [measured]

### 8.1 Method

Medium preset (Ψ = 27,443,200, P = 104.7 MiB), m = 8 rows per microbatch per rank, K ∈ {1, 4}, 25 optimizer steps with 5 excluded, 3 repeats (slow link: K = 4, 2 repeats), TF32 off, MATH attention, `condition=shared`. Phase boundaries synchronized identically in every variant. Core time: slowest rank per step, median over steps; [min–max] over repeats. Collective phases are host wall times given as max over ranks / min over ranks. The minimum is a heuristic for the later-arriving rank and reduces arrival-skew inflation, but still includes host dispatch and device synchronization; it is not an isolated NCCL kernel or wire-transfer time.

### 8.2 NVLink [measured, `timing_summary_a100_20261102T124935Z.json`]

| K | Variant | ms / opt step [min–max] | fwd+bwd | reduce max / last | optimizer | gather max / last |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `ddp` | 41.06 [41.04–41.43] | 30.17 | — | 10.93 | — |
| 1 | `torch_zero` | 46.70 [41.19–47.74] | 30.41 | — | 16.29 | — |
| 1 | `flat_ar` | 32.69 [32.53–32.80] | 29.73 | 1.34 / 1.01 | 1.83 | — |
| 1 | `zero1` | 32.04 [28.04–32.25] | 29.80 | 0.91 / 0.65 | 1.04 | 0.54 / 0.53 |
| 4 | `ddp` | 137.21 [137.20–138.18] | 123.01 | — | 14.37 | — |
| 4 | `torch_zero` | 143.97 [142.63–146.33] | 123.33 | — | 21.83 | — |
| 4 | `flat_ar` | 122.15 [106.22–123.11] | 119.17 | 2.02 / 1.01 | 1.85 | — |
| 4 | `zero1` | 121.49 [120.35–121.98] | 118.98 | 7.57 / 0.65 | 1.05 | 0.56 / 0.53 |
| 4 | `zero1_rs_every` | 120.41 [120.29–122.24] | 118.70 | — | 1.04 | 0.68 / 0.54 |

### 8.3 Slow link, `NCCL_P2P_DISABLE=1`, K = 4 [measured, `slowlink_summary_a100_20261102T124935Z.json`]

| Variant | ms / opt step [min–max] | fwd+bwd | reduce max / last | optimizer | gather max / last |
| --- | --- | --- | --- | --- | --- |
| `ddp` | 156.54 [155.56–157.52] | 148.91 | — | 10.39 | — |
| `flat_ar` | 158.53 [158.17–158.90] | 103.35 | 53.73 / 53.34 | 1.75 | — |
| `zero1` | 163.22 [163.06–163.37] | 103.52 | 29.70 / 29.05 | 1.01 | 29.65 / 29.57 |
| `zero1_rs_every` | 251.94 [251.36–252.52] | 221.20 | — | 0.95 | 29.73 / 29.69 |

### 8.4 What it supports

- **Most of the observed DDP → flat ZeRO speed-up is associated with flattening.** At K = 1, core time falls 41.06 → 32.69 → 32.04 ms (`ddp` → `flat_ar` → `zero1`): 8.37 ms of the 9.01 ms total difference is already present in the unsharded flat control. Optimizer phases are 10.93 → 1.83 → 1.04 ms; the additional local optimizer saving from sharding is about 0.80 ms. These phase medians support the mechanism but are not an additive causal decomposition of the core median.
- **Equal communication volume does not imply equal latency.** Last-arriving-rank phase medians on NVLink are 1.01 ms for the flat all-reduce, versus 0.65 + 0.53 = 1.18 ms for reduce-scatter + all-gather. On the slow link they are 53.34 versus 29.05 + 29.57 = 58.62 ms, about 9.9% more. The additional 5.28 ms on the slow path is measured, but its cause was not isolated: it cannot be assigned entirely to one extra API/kernel launch. Collective/protocol efficiency and fallback transport/progress costs may contribute. No slow-link profiler trace was saved.
- **Matched total times:** `zero1` is 0.64 ms faster than `flat_ar` at K = 1 on NVLink (32.04 vs 32.69); the three observed repeat ranges do not overlap, although the effect is small and the `zero1` repeats span 28.04–32.25 ms. At K = 4 the median difference is 0.66 ms (121.49 vs 122.15), with overlapping ranges and a fast 106.22 ms `flat_ar` repeat; that difference is unresolved. On the slow link `zero1` is 4.69 ms slower (163.22 vs 158.53), with disjoint repeat ranges.
- **The accumulation boundary trades memory for communication.** On the slow link, `zero1_rs_every` costs 251.94 − 163.22 = 88.73 ms more per step, close to the K − 1 = 3 additional reduce-scatters (3 × 29.05 = 87.15 ms). It saves 52.34 MiB between microbatches. On NVLink, 120.41 vs 121.49 ms has overlapping repeat ranges, so these runs do not resolve the total-time effect.
- **Rank skew can dominate an exposed collective phase.** For NVLink `zero1`, K = 4, reduce max / last is 7.57 / 0.65 ms: about 6.92 ms of rank waiting separates the two statistics. Across the flat NVLink reductions the gaps range from 0.26 to 6.92 ms. The maximum includes waiting and cannot be interpreted as pure link latency.
- **Framework ZeRO has a different cost profile.** `torch_zero` is slower in median than DDP at K = 1 (46.70 vs 41.06 ms; overlapping repeat ranges) and K = 4 (143.97 vs 137.21 ms; disjoint ranges). At K = 4 its optimizer phase is 21.83 ms versus DDP's 14.37 ms. The trace has 78 parameter broadcasts plus DDP's buffer broadcast; rank 1 owns 64 of the 78 tensors. This supports the extra synchronization/per-tensor launch cost, but does not isolate every contribution.
- **DDP's overlapped layout is consistent with lower slow-link core time:** 156.54 ms versus `flat_ar`'s 158.53 ms. Their two-repeat ranges are disjoint. The implementations also differ in optimizer and gradient layout, so these totals alone do not quantify the benefit of overlap.

### 8.5 What it does not establish

- NVLink K = 4 `zero1` vs `flat_ar`, and `zero1_rs_every` vs `zero1`: overlapping repeat ranges. K = 1's small observed advantage has only three repeats and does not establish a general speed-up.
- The fast NVLink repeats (`flat_ar`, K = 4: 106.22 ms; `zero1`, K = 1: 28.04 ms) are retained in the ranges. Shared-host scheduling, CPU launch and GPU frequency effects were not separately measured.
- DDP's optimizer phase varies 10.39–14.37 ms across stages and K; `torch_zero`, K = 4 reaches 21.83 ms. Causes were not isolated. Phase medians need not sum to core medians, because ranks, steps and repeats are aggregated separately.
- The 64.3 MiB residual above the flat memory ledger, the extra 7.31 MiB in the DDP-based rows and their +1.00 MiB first-step allocation excess were not independently isolated in this run. Allocator rounding and cuBLAS workspaces are interpretations, not separate measurements here.
- **[not tested]** bucketed or overlapped reduce-scatter / all-gather (`overlap_with_ddp`, FSDP2 prefetch), mixed precision with an fp32 master slice, gradient clipping on slices, a non-zero pad, W ≥ 3, sharded checkpoint save/restore. Slow-link costs are host-specific: 53.34 ms per 104.7 MiB flat all-reduce in this run; do not substitute Day 04's different-host timings.

### 8.6 Communication diagnosis: rank skew versus deliberately disabled P2P

Follow-up on 2026-11-03: the tens-of-ms slow-link figures must not be compared with normal NVLink kernel latency. `compare_zero1.py` sets `NCCL_P2P_DISABLE=1` only for slow-link workers; those workers' saved environment records confirm it. This disables CUDA GPU-to-GPU P2P transport (NVLink or PCIe P2P), not merely NVLink while retaining PCIe P2P. NCCL documents host-memory SHM as a fallback when P2P is unavailable; the exact selected fallback route was not established by this archive. [NVIDIA environment reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-disable).

K = 4, host-phase summary medians, ms:

| Mode | Operation | Max over ranks | Min over ranks |
| --- | --- | --- | --- |
| Normal NVLink | `flat_ar` all-reduce | 2.02 | 1.01 |
| Normal NVLink | `zero1` reduce-scatter | 7.57 | 0.65 |
| P2P disabled | `flat_ar` all-reduce | 53.73 | 53.34 |
| P2P disabled | `zero1` reduce-scatter | 29.70 | 29.05 |
| P2P disabled | `zero1` parameter all-gather (+ scheduler) | 29.65 | 29.57 |

Both timing repeats retain high min-rank slow-link durations: flat all-reduce 53.03/53.65 ms; reduce-scatter 28.80/29.30 ms; all-gather 29.55/29.60 ms. Thus arrival skew alone does not explain the slow-link phase scale. These are synchronized host wall-time phases, not isolated GPU transfer measurements; fallback transport, host progress and other overhead remain inside them.

The normal-link **count-stage profiler** does directly show the Day 03-style skew. Its two traces share the same `baseTimeNanoseconds`. For `zero1`'s first profiled reduce-scatter, rank 0's kernel lasts 36.866 ms, rank 1 starts 36.482 ms later and lasts 0.386 ms; kernel end times differ by only 1.60 microseconds. The second has 57.801/0.387 ms kernel durations, a 57.416 ms start gap, and end times within 1.89 microseconds. Normal `flat_ar` likewise has 21.644/0.706 ms for its first all-reduce. These profiled samples diagnose waiting; they are not the unprofiled timing-stage medians, and the profiler can perturb arrival skew.

Therefore: normal-link NCCL work on the later-arriving rank is still sub-ms in these samples; the earlier rank can spend tens of ms waiting inside the same NCCL kernel. Separately, disabling P2P produces tens-of-ms host communication phases even on the less-waiting rank. **The extra slow-link 58.62 − 53.34 = 5.28 ms cannot be attributed solely to an extra launch from the available data.** There are no slow-link GPU traces in this run, so its transport/launch/progress breakdown remains unresolved. No new GPU run was performed for this analysis (the later follow-up run is §8.7).

Derived evidence: [`communication_diagnosis.json`](../outputs/day06_downloads/a100_20261102T124935Z/communication_diagnosis.json), computed from the downloaded per-repeat metrics and both normal count traces.

### 8.7 Follow-up: alignment experiment (run `a100_sync_20261103T120618Z`, different host)

A separate diagnostic run tested whether synchronizing the ranks removes the waiting from the measured collective. The tables above were not re-measured. Host: Vast.ai `ssh2.vast.ai:12197`, 2 × A100-SXM4-40GB, NV12, both GPUs on NUMA node 0; torch 2.14.1+cu130, runtime NCCL 2.30.7. `flat_ar` and `zero1`, medium, K = 4; CPU Gloo barrier after a local synchronize; 64 jobs. Full report: [`COMM_SYNC.md`](COMM_SYNC.md).

- [measured] **Synchronizing after the previous iteration does not remove the waiting.** Iteration starts were already aligned to a median 0.05–0.18 ms without any barrier: the previous blocking collective plus local synchronize does that. Skew is created during forward/backward: the arrival gap at the collective equals the per-rank forward/backward difference to within 0.05 ms (median). With `prev_step`, per-job maximum arrival gaps stay 3–19 ms unprofiled, and paired-trace kernel start gaps are −6.7 / 1.3 / 37.6 / −9.8 ms.
- [measured] **A barrier immediately before the collective moves the waiting out of NCCL, not out of the iteration.** Kernel start gaps become −0.08 to +0.14 ms and both ranks' kernels last the same: normal all-reduce 0.65 / 0.67 ms, reduce-scatter 0.57 / 0.43, all-gather 0.40 / 0.40; disabled P2P 37.2 / 37.1, 18.5 / 18.5, 18.5 / 18.6 ms. The early rank's wait appears in the barrier, and loop time per step is unchanged within the repeat ranges. A 10 ms delay injected on rank 1 shows up in rank 0's NCCL kernel (10.93 vs 0.44 ms) or, with immediate alignment, in rank 0's barrier (median 8.62 ms). The iteration costs ~10 ms more in every mode.
- [measured] **Disabled P2P route:** `SHM/direct`, 4 channels, Ring/Simple (normal: `P2P/CUMEM/read`, 24 channels, Ring/Simple). With P2P disabled, rank 0's forward/backward was ~9 ms slower in 17 of 18 jobs; cause not isolated.
- [measured] **The slow-link RS + AG excess is not stable on that host.** Aligned traces: all-reduce 37.2 ms vs 36.9–37.1 ms for reduce-scatter + all-gather. Aligned host phases: +3.5 ms. Communication-only benchmark: +6.9 ms, with every operation drifting 25–35% within 30 s. This host did not reproduce a stable excess. That does not explain away §8.6: on the original host the excess was about 5.3 ms in both repeats (all-reduce 53.03 / 53.65 ms, reduce-scatter 28.80 / 29.30, all-gather 29.55 / 29.60), so the 5.28 ms remains a measured, host-specific result whose cause is unresolved, not shown to be one extra launch.
- [measured] Synchronization did not change results: tiny-run final parameters and gathered AdamW state are bitwise identical to the unmodified `train_zero1.py` in all three modes.

## 9. Completion assessment

The manual run meets the experiment criteria: ZeRO-1 matches unsharded AdamW bitwise (§5), state per rank is measured both from tensor bytes and first-step allocation (§6.1), and all count/timing stages pass. The communication follow-up further distinguishes rank waiting, profiling perturbation and disabled-P2P transport (§8.7). The assisted design, implementation-reading and core source walkthrough is now complete. No further server run is required to finish this walkthrough.

Independent practice still has two unrecorded items: reimplement the YOU WRITE sections without copying the reference, and write or give a brief unaided explanation of state ownership, the update window and matched performance controls. Pre-run predictions were not fully recorded before the run and must not be backfilled as historical predictions. The extensions below remain coverage limits or later-day topics, not prerequisites for starting Day 7.

| Gap | Kind | Where it gets resolved |
| --- | --- | --- |
| Padding path (Ψ is even in both presets, so the zero pad was never exercised) | correctness coverage | only if a later model has odd Ψ at W = 2; a W = 3 run is not possible on two GPUs |
| Gradient clipping under sharding (global norm from slices) | new mechanism | Day 9 |
| Mixed precision (bf16 parameters, fp32 master slice) | new mechanism | Day 9; DeepSpeed's version on Day 8 |
| Overlapping the reduce-scatter / all-gather with compute | performance | FSDP2 (Day 7), `overlap_with_ddp` (read only) |
| Sharded checkpoint and restart on a different W | new mechanism | not planned; note only |
| W ≥ 3 numerics | correctness | not possible on this host |

## Open questions

- [ ] Day 7: with `set_requires_gradient_sync(False)`, does FSDP2 keep unsharded gradients between microbatches (the `zero1` row) or reduce-scatter and accumulate (the `rs_every` row)? Measure the between-microbatch memory.
- [ ] Day 8: in DeepSpeed stage 1 vs stage 2, where are gradients reduced during accumulation, and is it a reduce-scatter or an all-reduce? Where are the updated fp32 partitions all-gathered into the bf16 parameters?
- [ ] Day 9: clip the accumulated gradient under ZeRO-1: one scalar all-reduce of the slice squared-norm sums, then a square root. Planted bug: clipping by the local slice norm.
