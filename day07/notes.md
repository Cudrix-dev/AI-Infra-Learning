# Day 07: FSDP2, `fully_shard` on the whole model vs per block

**Measured run: a100_manual_20261104_01, 2026-11-04.** Ran on Vast.ai with 2 × A100-SXM4-40GB over NV12, both GPUs on NUMA node 1. Python 3.12.15, PyTorch 2.14.1+cu130, CUDA 13.0, NCCL 2.30.7, `OMP_NUM_THREADS=1`, `condition=shared`.

The run passed 101 training runs, 38 trace files, 5 memory snapshots, 4 summary checks and 3 negative controls. The archive SHA256 and 322 extracted file hashes matched. Raw archive and full artifacts are in Windows Downloads under `day07_a100_manual_20261104_01`. Workspace evidence: [environment](../outputs/day07_downloads/a100_manual_20261104_01/environment.log), [parity](../outputs/day07_downloads/a100_manual_20261104_01/parity.log), [count](../outputs/day07_downloads/a100_manual_20261104_01/count.log), [memory](../outputs/day07_downloads/a100_manual_20261104_01/memory.log), [timing](../outputs/day07_downloads/a100_manual_20261104_01/timing.log), [slow link](../outputs/day07_downloads/a100_manual_20261104_01/slowlink.log), [output checks](../outputs/day07_downloads/a100_manual_20261104_01/verify_outputs.log).

Evidence labels:

- **[measured]** from this run's logs and summary JSONs.
- **[understanding]** mechanism or interpretation; not necessarily isolated by the run.
- **[not tested]** outside this experiment's coverage.

## 1. What systems problem did I solve today?

After Day 06, every rank still held the full parameters (P), the full gradients (P) and DDP's buckets (P). Only the AdamW state was sharded. FSDP2 shards all of it. Each rank keeps a `Shard(0)` DTensor slice of every parameter, receives only its slice of every gradient, and plain `AdamW` over those DTensors keeps only its slice of the state. Persistent model state per rank drops from 4P to 1.5P at W = 2.

To compute, FSDP2 rebuilds parameters just in time, one *group* per `fully_shard` call: an all-gather before the group's forward and again before its backward if the group released its full parameters, then a reduce-scatter when gradient synchronization is enabled. I compared two group layouts on Day 01's model inside Day 04's accumulation loop:
- **`fully_shard(model)`:** one group for all 78 tensors.
- **`fully_shard(block)` ×6 + `fully_shard(model)`:** seven groups.

I also ran two variants of the per-block layout: `reshard_after_forward=False` and explicit forward prefetch. I compared all of them with DDP using the same AdamW.

[measured] With gradient sync deferred to the window boundary (`set_requires_gradient_sync(False)`), all four FSDP layouts are **bitwise identical** to DDP + AdamW over 6 optimizer steps with accumulation and a short final window. Reduce-scattering every microbatch (FSDP2's default) is close but not bitwise: final parameters are 3.643e-7 apart. Each rank stores 13,721,600 of 27,443,200 parameter elements and 104.69 MiB of AdamW state.

## 2. Mechanism walkthrough [understanding]

Each `fully_shard` call claims parameters that a child group has not already claimed. Wrapping the blocks first gives each one its own communication schedule. The root then claims the embeddings, final normalization, and output head. Logits are activations, and softmax has no parameters. Initial broadcast gives all ranks the same starting tensors for parity. Without it, all-gather can still assemble a consistent model across ranks, but it will combine slices from different initializations. The replica check cannot catch this because FSDP ranks own different slices.

In the standard block layout, a block gathers its full weights before forward and releases them afterward. Its output gradient triggers the backward hook early enough to gather the weights again before computing the input gradient. Each rank computes the entire group's parameter gradient on its local data. Reduce-scatter combines contributions and gives AdamW just the slice it owns. With `nosync`, each rank keeps its complete local gradient until the last microbatch. Full parameters are still released after backward and gathered again as needed.

The optimizer is constructed after wrapping so it refers to the parameter shards that receive gradients. During backward, a prefetch can overlap the next group's gather with the current group's compute if it is issued early enough. Before AdamW reads the gradients, the compute stream waits for the reduction results.

## 3. Call-chain reference (pytorch@v2.14.1)

This table is the reference for Block 3. The per-stop source notes in your own words are still open.

| Item | Answer |
| --- | --- |
| Entry | `fully_shard` (`_fully_shard.py:98`) → `FSDPState.init` (forward pre/post hooks, `_fsdp_state.py:105`), `_init_param_group` (`_fsdp_init.py:433`), class swap `_apply_to_module` (`_fsdp_init.py:404`) |
| Shard | `FSDPParam._init_sharded_param` (`_fsdp_param.py:249`): dim-0 chunk of this rank's own tensor, padded to chunk 0's size, `DTensor(Shard(0))` |
| Forward gather | `FSDPState._pre_forward` (`:290`) → `FSDPParamGroup.pre_forward` (`_fsdp_param_group.py:550`) → `unshard` (`:382`) → `foreach_all_gather` (`_fsdp_collectives.py:325`) → `DefaultAllGather` → `dist.all_gather_single`; `wait_for_unshard` (`:427`) → `foreach_all_gather_copy_out` (`:431`) |
| Free | `post_forward` (`:570`) → `reshard` (`:509`) → `FSDPParam.to_sharded` (`_fsdp_param.py:921`) → `free_storage` (`:1306`, `storage.resize_(0)`) |
| Backward gather | output tensor hook (`_register_pre_backward_hook`, `_fsdp_state.py:440`) → `pre_backward` (`_fsdp_param_group.py:592`) → `unshard` + `_backward_prefetch` (`:821`) |
| Gradient reduction | `RegisterPostBackwardFunction` (`:1141`) on the module inputs → `post_backward` (`:607`) → `foreach_reduce` (`_fsdp_collectives.py:522`) → `chunk_cat` copy-in → `dist.reduce_scatter_single(ReduceOp.AVG)`; sharded grad = view of the output, or `+=` |
| Accumulation | `set_requires_gradient_sync(False)` (`_fully_shard.py:413`) → `reduce_grads = False` → `post_backward` returns before `foreach_reduce`; unsharded `.grad` kept |
| End of backward | `_root_post_backward_final_callback` (`_fsdp_state.py:401`): root group's post-backward (its inputs need no grad), `finalize_backward` waits for the reduce-scatters |
| Divide factor | `_get_gradient_divide_factors` (`_fsdp_collectives.py:812`): FP32/BF16 → `AVG`; custom factor → `PREMUL_SUM(1/f)`; FP16 → pre/post divide |

## 4. My setup vs DDP and Day 06

| Aspect | DDP | Day 06 `zero1` | FSDP2 `fsdp_block` |
| --- | --- | --- | --- |
| Persistent per rank | params P, buckets P, state 2P | params P, flat grad P, state P | param shards P/2, state P |
| Gradient reduction | 4 bucket all-reduces, overlapped with backward | 1 reduce-scatter per window | 1 reduce-scatter per group per microbatch (sync) or per window (nosync) |
| Parameter communication | none (replicated) | 1 all-gather after the optimizer | 1 all-gather per group before forward, 1 more before backward (not the root) |
| Bytes sent per rank per microbatch | P | P per window | ≈ 1.34 P (root not re-gathered) |
| Initial parameters | broadcast by the constructor | broadcast by `FlatZeroAdamW` | explicitly broadcast by `broadcast_module_state` before wrapping |
| Optimizer | AdamW on full tensors | AdamW on one flat slice | AdamW on 78 DTensor shards |

## 5. Evidence it is correct [measured, `parity.log`]

Tiny preset, 6 optimizer steps of G = 32 = 2 ranks × K 4 × m 4, last window G = 16 (windows `[4, 4, 4, 4, 4, 2]`). Reference: 1 process, one batch of G rows per step, Day 01's AdamW. A–C use Day 02 tolerances, D is exact, E uses A's tolerance. R: replicated parameters bitwise across ranks, vacuous under FSDP. S: state gathered from the shards vs DDP + AdamW (≤ 1e-4 × max).

| Variant | A grad | B params-1 | C | D | E | R | S vs DDP | Bitwise = DDP (same AdamW) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ddp` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | 30 tensors, yes | (reference) | — |
| `root_nosync`, `block_nosync`, `nrf_nosync`, `pf_nosync` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | vacuous | 0 | **yes**, all five quantities |
| `root_sync`, `block_sync` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.382e-9 | vacuous | 4.66e-10 / 7.1e-14 | no (final params 3.643e-7) |
| `block_nosync_foreach` vs `ddp_foreach` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.382e-9 | vacuous | — | **yes** |

| Negative control (on `block_nosync`) | Fails | Passes | Verdict |
| --- | --- | --- | --- |
| `no_init_bcast` | A (1.4e-2), B (5.07), C (6.6e-3), E, S | D, R | detected by C |
| `sync_never` | A (recorded gradient 0, missing shard grad), B (1.57e-5: no update at all, not even weight decay), D (adam step 0), E (ratio 0.000), S (no state) | C, R | detected by A |
| `sum_grads` | A, E (ratio **2.000**), S | **B** (2.6e-6), C, D, R | detected by A |

[understanding] Two of these are worth remembering:
- **`sum_grads` passes B.** Adam's first update is lr · m̂/(√v̂ + ε) ≈ lr · sign(g), so doubling g leaves it almost unchanged. A parameter check alone would miss a reduction-scale bug.
- **R passes in every control.** FSDP has no replicas that could drift apart.

## 6. Memory [measured]

### 6.1 Per rank, medium (Ψ = 27,443,200, P = 104.69 MiB)

| Variant | Elements stored | `exp_avg` + `exp_avg_sq` | After wrap | Across first step |
| --- | --- | --- | --- | --- |
| `ddp` | 27,443,200 (78 plain tensors) | 209.38 | 210.4 | 210.38 |
| `fsdp_root` | 13,721,600 (78 DTensors) | 104.69 | 53.9 | 110.19 |
| `fsdp_block`, `block_nrf`, `block_pf` | 13,721,600 (78 DTensors) | 104.69 | 52.9 | 111.19 |

MiB. After wrap, DDP holds P of parameters plus P of buckets, and FSDP holds the P/2 shards. The full parameters from `build()` are gone because the old optimizer was deleted first. FSDP's first-step allocation exceeds its state tensors by 5.5–6.5 MiB. That excess was not isolated.

### 6.2 Memory timeline: microbatch 0, K = 1, m = 8, rank 0 (`memory.log`)

| Variant | Start of step | Forward peak | Backward peak | Optimizer peak |
| --- | --- | --- | --- | --- |
| `ddp` | 484.9 | 1053.1 | **1117.1** | 703.6 |
| `fsdp_root` | 228.2 | 947.7 | **971.1** | 335.8 |
| `fsdp_block` | 228.2 | 828.9 | **905.0** | 335.8 |
| `fsdp_block_nrf` | 228.2 | 906.1 | **970.1** | 334.8 |
| `fsdp_block_pf` | 228.2 | 828.9 | **905.0** | 335.8 |

What the timeline lines show:
- **Start of step:** 484.9 − 228.2 = 256.7 MiB, against a 2.5P = 261.7 MiB ledger difference.
- **Saved activations:** +72 MiB per block forward in DDP. FSDP block forward adds only +59 MiB: the 72 MiB of activations minus the block's resharded 12 MiB.
- **`fsdp_root` holds 2P in forward.** At `fwd_pre model` it jumps by +215.4 MiB, the all-gather buffer plus the unsharded copy. At `fwd_post model` it is 36.6 MiB *lower* than at `fwd_post blocks.5`, while DDP adds 68 MiB there for the logits. 68 − 104.7 = −36.7: the all-gather buffer freed at the root's post-forward.
- **`bwd_done`:** 228.2 + 52.3 = start + P/2 of sharded gradients.

**Why the all-gather buffer survives copy-out [understanding].** The default FSDP2 path packs the local parameter shards into one contiguous input. The input is a view into this rank's slice of the group's all-gather output buffer, which FSDP2 allocates and passes to NCCL.

```text
local parameter shard data → copy-in to the rank-local view of the gather buffer
                          → all-gather into the full buffer
                          → copy-out to individual complete parameters
```

With implicit forward prefetch, group B's copy-in can run on a separate stream while group A's all-gather is still running. Because B's copy-in writes into B's gather buffer, FSDP2 keeps a reference to A's buffer so that the allocator gives B different storage. A's copy-out also reads A's buffer. The next group clears the reference after scheduling its own copy-in, at the start of `wait_for_unshard()`, with CUDA event ordering for A's copy-out. The CPU can enqueue that work without blocking until B's copy-in finishes. The dependency is about when the storage can be safely reused.

In `fsdp_root` there is no next group to take that cleanup path. The root's post-forward hook clears the last reference before backward, leaving only the complete weights for backward. This explains the roughly P-sized buffer alongside the complete weights during forward. See the installed source snapshot, `_fsdp_collectives.py` (`all_gather_copy_in`), `_fsdp_param_group.py` (`Note: Overlapping all-gather copy-in and all-gather`, `wait_for_unshard`), and `_fsdp_state.py` (`_post_forward`).

### 6.3 Between microbatches, K = 4 (timing stage)

| `ddp` | `root_nosync` | `block_sync` | `block_nosync` | `nrf_sync` |
| --- | --- | --- | --- | --- |
| 595.1 | 339.0 | 280.7 | 339.0 | 280.7 |

- **DDP vs `block_sync`:** 595.1 − 280.7 = 314.4 MiB, against a 5P − 2P = 3P = 314.1 ledger.
- **Answer to Day 06's open question:** `block_nosync` − `block_sync` = **58.3 MiB**, against P − P/2 = 52.3. With `set_requires_gradient_sync(False)`, FSDP2 keeps the **unsharded** gradient between microbatches, which is Day 06's `zero1` row. The 6 MiB excess was not isolated.
- **Peaks at K = 4:** `ddp` 1227, `root_nosync` 1080, `block_sync` 957, `block_nosync` 1016, `nrf_sync` 1023 MiB.
- **Peaks at K = 1:** at m = 32, 3014 (`ddp`) vs 2786 MiB (`fsdp_block`). The same ~210–230 MiB saving is 19% of the peak at m = 8 but 7.6% at m = 32.

## 7. Collectives [measured, `count.log`; medium, rank-0 trace, 2 profiled steps]

| K | Variant | AllGather / step | ReduceScatter / step | AllReduce / step |
| --- | --- | --- | --- | --- |
| 4 | `ddp` | 0 | 0 | 4 (+1 broadcast) |
| 4 | `root_sync` / `root_nosync` | 4 / 4 | 4 / 1 | 0 |
| 4 | `block_sync` / `block_nosync` | 52 / 52 | 28 / 7 | 0 |
| 4 | `nrf_sync` / `nrf_nosync` | 28 / 28 | 28 / 7 | 0 |
| 1 | `root_sync` | 1 | 1 | 0 |
| 1 | `block_sync`, `pf_sync` | 13 | 7 | 0 |
| 1 | `nrf_sync` | 7 | 7 | 0 |

All counts match Guide §1.3: For the standard block layout, 7 forward + 6 backward all-gathers (the root is not re-gathered) and 7 reduce-scatters per synchronized microbatch. `nosync` cuts only the reduce-scatters.

**Where they are** (`block_sync`, K = 1, m = 8):
- **Forward:** 7 all-gathers on the all-gather stream, launched from `FSDP::all_gather (root)`, then `(blocks.0)` … `(blocks.5)`.
- **Backward:** 6 re-gathers (blocks.5 → blocks.0) interleaved with 7 `FSDP::post_backward_reduce` reduce-scatters on a separate stream.
- **Last:** the root's reduce-scatter, from the final callback.

**Overlap** (`block_sync`, K = 1, rank-0 trace). Overlap ranges refer to the first profiled step. Step-level times average two profiled steps.

| | m = 8 | m = 32 |
| --- | --- | --- |
| Forward all-gather overlap, blocks 1–5 | 70–89% | 96–100% |
| Backward all-gather overlap, all 6 blocks | 0–100% | 97–100% |
| Reduce-scatter overlap, six blocks | 0% | 85–92% |
| Root reduce-scatter overlap | 0% | 0% |
| Exposed communication | 9.74 ms | 1.57 ms |
| Sum of min-rank kernel durations | 2.21 ms | 2.22 ms |
| GPU idle / step span | 22.21 / 67.45 ms | 6.88 / 101.24 ms |

The forward root and block 0 gathers had 0% overlap in both cases. At m = 8, the first two backward gathers overlapped substantially, while the last four and all reduce-scatters did not in the displayed step. This is consistent with too little compute available when those collectives were issued. The run did not measure CPU hook time directly. The similar min-rank sums at m = 8 and 32 help distinguish communication cost from changes in overlap.

Exposed communication is rank-0 NCCL kernel time without compute overlap, including time spent waiting for the other rank. At m = 8, 9.74 ms is the average across two profiled steps. Step 0 alone had 2.26 ms of total rank-0 communication and 1.54 ms exposed. The 2.21 ms min-rank sum in the table averages the per-collective shorter durations across both steps. It is a proxy for communication with less arrival waiting and may still include waiting or contention.

## 8. Timing [measured]

### 8.1 Method

Medium preset, 25 optimizer steps with 5 excluded, NVLink 3 repeats, slow link 2. TF32 off, MATH attention, `condition=shared`, `AdamW(foreach=True)` for every variant except the two `per-t` rows. Core time: slowest rank per step, median over steps, [min–max] over repeats. FSDP's communication is inside fwd+bwd.

### 8.2 NVLink

| K | m | Variant | ms / opt step [min–max] | fwd+bwd | optimizer | vs `ddp` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | `ddp` | 36.58 [36.33–37.03] | 33.34 | 3.53 | — |
| 1 | 8 | `fsdp_root` | 38.12 [37.99–38.46] | 33.18 | 4.91 | +1.54 |
| 1 | 8 | `fsdp_block` | 51.17 [50.96–51.22] | 46.30 | 4.83 | +14.59 |
| 1 | 8 | `fsdp_block_nrf` | 45.23 [45.11–46.84] | 40.45 | 4.80 | +8.65 |
| 1 | 8 | `fsdp_block_pf` | 51.33 [51.19–51.48] | 46.45 | 4.84 | +14.75 |
| 1 | 32 | `ddp` | 94.55 [94.50–96.30] | 91.99 | 2.52 | — |
| 1 | 32 | `fsdp_root` | 95.96 [95.87–96.08] | 92.79 | 3.16 | +1.41 |
| 1 | 32 | `fsdp_block` | 97.76 [97.73–97.83] | 94.56 | 3.23 | +3.21 |
| 1 | 32 | `fsdp_block_nrf` | 96.76 [96.68–97.77] | 93.52 | 3.23 | +2.21 |
| 1 | 32 | `fsdp_block_pf` | 98.36 [98.05–99.09] | 95.30 | 3.41 | +3.81 |
| 4 | 8 | `ddp` | 126.62 [126.12–129.60] | 123.07 | 3.52 | — |
| 4 | 8 | `root_nosync` | 146.61 [146.38–147.15] | 140.36 | 6.25 | +19.99 |
| 4 | 8 | `block_sync` | 216.71 [216.48–220.07] | 213.48 | 6.32 | +90.09 |
| 4 | 8 | `block_nosync` | 202.05 [200.25–208.75] | 198.69 | 6.44 | +75.44 |
| 4 | 8 | `nrf_sync` | 192.90 [183.35–219.47] | 189.76 | 6.30 | +66.28 |
| 4 | 8 | `pf_sync` | 217.49 [216.56–227.47] | 213.29 | 6.62 | +90.87 |
| 1 | 8 | `ddp`, per-tensor AdamW | 41.42 [41.18–42.65] | 30.51 | 10.95 | — |
| 1 | 8 | `fsdp_block`, per-tensor AdamW | 72.99 [64.74–73.18] | 47.92 | 25.03 | +31.57 |

### 8.3 Slow link (`NCCL_P2P_DISABLE=1`), K = 1, m = 8

| Variant | ms / opt step [min–max] | − NVLink | exposed comm (profiled) |
| --- | --- | --- | --- |
| `ddp` | 72.12 [72.01–72.24] | +35.54 | 17.36 |
| `fsdp_root` | 88.41 [88.32–88.50] | +50.28 | 58.43 |
| `fsdp_block` | 100.73 [100.43–101.02] | +49.55 | 27.50 |
| `fsdp_block_nrf` | 83.26 [83.23–83.29] | +38.03 | — |
| `fsdp_block_pf` | 100.71 [100.54–100.88] | +49.38 | 27.80 |

### 8.4 What it supports

- **NVLink, K = 1.** At m = 8, standard per-block FSDP took 51.17 ms vs 36.58 ms for DDP, a 14.59 ms gap. In the separate profiled run, the sum of shorter-rank NCCL durations was 2.21 ms and rank 0 spent 22.21 ms with no GPU kernels running. These measurements support a launch and scheduling explanation beyond transfer time. They come from different runs, so they cannot be subtracted to estimate CPU time. At m = 32, the step-time gap fell to 3.21 ms and more backward collectives overlapped compute.
- **Same byte volume, different grouping [derived].** Root-only and `block_nrf` each gather the model once and reduce-scatter its gradients once in total byte volume at K = 1. Root uses one group and costs 1.54 ms more than DDP at m = 8. `block_nrf` uses seven groups and costs 8.65 ms more. Their 7.11 ms difference is consistent with grouping and scheduling costs, though their communication schedules and memory lifetimes also differ.
- **Time saved per removed collective [derived].** At K = 1, m = 8, `block_nrf` removed six backward all-gathers and saved 5.94 ms versus `block_sync`, about 0.99 ms per removed call. The shorter-rank kernel durations for those gathers were around 0.1 ms each. At K = 4, `block_nosync` removed 21 reduce-scatters and saved 14.66 ms, about 0.70 ms per removed call. These are ratios of differences between layout or accumulation variants. They include any changes in hooks, buffering, waiting and overlap, and do not isolate the cost of an individual collective.
- **Explicit forward prefetch.** At m = 8 the median changed from 51.17 to 51.33 ms, with overlapping repeat ranges. The run does not show a reliable speedup from this variant.
- **Gradient accumulation.** At K = 4, `block_nosync` retained 58.3 MiB more per rank between microbatches than `block_sync`. `nrf_sync` removed 24 all-gathers and reached 192.90 ms, with a wide repeat range.
- **Slow link.** Standard per-block FSDP had 27.50 ms of exposed communication vs 58.43 ms for root-only, but still took 100.73 ms per unprofiled step vs 88.41 ms for root-only. `block_nrf` has the same modeled gradient and parameter traffic volume as DDP at K = 1, and its smaller groups allow overlap. It was the fastest measured FSDP layout on the slow link at 83.26 ms. Its slowdown from NVLink was 38.03 ms, close to DDP's 35.54 ms. The run did not profile `block_nrf` on the slow link to quantify its overlap.
- **Optimizer overhead.** At K = 1, m = 8, per-tensor AdamW took 25.03 ms in standard per-block FSDP and 10.95 ms in DDP. Foreach reduced these phases to 4.83 and 3.53 ms.

### 8.5 Limits

- CPU time per hook and the cause of timing outliers on the shared host were not isolated. Profiled times are used to inspect overlap, while the benchmark numbers come from the unprofiled runs.
- At m = 8, the root-only gap to DDP ranged from roughly 1.5 to 8 ms across the personal run and two pre-tests. Treat gaps below roughly 4 ms cautiously. The pre-tests provide context for run-to-run variability, not a calibrated noise bound.
- Evidence links into `outputs/` resolve in the local workspace. The directory is git-ignored, and the full archive is stored separately in Downloads.
- **[not tested]** `set_reshard_after_backward(False)`, deeper prefetch, mixed precision, CPU offload, HSDP, activation checkpointing, `torch.compile`, W ≥ 3.

## 9. Completion assessment

The experiment and measured write-up are complete. The personal run passed parity, detected all three controls, and captured collective counts, memory timelines and timings on both links. The guided walkthrough covered the wrapping order, parameter and gradient lifetimes, and the link between the hooks and the trace. Guide §1.8 now records a retrospective derivation and comparison with the run. The pre-run prediction checkbox remains open, along with personal Block 3 source notes, independent reimplementation and diffing of the `YOU WRITE` sections, and an unaided final explanation.

| Gap | Kind | Where it gets resolved |
| --- | --- | --- |
| `set_reshard_after_backward(False)` during accumulation (saves the forward all-gathers of microbatches 1..K−1) | performance knob | not run; read only |
| Mixed precision (`MixedPrecisionPolicy`: BF16 params, FP32 reduce) | new mechanism | Day 9 |
| Activation checkpointing composed with FSDP (re-gather during recompute: `_pre_forward`'s `PRE_BACKWARD` branch) | new mechanism | Day 9 |
| Sharded checkpoint (DCP / `get_model_state_dict`) | new mechanism | Day 16 (gather FSDP2 shards for weight sync) |
| Uneven dim-0 (padding path) | correctness coverage | not exercised: every dim 0 is even |
| HSDP / W ≥ 3 | coverage | not possible on this host |

## Open questions

- [ ] Day 8: does DeepSpeed stage 3 gather per module like FSDP2, and does it prefetch by a byte budget instead of one group ahead?
- [ ] Day 9: clipping under FSDP2: `clip_grad_norm_` on DTensor gradients computes the global norm (a reduction over shards); verify it against DDP.
- [ ] Day 16: gathering FSDP2 shards into a full state dict for the inference engine: `full_tensor()` per parameter vs `get_model_state_dict(full_state_dict=True)`.
