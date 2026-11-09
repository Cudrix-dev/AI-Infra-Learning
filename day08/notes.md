# Day 08: DeepSpeed ZeRO stages 0, 1, 2, 3 on the same model

> **Scope (plan change 2026-11-08, Option A).** Day 08 is complete. Sections 1–4 document the three required observations from `a100_core_20261108T090011Z`. Sections B1–B9 are **optional** follow-up (full walkthrough, predictions, timing and slow-link analysis). Numbers marked **[pre-test]** come from the assistant's run `pretest2`, not from yours.

Evidence labels:

- **[measured]** from your run's logs.
- **[pre-test]** from the assistant's `pretest2` logs; quote only for comparison.
- **[understanding]** mechanism or interpretation; not necessarily isolated by the run.
- **[not tested]** outside this experiment's coverage.

## 1. Run record (required)

| Item | Your value |
| --- | --- |
| Run ID | `a100_core_20261108T090011Z` |
| Date, host, NUMA placement | 2026-11-08 SGT, container `7690aac0bd70`, 2 × A100-SXM4-40GB, NV12, NUMA node 1 ([environment](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/environment.log)) |
| Software | Python 3.12, PyTorch 2.14.1+cu130, NCCL 2.30.7, DeepSpeed 0.19.9 (`DS_BUILD_OPS=0`), `condition=shared` |
| Stages finished | parity, count and memory ([verification](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/verify_core.log)) |
| `parity` verdict line | `PARITY: ALL CHECKS PASSED`; both controls detected |
| Archive SHA256 (if `run_server_once.sh` finished) | [`d86d83c8ed62be7edbc22375b6b0cd3e9d759d5b05b762f7b0da667b9fb362c7`](../outputs/day08_downloads/a100_core_20261108T090011Z/day08_a100_core_20261108T090011Z.tar.gz.sha256) |

The core run verified 27 child runs, 22 traces, 8 memory snapshots and 2 JSON summaries.

## 2. Observation 1: the collective behind ZeRO-1/2 gradient buckets (required)

**Evidence** (`count` log, table 1, K = 4, m = 8). Copy your rows:

| Variant | AllReduce | ReduceScatter | AllGather | bucket reductions per step (AllReduce − 1 scalar gradient-norm all-reduce) |
| --- | --- | --- | --- | --- |
| `z1` | 6 | 0 | 1 | 5 |
| `z2` | 21 | 0 | 1 | 20 |

Where they are launched: the K = 1 `SEQUENCE` blocks for `z1` and `z2` in the same log (the log prints sequences only at K = 1). Note the `launched inside` column for the bucket all-reduces: `## bwd 0` (main thread, boundary epilogue) or `bwd thread` (autograd hooks during backward).

**Answer these (3–6 sentences):**

- Which NCCL kernel reduces the gradient buckets, although the config's `reduce_scatter` is `true`? Which default makes it so, and what does each rank keep afterwards?
- How many bucket reductions does each stage issue per optimizer step at K = 4, and why: in which microbatch(es) does `z1` reduce, and in which does `z2`?
- Bytes per rank per step at W = 2, in units of P: `z1` = ___, `z2` = ___. Why does Day 06's hand-written ZeRO-1 send less?

**Observation [measured counts; bytes derived].** The K = 4 table shows 6 versus 21 AllReduces and zero ReduceScatters, although the config's `reduce_scatter` is `true`. With `use_multi_rank_bucket_allreduce=True`, each bucket is all-reduced and each rank retains its own slice. Excluding one scalar norm AllReduce, ZeRO-1 reduces five buckets at the window boundary, while ZeRO-2 reduces five on each of four microbatches. The K = 1 sequences show where: all five `z1` bucket all-reduces are launched from the main thread in `## bwd 0` after backward compute, while four of the five `z2` all-reduces come from autograd's thread during backward ([sequences](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/count.log#L191)). [derived] Including the post-update parameter AllGather, the two-rank ring estimates are 1.5P and 4.5P per rank per step. Day 06 used a true ReduceScatter for the gradients and sent P total per rank. [Measured count log](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/count.log#L169)

## 3. Observation 2: why ZeRO-3 starts each step above ZeRO-2 (required)

**Evidence** (`memory` log, K = 1, m = 8, rank 0, MiB):

| | `z2` | `z3` | `z3` − `z2` |
| --- | --- | --- | --- |
| start of step | 325.8 | 351.0 | +25.2 |

`z3` census at the start of a step (`memory` log, census block; MiB):

| Buffer (DeepSpeed's name) | `z2` (ledger) | `z3` (census) | difference |
| --- | --- | --- | --- |
| full parameters (`p.data`, flat buffer) vs `p.ds_tensor` partitions + persistent params | P = 104.7 | 52.3 + 0.2 | ≈ −52.2 |
| fp32 master partition | P/2 = 52.3 | 52.3 | 0 |
| AdamW state | P = 104.7 | 104.7 | 0 |
| `grad_partitions_flat_buffer` | — (z2 frees its partition after the step) | 52.3 | +52.3 |
| `ipg_buckets` | — (allocated in backward only) | 25.0 | +25.0 |
| unnamed constant | 64.1 | 64.1 | 0 |
| **measured total** | **325.8** | **351.0** | **+25.2** |

**Answer (3–6 sentences):** which buffers stage 3 keeps for the whole run that stage 2 allocates only during backward, or not at all; why the parameter saving (P/2) is cancelled in FP32; and whether the answer changes with DeepSpeed's default buckets (`z3_def` row).

**Observation [measured].** ZeRO-3 starts at 351.0 MiB against ZeRO-2 at 325.8 MiB. Sharding the model weights saves about half a model, and the 52.3 MiB persistent gradient partition consumes roughly the same space. Both stages have the same master and AdamW allocations, so the 25 MiB persistent bucket and about 0.2 MiB of persistent parameters account for the measured gap, with displayed components rounded. ZeRO-2 also needs a gradient partition and a bucket, but it allocates the bucket only during backward and frees its partition after the step, so neither appears at the start of a step. [derived] The cancellation is exact only at W = 2: in general z3 − z2 = B − P(1 − 2/W), so at W ≥ 3 the parameter saving P(1 − 1/W) would exceed the P/W gradient buffer by more than B on this model, and stage 3 would start lower (not testable on this host). The default bucket is 1907.3 MiB and raises `z3_def` to 2234.0 MiB at step start. [Measured census](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/memory.log#L61) · [Measured memory table](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/memory.log#L284)

## 4. Observation 3: one gradient-accumulation pitfall, shown by a negative control (required)

Pick **one** control: `double_scale` or `fixed_gas`.

**Evidence** (`parity` log, controls block). Copy your row:

| Control (on `z2`) | Checks failed (value) | Checks passed | Verdict |
| --- | --- | --- | --- |
| `fixed_gas` | D (Adam step 5 vs 6), E (gradient ratio 0.000), S | A, B, C, R | `DETECTED by check D` |

**Answer (3–6 sentences):** what DeepSpeed's default does (`scale_wrt_gas=True` divides by `gradient_accumulation_steps`; or the boundary comes from a micro-step counter), why it breaks Day 04's weighted, variable-length windows, which check caught it and which check it slipped past (and why), and the one-line fix in `ds_microbatch`.

**Observation [measured].** The deliberate `fixed_gas` control lets the engine infer update boundaries from a fixed interval of four microbatches. Its first five windows reach that boundary, so check A passes on the first step. The final window contains only two microbatches, so ZeRO-2 can reduce their gradients without triggering the sixth AdamW update. Check D detects five updates versus six in the reference. Check E's ratio of 0.000 is the harness's convention, not a measured zero gradient: when the engine reports no accumulation boundary, the harness reads the gradient as zeros, so E records that the short window never reached a boundary. The normal path passes the actual end of the window through `engine.set_gradient_accumulation_boundary(last)`. [Measured parity log](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/parity.log#L168)

## 5. Completion (required)

- [x] parity, count and memory stages finished; parity passed with both controls detected
- [x] §1–§4 filled with **[measured]** numbers
- Optional items done (tick any): §1.8 predictions · YOU WRITE · Block 3 notes · timing · slow link

---

## Optional reference: extended write-up (sections B1–B9)

Parity, the K = 1 memory matrix and collective counts below use the personal run. The K = 4 and m = 32 memory comparisons from the timing matrix, and the timing and slow-link results, retain their **[pre-test]** label.

### B1. What systems problem did I solve today?

Days 06 and 07 sharded model state by hand (ZeRO-1) and with PyTorch's FSDP2 (ZeRO-3). Today the same model ran under DeepSpeed, where one config number chooses what is sharded: stage 1 the AdamW state, stage 2 also the gradients, stage 3 also the parameters. I compared stages 0–3 with DDP inside Day 04's accumulation loop, with buckets sized for this 27M-parameter model and with DeepSpeed's defaults.

The plan's done-when has two parts: a memory table per stage, and why a higher stage isn't faster. Both turned out to depend on DeepSpeed's implementation more than on the ZeRO idea:
- in FP32, DeepSpeed keeps an fp32 master partition anyway, and stage 3 keeps a persistent gradient buffer and bucket, so stage 3 started every step with **more** memory than stages 1 and 2 and had a higher peak at K = 1, m = 8; [pre-test, timing stage] it had the lowest peak only at K = 4 and at m = 32;
- stage 1/2's gradient "reduce-scatter" is a bucket **all-reduce** by default, and stage 2 runs it every microbatch, so bytes per step grow with the stage and with K.

[measured] Stages 0 and 1 were **bitwise identical** to DDP + AdamW over 6 optimizer steps with accumulation and a short final window. Stages 2 and 3 were close but not bitwise (final parameters 3.643e-7 and 4.624e-7 apart). Both DeepSpeed-specific accumulation bugs were detected by their named checks.

### B2. Mechanism walkthrough [understanding]

`deepspeed.initialize` broadcasts rank 0's parameters and picks an optimizer class by stage. For stages 1 and 2 it is the same class: the model's parameters become views of one flat buffer, this rank's slice is cloned into an fp32 master partition, and my AdamW's parameter group is rewritten to hold only that partition. AdamW therefore updates one flat tensor per rank, which is why the optimizer can be built before wrapping, the opposite of FSDP2.

Stage 1 keeps the full local gradient until the window boundary, then reduces it once in 25 MiB buckets. Stage 2 cannot keep the full gradient, so its gradient hooks copy each gradient into a bucket during backward and reduce the bucket as soon as it is full, in every microbatch. In both, the default reduction is an all-reduce of the bucket after which each rank keeps the slice it owns. After AdamW, the updated partition is copied into the flat buffer and all-gathered.

Stage 3 replaces each parameter's data by an empty tensor and keeps this rank's partition in `ds_tensor`. Hooks on every submodule gather its parameters just before its forward and backward, prefetching upcoming parameters up to an element budget, from the module order recorded in the first step. A parameter is released after use unless it is reused within `max_reuse_distance` elements. Gradients are reduce-scattered in buckets into a persistent partition buffer. Parameters below `stage3_param_persistence_threshold` stay full on every rank and are all-gathered one by one after each step.

DeepSpeed's own loop divides the loss by `gradient_accumulation_steps` and decides the boundary from a micro-step counter. With my own row weights and a short final window, I passed `scale_wrt_gas=False` and set the boundary explicitly before every forward.

### B3. Call-chain reference (deepspeed 0.19.9)

This table is the reference for Block 3. The per-stop source notes in your own words are still open.

| Item | Answer |
| --- | --- |
| Entry | `deepspeed.initialize` (`__init__.py:433`) → `DeepSpeedEngine` (`runtime/engine.py:509`) → `_broadcast_model` (`:2037`), `_configure_zero_optimizer` (`:2632`): stages 1/2 one class with `partition_grads=(stage == 2)`, stage 3 `DeepSpeedZeroOptimizer_Stage3` |
| Stage 1/2 state | `DeepSpeedZeroOptimizer` (`runtime/zero/stage_1_and_2.py:158`): flat buffer (`:505`–`:524`), fp32 master partition clone (`:560`), `param_group['params'] = [partition]` (`:584`), hooks (`:713`) |
| Accumulation | `engine.backward` (`:3381`; `scale_wrt_gas`, `:3412`) → `_backward_epilogue` (`:3152`) → `allreduce_gradients` (`:3072`): stage ≥ 2 every microbatch, stage 1 `reduce_gradients` (`stage_1_and_2.py:958`) and stage 0 `buffered_allreduce_fallback` (`:3920`) at the boundary; `set_gradient_accumulation_boundary` (`:3466`) |
| Stage 1/2 reduction | `reduce_independent_p_g_buckets_and_remove_grads` (`:1414`) → bucket buffer (`setup_buckets`, `:3256`) → `reduce_ipg_grads` (`:2205`) → `average_tensor` (`:1590`) → `allreduce_and_scatter` (`:1545`) → `all_reduce` (default `use_multi_rank_bucket_allreduce`) |
| Stage 1/2 step | `step` (`:3013`) → `_optimizer_step` (`:2967`) → `_update_model_bit16_weights` (`:889`) → `all_gather_dp_groups` (`runtime/utils.py:1026`) |
| Stage 3 hooks | `DeepSpeedZeRoOffload` (`parameter_offload.py:136`): `mark_persistent_parameters` (`:317`), `_register_deepspeed_module` (`:337`, recursive: every submodule) |
| Stage 3 fetch / release | `fetch_sub_module` (`partitioned_param_coordinator.py:310`) → `_fetch_sub_module_impl` (`:353`, prefetch by `prefetch_bucket_size`) → `__all_gather_params_` (`:582`) → `all_gather_coalesced` (`partition_parameters.py:1540`); `release_sub_module` (`:494`) → `__params_to_release` (`:634`, `max_reuse_distance`) |
| Stage 3 gradients | `reduce_ready_partitions_and_remove_grads` (`stage3.py:2160`) → `__reduce_and_partition_ipg_grads` (`:1648`) → `__avg_scatter_grads` (`:1912`) → `reduce_scatter_coalesced` (`comm/coalesced_collectives.py:81`) → `partition_grads` (`:2006`) |
| Stage 3 step | `step` (`:2873`) → `_post_step` (`:2835`) → persistent `all_gather` → `_allgather_params_coalesced` (`partition_parameters.py:2022`): one `all_gather_into_tensor` per parameter |
| Evidence | `safe_get_full_grad` / `_fp32_param` / `_optimizer_state` (`utils/tensor_fragment.py:208`, `:135`, `:170`): fragment into a zero buffer + all-reduce; collective |

### B4. My setup vs DDP, Day 06 and Day 07

| Aspect | DDP | Day 06 `zero1` | Day 07 FSDP2 block | DeepSpeed stage 1 | DeepSpeed stage 3 |
| --- | --- | --- | --- | --- | --- |
| Persistent per rank (FP32) | params P, buckets P, state 2P | params P, flat grad P, state P | param shards P/2, state P | flat params P, master P/2, state P | partition P/2, master P/2, grad buffer P/2, state P, bucket B |
| Gradient reduction | 4 bucket all-reduces, overlapped with backward | 1 reduce-scatter per window | 1 reduce-scatter per group per microbatch (`sync`) | 5 bucket **all-reduces** at the boundary, not overlapped | bucketed reduce-scatters per microbatch |
| Parameter communication | none | 1 all-gather per window | 1 all-gather per group before forward and before backward | 1 all-gather after the step | coalesced all-gathers per submodule before forward and backward, + 1 per persistent param after the step |
| Bytes sent per rank, K = 1 | P | P | ≈ 1.34P | 1.5P | ≈ 1.5P |
| Initial parameters | DDP broadcast | `FlatZeroAdamW` broadcast | my `broadcast_module_state` | engine `_broadcast_model` | engine `_broadcast_model` |
| Optimizer | AdamW on 78 tensors | AdamW on one flat slice | AdamW on 78 DTensor shards | my AdamW, rewritten to one flat fp32 partition | same |

### B5. Correctness ([measured parity log](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/parity.log))

Tiny preset, 6 optimizer steps of G = 32 = 2 ranks × K 4 × m 4, last window G = 16 (windows `[4, 4, 4, 4, 4, 2]`). Reference: 1 process, one batch of G rows per step, Day 01's AdamW. A–C use Day 02 tolerances, D is exact, E uses A's tolerance. R: parameters held in full on every rank, bitwise across ranks. S: state gathered with `safe_get_full_optimizer_state` vs DDP + AdamW (≤ 1e-4 × max). Stage 3 ran with `stage3_param_persistence_threshold=1000`, so it partitioned the 11 weight matrices and embeddings and kept the 19 small tensors persistent.

| Variant | A grad | B params-1 | C | D | E | R | S vs DDP | Bitwise = DDP (same AdamW) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ddp` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | 30 tensors, yes | (reference) | — |
| `z0`, `z1` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | 30 tensors, yes | 0 | **yes**, all five quantities |
| `z2` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.382e-9 | 30 tensors, yes | 4.66e-10 / 7.1e-14 | no (final params 3.643e-7) |
| `z3` | 2.794e-9 | 2.906e-7 | **4.768e-7** | equal | 8.440e-9 | 19 persistent tensors, yes | 4.66e-10 / 8.5e-14 | no (final params 4.624e-7) |

| Negative control (on `z2`) | Fails | Passes | Verdict |
| --- | --- | --- | --- |
| `double_scale` | A (9.39e-3), E (ratio **0.250**), S | B (5.0e-6), C, D, R | detected by A |
| `fixed_gas` | D (adam step 5 vs 6), E (ratio 0.000), S | A, B, C, R | detected by D |

[understanding]
- **`double_scale` passes B.** Adam's first update is about lr · sign(g), so a gradient 4× too small barely changes it (Day 07's `sum_grads` lesson again).
- **`fixed_gas` passes A.** The first five windows have 4 microbatches and hit the counter's boundary exactly; only the short last window is lost. Only D (the Adam step count) and E see it.
- **`z2` is the same 3.643e-7 away from DDP as Day 07's FSDP2 `sync`**: both reduce every microbatch in FP32 at W = 2.
- **`z3`'s step-0 loss differs by 4.8e-7**, so its forward is not bitwise identical to the 1-process forward, unlike every other variant. Not isolated; a plausible cause is that gathered parameters live in different storage, which can change a kernel choice.

### B6. Memory ([measured memory log](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/memory.log))

#### B6.1 Per rank after wrapping, medium (Ψ = 27,443,200, P = 104.69 MiB)

| Variant | Param elements kept | fp32 master elements | `exp_avg` + `exp_avg_sq` | After wrap (MiB) | Across first step |
| --- | --- | --- | --- | --- | --- |
| `ddp` | 27,443,200 | — | 209.38 | 210.4 | 210.38 |
| `z0` | 27,443,200 | — | 209.38 | 105.8 | 209.4 |
| `z1`, `z2` | 27,443,200 (views of one flat buffer) | 13,721,600 | 104.69 | 157.1 | 52.34 |
| `z3` | 13,770,752 = 13,721,600 partition + 49,152 persistent (51 tensors) | 13,721,600 | 104.69 | 182.1 | 104.88 |

`z1`'s first step adds only P/2 because the state (P) arrives while `averaged_gradients` (P/2) is freed.

#### B6.2 Memory per stage: K = 1, m = 8, rank 0 (`memory.log`)

| Variant | Start of step | Forward peak | Backward peak | After step |
| --- | --- | --- | --- | --- |
| `ddp` | 484.9 | 1053.1 | 1117.1 | 594.9 |
| `z0` | 380.2 | 948.4 | 1012.4 | 380.2 |
| `z1` | 326.1 | 894.3 | 958.3 | 326.1 |
| `z2` | 325.8 | 894.4 | 958.4 | 325.8 |
| `z3` | **351.0** | **929.2** | **993.3** | 351.0 |
| `z1_def` (DeepSpeed's buckets) | 325.8 | 894.4 | **2730.4** | 325.8 |
| `z2_def` | 325.8 | 894.4 | **2710.4** | 325.8 |
| `z3_def` | **2234.0** | 2860.7 | 2924.8 | 2234.0 |
| Day 07 FSDP2 block (for reference, its own run) | 228.2 | 828.9 | 905.0 | — |

MiB. `ddp`'s row reproduces Day 07's personal run to 0.1 MiB.

#### B6.3 Who owns the memory: census at the start of a step, rank 0

| Buffer (DeepSpeed's name) | `z0` | `z1` / `z2` | `z3` | `z3_def` |
| --- | --- | --- | --- | --- |
| model params (`p.data`) | 104.7 | 104.7 (flat buffer) | 0.2 (persistent) | 0.2 |
| `p.ds_tensor` (param partitions) | — | — | 52.3 | 52.3 |
| fp32 master partition | — | 52.3 | 52.3 (`fp32_partitioned_groups_flat`) | 52.3 |
| AdamW state | 209.4 | 104.7 | 104.7 | 104.7 |
| `grad_partitions_flat_buffer` | — | — | 52.3 | 52.3 |
| `ipg_buckets` (reduction bucket) | — | — (allocated in backward) | 25.0 | **1907.3** |
| named total | 314.1 | 261.8 | 287.0 | 2169.3 |
| `memory_allocated` | 380.2 | 326.1 | 351.0 | 2234.0 |

Right before the optimizer, `z1` and `z2` add `zero.averaged_gradients` = 52.6 MiB (this rank's averaged gradient partition); `z0` adds the full `p.grad` (104.7).

#### B6.4 What the table supports

- **Start of step.** `ddp` − `z1` = 158.8 MiB against a 1.5P = 157.0 ledger. `z0` − `z1` = 54.1 MiB: the state saving (P) minus the master copy (P/2).
- **Stage 3 is higher than stages 1 and 2** by 24.9 MiB at the start of a step and 35 MiB at both peaks. Its parameter saving (P/2) is cancelled by the persistent gradient buffer (P/2), and the bucket (25 MiB) remains.
- **Forward.** `z1` and `z2` add DDP's +72 MiB per block: they shard nothing forward touches. `z3` gathers the token embedding (+16 MiB) at the model's pre-forward and prefetches the first block. Its per-block increments vary (+43 to +83 MiB) because gathered parameters, released ones and activations overlap.
- **Activations dominate.** The best stage cut the backward peak by 159 MiB against DDP, 14 % of 1117 MiB. Activations at m = 8 are about 630 MiB, and no stage touches them.
- **The defaults trap.** DeepSpeed's 5e8-element `reduce_bucket_size` is one 1907.3 MiB buffer. Stages 1 and 2 allocate it in backward and free it afterwards (+1772 MiB at the peak, start unchanged). Stage 3 keeps it for good (+1883 MiB at the start of every step). `z3_def` also gathers the whole model (+104.5 MiB) at the model's pre-forward and keeps it through backward (default prefetch 5e7 and reuse distance 1e9 elements).
- **FSDP2 (Day 07) vs DeepSpeed stage 3, both FP32.** FSDP2 started at 228.2 MiB, DeepSpeed stage 3 at 351.0. The 122.8 MiB difference ≈ master copy (52.3) + gradient buffer (52.3) + bucket (25.0) = 129.6. Different runs on different rentals; the ledger explains the gap, the runs were not interleaved.

#### B6.5 Between microbatches, K = 4 [pre-test, timing stage]

| | `ddp` | `z0` | `z1` | `z2` | `z3` |
| --- | --- | --- | --- | --- | --- |
| between microbatches (MiB) | 595.1 | 491.1 | 461.6 | 382.2 | 351.2 |
| − start of step (§B6.2) | 110.2 | 110.9 | 135.5 | 56.4 | 0.2 |
| K = 4 peak | 1227 | 1123 | 1095 | 1014 | **995** |

- **`z1` holds P + B between microbatches** (104.7 + 25.0 = 129.7 against 135.5): the full local gradient accumulates, and the bucket the first hook allocated stays until the boundary reduction. That is Day 06's `zero1` row.
- **`z2` holds P/2** (52.3 against 56.4): this rank's averaged gradient partition; the bucket is released after each backward's reduction. That is Day 06's `rs_every` row and Day 07's `sync`.
- **`z3` holds nothing new**: its gradient partition buffer exists from initialization.
- The 4–6 MiB excess in every column matches Day 07's 5.5 MiB for DDP and was not isolated.
- **Which stage has the lowest peak depends on where the peak falls.** At K = 1, m = 8 the peak is at the start of backward, before any gradient exists, and `z3` (993.3) is above `z1`/`z2` (958). At K = 4, later microbatches run with `z1`'s full gradient or `z2`'s partition already allocated, and `z3` is lowest (995 vs 1014 and 1095). At m = 32, K = 1, `z3` was also lowest (2747 vs 2855). Where exactly the m = 32 peak falls was not captured (no timeline at m = 32).

### B7. Collectives ([measured count log](../outputs/day08_downloads/a100_core_20261108T090011Z/session_a100_core_20261108T090011Z/count.log); medium, rank-0 trace, 2 profiled steps)

| K | Variant | AllReduce | ReduceScatter | AllGather | min-rank sum (ms) | exposed (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | `ddp` | 4 (+1 broadcast) | 0 | 0 | 0.92 | 21.49 |
| 4 | `z0` | 1 | 0 | 0 | 0.67 | 0.85 |
| 4 | `z1` | 6 | 0 | 1 | 1.30 | 1.30 |
| 4 | `z2` | **21** | 0 | 1 | 4.01 | 44.79 |
| 4 | `z3` | 1 | **20** | **155** | 10.38 | 73.73 |
| 1 | `ddp` | 4 (+1 broadcast) | 0 | 0 | 1.03 | 8.12 |
| 1 | `z0` | 1 | 0 | 0 | 0.72 | 1.34 |
| 1 | `z1` | 6 | 0 | 1 | 1.34 | 5.65 |
| 1 | `z2` | 6 | 0 | 1 | 1.38 | 4.84 |
| 1 | `z2_ovl` | 6 | 0 | 1 | 1.51 | 3.23 |
| 1 | `z3` | 1 | 5 | **77** | 3.00 | 29.35 |

**Where they are** (K = 1, m = 8, first profiled step):
- **`z1`:** 5 bucket all-reduces launched from the main thread inside `## bwd 0` (the boundary epilogue, after all backward compute: 0 % overlap), then in `## optimizer` one scalar all-reduce (gradient norm) and one all-gather of the updated parameters.
- **`z2`:** the same 5 + 1 + 1, but 4 of the 5 bucket all-reduces are launched from autograd's thread **during** backward, on the compute stream (0 % overlap: synchronous). With `overlap_comm` (`z2_ovl`) the bucket collectives can run on a side stream.
- **`z3`:** 83 collectives. Forward: 14 all-gathers (coalesced per module group and prefetch). Backward: 12 all-gathers and 5 reduce-scatters on separate streams. Optimizer: 1 scalar all-reduce and **51 all-gathers, one per persistent parameter**.

[understanding]
- The kinds confirm §1.3 of the guide: stage 1/2 gradient reduction is an `AllReduce` kernel, not `ReduceScatter`.
- K = 4 multiplies stage 2's all-reduces (4 × 5 + 1 = 21) and stage 3's gathers and reduce-scatters (4 × 26 + 51 = 155 all-gathers, 4 × 5 = 20 reduce-scatters). Stage 1 stays at 6 + 1.
- The min-rank sums (shorter of the two ranks per collective) rise with the stage at K = 4: 1.30 (`z1`), 4.01 (`z2`), 10.38 ms (`z3`). The `exposed` column is much larger because it includes rank 0 waiting for rank 1 in a profiled run.

### B8. Timing [pre-test]

#### B8.1 Method

Medium preset, 25 optimizer steps with 5 excluded, NVLink 3 repeats, slow link 2. TF32 off, MATH attention, `condition=shared`, `AdamW(foreach=True)` for every variant. Core time: slowest rank per step, median over steps, [min–max] over repeats. `step` is `engine.step()` for DeepSpeed (AdamW on the partition, copy into the model, parameter all-gather(s), `zero_grad`) and `optimizer.step()` for DDP.

#### B8.2 NVLink

| K | m | Variant | ms / opt step [min–max] | fwd+bwd | step | vs `ddp` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | `ddp` | 37.11 [36.90–37.91] | 33.97 | 3.46 | — |
| 1 | 8 | `z0` | 32.68 [31.40–34.73] | 29.24 | 3.01 | −4.43 |
| 1 | 8 | `z1` | 43.51 [43.33–44.75] | 37.31 | 6.53 | +6.40 |
| 1 | 8 | `z2` | 46.99 [46.82–47.20] | 38.74 | 8.12 | +9.88 |
| 1 | 8 | `z3` | 92.73 [84.81–97.37] | 79.43 | 13.38 | +55.62 |
| 1 | 8 | `z2_ovl` | 48.67 [48.62–50.30] | 40.73 | 8.18 | +11.56 |
| 1 | 8 | `z1_def` | 41.61 [34.60–42.65] | 35.37 | 6.46 | +4.50 |
| 1 | 8 | `z2_def` | 40.64 [40.54–41.61] | 34.52 | 6.98 | +3.53 |
| 1 | 8 | `z3_def` | 90.35 [90.12–90.56] | 77.08 | 14.31 | +53.24 |
| 1 | 32 | `ddp` | 94.44 [94.43–94.48] | 91.91 | 2.52 | — |
| 1 | 32 | `z0` | 94.54 | 91.99 | 2.52 | +0.10 |
| 1 | 32 | `z1` | 96.81 | 92.42 | 4.33 | +2.37 |
| 1 | 32 | `z2` | 96.57 | 92.18 | 4.37 | +2.13 |
| 1 | 32 | `z3` | 111.14 [110.31–111.91] | 97.63 | 13.58 | +16.70 |
| 4 | 8 | `ddp` | 127.47 [127.08–127.80] | 123.89 | 3.52 | — |
| 4 | 8 | `z0` | 128.72 [127.67–134.03] | 124.78 | 3.95 | +1.25 |
| 4 | 8 | `z1` | 148.77 [148.67–150.29] | 140.87 | 8.54 | +21.30 |
| 4 | 8 | `z2` | 164.73 [164.73–166.26] | 156.64 | 8.93 | +37.26 |
| 4 | 8 | `z3` | 319.20 [310.02–324.05] | 301.66 | 15.22 | +191.73 |

#### B8.3 Slow link (`NCCL_P2P_DISABLE=1`), K = 1, m = 8

| Variant | ms / opt step [min–max] | − NVLink | step | exposed comm (profiled) |
| --- | --- | --- | --- | --- |
| `ddp` | 72.40 [72.40–72.40] | +35.29 | 2.54 | — |
| `z0` | 81.84 [81.66–82.02] | +49.16 | 2.50 | — |
| `z1` | 112.29 [112.14–112.45] | +68.78 | 33.10 | 82.15 of 82.15 ms |
| `z2` | 111.00 [109.01–113.00] | +64.01 | 33.03 | — |
| `z3` | 114.67 [113.62–115.72] | +21.94 | 13.22 | 26.74 of 76.58 ms |

#### B8.4 What it supports

- **No stage beat DDP, except stage 0 at m = 8.** `z0` was 4.43 ms faster than DDP with non-overlapping repeat ranges: one all-reduce after backward instead of DDP's four bucket hooks. The cause was not isolated (DDP's Reducer work in a launch-bound regime is a plausible one). On the slow link the same design lost: `z0` 81.84 vs `ddp` 72.40, because nothing overlaps its single all-reduce.
- **The cost grows with the stage and with K.** At K = 1, m = 8: +6.4, +9.9, +55.6 ms for stages 1, 2, 3. At K = 4: +21.3, +37.3, +191.7 ms. Stage 1's extra time is its boundary all-reduces after backward (fwd+bwd +3.3 ms) plus a 3 ms longer step (parameter all-gather, gradient norm, copy).
- **Per removed collective [derived].** At K = 4, `z2` issues 15 more bucket all-reduces than `z1` and took 15.96 ms longer: about 1.06 ms per all-reduce, against min-rank kernel durations of 0.13–0.23 ms. Like Day 07's ratios, this includes hooks, synchronisation and waiting, not only transfer time.
- **Stage 3 is overhead-bound here.** Its step phase alone is 13.4 ms (51 per-parameter all-gathers plus AdamW), and its fwd+bwd is 45.5 ms above DDP at m = 8 but only 5.7 ms above at m = 32. More compute per gather hides more of it.
- **`overlap_comm` did not help stage 2 at m = 8** (48.67 vs 46.99 ms), although the trace showed its in-backward all-reduces moving to a side stream with 98 %, 28 %, 13 % and 2 % overlap. Not isolated.
- **Bigger buckets were faster.** DeepSpeed's 1907 MiB default bucket made `z2` 6.35 ms faster than 25 MiB buckets (one all-reduce instead of five) and `z1` 1.90 ms faster with a wide repeat range, at +1752 to +1772 MiB of backward peak. `z3_def` and `z3` repeat ranges overlapped.
- **Slow link.** Stage 3 lost only 21.9 ms to the slow link, stage 1 lost 68.8 ms: stage 1's 5 boundary all-reduces and its parameter all-gather are all exposed (profiled: 82.15 of 82.15 ms), while stage 3's gathers and reduce-scatters overlap compute (26.74 of 76.58 ms exposed). On this link stages 1, 2 and 3 ended within 4 ms of each other, all about 40 ms slower than DDP.

#### B8.5 Limits

- One pre-test rental; the shared host's noise was not characterised. Treat differences under about 4 ms at m = 8 cautiously (Day 07 §8.5). `z3` at K = 1, m = 8 had a 12.6 ms repeat range.
- Profiled numbers (exposed communication) come from separate runs and include waiting for the other rank; they rank variants, they are not benchmarks.
- CPU time per hook and per collective was not measured directly.
- **[not tested]** `use_multi_rank_bucket_allreduce: false`, `zero.Init`, BF16/FP16, offload, ZeRO++, gradient clipping, activation checkpointing, W ≥ 3.

### B9. Completion assessment

The personal core run and the required write-up are complete. The predictions, source walkthrough, reimplementation and timing analysis remain optional.

| Gap | Kind | Where it gets resolved |
| --- | --- | --- |
| `use_multi_rank_bucket_allreduce: false` (one `reduce` per owner: P/2 instead of P per reduction at W = 2) | communication knob | not run; read only |
| `zero.Init` (partition at construction) | memory at init | not needed at 27M; read only |
| BF16 / FP16 with ZeRO: the master copy becomes necessary, gradient and parameter bytes halve | new mechanism | Day 9 |
| Offload (ZeRO-Offload / Infinity), ZeRO++ (`zero_hpz_partition_size`, quantized weights/gradients) | new mechanism | not planned |
| Why `z3`'s forward is not bitwise identical | numerics | open |
| W ≥ 3 | coverage | not possible on this host |

### Open questions (optional)

- [ ] Day 9: with BF16 parameters, how much of stage 3's FP32 overhead disappears (master copy becomes needed, gradient buffer halves)?
- [ ] Day 9: `gradient_clipping` under ZeRO: the global norm is the scalar all-reduce seen in every `## optimizer` phase today; verify clipping parity against DDP.
- [ ] Would `use_multi_rank_bucket_allreduce: false` bring stage 1's bytes down to Day 06's P, and would it be faster at W = 2?
- [ ] Day 16: gathering a stage-3 model for an inference engine (`GatheredParameters`, `zero_to_fp32`) vs FSDP2's `full_tensor()`.
