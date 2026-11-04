# Day 06 — ZeRO-1 from scratch: AdamW state sharded across ranks (guide, dual-A100 edition)

> **Status (2026-11-04): complete; independent practice remains unrecorded.** Run `a100_20261102T124935Z` passed all 52 experiments and local archive/file checks; follow-up `a100_sync_20261103T120618Z` investigated rank arrival skew. Blocks 1–4, the core framework call chain, and the measurement conclusions have been discussed with the assistant. Independent reimplementation of the YOU WRITE sections and an unaided final explanation have not been recorded.

**Goal:** by tonight you can shard AdamW's two moment tensors across the two ranks, inside Day 04's accumulation loop, and show with your own checks that (1) the result is **bitwise identical** to DDP + unsharded AdamW, (2) each rank stores half of the optimizer state, measured twice (the state tensors, and the allocation across the first optimizer step), and (3) you know where the accumulation boundary sits for ZeRO-1: what is reduced, when, and what stays allocated between microbatches. You write `FlatZeroAdamW` (~60 lines: flat buffers, reduce-scatter, a sliced AdamW, all-gather), then read `torch.distributed.optim.ZeroRedundancyOptimizer` and see that the framework's ZeRO-1 makes a different choice: whole tensors per rank, a full all-reduce, and one broadcast per parameter.

Day 04 left the numbers this day builds on. Medium model: **Ψ = 27,443,200 parameters, P = 104.7 MiB** in FP32. DDP with `no_sync` at K ≥ 2 peaked at **1227 MiB** per rank, of which parameters, gradients, DDP buckets and AdamW state are 5P ≈ 524 MiB. AdamW's state is 2P, the largest single item, and it is identical on both ranks. That redundancy is today's target.

**Done means:**

- [ ] §1.8 predictions written (memory ledger per variant, collective counts, bitwise groups, which check catches which control) before you look at Appendix A
- [ ] `FlatZeroAdamW` written by you (the **YOU WRITE** parts of `zero1.py`), diffed against the reference
- [x] `compare_zero1.py --stage parity` prints `PARITY: ALL CHECKS PASSED`: every variant passes A–E, R, S; the boundary group is bitwise equal to DDP; all four controls are `DETECTED` by their named check
- [x] State per rank logged and explained in the assisted walkthrough: elements owned, `exp_avg` + `exp_avg_sq` MiB and the allocation at the first step, for `ddp`, `torch_zero`, `flat_ar`, `zero1`, on both ranks
- [x] `--stage count`: collectives per optimizer step for six variants, including the 79 broadcasts of `torch_zero`
- [x] `--stage timing` and `--stage slowlink`: flattening, state sharding, communication and `rs_every` costs discussed against the measured data; rank waiting and unresolved slow-link residual distinguished
- [x] Accumulation boundary: between-microbatch memory of `zero1` vs `zero1_rs_every` reasoned through and compared with measurements in the assisted walkthrough
- [x] Core Block 3 call chain traced in an assisted source walkthrough; `day06/notes.md` contains the measured results and assisted explanations
- [ ] You can answer, without notes: *what does ZeRO-1 shard, what does it cost in communication, and where does the accumulation boundary sit?*

The pre-run prediction box is historical: retrospective reasoning is not being relabeled as an advance prediction. The remaining implementation and unaided-explanation boxes distinguish independent practice from the completed assisted walkthrough; they do not require another GPU rental. The later-day extensions below are not prerequisites for moving on to Day 7.

---

## Read this first: same host type, fresh rental

Same software and rules as Days 02–04: **2 × A100-SXM4-40GB, NV12, default NCCL settings**, torch 2.14.1+cu130, NCCL 2.30.7, `--condition shared`. The current rental (`ssh -p 17591 root@ssh4.vast.ai`) has both GPUs on NUMA node 1, like Day 04's run host. The project directory was empty on this rental; the pre-test uploaded Days 01–04 and today's folder, and they are still there. On a different, fresh rental, from your Windows project root:

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/AI-Infra-Learning/experiments'
scp -P <port> requirements.txt root@<host>:/workspace/AI-Infra-Learning/
scp -P <port> -r experiments/day01_single_gpu experiments/day02_data_parallel experiments/day03_ddp_hooks experiments/day04_grad_accum experiments/day06_zero1 root@<host>:/workspace/AI-Infra-Learning/experiments/
```

If you rewrite `zero1.py`, copy only that file again before your run. Then, on the instance:

```bash
cd /workspace/AI-Infra-Learning
ln -sfn /venv/main .venv        # fresh rental only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
nvidia-smi topo -m              # record it: timings compare only within one host
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/hello_dist.py --device cuda
```

Today's code imports Day 01 (`train.py`), Day 02 (`ddp_manual.py`, `check_parity.py`, `launch_utils.py`, `find_symbols.py`, `gpu_probe.py`) and Day 04 (`compare_accum.py` for checks D/E, `count_collectives.py`), all unchanged.

**Measurement status.** Appendix A is your manual run `a100_20261102T124935Z`, completed at 21:00:35 Singapore time on 2026-11-02. Archive SHA256 and 179 manifest files verified; parity and all four controls passed. Raw evidence is in `C:/Users/qly24/Downloads/day06_a100_20261102T124935Z/results/`. The assistant's `pretest1` is retained separately as historical evidence. For the continuing design discussion, write reasoning in §1.8 before looking up the measured answer; the other prediction cells have not been backfilled from results.

---

## Files

| File | What it is |
| --- | --- |
| `experiments/day06_zero1/zero1.py` | **The core you write.** `FlatZeroAdamW`: flat parameter buffer (parameters become views), flat gradient buffer or shard accumulator, in-place reduce-scatter, AdamW on one slice, in-place all-gather; evidence helpers |
| `experiments/day06_zero1/train_zero1.py` | Day 04's accumulation loop with seven `--impl` choices, four `--bug` controls, synchronized phase timing, a per-rank memory ledger and state report, `--profile` |
| `experiments/day06_zero1/compare_zero1.py` | Orchestrator: smoke → `parity` → `count` → `timing` → `slowlink`. Plain `python` |
| `experiments/day06_zero1/find_symbols.py` | `file:line` of the Block 3 reading list in your install |
| `experiments/day06_zero1/run_server_once.sh` | All stages once, logs, source snapshot, output checks, checksummed archive |

---

## Block 1 — Design (30 min): reason first, then compare with the measured run

### 1.1 Where the memory goes, and which part is redundant

Per rank, FP32, after the first optimizer step (Ψ parameters, P = 4Ψ bytes):

| Item | DDP + AdamW | Redundant across ranks? |
| --- | --- | --- |
| parameters | P | yes, but every rank needs them in forward and backward |
| gradients (`p.grad`) | P | yes, but only after the reduction; before it each rank's are different |
| DDP buckets (`gradient_as_bucket_view=False`) | P | an implementation choice (Day 03) |
| AdamW `exp_avg`, `exp_avg_sq` | 2P | **yes, and only the optimizer reads them, once per step** |

ZeRO (Rajbhandari et al., 2020) removes the redundancy in stages: stage 1 shards the optimizer state, stage 2 also the gradients, stage 3 also the parameters. With **mixed precision** (bf16/fp16 parameters and gradients, fp32 master weights + two fp32 moments) the per-parameter cost is 2 + 2 + 12 = 16 bytes and stage 1 cuts it to 4 + 12/W. For a 7 B model: 112 GB per GPU without ZeRO, 38.5 GB with ZeRO-1 at W = 8. Today is FP32 (4 + 4 + 8 = 16 bytes per parameter, plus DDP's buckets), so stage 1 saves 8Ψ(W − 1)/W = P per rank at W = 2: **104.7 MiB on a ~1.2 GB peak**. Small here, because at m = 8 activations dominate; decisive when model states dominate.

### 1.2 One optimizer step of ZeRO-1

```
every rank:  K x backward  ->  flat_grad = sum_j w_j g_j  (full size, local)
reduce_scatter(flat_grad)  ->  rank r holds the SUM over ranks for ITS slice [r*S, (r+1)*S); divide by W
AdamW on the slice         ->  only S = Ψ/W elements updated; exp_avg, exp_avg_sq allocated for S elements
all_gather(flat_param)     ->  every rank receives the other slices: full, identical parameters again
```

**Communication volume is unchanged.** A ring all-reduce *is* a reduce-scatter followed by an all-gather: each rank sends (W − 1)/W · P in each half. ZeRO-1 splits the two halves and puts the optimizer between them: the reduce-scatter delivers exactly the gradient slice the rank will update, and the all-gather moves updated parameters instead of reduced gradients. Bytes sent per rank per window at W = 2:

| Variant | Collectives per window | Bytes sent per rank |
| --- | --- | --- |
| DDP `no_sync`, `flat_ar` | all-reduce of P | P |
| `zero1` | reduce-scatter + all-gather | P/2 + P/2 = P |
| `zero1_rs_every` (K microbatches) | K reduce-scatters + all-gather | (K + 1) P/2 |
| `torch_zero` (ZeroRedundancyOptimizer) | all-reduce of P (DDP) + broadcasts of updated parameters | P + P/2 = 1.5 P |

What changes is **when**: the all-gather comes after the optimizer and, in today's simple version, overlaps with nothing.

### 1.3 Why the slices may cut through tensors (and when they may not)

`FlatZeroAdamW` concatenates all 78 parameter tensors into one flat buffer, pads it to a multiple of W and gives each rank one contiguous slice, so a slice boundary can fall inside a weight matrix. That is only correct because **AdamW is elementwise**. `_single_tensor_adam` (Block 3, Stop 5) does, per tensor, `param.mul_(1 - lr*wd)`, `exp_avg.lerp_(grad, 1 - β1)`, `exp_avg_sq.mul_(β2).addcmul_(grad, grad, value=1 - β2)`, `denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)`, `param.addcdiv_(exp_avg, denom, value=-step_size)`. Every element depends only on its own four numbers and on scalars shared by all elements (lr, betas, eps, wd, the step count). Any partition gives the same numbers.

It stops being free when an update needs more than one element:

| Needs | Example | What ZeRO must add |
| --- | --- | --- |
| a global gradient norm | `clip_grad_norm_` (Day 9) | each rank sums its slice's squares; one scalar all-reduce; every rank applies the same factor. Clipping by the *local* slice norm is a silent bug |
| a per-tensor statistic | LAMB's trust ratio, Adafactor's row/column factors | whole tensors per rank, or a reduction per tensor |
| the whole matrix | Shampoo, Muon's orthogonalization | gather the matrix to its owner |

Two partition choices exist. Flat equal slices (DeepSpeed ZeRO, FSDP's flat parameter) balance memory exactly and make every collective equal-sized, but split tensors. Whole tensors per rank (`ZeroRedundancyOptimizer`, greedy by size) never split a tensor but balance only approximately. Tiny model, W = 2: 221,184 vs 216,704 elements.

### 1.4 The accumulation boundary (Day 04 → Day 06)

Day 04's question: *with accumulation, the sharded optimizer still needs the averaged gradient once per window. Does the reduction become a reduce-scatter at the boundary microbatch, and what stays allocated in between?* There are two consistent answers, and today's harness runs both:

| | `zero1` (boundary) | `zero1_rs_every` |
| --- | --- | --- |
| microbatches 1..K−1 | backward adds into a **persistent full flat gradient** (P); no communication | backward → `p.grad`; **reduce-scatter immediately**, add the averaged slice into a P/W accumulator, free the full gradient |
| boundary (microbatch K) | **one** reduce-scatter of the window sum | the K-th reduce-scatter |
| optimizer, all-gather | once per window | once per window |
| allocated between microbatches (gradient side) | P | P/W |
| reduce-scatters per window | 1 | K |
| Day 04 analogue | `no_sync` | sync every microbatch |
| whose pattern | DeepSpeed ZeRO-1 at its accumulation boundary (to verify on Day 8) | ZeRO-2 / FSDP with gradient sync on (Days 7–8) |

The parameters are full on every rank during the whole window: the all-gather is the last act of the window and must complete before any rank's next forward. Nothing is gathered *inside* the window. That is what changes with ZeRO-3/FSDP (Day 7), where parameters are gathered per forward and per backward, per microbatch.

`zero1_rs_every` is the honest price of sharded gradients: it saves P − P/W of gradient memory between microbatches and pays K − 1 extra reduce-scatters. Holding the full unreduced gradient is exactly what FSDP2's `set_requires_gradient_sync(False)` does (Day 7).

What each variant keeps allocated **between microbatches** (after the first step, activations freed), per rank:

| Variant | params | gradient side | DDP buckets | AdamW state | total |
| --- | --- | --- | --- | --- | --- |
| `ddp` (`no_sync`) | P | P (`p.grad`) | P | 2P | 5P = 523.4 MiB |
| `torch_zero` | P | P | P | ≈ P (whole tensors) | ≈ 4P |
| `flat_ar` | P | P (`flat_grad`) | — | 2P | 4P = 418.8 MiB |
| `zero1` | P | P (`flat_grad`) | — | P | 3P = 314.1 MiB |
| `zero1_rs_every` | P | P/2 (`grad_acc`) | — | P | 2.5P = 261.7 MiB |

`memory_allocated()` also counts things that are the same in every variant (cuBLAS workspaces, the causal mask, allocator slack), so predict **differences between rows**, then read the constant off one measured row.

### 1.5 Flat buffers: three tricks, one pitfall

1. **Parameters are views of `flat_param`.** `p.data = flat_param[off:off+n].view_as(p)`. The all-gather writes straight into the model; nothing is copied back.
2. **Gradients are views of `flat_grad`** (Day 03's `grad_as_view`). `AccumulateGrad` adds each microbatch in place, so after K backwards the buffer already holds the flat window sum.
3. **In-place collectives.** NCCL's reduce-scatter is in place when the output is the rank's own slice of the input (`recvbuff == sendbuff + rank · count`); same for all-gather. The reduced gradient slice lands in `flat_grad[lo:hi]`, the optimizer's slice *is* `flat_param[lo:hi]` (`nn.Parameter(view)` shares storage), and no extra P/W buffers exist.

**The pitfall:** PyTorch's default `zero_grad()` is `set_to_none=True`. It replaces the views with `None`, the next backward allocates fresh `p.grad` tensors, and `flat_grad` silently stays zero. Nothing crashes; AdamW steps with a zero gradient. That is the `grad_set_to_none` control.

### 1.6 Numerics: what is bitwise

| Comparison | Prediction | Why |
| --- | --- | --- |
| `zero1`, `flat_ar`, `torch_zero`, `torch_zero_bucket` vs `ddp` | **bitwise equal**: gradients, parameters, gathered AdamW state, every step | All reduce the same locally accumulated window sum once. At W = 2 every reduced element is one two-operand sum fl(a + b), whether the collective is an all-reduce, a reduce-scatter or DDP's bucket all-reduce, and ×½ (pre- or post-) is exact. AdamW is elementwise, so slicing does not change any element's arithmetic |
| `zero1_rs_every` vs `ddp` | **not** bitwise; close | Σⱼ fl(gⱼ⁰ + gⱼ¹)/2 vs fl(Σⱼ gⱼ⁰ + Σⱼ gⱼ¹)/2: Day 04's every-vs-`no_sync` rounding difference, now on slices |
| any of them vs 1 process, one big batch | within Day 02's tolerances (A–E) | big-batch matmul vs accumulated `+=` (Day 04) |
| state vs 1 process after 6 steps | close but drifting | the moments integrate 6 steps of a trajectory that differs at the 1e-7 level |

At W ≥ 3 the first row is no longer guaranteed: NCCL may sum three operands in different orders in an all-reduce and a reduce-scatter.

### 1.7 Checks R and S, and the four controls

- **R (replicas):** parameters bitwise equal across ranks after every optimizer step.
- **S (state):** `exp_avg` and `exp_avg_sq`, gathered from all slices (FlatZeroAdamW) or consolidated (`ZeroRedundancyOptimizer.consolidate_state_dict`), match the unsharded DDP + AdamW run, and Adam's step counter is equal.

In plain DDP a forgotten all-reduce makes the replicas diverge (Day 02's `no_allreduce` control was caught by the sync check). **In ZeRO-1 it does not.** Each rank updates its own slice with a wrong gradient, and the all-gather then copies every owner's wrong slice to everyone: identical replicas of a wrong model. Predict, for each control, which of A B C D E R S fail:

| `--bug` | Variant | What it does | Must be caught by |
| --- | --- | --- | --- |
| `no_reduce_scatter` | `zero1` | skips the reduce-scatter: each rank steps its slice with its local gradient / W | A |
| `no_allgather` | `zero1` | skips the all-gather: each rank's other slice stays at its old value | R |
| `grad_set_to_none` | `zero1` | `model.zero_grad()` with the default `set_to_none=True` (§1.5) | A |
| `rs_overwrite` | `zero1_rs_every` | `grad_acc.copy_(slice)` instead of `add_`: only the last microbatch survives | A |

Which control passes check A, and why? Which controls keep R green?

### 1.8 Your predictions (fill in before Block 4)

**Retrospective completion, 2026-11-04 (assisted).** This table consolidates the learner's reasoning during the guided discussion and the existing results; it was filled after the experiments and is **not a record of advance predictions or an independent assessment**. The original heading is retained to identify the exercise. Unless noted, measured values below come only from manual run `a100_20261102T124935Z`, not the later run on a different host. For the medium FP32 model, Ψ = 27,443,200 and P = 104.6875 MiB; W = 2.

| Quantity | How to estimate | Reasoned expectation (retrospective) | Measured check |
| --- | --- | --- | --- |
| `exp_avg` + `exp_avg_sq` per rank: `ddp`, `torch_zero` (rank 0 / 1), `zero1` | Each owned element has two FP32 moments: owned elements × 8 bytes. Framework ownership follows the whole-tensor greedy split. | `ddp`: 2P = 209.38 MiB per rank. `zero1`: 2P/W = P = 104.69 MiB per rank. `torch_zero`: approximately P; using its ownership counts gives 105.00 / 104.38 MiB. | State tensor bytes match these values. Framework ranks own 13,762,560 / 13,680,640 elements; flat ZeRO ranks each own 13,721,600. |
| Allocation across the first optimizer step, `ddp` vs `zero1` | AdamW allocates its moments lazily at the first `step()`; allocated-byte changes can also include other allocations and allocator rounding. | Approximately +2P = 209.38 MiB for `ddp`, versus +P = 104.69 MiB for `zero1`. This estimates allocation growth, not the temporary peak during the step. | `ddp`: +210.38 MiB; `zero1`: +104.69 MiB. DDP has a 1.00 MiB excess over moment tensor bytes; its cause was not isolated in this run. |
| Between-microbatch memory: `ddp` − `flat_ar`; `flat_ar` − `zero1`; `zero1` − `zero1_rs_every` | Subtract the §1.4 ledgers: 5P, 4P, 3P, 2.5P. | P = 104.69 MiB from eliminating duplicate bucket storage; P = 104.69 MiB from state sharding; P − P/W = 52.34 MiB from retaining only a gradient shard. | 112.00 / 104.69 / 52.34 MiB. The first difference includes another 7.31 MiB beyond the ideal bucket term; do not attribute the full 112 MiB to ZeRO state sharding. |
| Peak memory, same three differences | Begin with the same ledger differences if the peak occurs in the same phase and the other live working storage is comparable. | Roughly P / P / P/2 = 104.69 / 104.69 / 52.34 MiB; not guaranteed, since activation lifetimes and optimizer temporaries can change the peak. | 112.00 / 104.69 / 52.34 MiB, matching the between-microbatch differences here. Matched `flat_ar` → `zero1`: 1115.31 → 1010.63 MiB, about 9.4% lower peak. |
| Collectives per optimizer step at K = 4: AllReduce / ReduceScatter / AllGather / Broadcast | DDP: 4 bucket reductions and 1 buffer broadcast; framework ZeRO adds 78 parameter broadcasts or 2 owner buckets. Count collective types, not the internal phases of ring all-reduce. | `ddp`: (4,0,0,1); `torch_zero`: (4,0,0,79); `torch_zero_bucket`: (4,0,0,3); `flat_ar`: (1,0,0,0); `zero1`: (0,1,1,0); `zero1_rs_every`: (0,4,1,0). | All six tuples match the count summary. These are steady profiled training steps, excluding initialization, final state consolidation and control/validation communication outside the sampled steps. |
| Optimizer phase: `ddp` vs `flat_ar` vs `zero1` | The configured per-tensor AdamW repeatedly launches operations for 78 tensors; flat AdamW handles one large tensor; sharding halves its element count. | `ddp` much slower than `flat_ar`; `zero1` faster than `flat_ar` within the optimizer phase. Expect launch reduction to dominate the first change; do not require exact halving for the second. | NVLink K=1: 10.93 → 1.83 → 1.04 ms; K=4: 14.37 → 1.85 → 1.05 ms. These are optimizer phases, not whole-step speed-ups. |
| `zero1` vs `flat_ar` core time on NVLink | Weigh less local optimizer work against two exposed collective calls and rank/host overhead. Same ring traffic volume does not imply identical latency. | Similar total times; a small improvement from sharding is plausible, but no guaranteed overall speed-up. Separate phase medians are directional evidence, not an exact additive decomposition. | K=1: 32.04 vs 32.69 ms, about 0.64 ms faster; observed ranges are disjoint, but there are only 3 repeats. K=4: 121.49 vs 122.15 ms, about 0.66 ms faster; ranges overlap, so the advantage is unresolved. |
| `zero1_rs_every` − `zero1`, K = 4, P2P disabled | K−1 = 3 extra reduce-scatters; use the same host's reduce-scatter cost. Day 04 all-reduce timings from another host are only rough background. | Additional cost approximately 3 × t_RS, in exchange for P/2 = 52.34 MiB less retained gradient memory. | Same-run last-arriving-rank RS estimate 29.05 ms predicts about 87.15 ms; core difference is 251.94 − 163.22 ≈ 88.73 ms using unrounded values. The added-collective model is approximate and host-specific. |
| Bitwise groups | Same local accumulation and two-rank reduction operands; elementwise AdamW. Per-microbatch reductions change floating-point grouping. | `torch_zero`, `torch_zero_bucket`, `flat_ar`, `zero1` should match `ddp` bitwise in the tested W=2 setting. `zero1_rs_every` should be close, not necessarily bitwise. Single-process big-batch comparisons use tolerances; W≥3 is not covered. | All four boundary variants match DDP on the five saved quantities. `rs_every` does not; final parameter maximum difference is 3.643e−7. |
| Which checks each control fails | Follow when the fault first affects gradients, parameter replicas and the final gathered state. A/C inspect step 0; R checks every update; S here compares final state. | `no_reduce_scatter`: A/B/E/S; `no_allgather`: B/E/R/S; `grad_set_to_none`: A/B/E/S; `rs_overwrite`: A/B/E/S. Required detectors: A / R / A / A, respectively. | All four fault patterns and named detectors match. D passes all four; C passes because the first loss precedes the faulty update. With `no_allgather`, the first-step state can still be correct; the recorded S failure is after subsequent divergent-model steps. |

Previously derived by the learner: equal-slice state is `2P/W` and the saving is `2P(1−1/W)`; W=2 gives state P and saving P, while W=4 gives state P/2 and saving 3P/2. The W=4 result is theoretical, not a four-GPU measurement. A and S can fail while R passes because parameter all-gather distributes the same erroneous owner slices to every rank. The later synchronization study qualifies the latency interpretation; it does not turn the original slow-link residual into a measured fixed launch cost.

### 1.9 New rules (on top of Days 02–04)

1. **Shard only what is elementwise**, or add the reduction it needs (global norm, per-tensor statistics).
2. **Zero flat gradient buffers in place.** `set_to_none` detaches views.
3. **The all-gather is part of the step.** No rank may start the next forward on stale slices.
4. **Replicas in sync is not correctness** once an all-gather distributes each owner's result. Check the gradient and the state against an unsharded run.
5. **Compare sharded against matched unsharded code.** Flattening alone changes the optimizer's kernel count by two orders of magnitude.
6. **Compare both ranks for a flat collective.** The earlier rank can wait inside NCCL; min-rank host phase time reduces this skew but still includes dispatch/synchronization. Use both GPU traces to examine arrival times and kernel durations.

---

## Block 2 — Code (2 h)

### Stage 0 (15 min): the ledger by hand

From Ψ = 27,443,200, write the §1.4 table in MiB and the four collective rows of §1.2 for K = 4. Then predict the `[state rank r]` lines of a `ddp` and a `zero1` medium run: elements owned, `exp_avg` + `exp_avg_sq` MiB, allocation at the first step. Where does AdamW keep `step` for `foreach=False, fused=False, capturable=False`? (On the CPU: a 0-d float tensor per parameter.)

### Stage 1 (60 min): write the **YOU WRITE** parts of `zero1.py`

Copy `zero1.py` to a scratch name, delete the bodies under `YOU WRITE`, and rewrite them:

1. **Flat parameter buffer.** Size `padded = ceil(Ψ / W) · W`; copy each parameter in, re-point `p.data` to its view, then one `dist.broadcast` from rank 0 (ranks start from different seeds on purpose).
2. **Where gradients live.** Boundary variants: `flat_grad` with `p.grad` views. `rs_every`: no full buffer; a `grad_acc` of S elements.
3. **The optimizer's slice.** `nn.Parameter(flat_param[lo:hi])`, its `.grad` set to the matching slice of `flat_grad` (or to `grad_acc`), and a torch `AdamW(foreach=False, fused=False)` over that one parameter. Assert that it aliases `flat_param`.
4. **The gradient side.** `reduce_grads`: in-place reduce-scatter (or all-reduce for `shard=False`), then `div_(W)`. `reduce_microbatch` (rs_every only): concatenate, free `p.grad`, in-place reduce-scatter into the temporary's slice, divide, `add_` into `grad_acc`.
5. **The parameter side.** `step_local` (the AdamW step), `gather_params` (in-place all-gather).

Questions before moving on:

- After `reduce_grads`, what does `flat_grad[0:lo]` hold on rank 1? Does anything read it? (The input of the in-place reduce-scatter: this rank's local window sum. Nothing reads it before `zero_grad`.)
- Why must `zero_grad` run *before* the window and never between microbatches for the boundary variant? (It is the accumulation buffer.)
- AdamW's `exp_avg` is created with `zeros_like(opt_param)`. Why does that make the state S elements and not Ψ?
- Why is the zero pad harmless? (Zero gradient, zero moments, zero update, never copied to a parameter.) Both presets have an even Ψ, so neither this manual run nor the earlier pre-test exercised a non-zero pad.

### Stage 2 (25 min): read the harness

In `train_zero1.py` find:

- the three wrappers: DDP (`ddp`), DDP + `ZeroRedundancyOptimizer` (`torch_zero`, `torch_zero_bucket`), `FlatZeroAdamW` (`flat_ar`, `zero1`, `zero1_rs_every`), and that the LR scheduler always drives the optimizer that holds the state;
- the window: Day 04's `last`, weights, `no_sync` (DDP-based only), `reduce_microbatch` after each backward (rs_every only);
- the five synchronized timestamps t0..t4 and what each phase contains for each wrapper;
- `delta_first_step` (allocation across the first `optimizer.step()`) and `alloc_between_micro` (after microbatch 0 of the first window after warmup);
- the evidence path: `synced_grads()` and `full_state()` are extra all-gathers **outside** t0..t4; `torch_zero` uses ZeRO's own `consolidate_state_dict(to=0)`;
- `tm_min`: why the minimum over ranks can reduce arrival-skew inflation, while remaining a host phase duration that includes dispatch and synchronization rather than isolated transfer time.

### Stage 3 (20 min): controls

Fill the §1.7 table with your predictions for all seven checks, then compare with the parity output. Explain why `no_allgather` passes A (the step-0 gradient is computed before the first update) and why B is only barely above its tolerance for it (Adam's first update is about lr = 1.5e-5 per element; B's tolerance is 1e-5).

### Pitfalls (symptom → cause)

| Symptom | Likely cause |
| --- | --- |
| Training runs, loss falls slowly, gradient check shows exactly 0 | `zero_grad()` with `set_to_none=True` on flat-buffer views (§1.5) |
| Replicas identical, loss slightly worse than DDP | the reduction was skipped or wrong; the all-gather hides it (R stays green) |
| Replicas diverge after the first step | all-gather skipped, or a rank started the next forward before it completed |
| Hang at the first collective | ranks disagree on `padded` (different parameter sets) or one rank skipped a collective |
| "ZeRO-1 makes the step 22% faster than DDP" | This comparison combines flattening and sharding; use `flat_ar` to isolate sharding (§1.8, Appendix A.5) |
| A flat collective takes 15 ms on NVLink | rank skew: the phase on the first-arriving rank includes waiting for the other |
| Peak memory barely moves with ZeRO-1 | activations dominate the peak; ZeRO-1 removes (W − 1)/W of 2P only |
| `memory_allocated` grows by more than the state | Allocator block rounding is one possible cause; the measured +1.00 MiB per-tensor first-step excess was not isolated in this run (Appendix A.2) |
| Clipping changes behaviour after sharding | the norm was computed on the local slice, not all-reduced (§1.3) |
| Checkpoint restores only half the state | each rank saved its own slice; consolidate (`consolidate_state_dict`) or save shards with their ranges |

---

## Block 3 — Read the source (1 h): `ZeroRedundancyOptimizer`, AdamW, the collectives

Run `python experiments/day06_zero1/find_symbols.py` first. torch 2.14.1 line numbers below; your install prints its own.

### Stop 1 — `ZeroRedundancyOptimizer.__init__` → `_partition_parameters` (`zero_redundancy_optimizer.py:290`, `:651`)

The partition: per parameter group, `sorted(params, key=numel, reverse=True)`, then each tensor to the rank with the smallest running size (`_get_min_index`). *Q: on the tiny model, which rank gets `token_emb` (16,384 elements, the largest)? Why does rank 0 end with 6 tensors and rank 1 with 24?* (Medium: 14 vs 64 out of 78.) *Q: what would a partition that splits tensors buy, and what would it break?*

### Stop 2 — `_init_local_optimizer` (`:1516`)

The local optimizer is a plain `optimizer_class(params_of_this_rank, **defaults)`. *Q: so where is the state for the other rank's parameters?* (Nowhere on this rank: that is the whole saving.)

### Stop 3 — `step` → `_local_step` → `_sync_params` → `_broadcast_params_from_rank` (`:1124`, `:1047`, `:813`, `:760`)

`_local_step` copies the LR from the exposed `param_groups` into the local optimizer (`_sync_param_groups`), steps it, copies back. `_sync_params` issues, for every rank, `dist.broadcast(param.data, src=owner, async_op=True)` per parameter, then waits on all handles. *Q: derive the 78 broadcasts (medium) and the 30 (tiny) per step. With `parameters_as_bucket_view=True` (`_build_param_buckets`, `:1317`), why does it drop to one per rank?* *Q: where is the gradient reduced?* (Not here: ZeRO relies on DDP's all-reduce of the full gradient, so it moves 1.5× the bytes of a reduce-scatter + all-gather, §1.2.)

### Stop 4 — `consolidate_state_dict` and `state_dict` (`:513`, `:1223`)

Every rank's local `state_dict()` is moved to rank `to` with `_broadcast_object` (a pickled tensor broadcast; the comment says broadcast is used instead of gather for NCCL compatibility). `state_dict()` then re-indexes into the global parameter order. *Q: what does a checkpoint of a ZeRO-1 run contain, and what do you need to restart on a different W?* (Full consolidated state, or per-rank slices plus their ranges and a resharding step. Not tested today.)

### Stop 5 — `_single_tensor_adam` (`torch/optim/adam.py:347`; `AdamW` is `Adam` with `decoupled_weight_decay=True`, `adamw.py:19`)

Find `param.mul_(1 - lr * weight_decay)`, `exp_avg.lerp_(grad, 1 - beta1)`, `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)`, the bias corrections from `step`, `denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)`, `param.addcdiv_(exp_avg, denom, value=-step_size)`. *Q: name the one line that would break if a slice cut through a tensor and the update needed the whole tensor. (None here. That is §1.3.)* *Q: which temporaries are full-size?* (`sqrt()` and the division: on a single flat tensor that is a transient of up to 2 × the slice, on 78 tensors at most 2 × the largest.)

### Stop 6 — the collectives (`distributed_c10d.py:5800` `reduce_scatter_single`, `:5061` `all_gather_single`)

torch 2.14 deprecates `reduce_scatter_tensor` / `all_gather_into_tensor` in favour of these names (`zero1.py` falls back to the old ones on older releases). *Q: what do the docstrings require of the input and output sizes? Why does that force the padding?* The in-place rule is NCCL's, not PyTorch's: output = this rank's slice of the input (reduce-scatter), input = this rank's slice of the output (all-gather).

### Stop 7 — `hook_with_zero_step` (`algorithms/ddp_comm_hooks/ddp_zero_hook.py:183`)

Constructing `ZeroRedundancyOptimizer(..., overlap_with_ddp=True)` must be paired with registering a compatible DDP communication hook; the constructor alone does not install that hook. In the archived `hook_with_zero_step` implementation, optimizer computation starts after the last gradient bucket's computation finishes, and may overlap outstanding backward communication and parameter broadcasts. This is not an unconditional rule to update every parameter as soon as its gradient arrives. Initialization waits for DDP's bucket layout to stabilize, so the documented first two or three iterations do not perform parameter updates. Ordinary explicit `zero.step()` is not the update path in this mode. *Q: compare with Day 03's optimizer graph: what does each overlap or remove?* This overlap path was read, not benchmarked in Day 06.

### Stop 8 (preview, 10 min) — FSDP2 and DeepSpeed

- **FSDP2** (`fsdp/_fully_shard/_fully_shard.py:413` `set_requires_gradient_sync`, `:490` `set_reshard_after_backward`). *Q for Day 7: with gradient sync off, is the unsharded gradient kept between microbatches (the `zero1` row of §1.4) or reduce-scattered and accumulated (the `rs_every` row)?*
- **DeepSpeed** `runtime/zero/stage_1_and_2.py` (GitHub). *Q for Day 8: for stage 1 vs stage 2, where are gradients reduced during accumulation, and which of today's two rows does each match? Is the reduction a reduce-scatter or an all-reduce? Find the all-gather of the updated fp32 → fp16/bf16 partitions in `step()`.*

### Fill this in `notes.md` §3

| Item | Your answer (draft) |
| --- | --- |
| Entry point | `optimizer.step()` on a `ZeroRedundancyOptimizer` wrapping `AdamW`, after DDP's backward |
| Partition | greedy, largest tensor first, to the least-loaded rank (`_partition_parameters`); whole tensors |
| Gradient reduction | DDP's bucket all-reduce of the full gradient (ZeRO does none itself) |
| Local step | `_local_step`: sync LR into the local AdamW, `self.optim.step()` on owned tensors |
| Parameter sync | `_sync_params`: one `broadcast(src=owner)` per parameter (or per rank with bucket views) |
| State | only owned tensors have `exp_avg`/`exp_avg_sq`; `consolidate_state_dict` broadcasts each rank's local state dict to one rank |
| Mine vs framework | flat equal slices, reduce-scatter + all-gather (1 × the bytes) vs whole tensors, all-reduce + broadcasts (1.5 × the bytes) |

---

## Block 4 — Run + summarize on the dual A100

Plain `python` for the orchestrator; it launches `torchrun` itself and runs the smoke test first. Or run `bash experiments/day06_zero1/run_server_once.sh "a100_${RID}"`, which does all of this, checks the outputs and writes an archive.

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p outputs
nvidia-smi topo -m > "outputs/day06_topo_${RID}.txt"
python experiments/day06_zero1/compare_zero1.py --stage parity   --condition shared --run-id "a100_${RID}"
python experiments/day06_zero1/compare_zero1.py --stage count    --condition shared --run-id "a100_${RID}" --no-smoke
python experiments/day06_zero1/compare_zero1.py --stage timing   --condition shared --run-id "a100_${RID}" --no-smoke --repeats 3
python experiments/day06_zero1/compare_zero1.py --stage slowlink --condition shared --run-id "a100_${RID}" --no-smoke --repeats 2
```

Manual run `a100_20261102T124935Z` took about 11 minutes from the run-ID timestamp to final status (20:49:35–21:00:35 Singapore time). Individual-stage durations were not separately recorded; the earlier 1.6/1.4/5.7/1.8 minute estimates belong to the assistant's pre-test. Timing and count runs save metrics only (`--no-tensors`). Download `outputs/day06_zero1/` (or the archive) before releasing the instance.

### How to read the parity output

A/B/C are Day 02's checks, D/E Day 04's, all against one process with one big batch per step (windows `[4, 4, 4, 4, 4, 2]`). R and S are today's. Each block also prints one `state rank r` line per rank and an `i` line with the state's drift against the 1-process run (information only). The BITWISE block compares each variant with `ddp` on five quantities; the boundary group must be all `True`. Controls print lower-case `pass`/`fail` and a `DETECTED by check X` verdict; a note marks controls that kept R green.

### How to read the count table

One `ncclDevKernel_*` per collective in the rank-0 trace, divided by the 2 profiled optimizer steps (Day 04's counter). Compare with your §1.8 row.

### How to read the timing tables

- **ms / opt step**: all K forward/backward passes, the reduction, the optimizer, the all-gather and the scheduler, with synchronized phase boundaries (these syncs are the same for every variant). Slowest rank per step, median over steps, [min–max] over repeats.
- **fwd+bwd** includes DDP's overlapped bucket all-reduces and `rs_every`'s per-microbatch reduce-scatters.
- **reduce max/last**, **gather max/last**: the phase on the slowest rank / on the last-arriving rank. The first number includes rank skew; the second reduces arrival waiting but is still a host phase duration, not isolated NCCL transfer time.
- **optim**: the local AdamW step; for `torch_zero` it also contains the 78 broadcasts and the wait for the other rank's step.
- **peak**, **between**, **state/rank**: §1.4 and §1.8.
- Shared host: differences of a few ms per optimizer step are within noise. Rank the phases, not only the totals.

Things to extract for the notes: the optimizer phase for `ddp`, `flat_ar`, `zero1` (flattening vs sharding); `zero1` − `flat_ar` in total and per phase; the memory differences against §1.4; on the slow link, `zero1` vs `flat_ar` (same bytes, two collectives instead of one) and `zero1_rs_every` − `zero1` (K − 1 extra reduce-scatters).

### Write `day06/notes.md`

[`notes.md`](notes.md) already contains your verified `a100_20261102T124935Z` tables and updated timing conclusions. Continue adding your §1.8 reasoning and personal Block 3 reading; keep measured results separate from interpretations and untested paths.

---

## Today's core question, answered

**What does ZeRO-1 shard, what does it cost in communication, and where does the accumulation boundary sit?**

It shards the optimizer state: each rank keeps AdamW's `exp_avg` and `exp_avg_sq` only for its 1/W of the parameters, which saves 2P(W − 1)/W per rank in FP32 (104.7 MiB here) and 12Ψ(W − 1)/W bytes with mixed precision. Parameters and the full gradient stay replicated during forward and backward. Because AdamW is elementwise, the update of one slice needs only that slice's gradient, so the all-reduce splits into its two halves with the optimizer in between: a reduce-scatter delivers each rank its averaged gradient slice, the rank updates its parameter slice, and an all-gather returns the full parameters. That is the same number of bytes as one all-reduce, with one extra collective after the optimizer that, without bucketing, overlaps nothing. `ZeroRedundancyOptimizer` takes another route: whole tensors per rank, DDP's full all-reduce, then a broadcast per parameter, 1.5× the bytes.

With gradient accumulation, the reduction belongs at the window boundary. ZeRO-1 can accumulate the full local gradient for K microbatches and reduce-scatter it once (the `no_sync` analogue; P of gradient memory between microbatches, one reduce-scatter per window), or reduce-scatter every microbatch into a P/W accumulator (ZeRO-2's pattern; P/W between microbatches, K reduce-scatters). Either way there is one optimizer step and one all-gather per window, and the parameters are full on every rank for the whole window.

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. What is redundant in DDP + AdamW, and what does each ZeRO stage remove?</summary>

Every rank holds the same parameters, the same reduced gradients and the same AdamW state. Stage 1 shards the state (2P in FP32, 12Ψ bytes with mixed precision), stage 2 also the gradients, stage 3 also the parameters. Only the state is needed by nobody except the optimizer once per step, so it is the cheapest to shard: it costs no extra communication volume.
</details>

<details><summary>2. Why does ZeRO-1 move the same bytes as DDP?</summary>

A ring all-reduce is a reduce-scatter followed by an all-gather, (W − 1)/W · P sent per rank in each half. ZeRO-1 keeps both halves and moves the optimizer between them: the reduce-scatter yields exactly the gradient slice a rank updates, and the all-gather sends updated parameters instead of reduced gradients.
</details>

<details><summary>3. Why may the flat slices cut through a weight matrix? When may they not?</summary>

AdamW's update is elementwise with shared scalars, so any partition of the elements gives the same numbers. Gradient clipping needs a global norm (one scalar all-reduce of the slices' sums of squares); LAMB, Adafactor need per-tensor statistics; Shampoo and Muon need the whole matrix. Those need whole-tensor ownership or extra collectives.
</details>

<details><summary>4. Why is zero1 bitwise equal to DDP + AdamW at W = 2, and when would it stop being so?</summary>

Both reduce the same locally accumulated window sum once; at W = 2 each element is fl(a + b) whichever collective computes it, and ×½ is exact. AdamW is elementwise, so slicing does not change any element's arithmetic. With W ≥ 3 the summation order of an all-reduce and a reduce-scatter may differ; with per-microbatch reduce-scatters (rs_every) the rounding differs already at W = 2.
</details>

<details><summary>5. Where is the accumulation boundary for ZeRO-1, and what stays allocated between microbatches?</summary>

At the last microbatch of the window: one reduce-scatter of the locally accumulated sum, the sharded step, one all-gather. Between microbatches the full gradient buffer (P) stays allocated. The alternative reduce-scatters every microbatch into a P/W accumulator and frees the full gradient: P − P/W less memory, K − 1 more reduce-scatters per window. Parameters are full throughout; nothing is gathered inside the window.
</details>

<details><summary>6. Why can a ZeRO-1 bug keep the replicas identical?</summary>

The all-gather copies each owner's slice to everyone. If an owner computed its slice from the wrong gradient (no reduction, zeroed views, overwritten accumulator), every rank receives the same wrong values. Replica agreement proves only that the all-gather ran.
</details>

<details><summary>7. How does ZeroRedundancyOptimizer differ from FlatZeroAdamW?</summary>

Whole tensors per rank (greedy by size), no gradient sharding (DDP's all-reduce of the full gradient), a local optimizer over owned tensors, and one broadcast per parameter from its owner (or one per rank with `parameters_as_bucket_view`). 1.5× the bytes of reduce-scatter + all-gather at W = 2, approximate balance, no split tensors. Overlap mode requires a compatible registered DDP hook; the archived `hook_with_zero_step` starts optimizer computation after the last bucket's gradient computation and overlaps outstanding communication.
</details>

<details><summary>8. Is ZeRO-1 faster than DDP?</summary>

Not by itself. It does the same communication volume plus one more collective, and the optimizer works on 1/W of the elements, which saves only a memory-bound elementwise pass (about 0.8 ms in the K = 1 matched optimizer-phase comparison here). A naive comparison against per-tensor AdamW shows a large speed-up that comes from running AdamW on one flat tensor instead of 78, i.e. from flattening. ZeRO-1 is a memory technique; its time cost is the extra all-gather unless it is overlapped.
</details>

**Next (Day 7):** FSDP2. `fully_shard` on the whole model vs per block; DDP vs FSDP2 memory and time; find the `all_gather` / `reduce_scatter` in the trace. Today's §1.4 table is the starting point: FSDP2 shards parameters too, so it gathers them per forward and per backward, and `set_requires_gradient_sync(False)` decides which of today's two accumulation rows it follows.

---

## Appendix A — My measured run (2026-11-02, `a100_20261102T124935Z`)

Manual run on **2 × A100-SXM4-40GB**, rental `ssh4.vast.ai:17591`, NV12, both GPUs on NUMA node 1, Python 3.12.15, torch 2.14.1+cu130, NCCL 2.30.7, `condition=shared`. Session: about 20:49:35–21:00:35 Singapore time (12:49:35–13:00:35 UTC). Exit code 0; **52 runs, 12 traces, 3 summary tables, 4 detected controls**.

Local archive: `C:/Users/qly24/Downloads/day06_a100_20261102T124935Z/day06_a100_20261102T124935Z.tar.gz`; extracted evidence is the sibling `results/` directory. Archive SHA256 `82a19058bcec33877bafa1c85769d77a9d7e79fc1cdf8f8b078f51513bb2622c` and all **179** manifest files verified. Derived data: [`analysis.json`](../outputs/day06_downloads/a100_20261102T124935Z/analysis.json). The earlier assistant run `pretest1` stays in `outputs/day06_downloads/pretest1/`; its timings are not mixed into these tables.

### A.1 Parity (tiny; 6 optimizer steps of G = 32 = 2 × K 4 × m 4; last window G = 16)

| Variant | A grad | B params-1 | C | D | E last window | R | S vs DDP | Bitwise = `ddp` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ddp` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | yes | (reference) | — |
| `torch_zero`, `torch_zero_bucket`, `flat_ar`, `zero1` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | yes | 0 | **yes**, all five quantities |
| `zero1_rs_every` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.382e-9 | yes | 4.7e-10 / 7.1e-14 | no (final params 3.643e-7) |

State vs the 1-process reference after 6 steps: 1.7e-4 (`exp_avg`) and 1.2e-5 (`exp_avg_sq`) relative, for every variant including DDP (trajectory drift, information only).

| Control | Fails | Passes | Verdict |
| --- | --- | --- | --- |
| `no_reduce_scatter` | A, B, E, S | C, D, **R** | DETECTED by A |
| `no_allgather` | B (1.52e-5 vs 1e-5), E, **R** (after step 1), S | **A**, C, D | DETECTED by R |
| `grad_set_to_none` | A (gradient exactly 0), B, E, S | C, D, **R** | DETECTED by A |
| `rs_overwrite` | A, B, E (ratio 0.650), S | C, D, **R** | DETECTED by A |

`PARITY: ALL CHECKS PASSED`. All §1.6 predictions held.

### A.2 State per rank and memory (medium)

| Variant | Elements per rank | `exp_avg` + `exp_avg_sq` per rank | Allocation across step 1 | Between microbatches (K = 4) | Peak (K = 4) |
| --- | --- | --- | --- | --- | --- |
| `ddp` | 27,443,200 (78 tensors) | 209.38 | 210.38 | 595.1 | 1227.3 |
| `torch_zero` | 13,762,560 (14 tensors) / 13,680,640 (64 tensors) | 105.00 / 104.38 | 106.00 / 105.38 | 490.7 | 1122.9 |
| `flat_ar` | 27,443,200 | 209.38 | 209.38 | 483.1 | 1115.3 |
| `zero1` | 13,721,600 | 104.69 | 104.69 | 378.4 | 1010.6 |
| `zero1_rs_every` | 13,721,600 | 104.69 | 104.69 | 326.0 | 958.3 |

MiB. State and first-step allocation use repeat 0's per-rank records; between-microbatch memory is the maximum over those ranks, and peak is the median over 3 repeat peaks. Flat state decreases 209.38 → 104.69 MiB per rank, matching P = 104.6875 MiB saved; `rs_every` saves another 52.34375 MiB between microbatches. Peak falls 1115.3 → 1010.6 MiB against the matched flat control (**9.4%**), or 1227.3 → 1010.6 MiB against DDP (which also changes bucket/layout overhead).

The flat memory ledger has a residual of about 64.3 MiB; DDP-based variants carry another 7.31 MiB after accounting for their actual state ownership. Per-tensor optimizers' first-step allocation exceeds their moment tensors by 1.00 MiB. Workspace and allocator explanations are interpretations; these residuals were not independently isolated in this run. Tiny-model state figures and additional per-rank details are in notes §6.

### A.3 Collectives per optimizer step (medium, K = 4, rank-0 trace)

| Variant | AllReduce | ReduceScatter | AllGather | Broadcast |
| --- | --- | --- | --- | --- |
| `ddp` | 4 | 0 | 0 | 1 |
| `torch_zero` | 4 | 0 | 0 | 79 |
| `torch_zero_bucket` | 4 | 0 | 0 | 3 |
| `flat_ar` | 1 | 0 | 0 | 0 |
| `zero1` | 0 | 1 | 1 | 0 |
| `zero1_rs_every` | 0 | 4 | 1 | 0 |

Counts match the earlier pre-test: `zero1` has one gradient reduce-scatter and one parameter all-gather per optimizer step; AdamW moments remain local. `torch_zero` has 78 parameter broadcasts plus one DDP buffer broadcast; bucket views reduce the 78 to 2.

### A.4 Timing (medium, m = 8, 25 steps, 5 warmup; NVLink 3 repeats, slow link 2)

NVLink, from `timing_summary_a100_20261102T124935Z.json`:

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

Slow link (`NCCL_P2P_DISABLE=1`), K = 4, from `slowlink_summary_a100_20261102T124935Z.json`:

| Variant | ms / opt step [min–max] | fwd+bwd | reduce max / last | optimizer | gather max / last |
| --- | --- | --- | --- | --- | --- |
| `ddp` | 156.54 [155.56–157.52] | 148.91 | — | 10.39 | — |
| `flat_ar` | 158.53 [158.17–158.90] | 103.35 | 53.73 / 53.34 | 1.75 | — |
| `zero1` | 163.22 [163.06–163.37] | 103.52 | 29.70 / 29.05 | 1.01 | 29.65 / 29.57 |
| `zero1_rs_every` | 251.94 [251.36–252.52] | 221.20 | — | 0.95 | 29.73 / 29.69 |

All times are ms. Core: slowest rank per step, median over steps, then median [min–max] over repeats. Phase values are aggregated separately and need not sum to the core median. DDP/torch ZeRO reductions are inside fwd+bwd; torch ZeRO's parameter broadcasts are inside optimizer. `rs_every` reductions are inside fwd+bwd. A dash indicates that communication is elsewhere or absent. The min-rank phase is a later-arriving-rank estimate and still includes dispatch/synchronization; max-rank phase also includes arrival waiting. Neither is automatically pure NCCL transfer time.

### A.5 Findings and limits from this run

- **Most of the observed DDP → flat ZeRO speed-up is associated with flattening.** At K = 1, core time falls 41.06 → 32.69 → 32.04 ms (`ddp` → `flat_ar` → `zero1`): 8.37 ms of the 9.01 ms total difference is already present in the unsharded flat control. Optimizer phases are 10.93 → 1.83 → 1.04 ms; the additional local optimizer saving from sharding is about 0.80 ms. These phase medians support the mechanism but are not an additive causal decomposition of the core median.
- **Equal communication volume does not imply equal latency.** Last-arriving-rank phase medians on NVLink are 1.01 ms for the flat all-reduce, versus 0.65 + 0.53 = 1.18 ms for reduce-scatter + all-gather. On the slow link they are 53.34 versus 29.05 + 29.57 = 58.62 ms, about 9.9% more. The additional 5.28 ms on the slow path is measured, but its cause was not isolated: it cannot be assigned entirely to one extra API/kernel launch. Collective/protocol efficiency and fallback transport/progress costs may contribute. No slow-link profiler trace was saved.
- **Matched total times:** `zero1` is 0.64 ms faster than `flat_ar` at K = 1 on NVLink (32.04 vs 32.69); the three observed repeat ranges do not overlap, although the effect is small and the `zero1` repeats span 28.04–32.25 ms. At K = 4 the median difference is 0.66 ms (121.49 vs 122.15), with overlapping ranges and a fast 106.22 ms `flat_ar` repeat; that difference is unresolved. On the slow link `zero1` is 4.69 ms slower (163.22 vs 158.53), with disjoint repeat ranges.
- **The accumulation boundary trades memory for communication.** On the slow link, `zero1_rs_every` costs 251.94 − 163.22 = 88.73 ms more per step, close to the K − 1 = 3 additional reduce-scatters (3 × 29.05 = 87.15 ms). It saves 52.34 MiB between microbatches. On NVLink, 120.41 vs 121.49 ms has overlapping repeat ranges, so these runs do not resolve the total-time effect.
- **Rank skew can dominate an exposed collective phase.** For NVLink `zero1`, K = 4, reduce max / last is 7.57 / 0.65 ms: about 6.92 ms of rank waiting separates the two statistics. Across the flat NVLink reductions the gaps range from 0.26 to 6.92 ms. The maximum includes waiting and cannot be interpreted as pure link latency.
- **Framework ZeRO has a different cost profile.** `torch_zero` is slower in median than DDP at K = 1 (46.70 vs 41.06 ms; overlapping repeat ranges) and K = 4 (143.97 vs 137.21 ms; disjoint ranges). At K = 4 its optimizer phase is 21.83 ms versus DDP's 14.37 ms. The trace has 78 parameter broadcasts plus DDP's buffer broadcast; rank 1 owns 64 of the 78 tensors. This supports the extra synchronization/per-tensor launch cost, but does not isolate every contribution.
- **DDP's overlapped layout is consistent with lower slow-link core time:** 156.54 ms versus `flat_ar`'s 158.53 ms. Their two-repeat ranges are disjoint. The implementations also differ in optimizer and gradient layout, so these totals alone do not quantify the benefit of overlap.

- NVLink K = 4 `zero1` vs `flat_ar`, and `zero1_rs_every` vs `zero1`: overlapping repeat ranges. K = 1's small observed advantage has only three repeats and does not establish a general speed-up.
- The fast NVLink repeats (`flat_ar`, K = 4: 106.22 ms; `zero1`, K = 1: 28.04 ms) are retained in the ranges. Shared-host scheduling, CPU launch and GPU frequency effects were not separately measured.
- DDP's optimizer phase varies 10.39–14.37 ms across stages and K; `torch_zero`, K = 4 reaches 21.83 ms. Causes were not isolated. Phase medians need not sum to core medians, because ranks, steps and repeats are aggregated separately.
- The 64.3 MiB residual above the flat memory ledger, the extra 7.31 MiB in the DDP-based rows and their +1.00 MiB first-step allocation excess were not independently isolated in this run. Allocator rounding and cuBLAS workspaces are interpretations, not separate measurements here.
- **[not tested]** bucketed or overlapped reduce-scatter / all-gather (`overlap_with_ddp`, FSDP2 prefetch), mixed precision with an fp32 master slice, gradient clipping on slices, a non-zero pad, W ≥ 3, sharded checkpoint save/restore. Slow-link costs are host-specific: 53.34 ms per 104.7 MiB flat all-reduce in this run; do not substitute Day 04's different-host timings.

### A.6 Communication diagnosis: rank skew versus deliberately disabled P2P

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

Therefore: normal-link NCCL work on the later-arriving rank is still sub-ms in these samples; the earlier rank can spend tens of ms waiting inside the same NCCL kernel. Separately, disabling P2P produces tens-of-ms host communication phases even on the less-waiting rank. **The extra slow-link 58.62 − 53.34 = 5.28 ms cannot be attributed solely to an extra launch from the available data.** There are no slow-link GPU traces in this run, so its transport/launch/progress breakdown remains unresolved. No new GPU run was performed for this analysis (the later follow-up run is A.7).

Derived evidence: [`communication_diagnosis.json`](../outputs/day06_downloads/a100_20261102T124935Z/communication_diagnosis.json), computed from the downloaded per-repeat metrics and both normal count traces.


### A.7 Follow-up: aligning the ranks before measuring a collective (run `a100_sync_20261103T120618Z`)

Separate diagnostic run on a different host (`ssh2.vast.ai:12197`, NV12, NUMA node 0, runtime NCCL 2.30.7). Code: `experiments/day06_comm_sync/`; report: [`COMM_SYNC.md`](COMM_SYNC.md). A.1–A.6 above are unchanged.

- **A barrier after the previous iteration is not enough.** Starts were already aligned to ~0.1 ms by the previous blocking collective. Skew builds during forward/backward, so `prev_step` left per-job maximum arrival gaps of 3–19 ms.
- **A barrier immediately before the collective** makes both ranks' NCCL kernels equal: normal reduce-scatter 0.57 / 0.43 ms, all-reduce 0.65 / 0.67 ms; disabled P2P 18.5 / 18.5 and 37.2 / 37.1 ms. The wait moves into the barrier, and loop time per step does not improve. To read communication cost, use the later-arriving rank's kernel, or align immediately before and report the barrier wait next to it.
- **Disabled P2P** runs over `SHM/direct` with 4 channels (normal: `P2P/CUMEM/read`, 24 channels), Ring/Simple. The RS + AG excess over one all-reduce ranged from −0.2 ms (aligned traces) to +3.5 / +6.9 ms (host phases / communication-only benchmark), within 25–35% drift of the slow transport: unresolved.
