# Day 04 — Gradient accumulation and `no_sync()`

> Setup: **Vast.ai, 2 × A100-SXM4-40GB**, GPU ids 0,1, NV12, both GPUs on NUMA node 1 (CPU affinity 32–63, 96–127); PyTorch 2.14.1+cu130 (git `5c48869`), CUDA build 13.0, NCCL 2.30.7, Python 3.12.15, `OMP_NUM_THREADS=1`, `condition=shared`.
> Run: **`a100_20261102T042757Z`**, 2026-11-02 04:28–04:47 UTC, server exit code 0. Downloaded to `outputs/day04_downloads/a100_20261102T042757Z/`; the archive and all 287 manifest files pass SHA-256 verification (`download_verification.json`). 103 sub-experiments, 14 traces, 10 original logs, 3 summary tables. The code snapshot in the archive is byte-identical to `experiments/day04_grad_accum/` in this repository.
> Code: [`experiments/day04_grad_accum/`](../experiments/day04_grad_accum/README.md) · Guide: [`GUIDE.md`](GUIDE.md) (its Appendix A is the assistant's 2026-11-01 pre-test on another host, kept for comparison only)

**Status: complete** (2026-11-02). The done-when criterion, "matches one equivalent large-batch step", is met by the measured parity below, including the short final window and the step counters.

How to read the evidence labels in this file:

- **[measured]** is a result of run `a100_20261102T042757Z`, with the log or table it comes from.
- **[understanding]** is what I can explain after the walkthrough discussion. It is consistent with the measurements but was not independently re-derived or re-implemented by me.
- **[not tested]** was not exercised by any run.

## 1. What systems problem did I solve today?

The global batch is an optimization choice; the microbatch is a memory choice. Gradient accumulation runs K forward/backward passes of m rows per rank and takes one optimizer step, so G = W × K × m. The same harness ran accumulation over Day 02's flat sync, over Day 03's MiniDDP extended with a `no_sync()` (the provided `MiniDDPAccum`, run unchanged), and over PyTorch DDP. Three things were checked against one big-batch step per optimizer step: loss scaling (including a short final window), step counters (LR schedule and Adam's `step`) and the communication pattern.

[measured] Every variant matches the big-batch reference, and each of the five planted loop bugs is caught by the check designed for it. With K = 4, DDP's collectives per optimizer step drop from 16 all-reduces + 4 broadcasts to 4 + 1 under `no_sync()`. On the P2P-disabled link this makes the step 1.87× faster for DDP and 2.01× for flat sync. On NVLink the per-sync saving is ~2.6 ms for flat sync and is not separable from noise for the bucketed wrappers. Whenever the big batch fits, it is 1.41–1.52× faster than accumulating it.

## 2. The mechanism in my own words [understanding]

1. **Zero once per window.** `AccumulateGrad` steals the first microbatch's gradient into `p.grad` (when `.grad` is `None`) and adds later ones into it. With bucket views, `p.grad` is a slot of the bucket tensor, so the addition writes straight into the bucket; nothing has to be copied in at the end. Accumulation is "not zeroing".
2. **Weight by the share of the actual window.** Microbatch j's mean loss is multiplied by n_j / N, its rows over this rank's rows in the window. That equals 1/K only for K equal microbatches; a short final window of n microbatches needs 1/n, uneven microbatches need row weights. DDP's 1/W then turns the per-rank means into the global mean.
3. **Step and schedule once per window.** The LR schedule and Adam's bias correction count optimizer steps, not microbatches.
4. **Communicate once, decided in forward.** `no_sync()` only sets `require_backward_grad_sync = False`. DDP reads the flag in `_post_forward`: set → `reducer.prepare_for_backward()` arms the Reducer (`expect_autograd_hooks_ = true`); clear → it is not armed. During backward the C++ `autograd_hook` still fires for every parameter and returns early when unarmed, so `p.grad` just accumulates locally. A backward under `no_sync()` whose forward ran outside it is already armed and communicates anyway. The last microbatch runs its forward outside the context, so its backward reduces the bucket contents, which by then hold the weighted sum Σ_j w_j g_j of the whole window, exactly as on Day 03.

### 2.1 Why "every" and "no_sync" agree, but not bitwise

After a sync every rank holds the same average ḡ₁; adding local gradients and averaging again is linear: ½[(ḡ₁ + g₂⁰) + (ḡ₁ + g₂¹)] = ḡ₁ + ḡ₂. The roundings differ: fl(fl(ḡ₁ + g₂⁰) + fl(ḡ₁ + g₂¹)) vs fl(fl(g₁⁰ + g₂⁰) + fl(g₁¹ + g₂¹)). Within each group, flat, MiniDDP and DDP perform the same two-operand sums at W = 2 (pre-scaling by ½ is exact), so they are bitwise equal. [measured] Both predictions hold; see §5.

### 2.2 Why a scaling bug can hide

AdamW's update m̂ / (√v̂ + ε) is invariant to a constant factor on every gradient (up to ε). [measured, `semantics.log`] After 10 steps a persistent ×4 gradient moved the parameters by 1.872e-3 of the total training motion with AdamW (9.156e-3 with clipping) but 2.948× the motion with SGD. A one-off ×0.5 on the last step, the size of a 1/K bug in a short final window, moved them by 1.191e-2 with AdamW: the moments remember the correctly scaled past. A loss curve does not prove the scaling is right; a gradient-level check does.

## 3. Call chain traced (pytorch@v2.14.1)

Python: installed torch 2.14.1 (`find_symbols.log`); C++: `reducer.cpp` / `reducer.hpp` / `accumulate_grad.h` at the installed git revision `5c48869` (`environment.log`).

| Item | Recorded answer |
| --- | --- |
| Entry | `with ddp.no_sync(): loss = f(ddp(x)); loss.backward()` (`distributed.py:1660`) |
| Flag | `require_backward_grad_sync = False`, restored in `finally`; written only by the constructor, `__setstate__` and `no_sync` |
| Forward | `_pre_forward` (1745): skips `reducer.prepare_for_forward()`; `_rebuild_buckets()` is still called (a no-op until a synced backward recorded the order); `_check_sync_bufs_pre_fwd()` → `will_sync_module_buffers()` (2383) requires `require_forward_param_sync` |
| | `_post_forward` (1816): flag set → `require_forward_param_sync = True`, `prepare_for_backward([])`; clear → `require_forward_param_sync = False` |
| C++ arm/disarm | `Reducer::prepare_for_backward`: `expect_autograd_hooks_ = true`, `grad_ready_order_indices_.clear()`, `reset_bucket_counting()`; `Reducer::finalize_backward`: `expect_autograd_hooks_ = false` |
| Backward | `AccumulateGrad` adds into `p.grad` → post hook → `Reducer::autograd_hook`: bookkeeping and unused-parameter marking, then `if (!expect_autograd_hooks_) return;`, before `grad_ready_order_indices_.push_back` / `push_rebuilt_params` / `mark_variable_ready` |
| Last microbatch | forward outside the context → `prepare_for_backward` → Day 03's chain: `mark_variable_ready` → buckets → all-reduce → `queue_callback(finalize_backward)` |
| Buffers | the broadcast before a forward is gated by the *previous* forward's `require_forward_param_sync`, so a `no_sync` window broadcasts once instead of K times |
| Other readers | join path (1917); compiled-autograd hook `_register_accum_grad_hook` (1173), which reads the flag at backward time. Not exercised |

## 4. My minimal version vs the framework

| Aspect | MiniDDPAccum | DDP |
| --- | --- | --- |
| Flag | `require_backward_grad_sync`; `no_sync()` saves and restores it | identical |
| Where it is read | `forward()`: armed → Day 03's reset + `_expect_hooks = True`; else `_expect_hooks = False` | `_pre_forward` / `_post_forward` → `prepare_for_backward` → `expect_autograd_hooks_` |
| Hook under `no_sync` | Python hook fires, counts itself, returns before recording the order | C++ hook fires, takes the Reducer mutex, unused-parameter bookkeeping if enabled, returns |
| Finish | `finish_grad_sync()` called by the loop after every backward; a no-op after a `no_sync` forward | `finalize_backward` queued only when the last bucket launches; never reached in a `no_sync` backward |
| Buffers | broadcast once at construction | per-forward broadcast, skipped after a `no_sync` forward: 1 per window |
| Unused parameters | not supported | `local_used_map_` ORs usage over the whole `no_sync` session |

The training-loop parts (weights n_j / N, "last" taken from the actual window, one optimizer and scheduler step per window) are not in DDP at all. They are the user's job, and four of the five negative controls are bugs in exactly that code.

## 5. Evidence it is correct [measured, `parity.log`]

Tiny preset, 6 optimizer steps of G = 32 = 2 ranks × K 4 × m 4; the last window is G = 16, i.e. 2 microbatches per rank (windows `[4, 4, 4, 4, 4, 2]`). Reference: 1 process, one batch of G rows per optimizer step. Day 02 tolerances for A/B/C; D exact; E uses A's tolerance.

| Variant | A grad max diff | B params-1 max diff | C loss-0 diff | D LR + Adam step | E last-window grad max diff | Ranks in sync |
| --- | --- | --- | --- | --- | --- | --- |
| 1 process, K = 8 × m 4 | 2.794e-9 | 2.608e-7 | 4.768e-7 | equal (6) | 8.382e-9 | — |
| flat / MiniDDP / MiniDDP view / DDP / DDP view, `no_sync` | 2.794e-9 | 2.608e-7 | 0 | equal (6) | 8.324e-9 | yes |
| flat / MiniDDP / DDP, every microbatch; DDP `bwd_only_nosync` | 2.794e-9 | 2.608e-7 | 0 | equal (6) | 8.440e-9 | yes |

Bitwise (`grads0`, `params1`, `grads_last`, final parameters): all five `no_sync` variants are identical, and all four "every" variants are identical. Every vs `no_sync` is **not** identical; final parameters differ by up to 4.230e-7. This matches §2.1.

| Negative control (DDP + `no_sync`) | Failing checks | Detail | Verdict |
| --- | --- | --- | --- |
| `no_loss_scale` | A, E | gradient ×4 (A 3.755e-2; E ratio 2.000 in the 2-microbatch window); **B passes** (5.0e-6): Adam hid it in the first update | detected by A |
| `zero_every_micro` | A, B, E | only the last microbatch survives; last-window gradient norm ratio 0.650 | detected by A |
| `sched_per_micro` | B, D, E | LR at optimizer step 0 is 7.5e-5 instead of 1.5e-5 | detected by D |
| `fixed_k_scale` | E only | last-window gradient ×0.500; A–D all pass | detected by E |
| `modulo_last_sync` | E only | the short window never syncs: gradient ratio 1.313, ranks **not** bitwise in sync at the end | detected by E |

`fixed_k_scale` and `modulo_last_sync` pass every full-window check. Without check E and the uneven last window, both would ship.

MiniDDP counters (tiny, 8 buckets): `no_sync` 8 collectives per optimizer step, with 30 armed and 80 skipped hook calls (30 parameter tensors × 16 unsynced backwards / 6 steps); every: 29.3 collectives per step (8 buckets × 22 backwards / 6).

`semantics.log` (one process) confirms the per-window weights directly: correct K = 4, short-window and uneven 8/8/4 weighting all give |g|/|g_ref| = 1.000; the "mean of microbatch means" for 8/8/4 rows gives 1.028.

## 6. What the performance data proves (and what it doesn't)

### 6.1 Method

Medium preset (27.4 M parameters, P = 104.7 MiB FP32 gradients), m = 8 rows per microbatch per rank, K ∈ {1, 2, 4, 8} (G = 16 … 128), 25 optimizer steps with 5 excluded, 2 repeats, TF32 off, MATH attention, `condition=shared`. Core time = all K forward/backward passes + syncs + optimizer/scheduler, with synchronization only at its two ends; slowest rank per step, median over steps; overhead vs `none` (no communication) is computed within the same repeat. Collective counts come from separate profiled runs.

### 6.2 Collectives per optimizer step [measured, `count.log`, K = 4, rank-0 trace, 2 profiled steps]

| | flat once | flat every | MiniDDP `no_sync` | MiniDDP every | DDP `no_sync` | DDP every | DDP `bwd_only_nosync` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AllReduce | 1 | 4 | 4 | 16 | 4 | 16 | 16 |
| Broadcast | 0 | 0 | 0 | 0 | 1 | 4 | 4 |

The 4 bucket all-reduces per sync and 1 broadcast per window are what §3 predicts. `bwd_only_nosync` is indistinguishable from syncing every microbatch, which is the forward-time decision observed in the trace.

### 6.3 Time per optimizer step on NVLink [measured, `timing.log`]

ms, median of 2 repeats; overhead vs `none` in parentheses.

| K (G) | none | flat once | flat every | MiniDDP `no_sync` | MiniDDP every | DDP `no_sync` | DDP every | DDP big batch | 1 GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (16) | 36.76 | 38.82 (+2.06) | — | 43.43 (+6.66) | — | 42.14 (+5.38) | — | — | 50.65 |
| 2 (32) | 63.55 | 75.97 (+12.41) | 63.16 (−0.39) | 81.01 (+17.46) | 82.95 (+19.40) | 79.52 (+15.97) | 81.77 (+18.22) | 52.20 | 98.69 |
| 4 (64) | 134.32 | 135.92 (+1.60) | 143.87 (+9.55) | 144.09 (+9.77) | 141.73 (+7.41) | 138.78 (+4.46) | 143.38 (+9.06) | 94.85 | 194.58 |
| 8 (128) | 251.92 | 258.56 (+6.64) | 276.74 (+24.82) | 270.64 (+18.72) | 262.60 (+10.69) | 256.97 (+5.05) | 274.97 (+23.05) | 181.64 | 386.73 |

The K = 2 row is noise-dominated: the two `none` repeats were 54.54 and 72.57 ms, so every K = 2 overhead has a ±9–10 ms range. Other overhead ranges: flat every K = 4 [+7.50, +11.59], K = 8 [+23.67, +25.97]; DDP every K = 4 [−0.37, +18.49], K = 8 [+12.08, +34.02]; DDP `no_sync` K = 4 [+2.72, +6.20], K = 8 [+1.26, +8.84].

Peak allocated memory: `none`/flat 1012–1014 MiB at K = 1 and 1123–1126 MiB at K ≥ 2; MiniDDP/DDP 1117 MiB at K = 1 and 1227–1230 MiB at K ≥ 2; DDP big batch 1750 / 3014 / 5543 MiB at K = 2 / 4 / 8.

### 6.4 Slow link, `NCCL_P2P_DISABLE=1` [measured, `slowlink.log`]

| K (G) | none | flat once | flat every | DDP `no_sync` | DDP every |
| --- | --- | --- | --- | --- | --- |
| 1 (16) | 38.26 | 78.87 (+40.62) | — | 69.53 (+31.28) | — |
| 4 (64) | 133.20 | 159.61 (+26.42) | 321.04 (+187.84) | 152.19 (+19.00) | 283.96 (+150.76) |

Repeat ranges at K = 4 are within ±3 ms for every cell.

### 6.5 What it proves

- **`no_sync` removes K − 1 syncs per optimizer step, and on a slow link that is most of the step.** At K = 4, every − `no_sync` is 131.8 ms for DDP and 161.4 ms for flat sync, i.e. 43.9 and 53.8 ms per avoided sync, with tight repeats. `no_sync` makes the step 1.87× (DDP) and 2.01× (flat) faster.
- **On NVLink, flat sync shows the (K − 1) × C growth:** every − once is 7.95 ms at K = 4 and 18.18 ms at K = 8, i.e. 2.65 and 2.60 ms per extra sync.
- **DDP's `no_sync` overhead does not grow with K:** +5.38, +4.46, +5.05 ms at K = 1, 4, 8, consistent with one bucketed sync per optimizer step.
- **Memory is flat in K for K ≥ 2.** K = 1 → 2 adds ≈ P (≈110 MiB), because with K ≥ 2 the accumulated `p.grad` already exists while a later microbatch's backward reaches its activation peak (Day 03 §7). `no_sync` itself saves no memory for DDP: buckets are allocated either way.
- **Accumulation is a memory tool, not a speed tool.** When the big batch fits it is faster: 52.20 vs 79.52 ms (1.52×) at K = 2, 94.85 vs 138.78 ms (1.46×) at K = 4, 181.64 vs 256.97 ms (1.41×) at K = 8, using 1.4–4.5× the memory. A linear fit of `none` gives ~30.7 ms per 8-row microbatch + ~6.0 ms per optimizer step. The big-batch runs give ~2.7 ms per row, ~21.6 ms per 8 rows. So roughly 9 ms of each small microbatch is a fixed cost that larger microbatches amortize. That fixed cost is consistent with Day 03's launch-bound diagnosis, but this run does not isolate it.
- **Two GPUs vs one at the same G:** DDP `no_sync` is 1.20× faster at K = 1 and 1.51× at K = 8 (strong scaling, same G and K).

### 6.6 What it doesn't

- **DDP every vs `no_sync` on NVLink is not resolved.** The medians point the expected way (+4.6 ms at K = 4, +18.0 ms at K = 8, ~2.6 ms per sync at K = 8), but DDP every's repeat ranges are 19–22 ms wide.
- **MiniDDP every vs `no_sync` on NVLink is not resolved, and the order is not reproducible.** Here every is *faster* than `no_sync` at K = 4 (−2.4 ms) and K = 8 (−8.0 ms), and MiniDDP `no_sync`'s overhead grows with K (+6.66 → +18.72 ms) while DDP's does not. In the 2026-11-01 pre-test on another host the K = 8 order was the opposite (MiniDDP `no_sync` −3.21 ms vs every +13.27 ms). With the sign flipping between runs, I do not attribute it to the implementation. Isolating it would need ≥ 3 repeats and a per-hook CPU microbenchmark.
- **The slow-link K = 1 vs K = 4 asymmetry is unexplained.** One `no_sync` sync of the same 105 MiB costs +31.3 / +40.6 ms (DDP / flat) at K = 1 but +19.0 / +26.4 ms at K = 4, and an avoided sync at K = 4 costs 44–54 ms. The pre-test host showed the same pattern, so it is not a one-off; the cause (CPU-side NCCL proxy work competing with launch-bound compute, or something else) was not isolated.
- **Slow-link costs are host-specific.** This host's 44–54 ms per sync differs from the pre-test host's ~74–79 ms; the pre-test host had its GPUs on different NUMA nodes and this one has both on node 1. That is consistent with the difference but not a one-variable comparison. Do not compare slow-link numbers across rentals.
- **[not tested]** masked-token weighting with a cross-rank valid-token count (only explained), AMP/`GradScaler` and gradient clipping on the accumulated gradient in DDP (clipping only in the one-process semantics demo), BatchNorm or dropout (the preset uses neither, which is why accumulation can match the big batch exactly), `find_unused_parameters=True` under `no_sync`, W > 2, and accumulation under FSDP2/ZeRO.

## 7. Completion assessment

Day 04's done-when criterion is met by measured evidence, and the mechanism is understood to the level of the call chain in §3. No further Day 04 exercise is required. Remaining gaps, none of which blocks Day 06:

| Gap | Kind | Where it gets resolved |
| --- | --- | --- |
| `no_sync()`/`forward()`/`finish_grad_sync()` were run from the provided code, not re-typed from memory | recall | Optional: rewrite the three methods without looking (~15 min) and rerun the tiny parity stage the next time the server is up |
| Per-hook CPU cost, MiniDDP NVLink ordering | performance attribution | Only if a later step is hook-bound; Day 03's open microbenchmark item covers it |
| Accumulation under sharding (what must stay unsharded or unreduced between microbatches) | new mechanism | Days 06–08: ZeRO-1 boundary, FSDP2 `set_requires_gradient_sync`, DeepSpeed `is_gradient_accumulation_boundary()` |
| Accumulation with mixed precision and clipping | new mechanism | Day 09 (`GradScaler.unscale_` once per window, clip the accumulated gradient) |
| Valid-token weighting across ranks | new mechanism | Day 18 (`response_mask`) |

## Open questions

- [ ] ZeRO-1 (Day 6): with accumulation, the sharded optimizer still needs the full averaged gradient once per window. Does the reduce become a `reduce_scatter` only at the boundary microbatch, and what stays allocated in between?
- [ ] FSDP2 `set_requires_gradient_sync(False)`: what has to stay unsharded between microbatches, and what does it cost in memory (Day 7)?
- [ ] DeepSpeed: where does `is_gradient_accumulation_boundary()` gate communication for ZeRO-1/2/3 (Day 8)?
- [ ] With token-level losses (Day 18), the window weight needs the global valid-token count before the first backward. Who computes it, and does that add a collective?
