# Day 04 — Gradient accumulation and `no_sync()` (guide, dual-A100 edition)

> **Status (2026-11-02): complete.** The learner's own run `a100_20261102T042757Z` passed every stage; its results and the completion assessment are in [`notes.md`](notes.md). Appendix A below remains the assistant's pre-test on a different host.

**Goal:** by tonight you can run K microbatches per optimizer step on two GPUs and show, with your own parity checks and profiler counts, that (1) the result equals one big-batch step, (2) the loss scaling, the step counters and a short final window are handled correctly, and (3) `no_sync()` cuts the collectives per optimizer step from K × (buckets + 1) to buckets + 1. You add `no_sync()` to Day 03's `MiniDDP` yourself (~20 lines), then read DDP's two flags and the Reducer's `expect_autograd_hooks_` to see that **the sync decision is taken in forward, not in backward**.

Day 03 left two numbers on the table: on the medium model DDP adds **+4.5 ms per step** over the no-communication floor at local batch 8, and **+15 ms** when P2P is disabled. With accumulation you pay that once per optimizer step, or K times, depending on one context manager. Today measures both.

**Done means:**

- [ ] `accum_semantics.py` run: you can state G = W × K × m, the number of optimizer steps per epoch with and without the short final window, and why AdamW hides a missing 1/K
- [ ] `MiniDDPAccum` written by you (the **YOU WRITE** parts of `mini_ddp_accum.py`), diffed against the reference
- [ ] `compare_accum.py --stage parity` prints `PARITY: ALL CHECKS PASSED`: every variant passes A/B/C/D/E against one big batch per step, and all five negative controls are `DETECTED` by their named check
- [ ] Bitwise lines explained: why the nosync group is bitwise equal, the every group is bitwise equal, and the two groups are not
- [ ] `--stage count`: collectives per optimizer step for flat, MiniDDP and DDP, every vs `no_sync`, plus the `bwd_only_nosync` trap
- [ ] `--stage timing` and `--stage slowlink`: ms per optimizer step vs K; you can say how much `no_sync` saves on NVLink and on the slow link, and why accumulation is slower than one big batch when the big batch fits
- [ ] Call chain traced from `with ddp.no_sync():` to `Reducer::autograd_hook`'s early return (Block 3); `day04/notes.md` written
- [ ] You can answer, without notes: *what exactly does `no_sync()` skip, why must the forward be inside it, and what does it not save?*

---

## Read this first: same host, new folder

Same machine and rules as Days 02–03: **2 × A100-SXM4-40GB, NV12, default NCCL settings**, torch 2.14.1+cu130, NCCL 2.30.7, `--condition shared`. Copy today's folder from your Windows project root (use the endpoint of your current rental; the pre-test used `ssh -p 20163 root@ssh6.vast.ai`):

```bash
scp -P <port> -r experiments/day04_grad_accum root@<host>:/workspace/AI-Infra-Learning/experiments/
```

Then, on the instance:

```bash
cd /workspace/AI-Infra-Learning
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
python -m torch.distributed.run --standalone --nproc_per_node 2 experiments/day02_data_parallel/hello_dist.py --device cuda   # smoke: stop if it fails
```

Today's code imports Day 01 (`train.py`), Day 02 (`ddp_manual.py`, `check_parity.py`, `launch_utils.py`, `find_symbols.py`) and Day 03 (`mini_ddp.py`), all unchanged.

**Measurement status.** Appendix A holds the **assistant's pre-test** on 2026-11-01 (run id `pretest1`), done to check that the code runs and the checks discriminate. Record your predictions in §1.9 before you look at it, then run with your own run id. Your numbers replace Appendix A in `notes.md`.

---

## Files

| File | What it is |
| --- | --- |
| `experiments/day04_grad_accum/accum_semantics.py` | One process, no NCCL. Bookkeeping table; accumulated vs big-batch gradient and each scaling bug; how SGD / AdamW / AdamW + clipping react to a gradient that is c times too large |
| `experiments/day04_grad_accum/mini_ddp_accum.py` | **The core you write.** `MiniDDPAccum(MiniDDP)`: `no_sync()`, a forward that arms the hooks only when syncing, a `finish_grad_sync` that is a no-op otherwise, counters |
| `experiments/day04_grad_accum/train_accum.py` | One harness: `--impl none / flat_sync / mini_ddp / torch_ddp` × `--sync nosync / every / bwd_only_nosync`; K derived from `--global-batch` and `--micro-batch`; `--last-global-batch` for a short final window; five `--bug` controls; `--profile` |
| `experiments/day04_grad_accum/compare_accum.py` | Orchestrator: smoke → `parity` → `count` → `timing` → `slowlink`. Launch with plain `python` |
| `experiments/day04_grad_accum/count_collectives.py` | NCCL kernels per optimizer step from a Chrome trace. Pure Python |
| `experiments/day04_grad_accum/find_symbols.py` | `file:line` for `no_sync`, `_pre_forward`, `_post_forward`, and every line touching the two sync flags; the C++ to open |
| `day04/assets/accum_schematic.svg` | §1.4 schematic, drawn by `draw_accum_schematic.py` |

---

## Block 1 — Design (30 min): draw these and write predictions before you run anything

### 1.1 The problem: the batch you want does not fit, or is not the batch you can afford per step

The global batch is an **optimization** choice (noise scale, LR, number of updates). The microbatch is a **memory and kernel-efficiency** choice: activations scale with the rows in one forward/backward (Day 01: at the start of backward, activations and the `[B·T, V]` logits gradients dominate the peak). Gradient accumulation decouples them: run K forward/backward passes on microbatches of m rows, let the gradients add up in `p.grad`, then take **one** optimizer step.

$$G = W \times K \times m \qquad \text{per-GPU batch per optimizer step} = K \times m \qquad \text{optimizer steps} = \text{windows}$$

| Symbol | Meaning | Today (timing) |
| --- | --- | --- |
| m | rows per microbatch per rank (one forward) | 8 |
| K | microbatches per optimizer step ("window") | 1, 2, 4, 8 |
| W | data-parallel ranks | 2 |
| G | rows per optimizer step over all ranks | 16, 32, 64, 128 |

`accum_semantics.py` Part 1 prints the epoch bookkeeping, for example N = 1000 rows, m = 8, K = 4, W = 2: each rank sees ⌈500 / 8⌉ = 63 microbatches, so 15 full windows and **one window of 3 microbatches** at the end. Keep it (16 optimizer steps, scaled correctly) or drop it (15). Both are defensible; scaling it as if it had 4 is not.

**The LR schedule and Adam's bias correction count optimizer steps.** In `train_accum.py`, `cfg.total_steps = --opt-steps`, and `scheduler.step()` runs once per window.

### 1.2 Why accumulation needs no code in autograd

From Day 03 §1.2: `AccumulateGrad` **steals** the incoming gradient when `p.grad` is None and **adds into** `p.grad` otherwise. Two backwards without `zero_grad` gave `w.grad = 5` then `10`. Accumulation is that, on purpose: zero once per window, backward K times.

The post-accumulate hook still fires on every backward. After microbatch j it sees $\sum_{i \le j} g_i$. That is why Day 03 said "final for *this* backward". With accumulation, "final for this window" is a decision the training loop has to communicate to the communication layer. That decision is `no_sync()`.

### 1.3 Loss scaling: which factor, and when it is not 1/K

The big-batch loss is the mean over all G rows. Each rank's contribution must be its own mean over its rows in the window, and DDP's 1/W average turns the per-rank means into the global mean (equal rows per rank, Day 02 §1.3). Microbatch j of a window with $N$ rows on this rank:

$$\mathcal{L}_\text{rank} = \sum_j \frac{n_j}{N}\,\overline{\ell}_j \qquad\Rightarrow\qquad \text{weight}_j = \frac{n_j}{N} = \frac{1}{K}\ \text{only if all K microbatches have } m \text{ rows}$$

`train_accum.py` uses `w = rows_j / rows_in_window`. Three ways to get it wrong, each a control today:

| Mistake | Gradient | Caught by |
| --- | --- | --- |
| no scaling (`loss.backward()`) | K × too large | A (step-0 gradient) |
| 1/K in a short final window of n < K microbatches | n/K × too small, in that window only | E (last-window gradient) |
| "mean of microbatch means" with an uneven last microbatch | slightly wrong weights | Part 2 of `accum_semantics.py` (1.028× here) |

With masked tokens (LLM SFT, RL with `response_mask`, Day 18) the same rule becomes **valid tokens in the microbatch / valid tokens in the window, over all ranks**. Per-rank means of different token counts are not the global mean, even without accumulation.

### 1.4 Where does communication go?

![One optimizer step of K = 4 microbatches on one rank. Top: DDP syncing every microbatch, with 4 bucket all-reduces on the NCCL stream under every backward and a buffer broadcast before every forward: 16 all-reduces and 4 broadcasts per optimizer step. Middle: DDP with no_sync around microbatches 1 to 3: their hooks return early, one buffer broadcast before the first forward, and 4 bucket all-reduces only under the last backward: 4 plus 1. Bottom: Day 02 flat sync, accumulate locally and call one all-reduce on the compute stream after the last backward: 1 all-reduce, not overlapped](assets/accum_schematic.svg)

Schematic, not to scale (redraw with `python day04/assets/draw_accum_schematic.py`).

**Syncing every microbatch is correct.** After microbatch 1, each rank holds $\bar g_1$ (the average). Microbatch 2 adds its local gradient, the all-reduce averages again:

$$\tfrac12\big[(\bar g_1 + g_2^{(0)}) + (\bar g_1 + g_2^{(1)})\big] = \bar g_1 + \bar g_2$$

Averaging is linear and $\bar g_1$ is identical on both ranks, so the result is the same average. It just moves K × the bytes and launches K × the collectives. Predict the counts per optimizer step at K = 4 (medium: 4 DDP buckets after the rebuild, plus DDP's per-forward buffer broadcast of the causal mask, Day 03 §Stage 4):

| Variant | All-reduces / opt step | Broadcasts / opt step | Your prediction |
| --- | --- | --- | --- |
| flat, sync once (Day 02 `sync_grads` after the last microbatch) | 1 | 0 | |
| flat, every microbatch | K | 0 | |
| MiniDDP, `no_sync` | buckets | 0 | |
| MiniDDP, every | K × buckets | 0 | |
| DDP, `no_sync` | buckets | ? (§1.5) | |
| DDP, every | K × buckets | ? | |
| DDP, `no_sync` around **backward only** | ? (§1.5) | ? | |

A rough cost model per optimizer step, with $T_\text{fb}(m)$ one forward+backward, $T_\text{opt}$ the optimizer and scheduler, and $C$ the per-sync overhead you measured on Day 03 (not just wire time; Day 03's ablation showed that most of DDP's +4.5 ms survives without any communication):

$$T_\text{nosync}(K) \approx K\,T_\text{fb}(m) + T_\text{opt} + C, \qquad T_\text{every}(K) \approx K\,T_\text{fb}(m) + T_\text{opt} + K\,C$$

On NVLink $C$ is a few ms, so the gap grows like $(K-1)\,C$. With P2P disabled $C$ was ~15 ms on Day 03's host (regimes table); it depends on the host, so measure it at K = 1. Either way `no_sync` should matter much more there. Per sample, `no_sync` amortizes both $C$ and $T_\text{opt}$ over K microbatches.

### 1.5 How `no_sync()` works: the decision is taken in forward

DDP's `no_sync()` only flips a flag (`torch/nn/parallel/distributed.py`, Block 3):

```python
old = self.require_backward_grad_sync
self.require_backward_grad_sync = False
try: yield
finally: self.require_backward_grad_sync = old
```

Nobody reads that flag during backward (on the C++ Reducer path used here). It is read in **forward**:

- `_pre_forward`: if grad is enabled and the flag is set, `reducer.prepare_for_forward()` (iteration counter, stats).
- `_post_forward`: if grad is enabled and the flag is set, `require_forward_param_sync = True` and **`reducer.prepare_for_backward(...)`**. Otherwise `require_forward_param_sync = False`.
- C++ `prepare_for_backward` sets `expect_autograd_hooks_ = true` and resets the bucket counts. `autograd_hook` starts with bookkeeping, then *"Ignore if we don't expect to be called. This may be the case if the user wants to accumulate gradients for number of iterations before reducing them."*: `if (!expect_autograd_hooks_) return;`. `finalize_backward` sets it back to false.

Three consequences to predict:

1. **Hooks still fire under `no_sync`.** They return early. `MiniDDPAccum` counts them (`hook_calls_skipped`): per optimizer step, (K − 1) × (number of parameters).
2. **The forward must be inside the context.** A forward outside `no_sync()` has already armed the Reducer. Its backward communicates even if it runs inside `no_sync()`. That is the `bwd_only_nosync` control: correct numbers, every-microbatch traffic. (The compiled-autograd "Python reducer" path reads the flag inside its hook instead, at backward time. Today's runs use the C++ Reducer.)
3. **Buffer broadcasts follow the same flag, one forward late.** `will_sync_module_buffers()` needs `require_forward_param_sync`, which the *previous* forward set. In a `no_sync` window only the first forward follows a synced one, so DDP broadcasts buffers **once per window**, not K times. Now fill the `?` in §1.4.

The ready order for the bucket rebuild is recorded after the early return (`grad_ready_order_indices_.push_back` comes after it). The first `no_sync` backwards therefore record nothing; DDP rebuilds from the first *synced* backward. `MiniDDP` behaves the same way, because its `_recording` code sits after the `_expect_hooks` check.

### 1.6 Numerics: what is bitwise, what is only close

Three comparisons, all on the tiny model, G = 32 per step (2 ranks × K = 4 × m = 4), final window G = 16:

| Comparison | Prediction | Why |
| --- | --- | --- |
| any variant vs 1 process, one big batch per step | within Day 02's tolerances | the big batch sums over 32 rows inside one matmul; accumulation sums 4 partial gradients with `+=`. Different float summation order |
| flat vs MiniDDP vs DDP, all `no_sync` | bitwise equal | each rank computes the same local sum $S = g_1 + g_2 + g_3 + g_4$ (same kernels, same `+=` order), then one two-operand all-reduce: fl(S⁰/2 + S¹/2) = fl(S⁰ + S¹)/2 (Day 03 §1.5) |
| all "every" variants vs each other | bitwise equal | same argument, applied after each microbatch |
| "every" vs `no_sync` | **not** bitwise | every: fl(fl(ḡ₁ + g₂⁰) + fl(ḡ₁ + g₂¹)); nosync: fl(fl(g₁⁰ + g₂⁰) + fl(g₁¹ + g₂¹)). Equal in exact arithmetic, different roundings |

Also write down: is a single-process run with K = 8 microbatches of 4 rows (`ref_accum8`) bitwise equal to the big batch? (No: same reason as row 1.)

### 1.7 Step counters

Two counters must advance **once per window**: the LR scheduler (`last_epoch`) and Adam's per-parameter `step` (bias correction $1 - \beta^t$). The second is automatic if `optimizer.step()` is called once per window. The first is the classic bug: `scheduler.step()` inside the microbatch loop makes warmup K × too short. In the `sched_per_micro` control the LR used for the very first optimizer step is already lr_lambda(4) instead of lr_lambda(0). Check D compares the LR of every optimizer step and Adam's counter with the reference, exactly.

### 1.8 Memory and speed: accumulation is not free

Accumulation keeps the **activation** memory of one microbatch: peak memory should be nearly flat in K. Parameters, gradients and Adam state (4P ≈ 420 MiB here, Day 01) do not change. DDP's buckets (P) do not change either, and `no_sync` does not remove them: it skips communication, not memory. (With **sharded** gradients, ZeRO-2/FSDP, skipping the sync means holding unsharded gradients: Days 6–8.)

The price is speed. K microbatches of 8 rows launch K × as many kernels as one batch of 8K rows, and Day 03 showed the medium model at local batch 8 is partly launch-bound (36 ms at local 8, 93 ms at local 32: 4 × the rows for 2.6 × the time). Predict: is `ddp_nosync` at K = 4 faster or slower than `ddp_bigbatch` (the same G = 64 as one microbatch of 32 rows per rank), and how much more memory does the big batch need?

### 1.9 Your predictions (fill in before Block 4)

| Quantity | How to estimate | Your prediction |
| --- | --- | --- |
| DDP `no_sync` collectives / opt step at K = 4 | §1.4, §1.5 | |
| DDP every collectives / opt step at K = 4 | §1.4, §1.5 | |
| `bwd_only_nosync` collectives | §1.5, consequence 2 | |
| `hook_calls_skipped` per opt step, MiniDDP `no_sync`, tiny (30 parameter tensors), 6 windows of 4,4,4,4,4,2 microbatches | §1.5, consequence 1 | |
| `ddp_every` − `ddp_nosync` at K = 4, NVLink | (K − 1) × C, C from Day 03 | |
| Same, P2P disabled | (K − 1) × C, C from the K = 1 slow-link row (Day 03's host: ~15 ms) | |
| Peak memory K = 8 vs K = 1 (`ddp_nosync`) | §1.8 | |
| `ddp_bigbatch` vs `ddp_nosync` at K = 4: time, memory | §1.8 | |
| Does AdamW hide a missing 1/K? A one-off 1/K in the last window? | §1.3, Part 3 | |

### 1.10 New rules (on top of Days 02–03)

1. **Zero once per window, step once per window, schedule once per window.** Everything that counts "steps" counts optimizer steps.
2. **Scale each microbatch by its share of the window**, computed from the actual window, not from K.
3. **Decide "last microbatch" from the actual window length**, not with `(i + 1) % K == 0`. A short final window that never syncs leaves the ranks with different gradients and different parameters, silently.
4. **Put the forward inside `no_sync()`**, not only the backward.
5. **Clip, unscale (AMP) and log gradient norms once per window**, after the last backward (when the gradient is synced), before `optimizer.step()`.

---

## Block 2 — Code (2 h)

### Stage 0 (15 min): one process, three questions

```bash
python experiments/day04_grad_accum/accum_semantics.py --device cuda
python experiments/day04_grad_accum/find_symbols.py
```

Part 2 shows that the correct rows differ from the big batch by summation-order noise (~3e-9), while each bug changes the gradient norm (4.000, 0.377, 0.500, 1.028). Part 3 is the surprise: **with AdamW a persistent ×4 gradient moves the parameters by ~0.2% of the training motion**, while SGD moves ~3 × further. Adam normalizes by $\sqrt{v}$, so a constant factor cancels up to ε. A one-off ×0.5 on the last step does not cancel, because m and v remember the correctly scaled past. Write in your notes why "the loss curve looks fine" does not prove the scaling is right.

### Stage 1 (20 min): the accumulation loop, from memory

Write this in a scratch file before opening `train_accum.py`. It is the whole of correct DDP gradient accumulation:

```python
optimizer.zero_grad(set_to_none=True)                   # once per window
micro = list(zip(x_rank.split(m), y_rank.split(m)))     # n <= K microbatches (short final window: n < K)
N = x_rank.size(0)                                      # this rank's rows in the window
for j, (xj, yj) in enumerate(micro):
    last = j == len(micro) - 1                          # from the ACTUAL window, not (j + 1) % K
    with (nullcontext() if last else ddp.no_sync()):    # forward AND backward inside
        loss = F.cross_entropy(ddp(xj).flatten(0, 1), yj.flatten())
        (loss * xj.size(0) / N).backward()              # weight = this microbatch's share of the window
optimizer.step()                                         # once per window
scheduler.step()                                         # once per window
```

Questions before moving on:

- What does `p.grad` hold on rank 0 just before the last backward, and just after it?
- Where would `clip_grad_norm_` go, and why not inside the loop?
- With `torch.amp.GradScaler`: which calls run per microbatch and which per window? (`scaler.scale(loss * w).backward()` per microbatch; `scaler.unscale_`, `scaler.step`, `scaler.update` once. Day 9.)
- Would this be exactly a big batch if the model had **BatchNorm**? (No: BN statistics are per forward, so per microbatch. LayerNorm is per row, so today's model decomposes exactly. Dropout would also draw different masks.)

### Stage 2 (30 min): `no_sync()` for MiniDDP — write the **YOU WRITE** parts of `mini_ddp_accum.py`

Copy `mini_ddp_accum.py` to a scratch name, delete the bodies under `YOU WRITE`, and rewrite them:

1. **`no_sync()`**: save `require_backward_grad_sync`, set it False, yield, **restore the saved value** in `finally` (nested contexts, exceptions).
2. **`forward()`**: if grad is enabled and the flag is set, call `super().forward()` (Day 03: rebuild if due, reset pending counts and `_next_bucket`, arm the hooks). Otherwise set `_expect_hooks = False` and call the module directly.
3. **`finish_grad_sync()`**: return at once if the last forward did not arm the hooks; otherwise count the buckets and call the parent's (check all launched, `wait`, copy back, schedule the rebuild).

Why does nothing in `_on_grad_ready` need to change? (Day 03 already returns early when `_expect_hooks` is False, before recording the ready order. With `grad_as_view`, the local accumulation happens straight in the bucket, and the sync-microbatch hook scales the accumulated value in place.)

### Stage 3 (25 min): read the harness

Read the microbatch loop in `train_accum.py` and find:

- `K = G // (world * m)` and the check that full windows are K equal microbatches;
- `xs, ys = x.split(m), y.split(m)`: the short final window simply has fewer microbatches;
- `last` and `comm`: which microbatch communicates, and how `modulo_last_sync` breaks it;
- `fwd_ctx` vs `bwd_ctx`: the correct use and the `bwd_only_nosync` trap;
- `w` vs `w_bwd`: the logged loss always uses the correct weights, so a scaling bug does **not** show up in the logged loss;
- `finish(comm)` after every backward: flat calls `sync_grads` only when `comm`, MiniDDP's finish is a no-op after a `no_sync` forward, DDP finalizes by itself;
- timing: one row per **optimizer step**, full windows after warmup only.

### Stage 4 (20 min): the negative controls

Predict, for each, which of A/B/C/D/E fails, then check against the parity output:

| `--bug` | What it does | Must be caught by |
| --- | --- | --- |
| `no_loss_scale` | backward of the unscaled loss | A |
| `zero_every_micro` | `zero_grad` inside the loop | A |
| `sched_per_micro` | `scheduler.step()` per microbatch | D |
| `fixed_k_scale` | 1/K in the short final window | E |
| `modulo_last_sync` | `last = (j == K - 1)`: the short window never syncs | E (and the ranks diverge) |

Which control passes check B, and why is that the Stage 0 lesson again?

### Pitfalls (symptom → cause)

| Symptom | Likely cause |
| --- | --- |
| Accumulated run correct but no faster, collective count K × buckets | forward outside `no_sync()` (only the backward wrapped), or no `no_sync()` at all |
| Logged loss K × too large, training still fine with AdamW | logging the scaled-for-backward value inconsistently, or no loss scaling (Adam hides it) |
| Warmup ends after a fraction of the planned steps | scheduler stepped per microbatch |
| Ranks drift apart only after the first epoch | "last microbatch" computed with a modulo: the short final window never syncs |
| Gradient norm / clipping behaves differently after changing K | loss weights depend on K incorrectly; clipping threshold applies to a K-scaled gradient |
| Clipping inside the microbatch loop | clips partial, unsynced sums; clip once after the last backward |
| `no_sync` did not reduce memory | it never does for DDP: buckets and `p.grad` stay allocated; only communication is skipped |
| Accumulation with BatchNorm differs from the big batch | BN statistics are per microbatch: not a bug in accumulation, a property of BN |
| `find_unused_parameters=True` plus `no_sync` behaves unexpectedly | the Reducer marks a parameter used if it was used in *any* microbatch of the `no_sync` session (comment in `autograd_hook`) |

---

## Block 3 — Read the source (1 h): from `no_sync()` to the early return

Run `python experiments/day04_grad_accum/find_symbols.py` first. It prints the defs and **every line that reads or writes the two flags** in your installed `distributed.py` (torch 2.14.1 line numbers: `no_sync` 1660, `_pre_forward` 1745, `_post_forward` 1816, `will_sync_module_buffers` 2383). The C++ is `torch/csrc/distributed/c10d/reducer.cpp` at **v2.14.1** on GitHub. Search by symbol name.

### Stop 1 — `DistributedDataParallel.no_sync` (Python)

Ten lines. *Q: why restore `old` instead of setting True? What else in the file writes `require_backward_grad_sync`?* (Constructor default; `__setstate__`; nothing else outside `no_sync`.) *Who reads it?* The `find_symbols` list answers it: `_pre_forward`, `_post_forward`, the join path, and the compiled-autograd hook (`_register_accum_grad_hook`).

### Stop 2 — `_pre_forward`

Find: `if torch.is_grad_enabled() and self.require_backward_grad_sync: ... self.reducer.prepare_for_forward()`, then `self.reducer._rebuild_buckets()` (runs whenever grad is enabled, but only acts once, after the order has been recorded), then `if self._check_sync_bufs_pre_fwd(): self._sync_buffers()`.

*Q: under `no_sync`, which of these still run? Why does `_rebuild_buckets` not need the flag?* (It is a no-op until a synced backward has recorded the order; `should_rebuild_buckets()` in `reducer.hpp` is true only until `has_rebuilt_bucket_`.)

### Stop 3 — `_post_forward`

Find the `if/else` on the same condition: `require_forward_param_sync = True` + `reducer.prepare_for_backward(...)`, or `require_forward_param_sync = False`. Then `will_sync_module_buffers()` (line ~2383): `require_forward_param_sync and forward_sync_buffers and len(self.modules_buffers) > 0`.

*Q: derive the broadcast count per window from these lines. Why is it 1 with `no_sync` and K without? Q: in the `bwd_only_nosync` control, which forward armed the Reducer?*

### Stop 4 — `Reducer::prepare_for_backward` and `Reducer::autograd_hook` (C++)

In `prepare_for_backward`: `expect_autograd_hooks_ = true;`, `grad_ready_order_indices_.clear();`, `reset_bucket_counting()` (`next_bucket_ = 0`, `bucket.pending = bucket.variables.size()`). That is MiniDDP's reset in `forward`.

In `autograd_hook`: the `first_autograd_hook_called_` / `num_bwd_calls_` bookkeeping, the unused-parameter marking (with its comment *"During no_sync session, the same var can be set multiple times ..."*), then the early return quoted in §1.5. Only after it: `grad_ready_order_indices_.push_back(index)`, `push_rebuilt_params(index)`, `mark_variable_ready(index)`.

*Q1: which line of MiniDDP is this early return? Q2: why is the ready order recorded after it, and what would go wrong if a `no_sync` backward recorded it? Q3: does the hook cost anything under `no_sync`?* (A lock, a few branches, per parameter. Compare `mini_nosync` with `flat_nosync` in the timing table for the Python version of that cost.)

### Stop 5 — `Reducer::finalize_backward`

`TORCH_INTERNAL_ASSERT(expect_autograd_hooks_); expect_autograd_hooks_ = false;`, then the waits and copy-backs from Day 03. *Q: after a synced backward, what state is the Reducer in when the next `no_sync` forward runs? Why can a `no_sync` backward never reach `finalize_backward`?* (Nothing marks a bucket ready, so `queue_callback` is never called.)

### Stop 6 — `AccumulateGrad::accumulateGrad` (`torch/csrc/autograd/functions/accumulate_grad.h`)

Find the "steal" branch (`variable_grad` undefined: take the incoming gradient, possibly without a copy) and the in-place add branch. *Q: with `zero_grad(set_to_none=True)`, which branch runs for microbatch 1 and which for microbatches 2..K? With `gradient_as_bucket_view`, why must zeroing be in place?* (Day 03 §1.6.)

### Stop 7 (optional, 10 min) — previews

- **FSDP2:** `FSDPModule.set_requires_gradient_sync(False)` plays the role of `no_sync()`. *Q for Day 7: if gradients are sharded, what must be kept between microbatches when the reduce-scatter is skipped, and what does that cost in memory?*
- **DeepSpeed:** `gradient_accumulation_steps` in the config, `engine.backward()` / `engine.step()` called every microbatch, and `is_gradient_accumulation_boundary()`. *Q for Day 8: where does the engine decide whether to communicate?*

### Fill this in `notes.md` §3

| Item | Your answer (draft) |
| --- | --- |
| Entry point | `with ddp.no_sync(): loss = f(ddp(x)); loss.backward()` |
| Critical path | `no_sync` sets `require_backward_grad_sync = False` → `_pre_forward` skips `prepare_for_forward` → `_post_forward` skips `prepare_for_backward`, sets `require_forward_param_sync = False` → backward: `AccumulateGrad` adds into `p.grad` → post hook → `Reducer::autograd_hook` → `if (!expect_autograd_hooks_) return;` … last microbatch outside: `prepare_for_backward` → hooks → buckets → all-reduce → `finalize_backward` |
| Tensor changes | `p.grad` accumulates $\sum_j w_j g_j$ locally (same shape, same buffer after the first microbatch); buckets untouched until the sync microbatch |
| State changes | `require_backward_grad_sync`, `require_forward_param_sync`, `expect_autograd_hooks_`; LR scheduler and Adam `step` once per window |
| Evidence | parity A–E, five controls, bitwise groups, NCCL kernel counts per optimizer step, MiniDDP hook counters |

---

## Block 4 — Run + summarize on the dual A100

Plain `python` for the orchestrator (it launches `torchrun` itself and runs the smoke test first):

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p outputs
python experiments/day02_data_parallel/gpu_probe.py --want 2 > "outputs/day04_probe_before_${RID}.txt"
python experiments/day04_grad_accum/compare_accum.py --stage parity   --condition shared --run-id "a100_${RID}"
python experiments/day04_grad_accum/compare_accum.py --stage count    --condition shared --run-id "a100_${RID}" --no-smoke
python experiments/day04_grad_accum/compare_accum.py --stage timing   --condition shared --run-id "a100_${RID}" --no-smoke --repeats 2
python experiments/day04_grad_accum/compare_accum.py --stage slowlink --condition shared --run-id "a100_${RID}" --no-smoke --repeats 2
```

In the pre-test, parity took ~3 min (16 tiny runs), count ~3 min (7 profiled medium runs), timing ~40 min with 2 repeats (32 medium runs per repeat) and slowlink ~10 min. On NVLink the every-vs-`no_sync` gap is close to the host noise (Appendix A.3), so consider `--repeats 3` for timing if you can afford ~60 min. Timing runs save metrics only (`--no-tensors`). Download `outputs/day04_grad_accum/` before recycling the instance (~400 MB including the count traces).

### How to read the parity output

A/B/C are Day 02's checks (same tolerances, same code). D is exact: the LR at every optimizer step and Adam's step counter. E compares the gradient of the short final window, the only window where a fixed 1/K is wrong. Each block also prints the windows (`[4, 4, 4, 4, 4, 2]`), the in-process counters and whether the ranks end bitwise in sync. Controls print lower-case `pass`/`fail`; each must fail its named check.

### How to read the count table

One `ncclDevKernel_*` per collective, divided by the profiled optimizer steps (2), on rank 0. Compare with §1.4. The `backwards` column checks that the trace holds K backward ranges per step.

### How to read the timing tables

- **ms / opt step** covers K forward/backward passes, the syncs and the optimizer, slowest rank. **ms/micro** = that / K.
- **overhead vs none** = core(variant) − core(none) at the same K, in the same repeat: what data parallelism adds per optimizer step.
- **vs 1gpu**: strong scaling at the same G and K (one GPU with microbatches of 16).
- **coll/step**: in-process counters (flat, MiniDDP); for DDP use the count stage.
- **ddp_bigbatch**: the same G as one microbatch per rank (no accumulation).
- Shared host: spreads of a few tenths of a ms per microbatch are normal. Do not rank two variants whose ranges overlap.

Things to extract for the notes: the every − nosync gap vs K (does it grow like (K − 1) × C?); the overhead of `mini_nosync` vs `flat_nosync` at K = 8 (the price of hooks that return early, in Python); peak memory vs K; `ddp_bigbatch` vs `ddp_nosync`.

### Write `day04/notes.md`

Use the draft: it has the structure, the mechanism, the call chain and the pre-test tables. Replace the pre-test tables with your run, keep the predictions you wrote in §1.9 next to the results, and state what the data does not show.

---

## Today's core question, answered

**What exactly does `no_sync()` skip, why must the forward be inside it, and what does it not save?**

`no_sync()` sets one flag, `require_backward_grad_sync = False`. DDP reads it in forward: `_post_forward` then does not call `Reducer::prepare_for_backward`, so `expect_autograd_hooks_` stays false, and in the following backward every `autograd_hook` returns before touching a bucket. Gradients keep accumulating in `p.grad` through `AccumulateGrad`. The first forward outside the context arms the Reducer again; its backward finds $\sum_j w_j g_j$ in `p.grad` and all-reduces it in buckets as usual. Because `_post_forward` also clears `require_forward_param_sync`, the per-forward buffer broadcast is skipped too, except before the first microbatch of a window.

The forward must be inside because that is where the decision is taken. A backward inside `no_sync()` whose forward ran outside it still communicates.

It does not save memory (DDP's buckets and `p.grad` are allocated either way), compute (K forward/backward passes either way) or the optimizer step (once per window either way). It saves (K − 1) syncs per optimizer step: their collectives, their bucket copies and copy-backs, and their exposed tails and rank-skew waits. How much that is depends on C. On NVLink it is a few ms per sync, about the size of the shared host's noise in the pre-test. With P2P disabled it was ~74–79 ms on the pre-test rental, where `no_sync` made a K = 4 optimizer step 2.3× faster (Appendix A.4).

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. Relate m, K, W, G and the number of optimizer steps. What happens at the end of an epoch?</summary>

G = W × K × m rows per optimizer step; per-GPU batch per step = K × m. Each rank sees ⌈N/W⌉ rows per epoch (DistributedSampler pads), i.e. ⌈⌈N/W⌉/m⌉ microbatches, i.e. that / K windows. If K does not divide it, the last window is short: drop it, or step with it and weight its microbatches by their share of that window. The LR schedule and Adam's bias correction count windows, not microbatches.
</details>

<details><summary>2. Why multiply each microbatch loss by n_j / N, and when is that not 1/K?</summary>

The big-batch loss is a mean over all rows; the accumulated gradient must be the gradient of that mean. Summing per-microbatch means weighted by their share of the window gives this rank's mean, and DDP's 1/W gives the global mean (equal rows per rank). It equals 1/K only for K equal microbatches. A short final window (n < K microbatches) needs 1/n, an uneven last microbatch needs row (or token) weights, and masked tokens need valid-token weights over the whole window and all ranks.
</details>

<details><summary>3. Syncing every microbatch gives the right answer. Why, and why is it not bitwise equal to no_sync?</summary>

After each sync every rank holds the same average; adding local gradients and averaging again is linear, so the final value is the average of the sums. Rounding differs: every-sync rounds fl(ḡ₁ + g₂^r) per rank before the next two-operand sum, no_sync rounds fl(g₁^r + g₂^r). Same value in exact arithmetic, different float results (4.2e-7 on final parameters after 6 tiny steps in the pre-test). Within each group, flat, MiniDDP and DDP are bitwise equal at W = 2.
</details>

<details><summary>4. What does no_sync change inside DDP, step by step?</summary>

The flag is False → `_pre_forward` skips `prepare_for_forward` → `_post_forward` skips `prepare_for_backward` and sets `require_forward_param_sync = False` → `expect_autograd_hooks_` stays false → each `autograd_hook` returns early (after unused-parameter bookkeeping, before recording the order or touching buckets) → no bucket gets ready, no all-reduce, no `finalize_backward`. The next forward outside the context arms everything; its backward reduces the accumulated `p.grad`.
</details>

<details><summary>5. Why must the forward be inside no_sync()? What did the bwd_only control show?</summary>

The flag is read only in forward (C++ Reducer path). A forward outside the context calls `prepare_for_backward`, so the backward communicates even inside the context. The control gives exactly the every-microbatch numbers and the every-microbatch kernel count (16 all-reduces + 4 broadcasts per step at K = 4 in the pre-test): correct and silently K times the traffic.
</details>

<details><summary>6. How many buffer broadcasts per window, and why?</summary>

`_sync_buffers` runs before a forward when `require_forward_param_sync` is True, and that flag is set by the previous forward's `_post_forward`: True after a synced forward, False after a `no_sync` one. In a `no_sync` window only the first forward follows a synced forward, so 1 per window; syncing every microbatch gives K.
</details>

<details><summary>7. Why can a missing 1/K go unnoticed, and what catches it?</summary>

Adam's update is m̂ / (√v̂ + ε). Multiplying every gradient by c multiplies m by c and √v by c, so the update is unchanged up to ε. In the pre-test a persistent ×4 moved the AdamW parameters by 0.19% of the training motion (SGD: 295%). Weight decay is unaffected; clipping is (its threshold now applies to a scaled gradient). A one-off factor in the last window does not cancel. Only a gradient-level check catches both: A for the persistent one, E for the short window.
</details>

<details><summary>8. When is accumulation exactly a big batch, and when is it better or worse than one?</summary>

Exact (up to summation order) when the loss is a sum over rows and the forward of a row does not depend on the other rows in its batch: true here (LayerNorm, no dropout), false with BatchNorm (per-microbatch statistics) or dropout (different masks). Accumulation keeps one microbatch of activations, so peak memory is nearly flat in K. When the big batch fits, it is usually faster: fewer, larger kernels and one launch sequence instead of K. Use accumulation to reach a batch that does not fit, not to speed up one that does.
</details>

<details><summary>9. What does no_sync not save?</summary>

Memory (buckets and gradients stay allocated for DDP; with sharded gradients skipping the sync even costs memory), compute and the optimizer step. It saves the (K − 1) syncs: collectives, bucket copies, copy-backs, exposed tails and peer waits.
</details>

**Tomorrow (Day 5):** CUDA Graph pitfalls. Day 03's optimizer-only graph capture showed one side (submission gaps); tomorrow compares eager, `torch.compile` and manual capture on a small MLP or PPO update, and separates a graph break, a capture failure and a skipped Python side effect.

---

## Appendix A — Assistant's pre-test (2026-11-01, run id `pretest1`)

These are the **assistant's** numbers, measured to validate the code before handing it over. They are not your measurements. Your own run is reported in [`notes.md`](notes.md) §5–§6. Same software as Days 02–03; **new rental** (`ssh6.vast.ai:20163`, container 58663121). `nvidia-smi topo -m` shows NV12 between the GPUs, but GPU0 on NUMA node 0 and GPU1 on NUMA node 1, which matters for the P2P-disabled path. Raw logs and `metrics.json` files are in `outputs/day04_grad_accum_pretest1/`; profiler traces were not downloaded.

### A.1 Parity (tiny; 6 optimizer steps of G = 32 = 2 × K 4 × m 4; last window G = 16)

| Variant | A grad | B params-1 | C loss-0 | D | E last window | Ranks in sync |
| --- | --- | --- | --- | --- | --- | --- |
| 1 process, K = 8 × m 4 (`ref_accum8`) | 2.794e-9 | 2.608e-7 | 4.768e-7 | equal | 8.382e-9 | — |
| flat / MiniDDP / MiniDDP view / DDP / DDP view, `no_sync` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.324e-9 | yes |
| flat / MiniDDP / DDP every; DDP `bwd_only_nosync` | 2.794e-9 | 2.608e-7 | 0 | equal | 8.440e-9 | yes |

Bitwise: the five `no_sync` variants are identical to each other, the four "every" variants are identical to each other, and every vs `no_sync` is **not** identical (final parameters up to 4.230e-7 apart). All as predicted in §1.6.

| Control | Fails | Notes | Verdict |
| --- | --- | --- | --- |
| `no_loss_scale` | A, E | **B passes** (5.0e-6 ≤ 1e-5): Adam hid the ×4 in the first update | DETECTED by A |
| `zero_every_micro` | A, B, E | last-window gradient norm ×0.65 | DETECTED by A |
| `sched_per_micro` | B, D, E | LR at optimizer step 0: 7.5e-5 instead of 1.5e-5 | DETECTED by D |
| `fixed_k_scale` | E | last-window gradient ×0.500, everything else passes | DETECTED by E |
| `modulo_last_sync` | E | last window unsynced; final parameters differ across ranks by 1.179e-4 | DETECTED by E |

`PARITY: ALL CHECKS PASSED`. MiniDDP counters (tiny, 8 buckets): `no_sync` 8 collectives, 30 armed and 80 skipped hook calls per optimizer step; every 29.3 collectives per step (8 buckets × 22 backwards / 6 steps).

### A.2 Collectives per optimizer step (medium, K = 4, rank-0 trace, 2 profiled steps)

| Variant | Backwards | AllReduce | Broadcast |
| --- | --- | --- | --- |
| flat, sync once | 4 | 1 | 0 |
| flat, every | 4 | 4 | 0 |
| MiniDDP `no_sync` | 4 | 4 | 0 |
| MiniDDP every | 4 | 16 | 0 |
| DDP `no_sync` | 4 | 4 | **1** |
| DDP every | 4 | 16 | 4 |
| DDP `bwd_only_nosync` | 4 | **16** | **4** |

Exactly the §1.4/§1.5 predictions: `no_sync` with the forward outside the context does not reduce traffic at all.

### A.3 Timing vs K, NVLink (medium, m = 8, 25 optimizer steps, 5 warmup, 2 repeats)

ms per optimizer step, median [min–max] over repeats; overhead vs `none` in brackets is computed per repeat.

| K (G) | none | flat sync once | flat every | MiniDDP `no_sync` | MiniDDP every | DDP `no_sync` | DDP every | DDP big batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (16) | 36.95 | 37.28 (+0.33) | — | 43.54 (+6.60) | — | 41.56 (+4.61) | — | — |
| 2 (32) | 73.06 | 75.26 (+2.20) | 69.14 (−3.92) | 80.37 (+7.31) | 86.31 (+13.25) | 78.78 (+5.72) | 77.66 (+4.60) | 51.83 |
| 4 (64) | 133.89 | 136.74 (+2.86) | 148.06 (+14.18) | 145.28 (+11.39) | 143.08 (+9.19) | 139.36 (+5.47) | 145.45 (+11.57) | 94.35 |
| 8 (128) | 252.89 | 258.30 (+5.41) | 279.28 (+26.39) | 249.68 (−3.21) | 266.16 (+13.27) | 267.72 (+14.82) | 254.81 (+1.91) | 181.23 |

Repeat ranges of the overheads (ms): flat every K = 4 [+10.65, +17.71], K = 8 [+20.83, +31.95]; DDP `no_sync` K = 4 [+4.28, +6.67], K = 8 [+10.40, +19.24]; DDP every K = 2 [−2.65, +11.84], K = 4 [+6.18, +16.95], K = 8 [−3.74, +7.56]; MiniDDP `no_sync` K = 8 [−18.63, +12.20]. 1 GPU (microbatches of 16, same G and K): 50.34, 98.16, 193.87, 385.25 ms.

| K | Peak MiB: none | flat | MiniDDP / DDP (both modes) | DDP big batch |
| --- | --- | --- | --- | --- |
| 1 | 1014 | 1012 | 1117 | — |
| 2 | 1125 | 1123 | 1227–1230 | 1750 |
| 4 | 1125 | 1123 | 1227–1230 | 3014 |
| 8 | 1126 | 1124 | 1228–1230 | 5543 |

What this pre-test supports, and what it does not:

- **Flat sync shows the (K − 1) × C growth.** every − once is +11.3 ms at K = 4 and +21.0 ms at K = 8, i.e. ~3–4 ms per extra sync, the size of Day 03's flat overhead. The K = 2 row (`flat_every` 3.9 ms *faster* than `none`) is a noise artifact; it shows that the noise floor on this shared host is several ms per optimizer step.
- **DDP and MiniDDP on NVLink: not resolved.** The expected gap is a few ms per extra sync, about the size of the repeat-to-repeat spread here. Only DDP at K = 4 separates (+5.47 vs +11.57, ranges barely touching); at K = 2 and K = 8 the ranges overlap and the medians even invert. Two repeats are not enough. Use `--repeats 3` or more (and longer runs) before ranking them.
- **Memory is flat in K for K ≥ 2.** The step from K = 1 to K = 2 is +110 MiB ≈ P (104.7 MiB): with K ≥ 2 the accumulated `p.grad` already exists while a later microbatch's backward reaches its activation peak, whereas at K = 1 (`set_to_none`) it does not (Day 03 §7). The big batch needs 1.6–5.5 GiB.
- **The big batch is much faster when it fits:** 181.23 vs 267.72 ms at K = 8 (1.48×), 94.35 vs 139.36 ms at K = 4. Microbatches of 8 rows are launch-bound (`none` costs ~30.9 ms per microbatch + ~6 ms per optimizer step from a linear fit over K).

### A.4 Slow link (`NCCL_P2P_DISABLE=1`, 2 repeats)

| K (G) | none | flat sync once | flat every | DDP `no_sync` | DDP every |
| --- | --- | --- | --- | --- | --- |
| 1 (16) | 37.28 | 116.30 (+79.02) | — | 110.95 (+73.68) | — |
| 4 (64) | 134.59 | 193.02 (+58.43) | 457.62 (+323.03) | 188.50 (+53.91) | 432.65 (+298.05) |

Here the effect is unambiguous: syncing every microbatch costs ≈ K × the single-sync cost (298 ≈ 4 × 74 ms), and `no_sync` makes the optimizer step **2.3× faster** for DDP (432.65 → 188.50 ms) and 2.4× for flat. One sync costs ~74–79 ms on this rental, about 5× Day 03's ~15–18 ms on the previous host. With P2P disabled, NCCL goes through host memory, and this host's two GPUs sit on different NUMA nodes. Do not compare slow-link numbers across rentals. Unexplained in this pre-test: the K = 4 `no_sync` overhead (+54 ms) is lower than the K = 1 overhead (+74 ms) for the same 105 MiB.

### A.5 What this does not establish

Two GPUs, one shared host, FP32, one small model, two repeats. NVLink every-vs-`no_sync` differences for DDP/MiniDDP are within noise. There is no per-hook CPU cost isolation for `no_sync` hooks, no profiler-based breakdown of the timing runs, no W ≥ 3 numerics, and no test with BatchNorm, dropout, AMP or gradient clipping.
