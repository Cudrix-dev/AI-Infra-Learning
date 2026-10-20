# Day 01 — Reproducible single-GPU baseline (guide)

**Goal:** by tonight you can say exactly when each memory category is allocated and freed in one training step, and you have a baseline whose resume is *bit-exact*. Every later day (DDP, ZeRO, FSDP, PP) is "this loop, but the state is split or the gradients travel". So today's numbers and vocabulary get reused for 19 days.

**Done means:**

- [x] `train.py` runs and logs step time, tokens/s and peak memory (warmup excluded)
- [x] `check_resume.py` prints `ALL CHECKS PASSED` with `atol=0` (bit-exact)
- [x] The negative control (`--with-bad-resume`) shows a large diff, so you know the check can actually fail
- [x] `--memory-probe --preset medium` table, with every row explained in your own words
- [ ] AdamW call chain traced (Block 3) and `day01/notes.md` written

Files:

| File | What it is |
| --- | --- |
| `experiments/day01_single_gpu/train.py` | Your file, finished. Training loop, checkpointing, timing, memory probe |
| `experiments/day01_single_gpu/check_resume.py` | Runs full / crash / resume and compares them |
| `day01/GUIDE.md` | This guide |
| `day01/notes.md` | Your summary. You write it at the end (Block 4) |

> **What changed in your original `train.py`:** three bugs would have stopped it running. `nn.Moduel` became `nn.Module`; `torch.arrange` (3×) became `torch.arange`; `main()` stopped after the warmup loop and `CHECKPOINT_PATH` was never used. Your model and comments are kept. Shapes now live in a `Config` dataclass so the same code runs the `tiny` and `medium` presets. The synthetic data uses a random stride per row, because your `+1` pattern is learned in a few steps, and a loss stuck at 0 makes a weak parity signal.

---

## Block 1 — Design (30 min): draw this before you read the code

### 1.1 One training step, as a memory timeline

```
                 zero_grad    forward            loss    backward                   step
                 (None)   ───────────────►       ───►   ◄────────────────           ─────►
params     P  ████████████████████████████████████████████████████████████████████████████  lives forever
grads      P              .                              ░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█  born in backward (AccumulateGrad),
                          freed here (set_to_none)                                          dies at next zero_grad
adam m,v  2P  ............................................................................█  born at FIRST step(), lives forever
acts       A              ░░░░▓▓▓▓▓▓▓▓▓▓▓████████████████▓▓▓▓▓▓░░░░                       grows in forward, shrinks in backward
logits    BTV                            ████  (dies when nothing references it)
```

What to notice:

- **Peak memory is usually at the start of backward.** All activations are still alive, the logits' gradient (another `B·T·V` tensor) gets created, and grads start to appear.
- **Between steps** only `params + adam state = 3P` stay alive in fp32. With `zero_grad(set_to_none=False)` the grads stay alive too, which makes it 4P.
- **Activations scale with batch × seq × layers, not with parameter count.** That is why ZeRO/FSDP (Days 6–8, which shard P, grads and Adam state) and activation checkpointing (Day 9, which cuts A) solve *different* problems.

### 1.2 What must be in a checkpoint to resume bit-exactly?

Before you read the code, write your own list, then compare it with this table:

| State | Needed? | Why |
| --- | --- | --- |
| Model params | yes | obviously |
| Adam `exp_avg`, `exp_avg_sq` | **yes** | 2× the model's size. Without them the first update after resume is a different update |
| Adam per-param `step` | **yes** | drives the bias correction `1 − β^t`. Reset it to 0 and the effective LR jumps |
| LR scheduler counters | **yes** | otherwise warmup restarts, and the cosine phase is wrong |
| Global step counter | yes | picks the next batch and the LR |
| Data position / sampler state | yes in general | here the batch is a pure function of `(seed, step)`, so the step counter is enough |
| RNG states (CPU, CUDA) | yes in general | dropout, sampling, random augmentation. Here dropout=0, but save them anyway |
| Gradients | **no** | `zero_grad` runs at the start of every step |
| Activations | **no** | rebuilt by the next forward |
| Config | sanity check | resuming with a different batch size or total_steps is not "the same run" |

### 1.3 Two different "warmups". Don't mix them up.

- **LR warmup** (`lr_warmup_steps`) is part of the *algorithm*. Adam's second-moment estimate is noisy in the first steps, so you start with a small LR. It changes the numbers.
- **Measurement warmup** (`timing_warmup`) is part of the *benchmark*. The first steps pay one-time costs (CUDA context, cuBLAS handle, allocator growing its pool, lazy Adam state, autotuning). They are excluded from timing and from peak memory.

---

## Block 2 — Code (2 h): read `train.py` in this order

Read it top to bottom once, then come back to the points below. For real practice, close the file and re-type `train_step`, `save_checkpoint`/`load_checkpoint` and the timing loop from memory.

**1. `CUBLAS_WORKSPACE_CONFIG` (top of file).** cuBLAS reads it when its handle is created, at the first matmul. That is why it is set before `import torch` does any CUDA work. Setting it later silently does nothing.

**2. `setup_reproducibility`.**
- `torch.manual_seed` seeds the CPU generator and *all* CUDA generators.
- `set_float32_matmul_precision("highest")` turns off TF32, so fp32 really is fp32. TF32 is faster on the 4090, but it changes the numbers, and then they can't be compared with other machines or later days.
- `use_deterministic_algorithms(True, warn_only=True)` picks deterministic kernels where they exist, and warns where they don't.

**3. `attention_ctx`.** The fused SDPA kernels (flash, mem-efficient) may use atomics in backward, so the order of float additions changes between runs. Floating-point addition isn't associative, so you get bit-level noise. The MATH backend is deterministic. This is your first systems trade-off: **reproducibility costs speed**. Run once with `--no-deterministic` later and measure what it costs.

**4. `get_batch(cfg, step)`.** A private `torch.Generator` is re-seeded from `(seed, step)`, so the batch doesn't depend on any global state. Resume only needs the step counter. (With a real `DataLoader` you'd have to save the sampler's epoch and offset. Remember this for Day 3's `DistributedSampler.set_epoch`.)

**5. `TinyTransformer`.** Your model. `causal_mask` became `persistent=False`, because it's derived from config and is not training state. The final `LayerNorm` exists because pre-norm blocks add to an un-normalised residual stream.

**6. `build`.**
- `foreach=False, fused=False` is deliberate: it routes `step()` to `_single_tensor_adam`, the readable Python loop you trace in Block 3. The CUDA default is `foreach=True`.
- The scheduler gets a *plain lambda*. `LambdaLR.state_dict()` stores plain functions as `None`, so the checkpoint only holds counters, and the function is rebuilt from config.

**7. `train_step`.** The four numbered comments are the memory timeline from 1.1. Note `return loss.detach()` *without* `.item()`. `.item()` copies a GPU value to the CPU, so the CPU must wait for the GPU to finish everything queued before it. Do that every step and the CPU can no longer run ahead and queue kernels.

**8. `save_checkpoint` / `load_checkpoint`.**
- Atomic save: write `.tmp`, then `os.replace`. A crash mid-save leaves the previous checkpoint intact.
- `torch.load(..., map_location="cpu", weights_only=True)`. `weights_only=True` is the safe default (no arbitrary pickle code). `map_location="cpu"` matters because `torch.set_rng_state` needs a *CPU* ByteTensor.
- Load order: the scheduler already exists (its `__init__` wrote an LR into `param_groups`), then the optimizer state (restores the saved LR), then the scheduler counters. The PyTorch docs warn about exactly this ordering (you'll see it in Block 3).
- `--bad-resume` loads only the weights. That's the negative control.
- **Pitfall (hit on the first run):** `weights_only=True` only unpickles tensors, primitives and containers. `torch.__version__` *looks* like a string but is a `TorchVersion` object, so storing it raw made every checkpoint unloadable. Store `str(torch.__version__)`. Rule: put only tensors, numbers, strings, bools, None, lists, tuples and dicts in a checkpoint, and test a load before you rely on one.

**9. The timing loop.**
- `torch.cuda.Event(enable_timing=True)` and `.record()` put a timestamp *into the GPU stream*, so the CPU doesn't block. One `synchronize()` at the end, then `elapsed_time()` on each pair.
- `reset_peak_memory_stats()` runs at `start_step + timing_warmup`, so peak memory is the steady state.
- Reported: first-step time (cold), median / mean / p90 (steady), tokens/s = `B·T·steps / Σ step time`.
- Why not `time.time()` around `train_step`? Without a sync you only measure how fast Python *queues* kernels. With a sync every step you destroy the overlap you're trying to measure.

**10. `memory_probe`.**
- `mark()` = synchronize + `memory_allocated()` after every phase, and resets the peak, so the peak column means "peak inside this phase".
- `saved_tensors_hooks(pack, unpack)` is the official hook autograd calls for every tensor it saves for backward. Summing unique non-parameter storages gives *activation memory held by the graph*. Same mechanism CPU activation offloading uses.
- `del logits` frees `B·T·V·4` bytes, because `cross_entropy` saved `log_softmax`'s *output*, not the logits. Memory is freed when the last reference dies, whether that reference is yours or autograd's.

---

## Block 3 — Read the source (1 h): `AdamW.step()` from top to bottom

Line numbers are for **pytorch v2.14.1**. Check your version (`python -c "import torch; print(torch.__version__, torch.__file__)"`) and open the files under `site-packages/torch/optim/`, which is the code you actually run. If lines moved, use `grep -n`.

Answer the **Q** at each stop in your notes before you move on.

**Stop 1: `torch/optim/adamw.py:19` `class AdamW(Adam)`**
In recent versions AdamW is only `Adam(..., decoupled_weight_decay=True)` (line 47). Older versions had a full copy of the code. Same algorithm either way.
*Q: what does "decoupled" weight decay mean? (Answer at Stop 5, line 419.)*

**Stop 2: `torch/optim/optimizer.py`, the wrapper around every `step()`**
- `Optimizer.__init__` (≈395) calls `_patch_step_function` (≈580), which wraps the class's `step` with `profile_hook_step` (≈527).
- The wrapper opens `record_function("Optimizer.step#AdamW.step")`; that's the block you'll see in the profiler on Day 3. It runs global and per-optimizer **pre-hooks**, the real `step`, then **post-hooks**.

*Q: why would a distributed framework want a step pre-hook or post-hook? (Hint: ZeRO-1 on Day 6 gathers updated params after the step.)*

**Stop 3: `torch/optim/adam.py:215` `Adam.step` → `_init_group` (139)**
- `step` loops over `param_groups`; `_init_group` collects params whose `p.grad is not None`.
- **Lazy state init (line 162, `if len(state) == 0`).** `exp_avg` and `exp_avg_sq` are `zeros_like(p)`, so optimizer memory first appears at the first `step()`, not in the constructor. Your memory probe shows this row.
- `state["step"]` is a **CPU** scalar tensor unless `capturable`/`fused` is set (see the comment "special device hosting for step"). Incrementing a CPU scalar costs no kernel launch. The flip side: a CUDA Graph can't capture a CPU tensor update, which is why `capturable=True` exists. Day 5 needs this.

*Q1: a frozen param (`requires_grad=False`) never gets a grad. How much optimizer memory does it use?*
*Q2: why does `state["step"]` live on the CPU by default?*

**Stop 4: `adam.py:903` `adam()`, the dispatcher**
- If neither `fused` nor `foreach` is set, `_default_to_fused_or_foreach` turns on foreach for CUDA tensors. Fused is never picked by default.
- Line ≈963–968: `_fused_adam` / `_multi_tensor_adam` / `_single_tensor_adam`. Your `foreach=False, fused=False` lands on the last one.

*Q: foreach launches one kernel per op for a whole list of tensors. Fused launches one kernel for the whole update. Why does that matter more for your `tiny` model than for a 7B model?*

**Stop 5: `adam.py:347` `_single_tensor_adam`, the actual math.** Map each line to the formula:

| Line | Code | Math |
| --- | --- | --- |
| 414 | `step_t += 1` | t ← t + 1 |
| 419 | `param.mul_(1 - lr * weight_decay)` | θ ← θ(1 − ηλ) — **decoupled**: decay hits the weights directly and never enters m or v |
| 456 | `exp_avg.lerp_(grad, 1 - beta1)` | m ← β₁m + (1−β₁)g |
| 475 | `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)` | v ← β₂v + (1−β₂)g² |
| ≈530–544 | `bias_correction1/2`, `denom = (v.sqrt() / sqrt(bc2)).add_(eps)` | m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ) |
| 546 | `param.addcdiv_(exp_avg, denom, value=-step_size)` | θ ← θ − η·m̂/(√v̂ + ε) |

*Q1: count the full-tensor reads and writes per parameter in one update (param, grad, m, v, the `denom` temporary). The update does almost no FLOPs per byte. Is the optimizer step compute-bound or memory-bandwidth-bound?*
*Q2: `denom` is a temporary the size of one param. In the single-tensor path only one exists at a time. In the foreach path, how many exist at once, and what does that do to peak memory during `step()`?*

**Stop 6: `optimizer.py:700` `state_dict` / `:900` `load_state_dict` / `:793` `_process_value_according_to_param_policy`**
- `state` is keyed by **integer position** in `param_groups`, not by parameter name. Change the module order and the loaded Adam moments land on the wrong parameters, silently.
- On load, float state is cast to the param's device and dtype, but `step` is deliberately *not* (it stays on the CPU unless fused/capturable).
- Read the `.. warning::` in `load_state_dict`'s docstring about the scheduler. That's why `load_checkpoint` uses the order it does.

**Stop 7 (bridge to Day 3): where does `p.grad` memory come from?**
- `optimizer.py:1048` `zero_grad(set_to_none=True)` simply does `p.grad = None`.
- In the next backward, `torch/csrc/autograd/functions/accumulate_grad.h:181` `accumulateGrad` sees `!variable_grad.defined()` and *steals* the incoming gradient buffer when nothing else references it, otherwise clones it. So the grad memory is the backward's own output buffer, not a fresh allocation.
- The `register_post_accumulate_grad_hook` you'll use for your mini-DDP on Day 3 fires right after this point.

**Fill this in `notes.md` §3:**

| Item | Your answer |
| --- | --- |
| Entry point | `optimizer.step()` |
| Critical path | `profile_hook_step.wrapper` → pre-hooks → `Adam.step` → `_init_group` (lazy state) → `adam()` → `_single_tensor_adam` → post-hooks |
| Tensor changes | `exp_avg`, `exp_avg_sq`: created `[same shape as p]` fp32 on the param's device at step 1; `step`: CPU fp32 scalar |
| State changes | `state[p]["step"]` += 1 per call; `param_groups[i]["lr"]` written by the scheduler |
| Evidence | memory-probe row `[it0] optimizer.step` grows by 2P; `[it1]` grows by 0 |

---

## Block 4 — Run + summarize (30–60 min)

Run from the repo root (`D:\AI Infra Learning`), in your conda env (or WSL2).

```bash
# 0. record versions (goes into notes §6)
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"

# 1. smoke test the logic in a few seconds
python experiments/day01_single_gpu/train.py --tag smoke --steps 30

# 2. resume parity: must print ALL CHECKS PASSED with atol=0
python experiments/day01_single_gpu/check_resume.py --with-bad-resume

# 3. memory categories: medium preset so the numbers are big enough to read
python experiments/day01_single_gpu/train.py --memory-probe --preset medium

# 4. baseline performance (keep these numbers; Day 2-4 compare against them)
python experiments/day01_single_gpu/train.py --tag base_tiny   --steps 300
python experiments/day01_single_gpu/train.py --tag base_medium --steps 300 --preset medium
python experiments/day01_single_gpu/train.py --tag fast_medium --steps 300 --preset medium --no-deterministic

# 5. (optional) is resume still exact without deterministic kernels?
python experiments/day01_single_gpu/check_resume.py --no-deterministic --atol 1e-5
```

### What to expect, and how to read it

**`check_resume.py`.** In deterministic mode on the same GPU, A, B and C should all be `0.000e+00`. The resumed process runs exactly the same kernels on exactly the same bits. If B or C isn't 0, look in this order: a missing state item (the 1.2 table), load order, a non-deterministic kernel (read the `warn_only` warnings), TF32. The bad-resume row should be clearly non-zero: fresh Adam moments plus a restarted warmup change the trajectory immediately.

*"Documented tolerance"* for your notes: deterministic → `atol = 0` (bitwise). Non-deterministic → losses typically differ around 1e-7…1e-5 and the gap grows slowly with steps (chaotic amplification). Write down what you actually observed.

**Memory probe (medium).** Theory first, then compare with what you measure:

| Quantity | Formula | Medium preset |
| --- | --- | --- |
| Params N | `V·D + T·D + L(12D² + 13D) + 2D + D·V + V` | 27,443,200 |
| P (params, fp32) | 4N bytes | ≈ 104.7 MiB |
| Grads | P | ≈ 104.7 MiB |
| Adam m + v | 2P | ≈ 209.4 MiB |
| Logits | B·T·V·4 | 128 MiB |
| Saved activations | measured by the `saved_tensors_hooks` column | (fill in) |

Expected pattern:
- `build` ≈ +P.
- `[it0] forward` ≈ +activations + logits.
- `del logits` ≈ −128 MiB.
- `backward` ≈ +P (grads) − activations; its *peak* column is the highest in the table.
- `[it0] step` ≈ +2P.
- `[it1] zero_grad` ≈ −P.
- `[it1] step` ≈ +0.

Measured values come out slightly higher than the formulas, because the caching allocator rounds every block up and a few small tensors stay alive.

**Baseline perf.**
- **tiny:** step time is dominated by Python and kernel-launch overhead, and the GPU is mostly idle. tokens/s here measures your CPU, not the 4090. That's the problem CUDA Graphs solve on Day 5.
- **medium:** the GPU does real work.
- **`fast_medium` vs `base_medium`:** the cost of determinism, mostly MATH attention vs fused attention.
- Also compare `first_step_ms` with the steady median: that gap is why you never report a first-step number as throughput.

### Write `day01/notes.md` (300–600 words)

Prompts for each section:

1. **Problem.** A baseline whose numbers you can trust: reproducible, resumable, timed correctly.
2. **Mechanism.** The memory timeline in your own words, plus why each checkpoint item is needed.
3. **Call chain.** The table from Block 3.
4. **Mine vs the framework.** What does `torch.distributed.checkpoint` or a real trainer (Lightning, Megatron) save that you don't? Data-loader state, sharded state, async save, versioning.
5. **Evidence.** The `check_resume` output, including the negative control.
6. **Performance.** The tiny / medium / fast numbers with the command, versions and GPU. What they **don't** prove: a single GPU says nothing about communication, and the tiny preset is launch-bound.

---

## Today's core question, answered

**When is each memory category allocated and freed?**

| Category | Size (fp32 AdamW) | Allocated | Freed |
| --- | --- | --- | --- |
| Parameters | P = 4N | `model.to(device)` | never (end of program) |
| Gradients | P | during `backward()`, when `AccumulateGrad` first writes `p.grad` (stealing the incoming buffer) | `zero_grad(set_to_none=True)` at the start of the next step (never, with `set_to_none=False`) |
| Optimizer state | 2P (m, v) + CPU scalar `step` | lazily at the **first** `optimizer.step()` (`_init_group`) | never |
| Activations | ∝ B·T·D·L (+ B·T·V for the logits/softmax) | during forward, as autograd saves tensors for backward | progressively during backward, as each node runs and drops its saved tensors. The whole graph is gone when `backward()` returns (`retain_graph=False`) |
| Temporaries | transient | inside kernels and ops (`denom` in Adam, the logits' grad in backward) | as soon as the op finishes |
| Outside PyTorch's count | — | CUDA context, cuBLAS workspace, allocator cache (`reserved − allocated`) | `empty_cache()` releases the cache only |

Steady state between steps: **3P** (params + Adam). The peak is usually early in backward: **3P + activations + logit-grad + growing grads**. For a 7B model in fp32 that's 3 × 28 GB = 84 GB before any activations, which is why Days 6–9 exist.

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. Your resume passed with atol=0. Name three things that would break it.</summary>
Not saving the Adam state or `step` (bias correction resets). Not restoring the scheduler (warmup restarts). Non-deterministic kernels or TF32 (bit noise that compounds). Also: data that depends on the global RNG without saving it, and changing the module order (optimizer state is keyed by position).
</details>

<details><summary>2. Why is `step()` memory-bound, and what do foreach / fused change?</summary>
Each element is read and written a handful of times for ~10 FLOPs, so arithmetic intensity is ≈ 1 FLOP/byte, far below what the 4090 needs to be compute-bound. foreach cuts kernel *launches* (one launch per op per tensor list). fused cuts both launches *and* memory passes (one kernel reads p, g, m, v once and writes p, m, v once).
</details>

<details><summary>3. Why does nvidia-smi show much more memory than max_memory_allocated?</summary>
The CUDA context plus the cuBLAS/cuDNN workspaces, plus the caching allocator holding freed blocks (reserved ≥ allocated). nvidia-smi sees the whole process; `memory_allocated` only counts live tensors.
</details>

<details><summary>4. You time a step with time.time() and no synchronize and get 0.3 ms. What did you measure?</summary>
How long the CPU took to *enqueue* the kernels. The GPU may still be running. Use CUDA events, or synchronize at the timing boundaries.
</details>

<details><summary>5. `zero_grad(set_to_none=True)` vs `False`: memory and behaviour?</summary>
True frees the grad memory between steps (−P at rest), and the next backward assigns instead of adding (saves a read-modify-write). False keeps the buffers allocated and zeroes them with a kernel. With True, code that reads `p.grad` must handle `None`. Frozen params, or params unused this step, also keep `grad=None`, so the optimizer skips them.
</details>

**Tomorrow (Day 2):** the same model in two processes. You'll broadcast these params, shard `get_batch` by rank, and `all_reduce` the grads by hand. Today's `check_resume` mindset becomes "1-GPU vs 2-GPU parity".
