# Day 09: mixed precision and activation checkpointing (single-A100 guide, one optional dual-GPU question)

> **Run status.** The full reference run `manual_full_20261109T065636Z` passed on 2026-11-09. The archive and all 79 manifest entries were verified locally. The **[measured]** checks below use that run. [Appendix A](#appendix-a-pre-test-results-pretest1-assistant-run-2026-11-09-pre-test) retains `pretest1` for comparison. Day 09 was closed on 2026-11-11. The two handwritten functions and their `--impl mine` checks are deferred as an open practice item.

## Required to close Day 09 (about 3.5 h including the run)

Day 09 is done when you can explain three things from your own run and your own code:

| # | You can explain | Evidence from your run | Where in this guide |
| --- | --- | --- | --- |
| 1 | **What changes dtype under autocast, and what does not.** Where the FP32 weights live, and whether `torch.autocast` makes a separate master copy (compare with Day 08's DeepSpeed) | `dtypes` log: the dtype table, the "weight casts" rows, the saved-tensor rows, the "unchanged by step 1" rows (including `bf16_pure`) | §1.1, §1.2, Block 3 Stop 1 |
| 2 | **Why fp16 unscaling must come before clipping.** The order scale → backward × K → `unscale_` → clip → step → update, and what the `clip_before_unscale` control does | `fp16_order` log: rows `ok` and `bug`, checks C0–C3; your `window_step` passing with `--impl mine` | §1.3, Block 2 YOU WRITE (1), Block 3 Stops 2–3 |
| 3 | **Where checkpointing trades memory for recomputation.** What is saved, what is recomputed, by how much the peak drops, what it costs in step time, and that the numbers do not change | `ckpt` log (census, recompute counts, bitwise check) and the `matrix` log (peak and ms/step, on vs off at the same precision); your `model_forward` passing with `--impl mine` | §1.4, §1.5, Block 2 YOU WRITE (2), Block 3 Stop 4 |

**Day 09 is done when:**

- [ ] Your predictions in §1.1–§1.5 are written down (one line each is enough) **before** you run anything.
- [ ] `my_amp.py` contains your `window_step` and `model_forward`. `fp16_order.py --impl mine` prints `FP16 ORDER: ALL CHECKS PASSED`, and `probe_amp.py --part ckpt --impl mine` prints `CKPT CHECK: PASSED`.
- [ ] Your run of `run_server_once.sh <RID> --core` (or the four commands in Block 4) finished, printed `OUTPUT CHECKS PASSED`, and was downloaded and verified with `download_results.ps1`.
- [ ] [`notes.md`](notes.md) §1–§5 are filled with **[measured]** numbers: the run record, the dtype table, the fp16-order results, the checkpointing correctness, memory and time, and the three explanations (3–6 sentences each, in your own words).

**Optional (not needed to close the day):**

- §1.6 and `compare_amp.py --stage ddp`: the one dual-GPU question (in which dtype does DDP all-reduce under autocast?).
- The optional matrix rows: K = 4 accumulation at the same global batch, and flash attention (`--sdpa auto`).
- `fp16_order.py --underflow`: fp16 without a scaler.
- Block 3 Stops 5–8 (AdamW state init, reentrant checkpointing, selective checkpointing, DDP comm hooks, FSDP2's `MixedPrecisionPolicy`, DeepSpeed's bf16 optimizer).
- The self-check questions at the end.

**Scope on purpose.** No `torch.compile`, no FP8, no Transformer Engine. One debugging exercise only (clip before unscale). Only one workload: Day 01's medium model at a global batch of 32 rows per optimizer step.

---

## Read this first: same host type, fresh rental

Same as Days 02–08: **2 × A100-SXM4-40GB, NV12**, torch 2.14.1+cu130, CUDA 13.0, NCCL 2.30.7, driver 595.71.05, `--condition shared`. No new package. The measured run had both GPUs on NUMA node 1. **One GPU** is enough for everything required. The second GPU is used only by the optional DDP stage.

The project directory arrives empty on a fresh rental. From your Windows project root:

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/AI-Infra-Learning/experiments'
scp -P <port> README.md requirements.txt root@<host>:/workspace/AI-Infra-Learning/
scp -P <port> -r experiments/day01_single_gpu experiments/day02_data_parallel experiments/day09_mixed_precision root@<host>:/workspace/AI-Infra-Learning/experiments/
scp -P <port> -r day09 root@<host>:/workspace/AI-Infra-Learning/
```

Then, on the instance:

```bash
cd /workspace/AI-Infra-Learning
ln -sfn /venv/main .venv        # fresh rental only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
nvidia-smi topo -m              # record it with the run
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported(), torch.backends.cuda.matmul.allow_tf32)"
```

Expected: `2.14.1+cu130 13.0 True False`. **TF32 is off** (Day 01 sets `float32_matmul_precision("highest")`), so "fp32" in this guide means true FP32 CUDA-core matmuls. That matters for the timing comparison (§1.5).

Today's code imports these files unchanged:

- Day 01: `train.py` (model, data, `build`, `setup_reproducibility`)
- Day 02: `ddp_manual.py` (`log0`, `run_metadata`), `launch_utils.py`, `find_symbols.py`, `gpu_probe.py`

---

## Files

| File | What it is |
| --- | --- |
| `experiments/day09_mixed_precision/amp_setup.py` | **Reference** for the two parts you write: `window_step` (one optimizer step: autocast, Day 04 weights, GradScaler order, clipping, the control) and `model_forward` (Day 01's forward with per-block checkpointing). Given: `autocast_ctx`, `make_scaler`, `sdpa_context`, `AmpNet` |
| `experiments/day09_mixed_precision/my_amp.py` | **Yours.** The same two functions with empty bodies. Every script takes `--impl mine` |
| `experiments/day09_mixed_precision/probe_amp.py` | `--part dtypes`: the dtype table, ops by dtype, weight casts, saved-tensor census, gradient error, "unchanged by step 1". `--part ckpt`: saved vs recomputed, memory, bitwise check |
| `experiments/day09_mixed_precision/fp16_order.py` | The debugging exercise: fp32 reference vs fp16 `ok` vs fp16 `bug`, checks C0–C3 |
| `experiments/day09_mixed_precision/train_amp.py` | Training harness: precision, checkpointing, accumulation, clipping, per-phase time and memory, optional DDP |
| `experiments/day09_mixed_precision/compare_amp.py` | `--stage matrix` (required rows with `--core`) and `--stage ddp` (optional) |
| `experiments/day09_mixed_precision/find_symbols.py` | `file:line` of Block 3's reading list in your install |
| `experiments/day09_mixed_precision/run_server_once.sh` | Everything once, logged, checked and archived |
| `experiments/day09_mixed_precision/download_results.ps1` | Download, SHA-256 check, extraction and manifest check on Windows |

---

## Block 1: Design (30 min). Predict first, then check

Each section opens with a question. Write your answer in one line **before** you run the code, then check it against your log. The **"Check"** results below come from the full measured run. [Appendix B](#appendix-b-full-reference-run-measured) links to its logs.

The shared numbers: Ψ = 27,443,200 parameters, **P = 104.69 MiB** in FP32. The matmul weights (QKV, out_proj, two FFN linears per block, plus the head) are 23.1 M of them, **44 MiB in bf16**. One [B, T, D] activation at m = 32 is 32 × 256 × 512 × 4 B = **16 MiB in FP32**. The attention matrix [B, H, T, T] at m = 32 is **64 MiB in FP32**.

### 1.1 What does autocast actually change?

**Predict.** Fill this table for bf16 autocast (fp32 or bf16 in each cell): the parameters; the output of the token embedding, of a LayerNorm, of a Linear, of a whole block (the residual stream), of the head (the logits); the attention softmax; the loss; `p.grad`; AdamW's `exp_avg`.

**Mechanism.** `torch.autocast` changes **no tensor you own**. It is a dispatcher key: inside the region, every op on its *lower-precision list* (matmul-like ops: `mm`, `addmm`, `bmm`, `linear`, `baddbmm`, `scaled_dot_product_attention`, ...) first casts its floating inputs to the low dtype. Every op on its *fp32 list* (`layer_norm`, `nll_loss`, `softmax` when called directly, ...) casts them to fp32. Every other op runs in whatever dtype its inputs already have, following the usual type-promotion rules (`bf16 + fp32 → fp32`).

**Check** (`dtypes` log, first table, [measured]):

| Tensor | fp32 | bf16 autocast | fp16 autocast | why |
| --- | --- | --- | --- | --- |
| parameters | fp32 | **fp32** | **fp32** | autocast never touches them |
| token embedding out | fp32 | fp32 | fp32 | `embedding` is on no list: it runs in the weight's dtype |
| LayerNorm out | fp32 | **fp32** | **fp32** | fp32 list |
| attention out, FFN linear1/linear2 out, logits | fp32 | bf16 | fp16 | `addmm` / `mm` on the low-precision list |
| **block out = residual stream** | fp32 | **fp32** | **fp32** | `h + attn(...)`: fp32 + bf16 promotes to fp32 |
| attention scores `bmm`, `_safe_softmax` | fp32 | **fp32** | **fp32** | the MATH SDPA backend **upcasts q, k, v to fp32 inside the op** |
| `log_softmax` (inside `cross_entropy`) | fp32 | bf16 | fp16 | runs in the logits' dtype |
| loss (`nll_loss_forward`) | fp32 | fp32 | fp32 | fp32 list: autocast casts the log-probs to fp32 first |
| `p.grad` | fp32 | **fp32** | **fp32** | the backward of autocast's weight cast casts the gradient back to the parameter's dtype |
| AdamW `exp_avg`, `exp_avg_sq` | fp32 | fp32 | fp32 | `torch.zeros_like(p)` (Block 3 Stop 5) |

Two results surprise most people. The **residual stream stays fp32**. And with the MATH backend, the **largest activation (the attention probabilities) stays fp32**. Read the "forward ops by output dtype" table in your log to see exactly which ops ran in which dtype. A `TorchDispatchMode` sees the ops *after* autocast has inserted its casts: in bf16 that is 81 casts to bf16 and 19 to fp32 per forward in the measured run.

### 1.2 Where do the FP32 weights live, and is there a master copy?

**Predict.** (a) Does bf16 autocast keep a second, fp32 "master" copy of the weights, as DeepSpeed did on Day 08? (b) How many low-precision weight copies does one forward make, and how many MiB of them does autograd keep until backward? (c) If you instead cast the **whole model** to bf16 (`bf16_pure`: parameters, gradients and AdamW state all bf16), what fraction of the parameter elements does the first AdamW step leave **unchanged**? The first step's learning rate is 3e-4 × 1/20 = 1.5e-5, and Adam's first update is about lr × sign(g).

**Mechanism.** With `torch.autocast`, the **fp32 parameters are the master weights**. There is no separate master copy. The optimizer updates the fp32 parameters in place. Instead, each forward makes a *transient* bf16 copy of every matmul weight. That is `cached_cast` in `autocast_mode.cpp`: once per autocast region, for fp32 leaf tensors that require grad. The matmul's backward needs that copy to compute the input gradient, so autograd saves it until backward. With checkpointing on, the recompute in backward casts the weights again.

| Bytes per parameter | fp32 | `torch.autocast` bf16 (today) | DeepSpeed bf16 (Day 08 design, ZeRO-0) | `bf16_pure` |
| --- | --- | --- | --- | --- |
| model parameter | 4 (fp32) | 4 (fp32 = master) | 2 (bf16) | 2 (bf16) |
| fp32 master copy | — | — (the parameter itself) | 4 (inside the optimizer) | **none** |
| gradient | 4 | 4 (fp32) | 2 bf16 (+ fp32 accumulation buffers in DeepSpeed's bf16 optimizer) | 2 |
| AdamW state | 8 | 8 | 8 | 4 |
| transient, forward → backward | — | 2 per matmul weight (the casts) | — | — |
| total persistent | 16 | 16 | 16 (the "16Ψ" of the ZeRO paper) | 8 |

The totals match, but the placement differs. DeepSpeed stores the low-precision copy as *the* parameter and hides the fp32 copy in the optimizer. Autocast stores fp32 as the parameter and makes the low-precision copy on the fly. The placement decides what a sharding or communication layer sees. DDP under autocast all-reduces **fp32** gradients (§1.6), while DeepSpeed and FSDP2's `MixedPrecisionPolicy` can communicate bf16.

**Check** (`dtypes` log, [measured], m = 8):
- 50 weight casts per forward (8 per block × 6 + the head's 2), **44.1 MiB** in total, of which 44.0 MiB is saved for backward ("saved: weight cast bf16"). In fp32, autograd saves the fp32 weights themselves instead (88.1 MiB, "saved: param fp32"). Those cost nothing extra, because they are the parameters.
- After the step, memory is 490.8 (fp32) vs 489.8 MiB (bf16): params + grads + AdamW state, **the same 4P in both**. Autocast saves no persistent memory.
- **`bf16_pure`: 90.7 % of all parameter elements unchanged by step 1** (matrix weights 89.1 %, embeddings 99.9 %, biases/LayerNorm 56.7 %), vs 0.0 % with fp32 master weights. A weight of size 0.02 in bf16 has a spacing of 2⁻¹³ ≈ 1.2e-4 between representable values. An update of 1.5e-5 is below half of that, so it rounds away. That is what the master copy is for: small updates accumulate in fp32, and only the forward sees the rounded value.

### 1.3 FP16: the order of scaling, accumulation, unscaling, clipping and the step (the debugging exercise)

**Predict.** fp16 autocast, K = 4 microbatches per window (G = 16, m = 4), `clip_grad_norm_(max_norm=1.0)`, GradScaler with S = 2¹⁶ = 65,536. The fp32 reference gradient norm is about 0.48, so clipping should **not** activate. If clipping runs **before** `scaler.unscale_`: (a) what norm does `clip_grad_norm_` report? (b) what gradient norm does AdamW receive? (c) does training crash, diverge, or quietly slow down?

**Mechanism.** Day 04's window, with the scaler threaded through it:

```
zero_grad                                          once per window
for j in 0..K-1:                                   [no_sync() for j < K-1 under DDP, forward included]
    with autocast: loss_j = CE(net(x_j), y_j)
    scaler.scale(loss_j * w_j).backward()          p.grad (fp32) += S * w_j * g_j      S constant over the window
scaler.unscale_(optimizer)                         p.grad /= S; found_inf if any inf/nan; ONCE per window
clip_grad_norm_(params, max_norm)                  sees the TRUE gradient
scaler.step(optimizer)                             skips the step if found_inf
scaler.update()                                    S /= 2 after a skip, S *= 2 after 2000 clean steps
```

Each line sits where it does for a reason:

- **Scaling** exists because fp16 has 5 exponent bits: values below about 6e-8 flush to zero, and activation gradients in fp16 can be that small. Multiplying the loss by S shifts every gradient up by S through the fp16 backward. bf16 has fp32's 8 exponent bits and needs no scaler.
- **S must stay constant over the window**, or the microbatches would be summed with different weights. This is Day 04's contract again: the gradient of the window is Σ w_j g_j. That is why `update()` runs once, after the step.
- **`unscale_` once per window, before anything reads the gradient's magnitude.** `clip_grad_norm_` computes ‖p.grad‖ and knows nothing about S. If it runs first, it sees S·‖g‖ ≈ 31,000, decides to clip by max_norm/(S‖g‖), and then `unscale_` divides by S again. AdamW receives a gradient of norm max_norm/S = 1/65,536 ≈ 1.5e-5.
- **Why it does not crash:** Day 04's Part 3 showed that AdamW cancels a *constant* factor on the gradient, up to its eps. Here the factor is so large that the gradients (≈ 3e-9 per element) fall far below eps = 1e-8. The update becomes g/eps instead of g/|g|, a fraction of its normal size. The loss still decreases, just slowly, so nothing fails loudly.

**Check** (`fp16_order` log, [measured]):

| variant | S | C1: norm seen by clip / true | C2: rel. error of the gradient handed to AdamW | ‖g‖ to AdamW | C3: parameter move / reference |
| --- | --- | --- | --- | --- | --- |
| `ref` fp32 | 1 | 1.000 | 0 | 0.4805 | 1.000 |
| `ok` fp16 | 65,536 | 1.000 | 4.9e-3 | 0.4806 | 1.000 |
| `bug` clip before unscale | 65,536 | **6.55e4** | **1.000** | **1.53e-5** | **0.189** |

`bug` is `DETECTED by C1 and C2`. C0 passes for both, so the call counts alone cannot catch the bug: scale ×4, `unscale_` ×1 and `update` ×1 per window, with S constant inside it. Only the *order* differs. Optional `--underflow`: without the scaler, fp16 gradient error vs fp32 rises from 4.9e-3 to **6.4e-2** (13×). Exactly-zero gradient elements rise from 8.73 % to 8.78 %. The difference is 0.05 percentage points. The aggregate zero count alone does not identify which entries underflowed.

### 1.4 Activation checkpointing: what is saved and what is recomputed?

**Predict** (fp32, m = 32, one step; the per-block number comes from the census):
(a) Without checkpointing, autograd saves about 300 MiB per block. With `checkpoint(block, h, mask, use_reentrant=False)` on every block, what does it keep per block?
(b) How many times does each block's forward run per step?
(c) What is still saved that checkpointing the blocks does not touch?
(d) Will the gradients be bitwise equal to the non-checkpointed path?
(e) **For bf16 autocast with checkpointing on, will "act held" be lower or higher than fp32 with checkpointing on?**

**Mechanism.** Non-reentrant `checkpoint` runs the block under its own saved-tensor hook. Each tensor an op would save is dropped and replaced by a handle. What the frame keeps is the block's **inputs**: the residual stream `h` [B, T, D] and the mask. When backward reaches the first dropped tensor of a block, the unpack hook re-runs that block's forward with the same autocast and RNG state, then continues. Peak memory during backward is then roughly: the kept inputs + **one** block's activations + the gradients built so far.

**Check** (`ckpt` log, [measured], m = 32):

| | fp32 off | fp32 on | bf16 off | bf16 on |
| --- | --- | --- | --- | --- |
| act held after forward (MiB) | 2016.8 | **384.1** | 1656.8 | **512.1** |
| peak during backward (MiB) | 2698.7 | **1065.9** | 2082.7 | **937.9** |
| saved per block (census) | 300.1 | 0 | 206.1 | 0 |
| checkpoint inputs (6 × [B, T, D] fp32) | — | 96.0 | — | 96.0 |
| final norm + head + loss (census) | 304.2 | 304.2 | 416.2 | 416.2 |
| block forwards: in forward / inside backward | 6 / 0 | 6 / **6** | 6 / 0 | 6 / **6** |
| weight casts: forward / backward | — | — | 50 / 0 | 50 / **48** |
| gradients vs the non-checkpointed path | | **bitwise** | | **bitwise** |

The answers:

- (a) **Only the inputs.** The residual stream stays fp32 under autocast (§1.1), so the inputs are fp32 in both precisions.
- (b) **Two calls per block.** The second call can stop once the required tensors have been regenerated. The census also shows what runs again inside backward: per step, 12 `addmm`, 6 `mm`, 12 `bmm`, 6 `_safe_softmax`, 12 `native_layer_norm` and 6 `relu`. In bf16, the 48 block weights are also cast again.
- (c) **The loss path.** At V = 8192, the log-probs [B·T, V] are 256 MiB in fp32.
- (d) **Yes.** The recompute runs the same deterministic kernels on the same inputs.
- (e) **Higher, 512 vs 384 MiB.** Under autocast, the loss path saves bf16 log-probs (128 MiB) *and* the fp32 copy that `nll_loss` (fp32 list) made of them (256 MiB). Once the blocks are checkpointed, that 384 MiB dominates. Autocast is not a memory optimization by itself: it shrinks the matmul activations, adds casts, and leaves fp32 ops in fp32. The next memory lever here would be the loss: chunked cross-entropy, or checkpointing the head and loss too.

### 1.5 The matrix: what is comparable, and what to expect

**Workload, fixed while one factor changes:** Day 01 medium, G = 32 rows per optimizer step, K = 1 (m = 32), 100 optimizer steps from the same seed on the same batches, AdamW with Day 01's LR schedule, `clip_grad_norm_(1.0)` in every precision, MATH attention. Each row changes **one** factor: precision, or checkpointing. Two repeats in alternating order. Timing is the median over steps 10–99, with a synchronize at each phase boundary (forward, backward, optimizer).

**Predict:**
- (a) bf16 vs fp32 step time. TF32 is off, so fp32 matmuls run on CUDA cores (19.5 TFLOP/s on A100) and bf16 on tensor cores (312 TFLOP/s). But part of the step is attention in fp32 (the MATH upcast), LayerNorm, softmax, casts and AdamW. Is the speedup 16×, 3×, or less?
- (b) checkpointing's cost in step time: one extra forward ≈ +1/3 of the compute?
- (c) peak memory with checkpointing at m = 32, in fp32 and in bf16.
- (d) loss drift of bf16 and fp16 from fp32 after 100 steps. Which is closer to fp32? Hint: bf16 has 8 mantissa bits (relative spacing 2⁻⁸ ≈ 3.9e-3), fp16 has 11 (2⁻¹¹ ≈ 4.9e-4); compare with the step-0 gradient errors in §1.2's log: bf16 1.4e-2, fp16 4.8e-3.

**Check [measured].** At fixed precision, checkpointing reduced peak memory and increased step time. The matrix summary reports FP32 **2911 → 1275 MiB** and **99.4 → 119.0 ms/step**, BF16 **2296 → 1148 MiB** and **46.6 → 62.4 ms/step**, and FP16 **2296 → 1148 MiB** and **46.8 → 63.5 ms/step**. All 100 losses and final parameters were bitwise equal with checkpointing on and off at each precision. Both repeats and phase times are in [notes §4](notes.md#4-activation-checkpointing-memory-for-recomputation).

### 1.6 Optional, dual GPU: in which dtype does DDP all-reduce under autocast?

**Predict.** DDP on 2 GPUs, bf16 autocast, G = 32 = 2 × 16. Does the gradient all-reduce move bf16 or fp32? How many MiB per step? What does `register_comm_hook(..., bf16_compress_hook)` change, and at what numerical cost?

**Mechanism.** DDP's buckets are views of `p.grad`, and `p.grad` is fp32 because the parameters are fp32 (§1.1). So autocast does not halve gradient communication. The bf16 compression hook casts each bucket to bf16, all-reduces it, and casts back, which halves the bytes at the cost of rounding the sum.

**Check [measured].** Both runs had FP32 `p.grad` and four gradient all-reduces per step. Default DDP used FP32 with **104.7 MiB** of gradient payload. The BF16 hook used BF16 with **52.3 MiB**, then wrote the result back into the FP32 bucket. The NCCL kernel names (`ncclDevKernel_AllReduce_Sum_f32_RING_LL` vs `..._bf16_...`) and PyTorch's `record_param_comms` events identify the communication dtype. These profiled runs establish the payload change, with throughput left unmeasured by a dedicated benchmark.

### 1.7 New rules (on top of Days 02–08)

1. **"Mixed precision" names a policy, not a dtype.** State which tensors are low precision (here: matmul inputs and outputs) and which stay fp32 (parameters, gradients, optimizer state, residual stream, LayerNorm, loss, and in MATH attention the softmax).
2. **A master copy exists whenever the parameters the optimizer updates are fp32.** Under autocast, that is the parameter itself. Without one (`bf16_pure`), small updates vanish.
3. **Anything that reads gradient magnitudes runs after `unscale_`:** clipping, norm logging, gradient-noise statistics, custom checks. Scale once per window, unscale once per window.
4. **Checkpointing must not change the numbers.** Verify it bitwise in a deterministic setup before you trust its memory savings.
5. **Report memory per phase.** Checkpointing moves the peak from "end of forward" to "during backward"; a single peak number hides which phase it came from.

---

## Block 2: Code (2 h)

### Stage 0 (10 min): read the given parts of `amp_setup.py`

`autocast_ctx` (which precision gets which context), `make_scaler` (fp16 only, the defaults), `sdpa_context` and `AmpNet`. DDP must wrap the module you *call*, so the checkpointed forward lives in a module. Then read the two docstrings below the YOU WRITE markers: they state the contract the harness relies on, including when to call `on(...)`.

### Stage 1 (40 min, required): **YOU WRITE (1)** `window_step` in `my_amp.py`

About 25 lines. Why it matters: this one function is the whole mixed-precision training contract. If the order inside it is wrong, nothing crashes. The loss still goes down, and the bug only shows in a gradient check. The C1 and C2 checks in `fp16_order.py` exist to catch that.

Write it from the order in §1.3, including the `clip_before_unscale` branch (the same calls, with the `unscale_` moved after clipping). Then run:

```bash
python experiments/day09_mixed_precision/fp16_order.py --impl mine
```

Expected: `C0 pass` for `ok` and `bug`; `ok : C0 pass C1 pass C2 pass C3 pass`; `bug : ... => DETECTED by C1 and C2`; `FP16 ORDER: ALL CHECKS PASSED`.

Questions to answer while writing:
- Why does `no_sync()` wrap the forward too, not just the backward? (Day 04: DDP arms its hooks in forward.)
- Why must the logged loss be accumulated as `loss.detach() * w`, not from the scaled loss?
- `scaler.step()` calls `unscale_` itself if you did not. Why is that a trap if you clip?

### Stage 2 (20 min, required): **YOU WRITE (2)** `model_forward` in `my_amp.py`

About 10 lines: Day 01's `TinyTransformer.forward` (`experiments/day01_single_gpu/train.py`, `forward`), with `checkpoint(block, h, mask, use_reentrant=False)` instead of `block(h, src_mask=mask)` when `ckpt` is set. Why it matters: the line you write decides which tensors survive until backward: one input per block instead of ≈ 300 MiB of internals.

```bash
python experiments/day09_mixed_precision/probe_amp.py --part ckpt --impl mine
```

Expected: `CKPT CHECK: PASSED (fp32: bitwise, bf16: bitwise)`. Also check `block forward calls with ckpt on: 6 in forward, 6 inside backward (recompute)`.

### Stage 3 (30 min): read the probe

`probe_amp.py` shows three instruments you can reuse on any model:

- `OpTally`: a `TorchDispatchMode`. It sees every aten op after autograd and autocast, with its output dtype, and it identifies autocast's casts by matching the input storage to a parameter.
- `Probe.pack`: `saved_tensors_hooks`. It records every tensor autograd saves, deduplicated by storage, attributed to the innermost module on a hook-maintained stack.
- module pre-hooks: they count block forwards per phase. A forward inside backward is a recompute.

Find where the census would be blind: inside a checkpointed block, only the innermost saved-tensor hook (checkpoint's) applies.

### Stage 4 (20 min): read the harness

`train_amp.py`: `StepRecorder.on` (phase timing with synchronize, per-phase peaks), the data split (Day 04: the global batch of step s, rank slice, microbatches of m with weights rows_j/rows), `param_sha256` (bitwise equality across processes). Optional: `comm_summary` (record_param_comms events).

### Pitfalls (symptom → cause)

| Symptom | Cause |
| --- | --- |
| Loss decreases, but much more slowly in fp16 than in bf16 | clip (or any gradient-norm logic) ran before `unscale_` (C1/C2) |
| `RuntimeError: unscale_() has already been called on this optimizer since the last update()` | `unscale_` inside the microbatch loop: once per window |
| `AssertionError: No inf checks were recorded prior to update.` | `scaler.update()` without a `scaler.step()`, e.g. per microbatch |
| fp16 loss `nan`, `skipped` steps > 0 at the start | S = 65,536 overflowed; GradScaler skips and halves S. A few early skips are normal; persistent ones mean real overflow in the model |
| `p.grad` is bf16 | the model itself was cast (`model.to(bf16)`); autocast never does that |
| checkpointing saves less memory than expected | the residual stream, the head and the loss are outside the blocks (§1.4), or the peak is in the optimizer step |
| checkpointed gradients differ from non-checkpointed | a non-deterministic kernel, or RNG (dropout) without `preserve_rng_state` |
| DDP + checkpointing errors about hooks firing twice | reentrant checkpointing; use `use_reentrant=False` |

---

## Block 3: Read the source (1 h; Stops 1–4 required, 5–8 optional)

Run `python experiments/day09_mixed_precision/find_symbols.py` for the line numbers in your install. The pre-test lines (torch 2.14.1) are given below. The C++ files are not in the wheel: open them on GitHub at tag `v2.14.1`.

### Stop 1 (15 min, required): autocast. `torch/amp/autocast_mode.py:52` (`autocast`), `__enter__` `:308`, `__exit__` `:343`; then `aten/src/ATen/autocast_mode.cpp`

- In `__enter__`, find what is actually set: thread-local "autocast enabled for device X" and the dtype, plus `torch.autocast_increment_nesting()` (`:319`). No tensor is touched.
- In v2.14.1, the CUDA lists are macros in **`aten/src/ATen/autocast_mode.h`**: `AT_FORALL_LOWER_PRECISION_FP` (`addmm`, `linear`, `bmm`, `scaled_dot_product_attention`, ...), `AT_FORALL_FP32` (`layer_norm`, `nll_loss`, ...) and `AT_FORALL_FP32_SET_OPT_DTYPE` (`softmax`, `log_softmax`, ...). They are registered for CUDA in `autocast_mode.cpp` under `TORCH_LIBRARY_IMPL(aten, AutocastCUDA, m)`. Is `cross_entropy_loss` itself on a CUDA list? Your op table showed `_log_softmax` running in bf16 and `nll_loss_forward` in fp32. Explain that from the lists, then check your explanation in `cross_entropy_loss` in `aten/src/ATen/native/LossNLL.cpp`: which dtype does it pass to `log_softmax` explicitly?
- Find `cached_cast` in `autocast_mode.cpp`: which tensors are cached (fp32 leaves that require grad and are not views, i.e. parameters, and only casts to the low dtype). In `__exit__` (`:343`), find when the cache is cleared: `torch.clear_autocast_cache()` once the nesting count returns to 0.

*Q: why do the 50 weight casts in your log happen once per forward and not once per use? Why do 48 of them happen again in backward with checkpointing?*

### Stop 2 (15 min, required): GradScaler. `torch/amp/grad_scaler.py`: `OptState` `:43`, `scale` `:193` (the overloads start at `:182`), `_unscale_grads_` `:246`, `unscale_` `:302`, `step` `:375`, `_maybe_opt_step` `:363`, `update` `:484`

- `scale`: multiplies by a **GPU** tensor `_scale` (no host sync), lazily created at `init_scale`.
- `unscale_`: the per-optimizer state machine `READY → UNSCALED → STEPPED`, and the error on a second call. `_unscale_grads_` → `torch._amp_foreach_non_finite_check_and_unscale_`: one fused kernel per device and dtype that multiplies by 1/S and sets `found_inf`.
- `step`: if the state is `READY`, it calls `unscale_` itself. `_maybe_opt_step` reads `found_inf` (**this is the host sync**) and skips `optimizer.step()` if it is non-zero.
- `update`: `torch._amp_update_scale_`: backoff on inf, growth after `growth_interval` clean steps.

*Q: where would a clip placed between `step`'s internal `unscale_` and the optimizer step have to go, and why can't it? Answer: it can't; you must call `unscale_` yourself, then clip, then `step`.*

### Stop 3 (10 min, required): clipping. `torch/nn/utils/clip_grad.py`: `clip_grad_norm_` `:185`, `_get_total_norm` `:49`, `_clip_grads_with_norm_` `:120`

- `_get_total_norm`: per-device, per-dtype grouped norms, then the norm of norms. It reads whatever is in `.grad`.
- `_clip_grads_with_norm_`: `clip_coef = max_norm / (total_norm + 1e-6)`, clamped to 1, multiplied in place. No host sync (the coefficient stays on the GPU).

*Q: with `clip_before_unscale`, compute the coefficient from your log's C1 value and check that AdamW received max_norm / S.*

### Stop 4 (20 min, required): checkpointing. `torch/utils/checkpoint.py`: `checkpoint` `:383`, `_checkpoint_without_reentrant_generator_impl` `:1818`, `_CheckpointFrame` `:975`, `_checkpoint_hook` `:1339`, `_recomputation_hook` `:1273`, `_get_autocast_kwargs` `:219`

- In `_checkpoint_without_reentrant_generator_impl`: the RNG state saved (`preserve_rng_state`), the autocast state captured (`_get_autocast_kwargs`), the `recompute_fn` closure, and `_NoopSaveInputs`, which saves the **inputs**. That is the 96 MiB your census attributed to "outside modules".
- `_checkpoint_hook`: its pack function returns a `_Holder` instead of the tensor. Nothing inside the block is kept.
- The unpack function: on the first unpack of a frame, it runs `recompute_fn` under `_recomputation_hook`, which fills the holders, and then returns the requested tensor. Find the early-stop mechanism (`_StopRecomputationError`): the recompute can stop as soon as the needed tensors exist.

*Q: why does `use_reentrant=False` work with DDP and with inputs that do not require grad, and the reentrant `CheckpointFunction` (`:238`) not as well?*

### Stops 5–8 (optional)

- **Stop 5:** `torch/optim/adam.py` `_init_group` `:139`: `torch.zeros_like(p, memory_format=torch.preserve_format)`. The state dtype follows the parameter, which is why `bf16_pure` has bf16 moments.
- **Stop 6:** selective checkpointing: `create_selective_checkpoint_contexts` `:1702`, `CheckpointPolicy` `:1480`. Save matmul outputs, recompute the cheap elementwise ops. Which of today's recomputed ops would you keep?
- **Stop 7:** `nn/functional.py` `multi_head_attention_forward` `:6664` and `_canonical_mask` `:6612`: where the bool causal mask becomes a float mask (the `zeros_like` + `masked_fill_` in your recompute list), and the call into SDPA. Optional C++: `_scaled_dot_product_attention_math` in `aten/src/ATen/native/transformers/attention.cpp`: the upcast of half/bf16 q, k, v.
- **Stop 8:** distributed mixed precision: `ddp_comm_hooks/default_hooks.py` `_compress_hook` `:57`, `bf16_compress_hook` `:116`; FSDP2 `MixedPrecisionPolicy` (`_fsdp_api.py:14`: `param_dtype`, `reduce_dtype`, `output_dtype`) vs Day 07; DeepSpeed `runtime/bf16_optimizer.py` vs Day 08's fp32 master partition.

---

## Block 4: Run + summarize (30–60 min)

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ALGO
RID=a100_$(date -u +%Y%m%dT%H%M%SZ)
# your code first (seconds each)
python experiments/day09_mixed_precision/fp16_order.py --impl mine --run-id "$RID"
python experiments/day09_mixed_precision/probe_amp.py --part ckpt --impl mine --run-id "$RID"
# the required stages, logged, checked and archived (~6 min); omit --core to add the optional stages (~9 min)
tmux new -s d9 "bash experiments/day09_mixed_precision/run_server_once.sh $RID --core"
```

The runner writes `outputs/day09_amp/session_$RID/`. It holds `commands.txt`, `environment.log` (versions, GPUs, `nvidia-smi topo -m`, a project and torch-source snapshot), one log per stage, `SHA256SUMS`, `status.txt` and `CORE_SUCCESS.txt`. It also writes `outputs/day09_exports/day09_$RID.tar.gz` (+ `.sha256`). Download from Windows PowerShell in the project root:

```powershell
experiments/day09_mixed_precision/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>
```

It checks the archive hash, extracts to `outputs/day09_downloads/<RID>/`, and re-hashes every file in the manifest. Stop the instance only after it prints `Manifest OK`.

### How to read the `dtypes` log

Four columns: fp32, bf16, fp16, bf16_pure. Copy the first table into `notes.md` §2. In "what autograd keeps", compare `saved: param fp32` (fp32: the weights themselves) with `saved: weight cast bf16` (autocast: copies). "Allocated after step" is params + grads + AdamW state: equal in fp32/bf16/fp16. In the "unchanged by step 1" rows, `bf16_pure` vs the rest is the master-copy argument.

### How to read the `fp16_order` log

Line 1: the true norm and whether clipping is active. Then the table (C1: what clipping saw / true; C2: the error of what AdamW got; C3: how far training moved), C0 per variant, the two verdict lines, and the explanation line with S. The run must end with `FP16 ORDER: ALL CHECKS PASSED`.

### How to read the `ckpt` log

Per precision: memory (start, after forward, peaks), the census by module (blocks 300 → 0 MiB, the "outside" row gains the checkpoint inputs), the recompute evidence (block calls, re-casts, forward ops inside backward) and the correctness line. It must end with `CKPT CHECK: PASSED`.

### How to read the `matrix` log

One row per configuration: `ms/step` for both repeats, the phase split (fwd / bwd / opt), tokens/s, memory (start of step, act held, peak in forward, peak in backward, step peak), final loss, skipped steps. Then the checkpointing block per precision: losses equal over 100 steps, final parameters bitwise (SHA-256), peak ratio, time ratio and backward-time ratio. `CKPT EQUALITY: PASSED`. Then drift from fp32. Read the backward column for the recompute cost. The forward time hardly changes with checkpointing; the backward grows by about one forward of the blocks.

### Write `day09/notes.md`

§1 run record, §2 the dtype table + master-copy answer, §3 fp16 order, §4 checkpointing (correctness, memory, time), §5 your three explanations. Label each number **[measured]** (your run), **[derived]** (computed from measured numbers) or **[pre-test]** (quoted from Appendix A for comparison only).

---

## Today's core questions, answered (compare with yours after your run)

**What changes dtype.** Under `torch.autocast(bf16)`, only the inputs and outputs of ops on autocast's low-precision list change: matmuls, so the QKV and FFN projections and the logits. Parameters, `p.grad`, AdamW's state, the residual stream, LayerNorm and the loss stay fp32. With the MATH SDPA backend, attention's scores and softmax stay fp32 too. The fp32 parameters are the master weights. Autocast makes no persistent copy, only a transient bf16 copy of each matmul weight per forward (44 MiB here), which autograd keeps until backward. DeepSpeed's design (Day 08) has the same 16 bytes per parameter in a different place: bf16 parameters, plus an fp32 master in the optimizer. Without any fp32 copy (`bf16_pure`), 90.7 % of the elements did not move in the first step.

**Why unscale precedes clipping.** The scaler multiplies every gradient by S so that fp16 does not underflow. `clip_grad_norm_` measures whatever is in `.grad`, so before `unscale_` it measures S‖g‖, clips a gradient that did not need clipping by a factor ≈ 1/(S‖g‖), and `unscale_` then divides by S again: AdamW receives max_norm/S. The loss still falls, because AdamW cancels most constant factors, but every update shrinks once eps dominates (to 19 % here). Scale once per window with S held constant, unscale once per window, then clip, step and update.

**Where checkpointing trades memory for compute.** Per checkpointed block, it keeps the block input (one fp32 [B, T, D], 16 MiB at m = 32) instead of every intermediate (≈ 300 MiB fp32, ≈ 206 MiB bf16). During backward it re-runs that block's forward (6 extra block forwards per step, the matmuls and attention included), which costs about one forward of compute. The gradients are bitwise identical. What remains is the part it does not cover (head, loss, inputs) and one block's working set during backward.

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. Name four tensors that stay fp32 under bf16 autocast in this model, and say why each does.</summary>

Parameters (autocast never touches tensors you own); `p.grad` (the cast's backward returns the gradient in the parameter's dtype); AdamW state (`zeros_like(p)`); the residual stream (fp32 + bf16 promotes to fp32); LayerNorm outputs and the loss (fp32 list); MATH attention's scores and softmax (the backend upcasts).
</details>

<details><summary>2. Is there an fp32 master copy under autocast? Under DeepSpeed bf16? Under model.to(bfloat16)?</summary>

Autocast: the fp32 parameters are the master; no extra copy, plus transient low-precision weight casts. DeepSpeed bf16: bf16 parameters plus a separate fp32 master in the optimizer. `model.to(bf16)`: no master at all, so small updates round away.
</details>

<details><summary>3. Why must the loss scale stay constant over K accumulated microbatches?</summary>

`p.grad` accumulates S_j·w_j·g_j. `unscale_` divides by one S. If S changed inside the window, the microbatches would be weighted by S_j/S instead of w_j. That is the same contract as Day 04, broken in a new way.
</details>

<details><summary>4. The fp16 loss goes down, just slower than bf16. Name the first check you would run.</summary>

Compare the norm clipping reports (or the gradient norm after clipping) with an fp32 reference on the same batch. A ratio near S means clipping ran before `unscale_`.
</details>

<details><summary>5. What does non-reentrant checkpointing keep, and what triggers the recompute?</summary>

It keeps the inputs of the checkpointed function, through `_NoopSaveInputs`, plus the RNG and autocast state. The first unpack of any dropped tensor of that frame during backward triggers the recompute.
</details>

<details><summary>6. Why did bf16 + checkpointing hold more memory than fp32 + checkpointing?</summary>

After the blocks are checkpointed, the loss path dominates. Under autocast, it saves both the bf16 log-probs and the fp32 copy that `nll_loss` makes of them (128 + 256 MiB), against 256 MiB in fp32.
</details>

<details><summary>7. Does bf16 autocast halve DDP's gradient all-reduce?</summary>

No. The buckets hold fp32 `p.grad`. A bf16 compression hook (or FSDP2's `reduce_dtype`, or DeepSpeed's bf16 mode) is needed, and it changes the numerics of the sum.
</details>

**Next (Day 10):** Triton fused softmax. Today's MATH attention showed a [B, H, T, T] softmax kept in fp32 and saved for backward. A fused kernel that never materializes it is the other way to cut that memory.

---

## Appendix A: pre-test results (`pretest1`, assistant run, 2026-11-09) [pre-test]

Host: Vast.ai container 59841443, 2 × A100-SXM4-40GB, NV12, both GPUs on NUMA node 1, driver 595.71.05, torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24, NCCL 2.30.7, `condition=shared`. Logs: [`outputs/day09_downloads/pretest1/`](../outputs/day09_downloads/pretest1/) (session folder `session_pretest1`).

### A.1 Dtypes ([log](../outputs/day09_downloads/pretest1/session_pretest1/dtypes.log))

See §1.1 and §1.2. Saved for backward at m = 8: fp32 592.3 MiB (activations 504.2 fp32 + weights 88.1), bf16 446.3 MiB (activations 284.2 fp32 + 118.0 bf16, weight casts 44.0), fp16 the same as bf16, bf16_pure 356.3 MiB. Step-0 gradient error vs fp32: bf16 1.42e-2, fp16 4.78e-3, bf16_pure 2.11e-2. Loss at step 0: 9.20471 / 9.20441 / 9.20470 / 9.18750.

### A.2 fp16 order ([log](../outputs/day09_downloads/pretest1/session_pretest1/fp16_order.log))

See §1.3. True norm 0.4805 (clipping inactive). `ok` C1 1.000, C2 4.9e-3, C3 1.000. `bug` C1 6.554e4, C2 1.000, ‖g‖ to AdamW 1.526e-5 (= 1/65,536), C3 0.189. No skipped steps.

### A.3 Checkpointing probe ([log](../outputs/day09_downloads/pretest1/session_pretest1/ckpt.log))

See §1.4. Bitwise gradients in fp32 and bf16.

### A.4 Matrix ([log](../outputs/day09_downloads/pretest1/session_pretest1/matrix.log))

One GPU, medium, G = 32 rows per optimizer step, 100 steps, timing = median of steps 10–99, 2 repeats in alternating order (both shown). Memory in MiB, rank 0. Optional rows are marked.

| row | K × m | ms/step (r0 / r1) | fwd | bwd | opt | tokens/s | start | act held | peak fwd | peak bwd | step peak | loss @100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fp32_off` | 1 × 32 | 99.0 / 98.9 | 30.5 | 60.1 | 8.4 | 82,772 | 382 | 2017 | 2655 | 2911 | **2911** | 8.1172 |
| `fp32_on` | 1 × 32 | 119.1 / 119.2 | 30.7 | 80.0 | 8.4 | 68,771 | 379 | 384 | 1019 | 1275 | **1275** | 8.1172 |
| `bf16_off` | 1 × 32 | 50.0 / 50.1 | 14.0 | 23.1 | 12.9 | 163,682 | 385 | 1655 | 2168 | 2296 | **2296** | 8.1169 |
| `bf16_on` | 1 × 32 | 62.4 / 61.5 | 15.6 | 33.5 | 12.9 | 132,230 | 379 | 513 | 1059 | 1148 | **1148** | 8.1169 |
| `fp16_off` | 1 × 32 | 50.0 / 50.1 | 13.8 | 23.0 | 13.3 | 163,627 | 385 | 1655 | 2168 | 2296 | **2296** | 8.1171 |
| `fp16_on` | 1 × 32 | 63.5 / 63.5 | 15.9 | 33.9 | 13.7 | 129,028 | 379 | 513 | 1059 | 1148 | **1148** | 8.1171 |
| `bf16_off_k4` [optional] | 4 × 8 | 171.3 / 135.8 | 54.7 | 84.5 | 15.2 | 54,064 | 380 | 449 | 968 | 1000 | 1000 | 8.1170 |
| `bf16_off_flash` [optional] | 1 × 32 | 43.6 / 43.1 | 9.9 | 20.6 | 12.9 | 189,005 | 381 | 1178 | 1687 | 1815 | 1815 | 8.1169 |
| `bf16_on_flash` [optional] | 1 × 32 | 55.1 / 53.9 | 13.1 | 28.5 | 13.0 | 150,222 | 382 | 512 | 1060 | 1150 | 1150 | 8.1169 |

No fp16 step was skipped. The scale stayed at 65,536 for all 100 steps.

**Checkpointing at the same precision (on / off):**

| precision | losses equal over 100 steps | final params bitwise (SHA-256) | step peak | act held | ms/step | backward |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | yes | yes | 2911 → 1275 (× 0.44, −1636 MiB) | 2017 → 384 | 99.0 → 119.1 (× 1.20) | × 1.33 |
| bf16 | yes | yes | 2296 → 1148 (× 0.50, −1148 MiB) | 1655 → 513 | 50.0 → 62.0 (× 1.24) | × 1.45 |
| fp16 | yes | yes | 2296 → 1148 (× 0.50) | 1655 → 513 | 50.1 → 63.5 (× 1.27) | × 1.47 |

The forward time barely changes (fp32 +0.2 ms, bf16 +1.6 ms: checkpoint's own bookkeeping per block); the backward grows by about one forward (fp32: +19.9 ms against a 30.5 ms forward; bf16: +10.4 ms against 14.0 ms). That is less than one forward, because the recompute skips the head and the loss.

**Drift from fp32 (ckpt off, same init and batches):** mean |loss − loss_fp32| over the last 10 steps is **bf16 4.4e-4** and **fp16 7.8e-5**. ‖θ − θ_fp32‖ / ‖θ_fp32 − θ_0‖ after 100 steps is bf16 1.6e-2 and fp16 1.1e-2. The fp32 loss went from 9.181 to 8.117 and moved the parameters by 20.2. fp16 stays closer to fp32 than bf16, matching their mantissas (11 vs 8 bits), because nothing overflowed.

**Answers to §1.5:**
- (a) bf16 is **2.0×** faster than true fp32, not 16×. Only the matmuls get faster. MATH attention still runs fp32 `bmm` and softmax, and the optimizer phase does not speed up at all.
- (b) Checkpointing costs **+20 % to +27 %** per step.
- (c) Checkpointing roughly halves the step peak, to 1275 MiB (fp32) and 1148 MiB (bf16).
- (d) fp16 drifts less than bf16 here.

Optional rows: K = 4 at the same G gives the lowest peak (1000 MiB), but at m = 8 the step takes 136–171 ms. Launch and phase-boundary synchronization overhead are possible contributors that these timings do not isolate. Checkpointing reaches 1148 MiB for +24 %. Flash attention removes the fp32 [B, H, T, T] matrix: 1815 MiB and 43 ms without checkpointing.

**One unexplained observation [pre-test, not isolated].** The `opt` phase (clip + AdamW, from the end of backward to the end of the step) takes 12.9 ms under bf16 autocast against 8.4 ms in fp32, with fp32 gradients of the same size in both. Two checks did not explain it. First, a standalone loop (`window_step` with a synchronize at every event) timed `clip_grad_norm_` at 1.3 ms and AdamW at 6.8–8.8 ms in **both** precisions. Second, disabling Python's garbage collector left the harness at 8.3 vs 12.9 ms. A profiler trace of that phase is an optional follow-up. It does not affect the checkpointing comparison, which is made at the same precision.

> **Update from run `manual_full_20261109T065636Z` [measured].** The slow `opt` phase is not a fixed cost of autocast; it varies from process to process. Two reduced-precision runs had an FP32-level `opt` phase: `bf16_off_r0` at 8.33 ms (43.1 ms/step) and `fp16_off_r1` at 8.73 ms (42.9 ms/step). Their repeats, `bf16_off_r1` and `fp16_off_r0`, took 13.1 and 13.7 ms (50.2 and 50.7 ms/step). In those slow runs, backward was also about 2 ms longer. A slow `opt` phase appeared in 8 of the 10 reduced-precision runs and in none of the 4 FP32 runs, so precision may still be a factor, but it is not sufficient on its own. The pretest1 sentence above, "It does not affect the checkpointing comparison", is also wrong. The checkpoint-off baseline can fall in either mode, so the BF16 checkpointing overhead lies anywhere from +24 % to +45 %. A steadier measure is the extra backward time: about one forward pass of the blocks in both precisions. The next check, still optional, is to profile one fast and one slow `bf16_off` process and compare the `opt` phase between them: which kernels run, and the gaps between them.

### A.5 DDP question, optional ([log](../outputs/day09_downloads/pretest1/session_pretest1/ddp.log))

2 GPUs, NVLink, bf16 autocast, G = 32 = 2 ranks × 16 rows, K = 1, profiled steps 4–5 (not a benchmark). The 1-GPU reference ran bf16 on the same 32 rows.

| | `p.grad` | gradient all-reduce per step | NCCL kernel | payload per step | step-0 gradient vs 1-GPU bf16 |
| --- | --- | --- | --- | --- | --- |
| DDP | fp32 | 4 buckets (4.19 M, 7.35 M, 7.36 M, 8.53 M elements), `Float` | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | **104.7 MiB** = P | rel err 3.7e-3 |
| DDP + `bf16_compress_hook` | fp32 | the same 4 buckets, `BFloat16` | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | **52.3 MiB** = P/2 | rel err 3.9e-3 |

The hook's gradients differ from plain DDP's by 1.8e-3 (relative). Each step also has one `broadcast` of a `Bool` tensor of 65,536 elements: that is DDP re-broadcasting the module's buffers before every forward (`broadcast_buffers=True`), here the 256 × 256 causal mask. Plain DDP vs one GPU differs by 3.7e-3 because each rank's bf16 matmuls sum 16 rows instead of 32, so the rounding falls differently. That is the bf16 noise floor, larger than what the hook adds.

**Answer to §1.6:** autocast does not change what DDP communicates. Its buckets hold the fp32 `p.grad`, so the all-reduce moves P bytes per step. Halving that takes an explicit choice (a compression hook, FSDP2's `reduce_dtype`, DeepSpeed's bf16 mode), at the price of rounding the reduction.

## Appendix B: full reference run [measured]

Run `manual_full_20261109T065636Z` completed in full mode on two A100-SXM4-40GB GPUs with exit code 0. The single-GPU matrix covered nine rows with two repeats, followed by three runs in the DDP stage. The archived data and all 79 manifest files were verified. See the [run record and completed tables](notes.md).

| Evidence | Result |
| --- | --- |
| [Dtype probe](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/dtypes.log) | Autocast left parameters, gradients and AdamW moments FP32. BF16 autocast created 44.1 MiB of weight casts. With pure BF16, 90.7% of parameter elements were unchanged after one step. |
| [FP16 order](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/fp16_order.log) | Correct order delivered a norm of 0.48060 to AdamW. The deliberate bug delivered 1.5259e-5, with zero skipped steps in both cases. |
| [Checkpoint probe](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/ckpt.log) | Loss and gradients matched bitwise at fixed precision. Six block inputs occupied 96 MiB and the BF16 recompute made 48 weight casts. |
| [Training matrix](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/matrix.log) | The six core rows passed the checkpoint equality check. The three optional rows covered accumulation and Flash attention. |
| [DDP trace summary](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/ddp.log) | The BF16 hook halved the gradient payload while the parameter gradients stayed FP32. |
| [Output checks](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/verify_outputs.log) | All requested stages passed. |

**Reading the optional comparisons.** Compare `bf16_off` with `bf16_off_k4` to hold MATH attention and checkpointing fixed. The peak dropped from 2296 to 1000 MiB, while the two step times rose from 43.1 / 50.2 to 169.4 / 123.0 ms. The recorder synchronizes after each forward and backward, so attributing the slowdown requires a separate step-timed comparison. In the Flash rows, the peak was 1815 MiB with checkpointing off and 1150 MiB with it on. Flash tiles both score and probability calculations and retains the attention output, whose size scales linearly with sequence length. The savings from Flash and block checkpointing overlap in these measurements.
