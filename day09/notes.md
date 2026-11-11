# Day 09: mixed precision and activation checkpointing

The results below are from the full run `manual_full_20261109T065636Z`. **[measured]** identifies values read from its logs. **[derived]** identifies arithmetic on those values. Mechanism explanations and untested follow-ups are labeled separately. The earlier `pretest1` is retained in [Guide Appendix A](GUIDE.md#appendix-a-pre-test-results-pretest1-assistant-run-2026-11-09-pre-test).

## 1. Run record

| Item | Record |
| --- | --- |
| Run ID | `manual_full_20261109T065636Z` |
| Time | 2026-11-09 06:56:36–07:01:43 UTC (14:56:36–15:01:43 SGT) |
| Host | Vast.ai container 59841443, hostname `f46fb89171b7`, 2 × A100-SXM4-40GB, NV12, both GPUs on NUMA node 1 |
| Software | Python 3.12.15, PyTorch 2.14.1+cu130, CUDA 13.0 build, cuDNN 9.24, NCCL 2.30.7, driver 595.71.05, TF32 off |
| Scope | `--full`, `condition=shared`, `impl=ref`. Dtype probe, FP16 order and no-scaler control, checkpoint probe, 9 matrix rows × 2 repeats, and 3 DDP-stage runs |
| Verdict | Exit code 0. `FP16 ORDER: ALL CHECKS PASSED`, `CKPT CHECK: PASSED`, `CKPT EQUALITY: PASSED`, and `OUTPUT CHECKS PASSED` |
| Archive | SHA-256 `458aee23578a3dcf1937445969e8e66135a6171a605608f6f6fb83b8d9477f5a`. The local archive and all 79 extracted files in the manifest were independently reverified |
| Handwritten implementation | `window_step` and `model_forward` are still stubs in the workspace. No `--impl mine` verification is recorded |

[Environment](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/environment.log) · [Commands](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/commands.txt) · [Output checks](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/verify_outputs.log) · [Manifest](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/SHA256SUMS)

## 2. What changes dtype

**[measured]** One optimizer step with the same initialization and batch, medium preset, microbatch 8, MATH SDPA.

| Tensor | FP32 | BF16 autocast | FP16 autocast | `bf16_pure` |
| --- | --- | --- | --- | --- |
| Parameters | FP32 | FP32 | FP32 | BF16 |
| Token embedding and block 0 LayerNorm outputs | FP32 | FP32 | FP32 | BF16 |
| Block 0 attention output after projection and FFN linear outputs | FP32 | BF16 | FP16 | BF16 |
| Block 0 residual output and final LayerNorm output | FP32 | FP32 | FP32 | BF16 |
| Attention scores and softmax (MATH SDPA) | FP32 | FP32 | FP32 | FP32 |
| Logits and `log_softmax` inside cross-entropy | FP32 | BF16 | FP16 | BF16 |
| Loss from `nll_loss_forward` | FP32 | FP32 | FP32 | BF16 |
| Parameter gradients | FP32 | FP32 | FP32 | BF16 |
| AdamW `exp_avg` and `exp_avg_sq` | FP32 | FP32 | FP32 | BF16 |
| AdamW step counter | FP32 | FP32 | FP32 | FP32 |
| GradScaler scale | — | — | 65,536 (FP32 tensor) | — |

| Memory and numerics | FP32 | BF16 autocast | FP16 autocast | `bf16_pure` |
| --- | ---: | ---: | ---: | ---: |
| Forward weight casts (count / MiB) | 0 / 0.0 | 50 / 44.1 | 50 / 44.1 | 0 / 0.0 |
| Saved low-precision weight casts (MiB) | 0.0 | 44.0 | 44.0 | 0.0 |
| Saved tensors, unique storage (MiB) | 592.3 | 446.3 | 446.3 | 356.3 |
| Allocated after forward and loss (MiB) | 675.0 | 620.0 | 620.0 | 433.6 |
| Allocated after step (MiB) | 490.8 | 489.8 | 489.8 | 283.0 |
| Step-0 gradient relative error vs FP32 | 0 | 1.42e-2 | 4.78e-3 | 2.11e-2 |
| Parameter elements unchanged by step 1 | 0.0% | 0.0% | 0.0% | 90.7% |

**[understanding]** CUDA autocast applies registered rules per operator. Linear layers use low-precision inputs, ordinary residual addition promotes BF16 plus FP32 to FP32, and MATH SDPA upcasts its internal low-precision attention calculations. Native autocast updates the model's original FP32 parameters, while the DeepSpeed experiment used FP32 master shards to retain small updates. A temporary weight cast can share the same storage between the autocast cache and autograd. Clearing the cache leaves any autograd-held copy alive. The BF16 autocast run retained 44.0 MiB of cast weights, yet its forward allocation was lower because saved activations shrank.

[Raw dtype log](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/dtypes.log)

**[derived] Model-state bytes per parameter.** In Day 09, native autocast keeps FP32 weights (4 bytes), FP32 gradients (4 bytes), and two FP32 AdamW moments (8 bytes), for 16 bytes per parameter. The mixed-precision ZeRO design in Day 08 also totals 16 bytes before sharding: low-precision weights and gradients (2 bytes each), an FP32 master weight (4 bytes), and two FP32 moments (8 bytes). Sharding and buffers determine the actual per-rank allocation. The Day 09 step-end allocations of 490.8 MiB in FP32 and 489.8 MiB under BF16 autocast compare two native PyTorch runs and include allocations beyond this model-state count. [Day 08 Guide](../day08/GUIDE.md#12-what-each-stage-keeps-per-rank-in-fp32)

## 3. FP16: unscale before clip

**[measured]** Global batch 16, four microbatches of 4, clipping threshold 1.0. The FP32 step-0 reference norm was 0.48053, below the clipping threshold.

| Variant | Scale | Norm seen by clip | C1 ratio vs reference | C2 gradient relative error | Norm to AdamW | C3 move ratio | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ref` | 1 | 0.48053 | 1.000 | 0 | 0.48053 | 1.000 | 0 |
| `ok` | 65,536 | 0.48060 | 1.000 | 4.910e-3 | 0.48060 | 1.000 | 0 |
| `bug` | 65,536 | 31,496 | 6.554e4 | 1.000 | 1.5259e-5 | 0.1886 | 0 |
| `noscale` (optional) | 1 | 0.48308 | 1.005 | 6.399e-2 | 0.48308 | 1.002 | 0 |

C0 passed for both `ok` and `bug`: four scale calls, one unscale call, one update call, and a constant scale within each window. `ok` passed C0–C3 and C1/C2 detected `bug`. Without the scaler, exactly-zero gradient elements increased from 8.73% to 8.78% and the relative gradient error increased from 4.910e-3 to 6.399e-2.

**[understanding]** Clear gradients once at the start of a window and weight each mean loss by its microbatch's share of the samples. Scale and backpropagate each weighted loss with a fixed scale, then unscale once, clip, step and update the scaler. In the bug path, global-norm clipping sees the scaled gradient and reduces its norm to about 1 before unscale divides it by 65,536. The gradients remain finite, so the scaler permits the update. Logging the norm immediately before clipping and immediately before stepping exposes the error even when skipped steps stay at zero.

**[understanding] Why the parameter move ratio was 0.1886.** In AdamW's gradient-driven update, the first moment is divided by the square root of the second moment plus `eps`. If the whole gradient history shrinks by the same factor, the numerator and the square root shrink together. They largely cancel when `eps` is negligible. The bug makes many gradient components small enough for the fixed `eps = 1e-8` to matter. The numerator shrinks while the denominator approaches `eps`, so the update gets smaller. This explains why the parameter displacement shrank even with AdamW. The ratio of 0.1886 is measured over four steps. [PyTorch AdamW formula](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

[Raw FP16 log](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/fp16_order.log)

## 4. Activation checkpointing: memory for recomputation

**[measured]** Checkpoint probe with one step, microbatch 32, MATH SDPA.

| Metric | FP32 off | FP32 on | BF16 off | BF16 on |
| --- | ---: | ---: | ---: | ---: |
| Activation memory held after forward (MiB) | 2016.8 | 384.1 | 1656.8 | 512.1 |
| Peak during backward (MiB) | 2698.7 | 1065.9 | 2082.7 | 937.9 |
| Saved inside each block (MiB) | 300.1 | 0.0 | 206.1 | 0.0 |
| Saved outside modules, including loss and checkpoint inputs (MiB) | 256.1 | 352.1 | 384.1 | 480.1 |
| Block calls in forward / during backward | 6 / 0 | 6 / 6 | 6 / 0 | 6 / 6 |
| Weight casts in forward / during backward | 0 / 0 | 0 / 0 | 50 / 0 | 50 / 48 |
| Loss and gradients vs same-precision baseline | Baseline | Bitwise equal | Baseline | Bitwise equal |

**[measured]** Training matrix, single GPU, global batch 32, 100 steps, 10 warmup steps, two repeats. Phase times are the log's summaries. Memory columns are in MiB.

| Configuration | K × m | ms/step (r0 / r1) | Forward (ms) | Backward (ms) | Optimizer (ms) | Act held | Step peak | Final loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32_off` | 1 × 32 | 99.0 / 99.8 | 30.5 | 60.1 | 8.7 | 2017 | 2911 | 8.1172 |
| `fp32_on` | 1 × 32 | 119.1 / 119.0 | 30.7 | 80.0 | 8.4 | 384 | 1275 | 8.1172 |
| `bf16_off` | 1 × 32 | 43.1 / 50.2 | 13.8 | 22.1 | 10.7 | 1655 | 2296 | 8.1169 |
| `bf16_on` | 1 × 32 | 62.6 / 62.2 | 15.7 | 33.8 | 13.0 | 513 | 1148 | 8.1169 |
| `fp16_off` | 1 × 32 | 50.7 / 42.9 | 13.6 | 22.0 | 11.2 | 1655 | 2296 | 8.1171 |
| `fp16_on` | 1 × 32 | 63.3 / 63.7 | 15.9 | 33.9 | 13.7 | 513 | 1148 | 8.1171 |
| `bf16_off_k4` (optional) | 4 × 8 | 169.4 / 123.0 | 54.5 | 84.4 | 15.4 | 449 | 1000 | 8.1170 |
| `bf16_off_flash` (optional) | 1 × 32 | 43.0 / 43.0 | 9.9 | 20.2 | 12.9 | 1178 | 1815 | 8.1169 |
| `bf16_on_flash` (optional) | 1 × 32 | 47.1 / 54.7 | 13.2 | 24.6 | 13.0 | 512 | 1150 | 8.1169 |

**[measured]** With checkpointing toggled at the same precision, all 100 losses and the final parameters were bitwise equal in FP32, BF16 and FP16. No matrix step was skipped. The mean absolute loss drift from FP32 over the last ten steps with checkpointing off was 4.35e-4 for BF16 and 7.82e-5 for FP16. The corresponding final parameter differences relative to the FP32 training displacement were 1.643e-2 and 1.108e-2.

**[derived]** FP32 peak fell by 1,636 MiB (56.2%) while the matrix summary's step time rose from 99.4 to 119.0 ms (about 20%). BF16 peak fell by 1,148 MiB (50%), with step time rising from 46.6 to 62.4 ms (about 34%). FP16 had the same peak reduction and went from 46.8 to 63.5 ms (about 36%).

**[understanding]** The six checkpoint boundaries preserve six FP32 residual inputs of 16 MiB each. Backward restores the forward autocast and RNG context, then recomputes needed block internals. The probe counted six recompute calls and 48 new BF16 block-weight casts.

**[derived] Why BF16 checkpointing held an extra 128 MiB after forward.** The head and loss sit outside the checkpoint boundaries. With 32 rows, 256 tokens per row and a vocabulary of 8192, the log-probabilities are shaped `[8192, 8192]`. The FP32 path saves one 256 MiB tensor. The BF16 autocast path saves the 128 MiB BF16 `log_softmax` output and the 256 MiB FP32 copy used by `nll_loss`. This accounts for the 128 MiB increase in the outside-modules census (352.1 → 480.1 MiB with checkpointing on). BF16 still had the lower backward peak (937.9 MiB versus 1065.9 MiB).

[Raw checkpoint log](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/ckpt.log) · [Raw matrix log](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/matrix.log)

## 5. Three explanations from the walkthrough

These explanations are based on the walkthrough and the run.

1. **What autocast changes.** I follow the dtype through each operation. A linear layer can produce BF16, and adding its output to an FP32 residual gives FP32 by type promotion. LayerNorm follows the FP32 autocast rule. In my run, the MATH attention backend computed scores and softmax in FP32, and the BF16 `log_softmax` output went into an FP32 loss operation. The optimizer updates the original FP32 weights, so a small update can build up over steps even while a linear layer keeps seeing the same rounded BF16 value.
2. **Why order matters in FP16.** Each microbatch contributes a weighted gradient at the same scale, and those gradients add up in `.grad`. I divide the sum by the scale once and clip after that. If I clip first, the scaled norm looks too large, so clipping shrinks the whole gradient before unscale divides it again. The gradient can stay finite and the optimizer can still run, so I would check the norms before clipping and before the update.
3. **What checkpointing buys.** I save each block's input, then rerun the needed part of the block during backward with the same autocast settings and random-number state. In my FP32 matrix, that lowered the peak from 2911 to 1275 MiB, while backward went from 60.1 to 80.0 ms. The loss and final parameters matched bit for bit at the same precision. When choosing a checkpoint boundary, I care about the peak over the whole step and the extra time spent on recomputation.

## 6. Optional results and follow-ups

- **DDP [measured].** Both cases had FP32 `p.grad` and four all-reduces over 27,443,200 elements per step. The default all-reduce used FP32 (104.7 MiB payload), while the BF16 hook used BF16 (52.3 MiB) and wrote the result back to the FP32 bucket. Gradient relative errors versus single-GPU BF16 were 3.670e-3 and 3.943e-3. The hook versus plain DDP error was 1.771e-3. These profiled runs establish dtype and payload. [DDP log](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/ddp.log)
- **Accumulation [measured].** At fixed BF16, MATH, and checkpoint off, `4×8` used 1000 MiB versus 2296 MiB for `1×32`, while taking 123.0–169.4 ms versus 43.1–50.2 ms. Against `1×32` with checkpoint on, `4×8` saved another 148 MiB and still ran slower. Kernel launches and intermediate timing synchronization are candidate explanations **[not isolated]**.
- **Flash [measured].** With checkpoint off, the peak fell from 2296 to 1815 MiB when switching from MATH to Flash. With checkpoint on, the two peaks were 1148 and 1150 MiB, while Flash reduced step time in both repeats. Flash computes score and probability tiles without materializing complete `[B,H,T,T]` matrices in HBM. The retained attention output `O = PV` scales linearly with sequence length. Day 01 profiled a different fused backend, CUTLASS memory-efficient attention. [Day 01 profiler record](../day01/notes.md)
- **Open timing question.** The optimizer phase in the two `bf16_off` repeats took 8.33 and 13.06 ms, and backward also rose from 21.09 to 23.19 ms. Both used the same precision and configuration. The `fp16_off_r1` optimizer phase took 8.73 ms. The timing varies across repeats and the cause is unresolved. The BF16 checkpointing overhead of about 34% uses the run's combined summary. Comparing the two pairs of repeats gives roughly +24% and +45%, showing how sensitive the percentage is to the baseline. [Matrix log](../outputs/day09_downloads/manual_full_20261109T065636Z/session_manual_full_20261109T065636Z/matrix.log)
- **Follow-ups [not tested].** The matrix has no Flash `4×8` row. To test whether timing synchronization magnifies the cost of splitting a batch, measure both `1×32` and `4×8` with phase-boundary synchronization and again with step-boundary synchronization, then compare the two splitting penalties.

## Completion

- [ ] Pre-run predictions for Guide §1.1–§1.5 recorded before execution. The workspace has no record of them, so the explanations above are a post-run review.
- [ ] **Deferred (open practice item, 2026-11-11):** fill `my_amp.py` and pass both `--impl mine` checks. Day 09 was closed without it to keep to the schedule. On the next GPU rental, the two checks take seconds: `fp16_order.py --impl mine` and `probe_amp.py --part ckpt --impl mine`.
- [x] Full reference run finished, downloaded and independently verified against the archive checksum and manifest.
- [x] Record the actual results and walkthrough explanations in §1–§5.
