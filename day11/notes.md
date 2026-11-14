# Day 11: tensor-parallel MLP (column → GELU → row) with explicit autograd communication

The manual reference run `manual_ref_20261113T062618Z` passed on two A100s. My `my_tp.py` is complete: all six functions are written, and the assistant reviewed them against the reference on 2026-11-14. It has not been executed yet. The `--impl mine` GPU run is pending. The earlier runs are documented in [GUIDE Appendix A](GUIDE.md#appendix-a-pre-test-results-pretest1-assistant-run-2026-11-13-pre-test).

Labels, as on Day 10: **[measured]** values from my run, **[pre-test]** values from the assistant's run, **[derived]** arithmetic, **[understanding]** my reasoning, **[not isolated]** a cause the run does not separate.

## 1. Predictions and checks

The reasoning column reconstructs the predictions after the guided walkthrough. The observations come from the manual reference run. [GUIDE Block 1](GUIDE.md#block-1-design-45-min-predict-first-then-open-the-answer)

| # | Question | Prediction and reasoning [understanding] | Observation [measured] |
| --- | --- | --- | --- |
| P1 | Rank 1 ownership and local shapes | In the tiny example, rank 1 owns hidden features 2–3. It takes `W1[2:4, :]` of shape [2, 4], `b1[2:4]` of shape [2], and `W2[:, 2:4]` of shape [4, 2]. In the GPU run it owns features 1024–2047, so the weight shards and their gradients have shapes [1024, 512] and [512, 1024]. | The per-rank table printed those gradient shapes and showed `db1_r` [1024] and `db2` [512] on both ranks. |
| P2 | Activation shapes and layout | `x'` is replicated. `h_r` and `a_r` each contain a different half of the hidden features. `z_r` is a full-shaped partial sum. After summation, `z` is replicated, and adding the same bias gives replicated `y`. | `h_r` and `a_r` were [256, 2, 1024] per rank. `y` was [256, 2, 512] and bitwise equal across ranks. |
| P3 | Adding `b2` before `g` | Add `b2` after the sum. In the tiny example, the first output should be `1 + 2 + 1 = 4`. Adding the bias to each partial gives `2 + 3 = 5`. The incorrect `y` changes the MSE gradient, so output and gradient comparisons should fail even though both ranks agree. | With one block, `y` had `rel = 8.98e-2`. The numeric comparisons failed, and the cross-rank replication check passed. |
| P4 | Forward collectives per block | Only `g` communicates. Its SUM all-reduce acts on [256, 2, 512], a tensor of `256 × 2 × 512 × 4 B = 1 MiB` in FP32. | One SUM all-reduce on [256, 2, 512], with 1,048,576 tensor bytes. |
| P5 | Communication in `g.backward()` | Either partial output changes the sum one-for-one. Each rank already has the full upstream gradient and passes it through. An extra SUM would turn two copies of 2 into 4. | The unit test recorded `bwd []`, and each rank kept its incoming gradient. |
| P6 | Parameter-gradient communication | Each weight or first-bias shard can compute its own gradient from local tensors. The full `db2` comes from the same replicated `dy` on each rank. These parameter gradients need zero TP collectives. | All parameter gradients passed. The only backward collective in the one-block main run was the input-gradient all-reduce. |
| P7 | Communication in `f.backward()` | Each `dx_r` contains the paths through that rank's hidden features. SUM combines them into the full `dx` on both ranks. For the first token's input feature 2, the contributions `−1` and `+2` give `1`. | The unit test recorded one all-reduce. The one-block parity run recorded a SUM on [256, 2, 512]. |
| P8 | Effect of `no_dx_allreduce` | With one block, `dx` is incomplete and differs across ranks. The block's own parameter gradients have already been computed. With two blocks, block 1 passes its partial input gradient back as block 0's upstream gradient, so block 0's parameter gradients also fail. | One block failed `dx` and `replicated`. Two blocks also failed every block 0 parameter gradient. |
| P9 | Scale bugs | Extra SUM in `g.backward()` gives ×2 to `dx`, `dW1`, `db1` and `dW2` for one block. With two blocks, those three parameter gradients in block 1 are ×2, block 0's `db2` is ×2, and block 0's other gradients and model `dx` are ×4. Averaging in `f.backward()` gives `dx` ×0.5 for one block. With two blocks it gives all block 0 parameter gradients ×0.5 and model `dx` ×0.25. | Both controls matched those ratios and failure sets. [Per-parameter results](#5-negative-controls) |
| P10 | Bitwise parity against the single-GPU baseline | FP32 and FP64 can both differ bitwise because sharding changes where the sums round. Expect a smaller error in FP64. A candidate for the largest relative error is `dx`, which receives error from the intervening backward computations and the final sum across ranks. | Every tensor had `bitwise=no`. The largest `rel` was on `y` for FP32 ×1 (8.06e-7), on `dx` for FP32 ×2 (9.38e-7), and on `dx` for FP64 ×1 (3.17e-15). The `dx` guess held for FP32 ×2 and FP64 but **missed FP32 ×1**, where `y` (8.06e-7) beat `dx` (6.61e-7). Errors this size depend on summation order, so the ranking of tensors is not something I can predict reliably. |

## 2. Run record

| Item | Manual reference run [measured] |
| --- | --- |
| Run ID | `manual_ref_20261113T062618Z`, `--impl ref`, TP = 2 |
| Time (UTC) | 2026-11-13 06:27:07–06:27:49 (42 s for the runner [derived]) |
| Host | Vast.ai container 60454650, hostname `7e92845066ae` |
| GPUs and topology | 2 × A100-SXM4-80GB, NV12, both GPUs on NUMA node 0, bidirectional P2P |
| Driver and software | Driver 580.178.04, Python 3.12.15, torch 2.14.1+cu130, CUDA build 13.0, NCCL 2.30.7 |
| Megatron stage | Skipped (`megatron=none` in `commands.txt`) |
| Verdict | Exit 0. Tiny example, all four unit checks, three parity cases and all eight negative-control cases passed their expected checks. |
| Archive | SHA-256 `f85617a01ae5909f5bf10d11ad385a67c542f7f0cc68a737e4e91d618f839668`. All 37 manifest entries verified locally. |

[Download folder and result JSONs][run] · [Archive][archive] · [Environment][env] · [Commands][commands] · [Verification log][verify]

## 3. Ownership drawing (tiny example, my own)

**[understanding]** Two tokens and four hidden features. **R** means replicated, **S0/S1** means sharded along dimension 0/1 of the corresponding full tensor, and **P** means a full-shaped partial sum. The matrices use PyTorch's layout. The tiny example uses ReLU.

```text
                    Rank 0 · hidden features 0–1     Rank 1 · hidden features 2–3
Parameters
W1_r / dW1_r        [2,4] S0, rows 0–1               [2,4] S0, rows 2–3
b1_r / db1_r        [2]   S0, entries 0–1            [2]   S0, entries 2–3
W2_r / dW2_r        [4,2] S1, columns 0–1            [4,2] S1, columns 2–3
b2 / db2            [4]   R                          [4]   R

Forward ↓
x → f → x'          [2,4] R                          [2,4] R
fc1 → h_r           [2,2] S1                         [2,2] S1
ReLU → a_r          [2,2] S1                         [2,2] S1
fc2 → z_r           [2,4] P                          [2,4] P
                        └── g: all-reduce SUM ────────────┘
                  both get z = z_0 + z_1 [2,4] R
z + b2 → y          [2,4] R                          [2,4] R

Backward ↓
dy → g backward     [2,4] R, pass dy to dz_0         [2,4] R, pass dy to dz_1
fc2 → da_r          [2,2] S1                         [2,2] S1
ReLU → dh_r         [2,2] S1                         [2,2] S1
fc1 → dx_r          [2,4] P                          [2,4] P
                        └── f backward: all-reduce SUM ───┘
                  both get dx = dx_0 + dx_1 [2,4] R
```

Column-parallel splits the output features, which are columns of `W1.T` and rows of PyTorch's `W1`. Row-parallel splits the input features, which are rows of `W2.T` and columns of PyTorch's `W2`.

The first token makes the two sums concrete. For output feature 0, `g` adds `1 + 2 = 3`, then each rank adds `b2[0] = 1` to get `4`. For input feature 2, `f.backward()` adds `−1 + 2 = 1`. [Tiny example log](../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/tiny_example.log)

## 4. Parity (TP = 2 vs one GPU)

Pass criterion is `max|tp − 1gpu| / max|1gpu| ≤ 1e-5` in FP32 or `≤ 1e-12` in FP64 for every tensor, bitwise equality of replicated tensors across ranks, and one all-reduce per block in each direction.

| Case | Tensor | Max absolute error | `rel` [measured] | Bitwise vs single GPU | `ratio` |
| --- | --- | ---: | ---: | --- | ---: |
| FP32 ×1 | `y` | 2.74e-6 | 8.06e-7 | no | 1.0000 |
| FP32 ×1 | `dx` | 2.61e-11 | 6.61e-7 | no | 1.0000 |
| FP32 ×1 | `dW1` | 3.49e-10 | 6.87e-7 | no | 1.0000 |
| FP32 ×1 | `db1` | 2.33e-10 | 1.35e-7 | no | 1.0000 |
| FP32 ×1 | `dW2` | 5.82e-10 | 2.98e-7 | no | 1.0000 |
| FP32 ×1 | `db2` | 4.66e-10 | 1.35e-7 | no | 1.0000 |
| FP32 ×2 | `dx` (worst `rel`) | 2.23e-11 | 9.38e-7 | no | 1.0000 |
| FP64 ×1 | `dx` (worst `rel`) | 1.25e-19 | 3.17e-15 | no | 1.0000 |

The single-GPU baseline repeated bitwise and the rank-0 dense replay matched it bitwise in all three cases. The replicated `y`, `dx` and `db2` were bitwise equal across TP ranks in every case. [FP32 ×1][fp32b1] · [FP32 ×2][fp32b2] · [FP64 ×1][fp64b1]

**[understanding]** A dense matmul and a sharded matmul can group their sums differently. Floating-point addition can round differently with each grouping. The much smaller FP64 error is consistent with rounding. **[derived]** The unit roundoff is 2⁻²⁴ ≈ 6.0e-8 in FP32 and 2⁻⁵³ ≈ 1.1e-16 in FP64, a ratio of 2⁻²⁹ ≈ 1.9e-9. Scaling the FP32 ×1 errors (6.6e-7 for `dx`, 8.1e-7 for `y`) by that ratio predicts about 1.3e-15 to 1.5e-15. The measured FP64 values were 3.17e-15 for `dx` and 1.76e-15 for `y`, the same order. If the TP code dropped or added a term, the FP64 error would stay near the size of that term instead of falling by ~10⁹.

## 5. Negative controls

| Control | Blocks | Failed checks and ratios [measured] |
| --- | ---: | --- |
| `no_dx_allreduce` | 1 | `dx` (0.4976), `replicated` |
| `no_dx_allreduce` | 2 | `dx` (0.2480), `replicated` and block 0's `W1` (0.4971), `b1` (0.4872), `W2` (0.5002), `b2` (0.5102) |
| `g_bwd_allreduce` | 1 | `dx`, `W1`, `b1`, `W2` (all 2.0) |
| `g_bwd_allreduce` | 2 | `dx` and block 0's `W1`, `b1`, `W2` (4.0), block 0's `b2` and block 1's `W1`, `b1`, `W2` (2.0) |
| `avg_dx` | 1 | `dx` (0.5) |
| `avg_dx` | 2 | `dx` (0.25), all block 0 parameter gradients (0.5) |
| `bias_before_reduce` | 1 | `y` (ratio 1.0233, `rel` 8.98e-2), `dx` and every parameter gradient |
| `bias_before_reduce` | 2 | `y` (ratio 1.0891, `rel` 1.75e-1), `dx` and every parameter gradient |

All eight cases matched the expected failures. [Single-block controls][fp32b1] · [Two-block controls][fp32b2]

**[understanding]** A block computes its own parameter gradients from the incoming gradient. Its `f.backward()` then combines the contributions to its input gradient. With the reduction missing in block 1, block 0 receives an incomplete upstream gradient and computes wrong parameter gradients. With `avg_dx` in both blocks, the first extra division halves the gradient entering block 0, and the second halves it again before it reaches the model input.

## 6. Collectives

| Case | Forward SUM all-reduces | Backward SUM all-reduces | Tensor shape per call |
| --- | ---: | ---: | --- |
| FP32 ×1 | 1 | 1 | [256, 2, 512] |
| FP32 ×2 | 2 | 2 | [256, 2, 512] |
| FP64 ×1 | 1 | 1 | [256, 2, 512] |

Each call reduced a tensor of 1 MiB in FP32 or 2 MiB in FP64. The size is `256 × 2 × 512 × bytes_per_element` [derived]. Both reductions operate on all token positions at the model width of 512. The intermediate FFN features remain sharded. [Unit checks][unit] · [FP32 ×1][fp32b1] · [FP32 ×2][fp32b2] · [FP64 ×1][fp64b1]

## 7. My code ↔ Megatron Core (`core_v0.19.2`)

The comparison summarizes the guided walkthrough of the pinned Megatron source.

| My code | Megatron Core | Location | What is the same, what differs (in my words) |
| --- | --- | --- | --- |
| `shard_column_parallel` / `shard_row_parallel` | `_initialize_affine_weight_cpu`, `partition_dim` 0 / 1 | `tensor_parallel/layers.py:184`, `:1008`, `:1371` | Both split `W1` along its output dimension and `W2` along its input dimension. |
| Clone before SUM all-reduce | `_reduce` | `tensor_parallel/mappings.py:22` | Our code reduces into a clone. Megatron reduces its contiguous tensor in place and bypasses communication when the TP group has size one. |
| `CopyToTP` (f) | `_CopyToModelParallelRegion` and the `allreduce_dgrad` path | `mappings.py:201`, `layers.py:1066`, `:1139–1147`, `:637–639`, `:749–750` | In this TP = 2 configuration, Megatron starts the `dx` all-reduce inside the linear backward, computes `dW` using the saved input and local upstream gradient, then waits before returning `dx`. |
| `ReduceFromTP` (g) | `_ReduceFromModelParallelRegion` | `mappings.py:221`, `layers.py:1480` | Forward adds partial outputs. Backward passes the complete upstream gradient to each rank. |
| `ColumnParallelLinear.forward` | `ColumnParallelLinear.forward` (`gather_output=False`) | `layers.py:1100` | The first layer returns only its own hidden features. |
| `RowParallelLinear.forward`, bias after `g` | `RowParallelLinear.forward` (`input_is_parallel=True`) | `layers.py:1430`, `:1482` | The second layer consumes its local features directly and adds bias once after the forward reduction. |
| `TPMLPBlock` | `MLP` (`linear_fc1` → activation → `linear_fc2`) | `transformer/mlp.py:155`, `:222`, `:242`, `:258` | Each rank applies the elementwise activation to its own features and passes them straight to the row-parallel layer. |
| Replicated `b2` | `param_is_not_tensor_parallel_duplicate` in the grad norm | `layers.py:95`, `optimizer/clip_grads.py:246` | TP rank 0 counts the replicated gradient in the norm. Every rank applies the same clipping factor to its copy. |

For Day 13: which TP collective I expect to find in a trace inside one pipeline stage, its shape for Day 13's model, and the Megatron line that issues it.

**[understanding]**, a prediction to check on Day 13. Assume TP = 2, sequence parallelism off, and activations of shape `[s, b, h]` (sequence length, micro-batch size, hidden size). The model shape is not fixed yet.

- **Forward:** one all-reduce SUM of shape `[s, b, h]` per row-parallel layer. In each transformer layer those are the attention output projection `linear_proj` and the MLP's `linear_fc2`. Each is `g`, issued by `reduce_from_tensor_model_parallel_region` in `RowParallelLinear.forward` (`layers.py:1480`).
- **Backward:** one all-reduce SUM of shape `[s, b, h]` per column-parallel layer, for `linear_qkv` and `linear_fc1`. Each is `f`, issued asynchronously inside `LinearWithGradAccumulationAndAsyncCommunication.backward` (`layers.py:637–639`, wait at `:749–750`). In the trace it should overlap the weight-gradient GEMM.
- **Count and size [derived]:** 2 forward and 2 backward TP all-reduces per transformer layer, per micro-batch, on every stage. Each moves `s × b × h × 2 B` in BF16. The size depends on `h`, not on the FFN width.
- **How to tell TP from PP in the trace:** the TP all-reduce runs over the TP group (`[0,1]` or `[2,3]`). The pipeline transfer is a send/recv over a PP group (`[0,2]` or `[1,3]`).

## 8. Interpretation (my explanations)

These explanations summarize the guided walkthrough.

1. **Ownership and layout.** In PyTorch's `[out, in]` layout, rank 1 takes the last 1024 rows of `W1` and the matching bias entries, then takes the last 1024 columns of `W2`. GELU works per element, so the features stay on their owning ranks between the layers.
2. **The two all-reduces.** Forward sums the rank-local contributions to the second layer's output before adding `b2`. Backward sums contributions to the first layer's input gradient `dx` after the local gradient has passed through `W1_r`. The gradient of the intermediate activation stays sharded. *Why each runs in only one direction:* the backward of an operation is its transpose (its adjoint). `g`'s forward is a sum, `z = z_0 + z_1`, so ∂z/∂z_r = 1. Each rank's `dz_r` is just the replicated `dz`, and the backward needs no communication (P5). `f`'s forward is a copy: the same `x` goes into every rank's branch. The transpose of a copy is a sum, so the backward must add the per-branch gradients, `dx = dx_0 + dx_1` (P7). Because each operator's backward is the other's forward, `f` and `g` are called *conjugates*. One block therefore issues exactly one all-reduce in each direction.
3. **Gradient scaling.** TP ranks handle different paths through the same example, so their contributions to `dx` add. In equal-batch DDP, ranks average gradients of local mean losses across the data-parallel group. `g_bwd_allreduce` adds copies of the same gradient and doubles the gradients upstream of `g`. With `bias_before_reduce`, each rank contributes another copy of `b2` to the forward sum.

### 8.1 Optional walkthrough with LayerNorm and sequence parallelism

**[understanding]** This is the simplified LayerNorm → MLP path from the walkthrough. The manual run measured ordinary TP. Shapes below are per rank in `[sequence, batch, features]` order. At the sequence-sharded boundaries, rank 0 owns positions 0–127 and rank 1 owns 128–255. Each token still has all 512 model features, so its LayerNorm statistics can be computed locally.

| Forward step | What each rank holds afterward | Shape |
| --- | --- | --- |
| Local LayerNorm | Its 128 normalized token positions, with all model features | `[128, 2, 512]` |
| Sequence all-gather | All normalized token positions | `[256, 2, 512]` |
| Column-parallel `fc1` and GELU | All positions, with 1024 of the 2048 FFN features | `[256, 2, 1024]` |
| Row-parallel `fc2` matmul | Its partial contribution to every position and model feature | `[256, 2, 512]` |
| Sequence reduce-scatter SUM, then add `b2` | Complete MLP outputs for its 128 positions | `[128, 2, 512]` |

The next per-token operation, such as another LayerNorm, can use those sequence shards directly. The next column-parallel projection gathers along the sequence dimension at its own entry.

| Backward step, in execution order | What each rank holds afterward | Shape |
| --- | --- | --- |
| Gradient reaches the MLP exit | The upstream gradients for its own token positions | `[128, 2, 512]` |
| Sequence all-gather, the backward of the exit reduce-scatter | The upstream gradients for all positions, ready for the local `fc2` backward | `[256, 2, 512]` |
| Local `fc2` and GELU backward | The gradient for its own 1024 FFN features at every position | `[256, 2, 1024]` |
| Local `fc1` backward | Its contribution to the normalized input gradient at every position | `[256, 2, 512]` |
| Sequence reduce-scatter SUM, the backward of the entry all-gather | The full normalized input gradient for its own 128 positions | `[128, 2, 512]` |
| Local LayerNorm backward | Input gradients for those 128 positions | `[128, 2, 512]` |

**Follow token 200.** Rank 1 normalizes it, then both ranks receive it at the entry all-gather. If their local `fc2` outputs contribute 1 and 2 to one component, the exit reduce-scatter sends the sum 3 to rank 1. In backward, if rank 1 receives an upstream gradient of 7 for that component, the exit all-gather gives both ranks the same 7. At the entrance, if the local `fc1` backward passes contribute −1 and +2 to one normalized input component, reduce-scatter sums them to 1 and sends it to rank 1 for LayerNorm backward.

For a LayerNorm with trainable scale and bias, each rank also computes parameter-gradient contributions from its own tokens. Summing those contributions over the sequence-parallel group gives the full parameter gradients.

## 9. Not covered or not isolated

- Bitwise differences in the local weight gradients: likely cuBLAS kernel choice for different matmul shapes **[not isolated]**.
- No timing, no overlap measurement, no BF16/FP16, no TP > 2.
- The Megatron layer results are from the assistant pre-test. The manual archive covers the reference implementation at TP = 2.
- My own `my_tp.py` has been reviewed but not executed. No `--impl mine` numbers appear in this file yet.

## Completion

- [x] Record reasoning and observations for P1–P10 after the walkthrough (§1).
- [x] Record the two-rank ownership drawing for the tiny example (§3).
- [x] `my_tp.py` written (all six functions; reviewed against the reference, 2026-11-14).
- [ ] `--impl mine` unit and parity tests pass on the GPUs (FP32 ×1, ×2, FP64). Pending the next rental: `bash experiments/day11_tensor_parallel/run_server_once.sh "mine_<UTC>" --impl mine` (~40 s).
- [x] Reference run with controls downloaded and verified, with results recorded in §2 and §4–§6.
- [x] Review Block 3 Stops 1–5 and record the mapping in §7.
- [x] Record the optional TP + SP walkthrough with LayerNorm (§8.1).
- [ ] Run the optional `--megatron` stage.

[run]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/
[archive]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/day11_manual_ref_20261113T062618Z.tar.gz
[env]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/environment.log
[commands]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/commands.txt
[verify]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/verify_core.log
[unit]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/unit.log
[fp32b1]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/parity_float32_b1.log
[fp32b2]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/parity_float32_b2.log
[fp64b1]: ../outputs/day11_downloads/manual_ref_20261113T062618Z/session_manual_ref_20261113T062618Z/parity_float64_b1.log
