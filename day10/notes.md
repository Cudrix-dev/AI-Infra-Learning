# Day 10: row-wise fused softmax in Triton

The learning walkthrough, handwritten kernel and reference GPU experiment are complete. The downloaded run `ref_20261111T091125Z` used `--impl ref --full`. The handwritten `my_softmax.py` passed `check_softmax.py --impl mine --quick` (159 cases, 0 failed) on an A100-SXM4-80GB on 2026-11-13. Its full check and core benchmark are pending. The GPU instance has been destroyed. Earlier assistant runs are recorded in [Guide Appendix A](GUIDE.md#appendix-a-pre-test-results-pretest2-assistant-run-2026-11-11-pre-test).

**[measured]** denotes values recorded in this run, including compiler-reported metadata and ratios of measured times. **[derived]** denotes calculated values such as bytes and effective GB/s. **[understanding]** denotes reasoning from the implementation. **[not isolated]** marks a cause whose contribution the experiment did not separate.

## 1. Predictions and checks

The predictions below reconstruct the reasoning from the guided review after the reference run. The observations come from that run.

| Guide § | Question | Prediction [retrospective] | Reference observation [measured] | Interpretation |
| --- | --- | --- | --- | --- |
| 1.1 | Warps at N = 16,384 | One warp gives each thread more elements and raises register pressure. Four should help. At 32, the benefit may level off as cross-warp reduction and SM residency matter more. | 1 / 4 / 32 warps took 1158.0 / 201.5 / 201.5 µs per matrix call. The inspector reported spill counts of 916 / 0 / 0. | Four and 32 were effectively tied. The share of the slowdown caused by spill is [not isolated]. |
| 1.2 | Zero-filled masked load | With no padding, the fill value is unused. With padding, zero adds positive exponent mass in ordinary finite cases. Extreme finite inputs can round the added mass away. | The `zero_fill` control passed at N = 1,024 and failed at N = 1,000. | The extra contribution changes the denominator. A zero can also become the maximum when all real entries are negative. |
| 1.3 | Fusion and attention | Five reads and three writes versus one read and one write suggest about 4× the time for `naive` if both run at the same effective transfer rate. The adjacent attention matmuls still exchange S and P through global memory. | `naive` took 815.8 µs, Triton took 198.6 µs, and `copy` took 197.1 µs for FP32 32,768 × 1,024. | The measured time ratio was 4.11×. The traffic ratio is a model. The attention conclusion follows from the kernel boundaries. |
| 1.4 | Host submission, GPU time and L2 | Expect a launch floor at small M and convergence of loop and graph times at large M. A repeatedly used working set within L2 can exceed the HBM peak in effective GB/s. | For N = 1,024, M = 1 / 108 / 1,728 / 32,768 gave 31.1 / 32.2 / 31.5 / 198.5 µs in the loop. The first close loop/graph pair was at M = 6,912. | M = 1,728 used a 13.5 MiB working set and reached 2,546 effective GB/s with graph replay. Cache hit rate was not measured. |
| 1.5 | Graph replay for a single kernel | Replaying a whole chain can amortize host submission. Replaying one Softmax per Python call may cost more than it saves. | At 1 × 128, eager / compile / compile_ro / compile_ro_static took 8.4 / 54.5 / 102.9 / 81.3 µs. | Default compile recorded into one graph with many calls reached 2.1 µs per call. Fixing the input address reduced the graph-tree time by 21.6 µs, consistent with skipping the input copy **[not isolated]**. |

## 2. Run record

| Item | Reference run [measured] |
| --- | --- |
| Run ID | `ref_20261111T091125Z` |
| Time (UTC) | 2026-11-11 09:11:25–09:13:41 |
| Host | Vast.ai container 60186096, hostname `ca5090a08e4d`, 2 × A100-SXM4-40GB connected by NV12 |
| GPU used | GPU 0, NUMA node 0, 108 SMs, 40 MiB L2, max SM/memory clocks 1410/1215 MHz, 400 W power limit |
| Software | Python 3.12.15, PyTorch 2.14.1+cu130, CUDA build 13.0, cuDNN 9.24, Triton 3.8.0, NVIDIA driver 595.71.05 |
| Mode | ref, --full, fresh Triton and Inductor cache directories |
| Verdict | Exit 0. Softmax 381 cases, RMSNorm 369 cases, all three negative controls detected, 192 core and 24 launch benchmark entries |
| Archive verification | SHA-256 `f061834bc5ce40ed70b8980ff7c77d3936cbbe9e3c3636c266bbd70d1dc9bc93`. All 51 manifest entries re-verified locally. |

[environment](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/environment.log) · [check](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/check.log) · [inspect](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/inspect.log) · [bench](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/bench.log) · [bench_launch](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/bench_launch.log) · [check_rmsnorm](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/check_rmsnorm.log)

[Run status](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/status.txt) · [Core CSV](../outputs/day10_downloads/ref_20261111T091125Z/bench_ref_core_ref_20261111T091125Z.csv) · [Launch CSV](../outputs/day10_downloads/ref_20261111T091125Z/bench_ref_launch_ref_20261111T091125Z.csv) · [Kernel dumps](../outputs/day10_downloads/ref_20261111T091125Z/inspect_ref_ref_20261111T091125Z/)

## 3. Correctness

| dtype | Cases | Failed | Max abs. error vs torch | Triton vs fp64 | Torch vs fp64 | Max abs. row-sum error | rtol / atol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 127 | 0 | 1.192e-7 | 7.882e-8 | 9.144e-8 | 1.777e-7 | 1e-5 / 1e-6 |
| fp16 | 127 | 0 | 7.629e-6 | 2.440e-4 | 2.440e-4 | 4.044e-4 | 1e-3 / 1e-5 |
| bf16 | 127 | 0 | 1.526e-5 | 1.950e-3 | 1.950e-3 | 2.930e-3 | 1.6e-2 / 1e-5 |

**[measured]** The reference checked 33 row lengths across the contiguous, single-row, row-strided and N-D layouts. The suite passed all 12 stability cases and preserved the sentinel in all 99 row-strided cases. Unsupported inputs tested by the contract check raised `ValueError`. `no_max` failed on values near 10,000. `zero_fill` and `no_store_mask` passed at N = 1,024 and failed at N = 1,000, where the former changed probabilities and the latter overwrote sentinel values. [Check log](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/check.log)

**[understanding]** The kernel supports CUDA FP32, FP16 and BF16 inputs, with a contiguous last dimension and up to 65,536 columns. A 2-D input can have a padded row stride. Higher-dimensional inputs must be viewable as rows without copying. A fully masked row of real `-inf` values gives NaN, matching the tested `torch.softmax` behavior. [Input contract](../experiments/day10_triton_softmax/README.md#contract-what-the-kernel-supports)

In the handwritten kernel I initially spelled `tl.arange` as `tl.arrange`. Static review caught it and I corrected it. I use the input and output strides separately, and apply the same column mask to both accesses. [Handwritten kernel](../experiments/day10_triton_softmax/my_softmax.py)

**[measured] Handwritten kernel, quick check.** On 2026-11-13, on a different host (A100-SXM4-80GB, torch 2.14.1+cu130, Triton 3.8.0), `check_softmax.py --impl mine --quick` passed 159 cases with 0 failures. That covers 53 per dtype, plus all 12 stability cases, the sentinel in 39/39 row-strided cases, and the contract check. The output was copied from the terminal; no log archive was downloaded.

| dtype | Cases | Failed | Max abs. error vs torch | Mine vs fp64 | Torch vs fp64 | Max abs. row-sum error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 53 | 0 | 5.960e-8 | 9.172e-8 | 9.172e-8 | 1.578e-7 |
| fp16 | 53 | 0 | 1.221e-4 | 2.349e-4 | 2.349e-4 | 4.044e-4 |
| bf16 | 53 | 0 | 1.526e-5 | 1.950e-3 | 1.950e-3 | 2.686e-3 |

The kernel's error against fp64 equals `torch.softmax`'s in every dtype, so computing in FP32 and casting once at the store matches PyTorch's rounding **[derived]**. The quick suite skips some row lengths and layouts, so the full 381-case check is still open.

## 4. Timing and bandwidth

Times come from the benchmark **[measured]**. Effective GB/s **[derived]** uses `2 × M × N × element_size` divided by time, in decimal GB/s. All implementations use the same byte model. The copy baseline uses the same shape and dtype. The L2 flag means the modeled input plus output footprint fits within 40 MiB.

### 4.1 Rows stage (2²⁵ elements per call)

Cells show `loop_us` in µs with effective GB/s in parentheses. Every listed footprint exceeds L2.

| dtype | N | triton | torch | naive | compile | copy |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 128 | 244.8 (1,097) | 199.8 (1,344) | 901.9 (298) | 206.3 (1,301) | 197.3 (1,361) |
| fp32 | 1,024 | 198.6 (1,352) | 203.0 (1,322) | 815.8 (329) | 207.2 (1,296) | 197.1 (1,362) |
| fp32 | 16,384 | 201.6 (1,332) | 216.8 (1,238) | 818.2 (328) | 256.9 (1,045) | 197.1 (1,362) |
| fp32 | 65,536 | 283.0 (949) | 394.0 (681) | 829.6 (324) | 313.4 (857) | 197.1 (1,362) |
| bf16 | 128 | 238.0 (564) | 122.8 (1,093) | 612.5 (219) | 105.6 (1,272) | 100.6 (1,334) |
| bf16 | 1,024 | 101.2 (1,327) | 122.6 (1,095) | 527.3 (255) | 105.6 (1,272) | 100.8 (1,331) |
| bf16 | 8,192 | 102.3 (1,312) | 233.4 (575) | 537.8 (250) | 132.0 (1,016) | 100.7 (1,333) |
| bf16 | 65,536 | 141.7 (947) | 196.7 (682) | 539.5 (249) | 169.0 (794) | 100.7 (1,333) |

At FP32 N = 1,024, the ratio of the measured times is 4.11×. The byte model for naive versus fused is 32 versus 8 bytes per element **[derived]**.

### 4.2 Count stage (FP32)

| N | M | MiB | L2 flag | Triton loop / graph µs | Torch loop / graph µs | Copy loop / graph µs | Triton graph GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,024 | 1 | 0.008 | yes | 31.1 / 1.8 | 8.4 / 5.0 | 6.9 / 1.4 | 5 |
| 1,024 | 16 | 0.125 | yes | 31.0 / 1.9 | 8.6 / 5.2 | 6.9 / 1.5 | 68 |
| 1,024 | 108 | 0.844 | yes | 32.2 / 2.1 | 8.4 / 5.3 | 6.8 / 1.7 | 419 |
| 1,024 | 432 | 3.375 | yes | 32.1 / 2.8 | 8.4 / 5.5 | 6.8 / 2.0 | 1,256 |
| 1,024 | 1,728 | 13.5 | yes | 31.5 / 5.6 | 9.3 / 8.1 | 6.8 / 4.8 | 2,546 |
| 1,024 | 6,912 | 54 | no | 44.8 / 43.1 | 47.3 / 45.9 | 45.0 / 43.5 | 1,314 |
| 1,024 | 32,768 | 256 | no | 198.5 / 196.6 | 202.9 / 201.3 | 197.1 / 195.6 | 1,366 |
| 16,384 | 1 | 0.125 | yes | 33.7 / 5.2 | 9.6 / 8.6 | 6.8 / 1.5 | 25 |
| 16,384 | 16 | 2 | yes | 31.0 / 5.4 | 10.1 / 8.9 | 6.8 / 1.7 | 389 |
| 16,384 | 108 | 13.5 | yes | 32.7 / 7.6 | 11.1 / 9.9 | 6.8 / 4.8 | 1,854 |
| 16,384 | 432 | 54 | no | 42.9 / 41.5 | 55.0 / 53.5 | 42.2 / 40.7 | 1,365 |
| 16,384 | 1,728 | 216 | no | 171.0 / 169.3 | 185.8 / 185.2 | 166.9 / 165.4 | 1,338 |

The first sampled shapes where Triton's loop and graph times are within about 4% are M = 6,912 at N = 1,024 and M = 432 at N = 16,384. The highest Triton graph rate among L2-flagged count entries is 2,546 GB/s at 1,728 × 1,024 **[derived]**.

### 4.3 Padding stage (FP32, M = 16,384)

Cells show `loop_us` in µs with effective GB/s in parentheses.

| N | BLOCK_SIZE | triton | torch | compile | copy |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 1,024 | 99.1 (1,323) | 101.7 (1,289) | 104.3 (1,257) | 98.3 (1,333) |
| 1,024 | 1,024 | 101.2 (1,327) | 104.8 (1,281) | 106.1 (1,265) | 100.7 (1,333) |
| 1,025 | 2,048 | 101.2 (1,328) | 105.0 (1,279) | 147.7 (910) | 100.8 (1,333) |
| 2,048 | 2,048 | 198.9 (1,350) | 204.7 (1,311) | 198.9 (1,350) | 197.2 (1,362) |

### 4.4 Warp sweep (FP32)

Cells show `loop_us` in µs with effective GB/s in parentheses. Spill counts are compiler-reported, in warp order.

| N | 1 warp | 2 warps | 4 warps | 8 warps | 16 warps | 32 warps | Spills |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,024 | 201.7 (1,331) | 199.8 (1,344) | 198.6 (1,352) | 197.1 (1,362) | 202.9 (1,323) | 309.4 (867) | 0 / 0 / 0 / 0 / 0 / 0 |
| 16,384 | 1158.0 (232) | 1176.8 (228) | 201.5 (1,332) | 201.3 (1,333) | 201.9 (1,330) | 201.5 (1,332) | 916 / 616 / 0 / 0 / 0 / 0 |
| 65,536 | — | — | 1286.8 (209) | 1122.0 (239) | 283.0 (949) | 284.1 (945) | — / — / 1,338 / 614 / 78 / 34 |

### 4.5 First call and warmup

- **[measured]** FP32 Triton in the rows sweep took 68.5 ms on the first N = 512 call and 2.83 ms at N = 1,024. Earlier check and inspect stages share the runner's cache. N = 512 has different stride and mask-boundary divisibility properties from the N = 500 version with the same block size **[understanding]**.
- **[measured]** Default `compile` first calls in the rows stage ranged from 0.339 to 2.203 s. For the launch shapes that also appeared in the core run, they were about 40–43 ms.
- **[measured]** First calls for `compile_ro` and `compile_ro_static` were 131–307 ms. Their slowest warmup calls ranged from 1.9 to 34.6 ms across the launch shapes.

### 4.6 Launch stage (FP32, µs per call)

Times are `loop_us`, except the last column.

| M × N | triton | triton_raw | torch | compile | compile_ro | compile_ro_static | Triton graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 × 128 | 31.9 | 13.5 | 8.4 | 54.5 | 102.9 | 81.3 | 2.1 |
| 16 × 1,024 | 32.9 | 13.6 | 8.3 | 51.1 | 102.4 | 81.8 | 2.5 |
| 108 × 1,024 | 39.9 | 13.8 | 8.5 | 52.9 | 102.7 | 83.0 | 2.7 |
| 1 × 16,384 | 31.6 | 13.6 | 12.5 | 75.0 | 105.6 | 83.9 | 6.7 |

## 5. Interpretation

1. **Grid, block and warps.** I use one program per row. `BLOCK_SIZE` rounds the row length up to a power of two, while `num_warps` determines how many threads share that row. At N = 16,384, one warp assigns 512 logical elements per thread and four warps assign 128 **[derived]**. The first configuration spilled and ran about 5.75× slower **[measured]**. Changing the warp count also changes cross-warp reduction and how many blocks can reside on an SM, so the spill contribution is **[not isolated]**.
2. **Stability and masks.** I subtract the row max before taking exponentials. The largest exponent is then zero for a row with a finite maximum, which avoids overflow on the large-offset case. Padding with `-inf` makes its exponent zero, and masking the store protects the row tail. FP16 and BF16 inputs are widened to FP32 after loading, with the final value rounded at the store. The controls show why ordinary random rows and power-of-two lengths alone miss bugs **[understanding]**.
3. **Fusion and attention.** The naive implementation makes five full-tensor reads and three writes, while the ideal fused path makes one read and one write **[derived]**. At FP32 32,768 × 1,024, measured times of 815.8 and 198.6 µs are consistent with that model. The adjacent attention matmuls still write S and read P across kernel boundaries. Tiled attention can use online max and sum updates to accumulate the final output without materializing full S and P **[understanding]**. In training, autograd still saves P for the backward pass, so a standalone fused Softmax cuts forward traffic but not activation memory. Attention-level fusion avoids storing P by recomputing it in the backward pass **[understanding]**.
4. **Reading the sweep.** At N = 1,024, M = 1 has a 31.1 µs loop and a 1.8 µs graph time **[measured]**, consistent with host submission limiting throughput. At N = 16,384, M = 1 to 16 raises the row count sixteenfold while graph time goes from 5.19 to 5.39 µs **[measured]**. This is consistent with an under-filled GPU. Both sizes carry the L2 flag, so cache and small-workload effects are **[not isolated]**. For FP32 32,768 × 1,024, the 256 MiB footprint **[derived]** and the 198.6 µs Triton time near the 197.1 µs copy **[measured]** support a bandwidth limit. For FP32 N = 65,536, Triton loop and graph times are 283.0 and 280.9 µs while copy takes 197.1 µs in the loop **[measured]**. The GPU kernel is the limit there and the contributions of spill and residency are **[not isolated]**. Short rows show a second kernel-limited case: at FP32 262,144 × 128 (also 256 MiB), Triton took 244.8 µs against 199.8 µs for eager PyTorch and 197.3 µs for copy **[measured]**. One program per 128-element row is the suspected cause **[not isolated]**; see §6.
5. **CUDA Graphs.** Recording many calls into one graph amortizes the CPU submission cost. The graph tree replays one Softmax per Python call here and can copy a dynamic input into its captured buffer. At 1 × 128, fixing the input address reduces `compile_ro` time by 21.6 µs **[measured]**, consistent with removing the copy path. The remaining overhead was **[not isolated]**. Day 03's optimizer graph had many kernel submissions to amortize.

## 6. Open measurements

| Observation | What remains [not isolated] | Controlled follow-up |
| --- | --- | --- |
| At N = 128, Triton trails eager PyTorch | Row packing, per-block overhead and cross-warp reductions change together | Compare fixed-shape variants with one row per warp and multiple rows per block. |
| At N = 16,384, 1 warp spills and is much slower than 4 | Local-memory traffic, reduction cost and residency contributions | Measure local-memory traffic, achieved occupancy and stall counters at fixed M and N. |
| At N = 65,536, Triton and eager torch trail copy | Triton spill and residency, plus the eager kernel's actual DRAM traffic | Profile each kernel's local-memory and DRAM traffic. |
| At N = 1,025, `compile` drops to 910 GB/s **[derived]** | The specific Inductor kernel and its bottleneck at this shape | Dump the generated code for 16,384 × 1,025 and compare with N = 1,024. |

## Completion

- [x] Complete the guided design and source walkthrough, including the optional graph and online Softmax discussions.
- [x] Write the Softmax kernel and correct the `tl.arange` spelling after static review.
- [x] Run, download and verify `ref --full`. Document correctness, performance and interpretation.
- [x] Record the Block 1 predictions as a retrospective.
- [x] Run the handwritten kernel's quick GPU correctness check: `--impl mine --quick` passed 159 cases on 2026-11-13.
- [ ] Run the full `--impl mine` correctness check and core benchmark.
- [ ] Optional. Implement and test the handwritten RMSNorm kernel.
