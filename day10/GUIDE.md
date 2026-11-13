# Day 10: a row-wise fused softmax in Triton (single-A100 guide)

> **Run status.** The full reference run `ref_20261111T091125Z` passed on 2026-11-11. The archive and all 51 manifest entries were verified locally. The guided walkthrough and handwritten Softmax are complete. GPU correctness and the core benchmark for the handwritten `--impl mine` remain pending. [Measured results](notes.md#2-run-record) · [Earlier assistant pre-test](#appendix-a-pre-test-results-pretest2-assistant-run-2026-11-11-pre-test)

## Required to close Day 10 (about 4 h including the run)

The verified reference run supplies the GPU evidence for the four learning outcomes below.

| # | You can explain | Evidence from the reference run | Where in this guide |
| --- | --- | --- | --- |
| 1 | **How the grid maps programs to rows**, and what BLOCK_SIZE and num_warps change inside one program | `inspect` log Part 1/1b (elements per thread, registers, spills); `bench` warps stage | §1.1, Block 2 Stage 1, Block 3 Stops 1 and 3 |
| 2 | **How the kernel stays stable and correct for any N**: max subtraction, the masked-load fill value, the masked store | `check` log: stability cases, non-power-of-two rows, sentinel padding, the three negative controls | §1.2, Block 2 Stages 1–2 |
| 3 | **What fusion removes**, in bytes, and why this kernel alone does not remove attention's score matrix | `bench` rows stage: `triton` vs `naive` vs `copy` | §1.3 |
| 4 | **How to read the sweep**: where a row is launch-bound, under-filled, bandwidth-bound or limited by the kernel itself, and which of those claims the measurements isolate | `bench` count, rows, padding and warps stages (`loop_us` vs `graph_us`, GB/s vs `copy`, the `L2` flag) | §1.4, Block 4 |

**Completion record**

- [x] Record the predictions as a retrospective from the guided review. [Predictions](notes.md#1-predictions-and-checks)
- [x] Implement `my_softmax.py` and review its addressing, masks and FP32 arithmetic.
- [x] Run, download and verify `--impl ref --full` and document its correctness and performance.
- [ ] Verify the handwritten kernel on GPU with `check_softmax.py --impl mine` and a downloaded, verified `--impl mine --core` run.

**Optional (not needed to close the day):** the reference runner's `--full` mode (the `launch` stage with `torch.compile(mode="reduce-overhead")`, §1.5); the RMSNorm stretch (Block 2 Stage 5); Block 3 Stops 5–6; the self-check at the end.

**Scope on purpose.** One kernel, forward only, one GPU, one program per row, the whole row in registers. No backward kernel, no online (looped) softmax, no multi-row programs, no autotuning, no Nsight Compute. The ones worth doing next are listed as follow-ups in §1.4 and in the notes template.

---

## What is new today (and what you already have)

You have used softmax many times without writing it. Day 01 found that MATH attention's `[B, H, T, T]` probability matrix was the largest saved activation, and that fused attention saved about 240 MiB by not keeping it. Day 03 showed that a CUDA Graph removes host submission gaps without making a single kernel faster. Day 09 showed that autocast leaves MATH attention's softmax in fp32, and that Flash attention computes score and probability tiles without writing the full matrix to HBM.

New today:

- **Writing a GPU kernel.** Triton's programming model: a grid of *program instances*, each operating on a block of elements with vector operations, compile-time constants, masks, and pointer arithmetic.
- **Reductions inside a kernel**, and the numerics of the stable softmax.
- **Specialization and code generation.** What Triton compiles per shape, alignment, and dtype; what the PTX looks like; what `torch.compile` generates for the same operation.
- **Memory traffic as the performance model.** The byte count for effective bandwidth, comparison against a copy of the same bytes, and the L2 trap.
- **A benchmark that separates host from device time.** The same calls timed one by one, back to back, and replayed from a CUDA Graph. Day 03's tool is used here as a measuring instrument.

---

## Read this first: same host type, one GPU

The host type is the same as on Days 02–09: 2 × A100-SXM4-40GB, torch 2.14.1+cu130. **Triton 3.8.0 ships with that torch**, so nothing new needs to be installed. Today uses **one GPU** (`CUDA_VISIBLE_DEVICES=0`); the second one stays idle. The Day 10 code is self-contained and imports nothing from earlier days.

The project directory is empty on a fresh rental. From your Windows project root:

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/AI-Infra-Learning/experiments'
scp -P <port> README.md requirements.txt root@<host>:/workspace/AI-Infra-Learning/
scp -P <port> -r day10 root@<host>:/workspace/AI-Infra-Learning/
scp -P <port> -r experiments/day10_triton_softmax root@<host>:/workspace/AI-Infra-Learning/experiments/
```

Then, on the instance:

```bash
cd /workspace/AI-Infra-Learning
ln -sfn /venv/main .venv        # fresh rental only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
nvidia-smi topo -m              # record it with the run
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
```

Expected: `2.14.1+cu130 3.8.0 NVIDIA A100-SXM4-40GB`.

---

## Files

All in `experiments/day10_triton_softmax/`:

| File | What it is | Read it when |
| --- | --- | --- |
| `softmax_ref.py` | **Reference.** `softmax_kernel`, `launch_config`, `prepare_rows` (the contract), `softmax` (the wrapper), RMSNorm reference | Contract and wrapper in Stage 0; the kernel only after yours passes |
| `my_softmax.py` | **Yours.** The same wrapper; the kernel body is **YOU WRITE** | Stage 1 |
| `softmax_controls.py` | Three broken kernels the checks must catch | Stage 3 |
| `check_softmax.py` | Correctness sweep, stability, masks, contract, controls | Stage 2 (read the docstring first) |
| `inspect_kernels.py` | Registers, spills, occupancy estimate, PTX counts, Inductor's code | Stage 4, Block 3 |
| `bench_softmax.py` | The benchmark (read its docstring before Block 4) | §1.4, Block 4 |
| `find_symbols.py` | `file:line` of Block 3's reading list in your install | Block 3 |
| `run_server_once.sh`, `download_results.ps1` | Everything once, logged and archived; download and verify on Windows | Block 4 |

---

## Block 1: Design (40 min). Predict first, then check

The predictions below reconstruct the guided reasoning after the run. The **Check [measured]** paragraphs use `ref_20261111T091125Z`.

Shared numbers: A100-SXM4-40GB has **108 SMs**, **40 MiB of L2**, a data-sheet HBM bandwidth of **1555 GB/s**, and per SM at most 64 resident warps, 32 resident blocks and 65,536 32-bit registers. One fp32 element is 4 bytes.

### 1.1 Programs, rows, blocks and warps

**Mechanism.** A Triton kernel is written for **one program instance**. The launch `softmax_kernel[grid](...)` with `grid = (n_rows,)` starts `n_rows` independent instances; inside, `tl.program_id(0)` tells an instance which row it owns. So the mapping is **one program ↔ one row**, and nothing in the kernel communicates between rows.

Inside a program, `cols = tl.arange(0, BLOCK_SIZE)` is a *vector* of BLOCK_SIZE indices, and every operation on it (`load`, `-`, `exp`, `max`) is a vector operation. The program runs as a CUDA thread block of `32 × num_warps` threads. The compiler spreads the BLOCK_SIZE lanes across those threads: **each thread owns BLOCK_SIZE / (32 · num_warps) elements**, held in registers. A reduction such as `tl.max(x, axis=0)` combines values first inside a thread, then across the 32 lanes of a warp with shuffles, then across warps through shared memory with a barrier.

`BLOCK_SIZE` is a `tl.constexpr`: a compile-time constant, so a power of two ≥ N is needed (`triton.next_power_of_2(N)`) and each distinct value is a separately compiled kernel. `launch_config` uses 4 warps up to BLOCK 1024, 8 at 2048, and 16 from 4096 up.

Three consequences you will see in the logs:

- **Work per program grows with N.** At N = 65,536 with 16 warps a thread is assigned 128 fp32 elements. The compiled kernel also needs registers for addresses and intermediate values. It reports spills to local memory, which is backed by the cache hierarchy and DRAM.
- **How many programs fit on an SM at once** (occupancy) is limited by warps, registers and shared memory. `inspect_kernels.py` estimates it from the compiler's metadata. If the grid has fewer rows than SMs × programs per SM, part of the GPU has nothing to run.
- **num_warps trades per-thread work for per-program overhead.** More warps means fewer elements per thread (fewer registers) but a more expensive cross-warp reduction and larger blocks, so fewer of them fit on an SM.

**Question.** For N = 16,384 in fp32, predict what happens to the time per row at num_warps = 1, 4 and 32. At which end does register pressure dominate, and at which end does the reduction or the occupancy limit start to cost?

**Prediction [retrospective].** One warp gives each thread more elements and raises register pressure. Four should help. At 32, the benefit may level off as cross-warp reduction and SM residency matter more.

**Check [measured].** At FP32 N = 16,384, 1 / 4 / 32 warps took 1,158.0 / 201.5 / 201.5 µs per matrix call and reported 916 / 0 / 0 spills. At N = 1,024, 1–16 warps gave 1,323–1,362 effective GB/s and 32 warps gave 867 GB/s **[derived]**. Changing warp count also changes work per thread, reduction and residency. Their separate contributions are **[not isolated]**. [Benchmark](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/bench.log) · [Compiler report](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/inspect.log)

### 1.2 Stable softmax inside one program, and masks

**Mechanism.** For one row x of length N:

```
m = max_j x_j                         (reduction 1)
e_j = exp(x_j − m)                    every exponent ≤ 0, so e_j ∈ (0, 1] and the largest is exactly 1
s = Σ_j e_j                           (reduction 2)   s ≥ 1, so the division is safe
y_j = e_j / s
```

Subtracting m does not change the result, because the factor exp(−m) cancels between numerator and denominator. It changes what the hardware computes: fp32 `exp` overflows above about 88.7, and a row of attention logits near 1e4 would otherwise become `inf / inf = NaN`.

**Masks.** BLOCK_SIZE ≥ N, so lanes N … BLOCK_SIZE−1 do not exist. `tl.load(ptr, mask=cols < N, other=v)` reads no memory for masked lanes and gives them the value v. Choose v so the padding lanes change neither reduction: v = −inf is the identity of `max`, and exp(−inf − m) = 0 adds nothing to the sum. The store must be masked too, or the padding lanes are written past the end of the row, into the next row or past the allocation.

**Precision.** fp16/bf16 rows are converted to fp32 right after the load, reduced and divided in fp32, and rounded once on the store. That is also what ATen's CUDA softmax does internally, so both differ from a float64 softmax by about the output dtype's rounding. The tolerances in `day10_common.TOLERANCES` allow about one ulp (fp16) or two (bf16) between the two implementations.

Two edge cases follow from the formula, and torch has the same behaviour. An entry of −inf gives exactly 0, as in a causal mask. A row that is entirely −inf gives m = −inf and −inf − (−inf) = NaN, so the whole row is NaN.

**Question.** If you load masked lanes with `other=0.0` instead of −inf, for which row lengths and which input values does the result stay correct? Which check case should fail, and why does a row length of 1,024 not catch the bug?

**Prediction [retrospective].** With no padding, the fill value is unused. With padding, zero adds positive exponent mass in ordinary finite cases. Extreme finite inputs can round the added mass away.

**Check [measured].** `CONTROLS: ALL DETECTED`. `zero_fill` and `no_store_mask` passed at N = 1,024 and failed at N = 1,000. The former changed valid probabilities and the latter overwrote sentinel padding. All 99 reference `row_strided` cases preserved the sentinel. `no_max` failed on large offsets, where naive FP32 Softmax produced 100% non-finite results. [Correctness log](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/check.log)

### 1.3 What fusion changes: intermediates and bytes

**Mechanism.** Written as separate PyTorch ops (`naive_softmax` in the benchmark):

| Kernel | Reads | Writes |
| --- | --- | --- |
| `amax` | x | m (one value per row) |
| `x − m` | x | z |
| `exp` | z | e |
| `sum` | e | s (one value per row) |
| `e / s` | e | y |
| **total, per element** | **5** | **3** |

That is 8 full passes over M × N elements with two full-size intermediates (`z` and `e`) plus the output `y`. The fused kernel loads each element once, keeps the row in registers through both reductions, and stores once: **2 passes, no intermediates**. Both versions do the same arithmetic; the difference is memory traffic, about 4 × here. Softmax does a handful of FLOPs per byte, so it is memory-bound, and in the bandwidth-bound regime its time should scale with the bytes moved.

Note that eager `torch.softmax` is **not** the naive version. ATen already ships fused CUDA softmax kernels. The fair comparison for your kernel is `torch`, and `naive` shows what fusion buys.

**Connection to attention, and the limits of this kernel.** In MATH attention (Day 01, Day 09) the chain is `S = QKᵀ/√d → mask → P = softmax(S) → O = PV`, and S and P are both `[B, H, T, T]` tensors in HBM. A standalone fused softmax makes the *softmax step* cost one read of S and one write of P instead of eight passes, but **S is still written by the matmul and P is still read by the next matmul**. The `[B, H, T, T]` tensors are still materialized, and autograd would still save P. Removing them takes fusing across the matmuls. That is what Flash attention does: it computes S in tiles, keeps a running max and sum (the *online softmax*), rescales the partial output, and never writes S or P to HBM. That needs a looped softmax, which today's whole-row kernel is not. Inductor's long-row kernel (Block 3 Stop 5) is a small version of it.

**Question.** For the fp32 shape M × N = 2²⁵ (128 MiB in, 128 MiB out), predict the time of `naive` relative to the fused kernel. Then predict the ratio for the same softmax inside MATH attention at Day 01's medium shape: would fusing only the softmax step remove the 32 MiB `[B, H, T, T]` saved activation per layer that Day 01 measured?

**Prediction [retrospective].** Five reads and three writes versus one read and one write suggest about 4× the time for `naive` if both run at the same effective transfer rate. The adjacent attention matmuls still exchange S and P through global memory.

**Check [measured].** At FP32 32,768 × 1,024, `naive` took 815.8 µs, Triton 198.6 µs and copy 197.1 µs. The measured time ratio is 4.11×. The 8-versus-2 pass count is **[derived]** from the implementation and models full-tensor traffic. [Rows benchmark](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/bench.log)

### 1.4 Reading a performance sweep

**Mechanism: what each timing contains.**

| Column | How it is measured | What it contains |
| --- | --- | --- |
| `first_ms` | first call, synchronized | compilation if this specialization is new (Triton JIT; Dynamo + Inductor for `compile`) + allocation + the call |
| `warm_ms` | slowest of the next 10 synchronized calls | leftover one-time work (for `compile_ro`: CUDA Graph recording) |
| `sync_us` | one call + `synchronize`, median of 30 | host launch path + GPU time + synchronization latency |
| `loop_us` | `iters` calls between two CUDA events, ÷ iters, median of 5 | **max(host submission per call, GPU time per call)**: the CPU queues ahead while it is faster than the GPU |
| `graph_us` | the same `iters` calls captured in one CUDA Graph, replayed | ≈ **GPU time per call**: host submission is gone (Day 03) |

So `loop_us ≫ graph_us` means **launch-bound**: the GPU waits for the CPU. `loop_us ≈ graph_us` means the GPU is the bottleneck, and the question becomes *which* part of the GPU.

**Effective bandwidth.** `bytes = 2 · M · N · element_size` (one read and one write per element, the minimum for any softmax), and `GB/s = bytes / time`. This is a property of the *workload*, not a measured DRAM byte count, and it is reported for every implementation with the same definition. The **measured hardware reference** is the `copy` row: `out.copy_(x)` moves exactly those bytes with no math, giving an achieved transfer rate of 1,362 GB/s **[derived]** at FP32 32,768 × 1,024 in the reference run. A softmax near copy speed at the same shape is consistent with a bandwidth limit. Rows flagged **L2** have `bytes ≤ 40 MiB`: back-to-back calls can then hit in L2, and effective GB/s can exceed DRAM bandwidth. Those rows are not DRAM measurements.

**Four regimes, and what each sweep isolates:**

| Regime | Signature in the log | Stage that shows it |
| --- | --- | --- |
| Launch-bound | `loop_us` flat as M grows, `graph_us` much smaller | `count` (small M), `launch` |
| Under-filled GPU | Few programs relative to SM count, `graph_us` nearly flat as M grows | `count` with long rows and few rows |
| Bandwidth-bound | GB/s ≈ `copy` | `rows` (mid N), `padding` |
| Kernel-limited | GB/s well below `copy` at the same bytes, `loop_us ≈ graph_us` | `rows` at N = 128 and N = 65,536, `warps` |

**How the number of rows changes the result.** With fixed N, M decides three things at once: the bytes (time per byte only becomes meaningful once the kernel is longer than the launch), the number of programs (fewer rows than SMs leaves SMs idle), and whether the data fits in L2. The `count` stage moves M from 1 to 32,768 so you can watch all three happen.

**Question.** In the `count` stage at N = 1,024, predict `loop_us` for the Triton kernel at M = 1, 108, 1,728 and 32,768. At which M does the loop stop being launch-bound? At which M do you expect GB/s *above* the copy's DRAM figure, and why must that row not be reported as bandwidth?

**Prediction [retrospective].** Expect a launch floor at small M and convergence of loop and graph times at large M. A repeatedly used working set within L2 can exceed the HBM peak in effective GB/s.

**Check [measured].** At N = 1,024, M = 1 / 108 / 1,728 / 32,768 gave 31.1 / 32.2 / 31.5 / 198.5 µs in the Triton loop. At M = 6,912, loop and graph first converged among the sampled shapes at 44.8 / 43.1 µs. M = 1,728 has a 13.5 MiB footprint and reaches 2,546 effective GB/s with graph replay **[derived]**. At 54 MiB the whole footprint exceeds L2, and individual accesses can still hit. [Count benchmark](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/bench.log)

**Labels to use when you explain the sweep.** Write **[measured]** for a timing or a ratio of two timings. Write **[derived]** for arithmetic such as GB/s or "8 passes vs 2". Write **[not isolated]** for any cause the sweep cannot separate. Examples are "the N = 65,536 slowdown is caused by spills" or "the N = 128 slowdown is caused by per-program overhead". The warps stage and Part 1b *correlate* spills with time, which is evidence, not isolation. Isolating it would take something like Nsight Compute's `lmem`/`dram__bytes` counters, or a multi-row-per-program variant for short rows.

### 1.5 Optional: `torch.compile(mode="reduce-overhead")` where launches dominate

**Mechanism.** `reduce-overhead` makes Inductor replay its kernels through **CUDA Graph trees**. That is Day 03's capture and replay, managed for you: static memory pools, recording on an early call, then replay. Day 03's lesson was that replay saves *host submission per kernel*: 1,953 launches became one. A softmax is **one** kernel, so the most a graph can save is one launch per call, and the graph tree adds its own per-call work. Before each replay it checks the inputs and copies any input that is not in a static buffer into one. Day 05's notes deferred this exact case to today.

**Question.** For a 1 × 128 softmax, predict the per-call loop time of `compile` and `compile_ro` relative to eager torch. How much host submission can a graph save when the function launches one kernel?

**Prediction [retrospective].** Replaying a whole chain can amortize host submission. Replaying one Softmax per Python call may cost more than it saves.

**Check [measured].** At the sampled 1 × 128 shape, eager / default compile / compile_ro / compile_ro_static took 8.4 / 54.5 / 102.9 / 81.3 µs per loop call. Fixing the input address reduced time by 21.6 µs, consistent with skipping the input copy **[not isolated]**. Default compile recorded into one graph containing many calls reached 2.1 µs per call. The remaining per-call costs are **[not isolated]**. [Launch benchmark](../outputs/day10_downloads/ref_20261111T091125Z/session_ref_20261111T091125Z/bench_launch.log)

### 1.6 New rules (on top of Days 01–09)

1. **Name the byte count behind every GB/s**, and put a measured reference (a copy of the same bytes) next to it. A data-sheet peak is not a measurement.
2. **Check that the working set exceeds L2** before calling a number "bandwidth", or flag it.
3. **Report host and device time separately for small inputs.** `loop_us` vs `graph_us` is the cheapest way to do that.
4. **Every kernel check needs a case designed to fail.** Non-power-of-two lengths for masks, sentinel padding for stores, large offsets for stability. Random inputs of size 1,024 pass most bugs.
5. **Compilation is a separate number.** Report it from cold caches and keep it out of steady-state timing.

---

## Block 2: Code (2 h)

### Stage 0 (15 min): read the contract and the wrapper

In `softmax_ref.py`, read the module docstring, `prepare_rows`, `launch_config` and `softmax`. **Do not read `softmax_kernel` yet.** Look for:

- **The shape assumptions the kernel relies on.** The last dimension is contiguous, so the column offset is simply `+ cols`. A 2-D input may have any row stride, which the kernel receives as `in_row_stride` **in elements**. N-D inputs are flattened with `view`, never with a copy.
- **What is rejected, and why.** float64, CPU tensors, `N > 65536`, and a non-contiguous last dimension all raise `ValueError` before any launch. The check script tests that.
- **How the grid is built.** `grid = (n_rows,)`, and BLOCK_SIZE and num_warps come from `launch_config`.

### Stage 1 (45 min, required): **YOU WRITE** `my_softmax_kernel` in `my_softmax.py`

About 12 lines; the comments in the stub list the six steps. The loop:

```bash
python experiments/day10_triton_softmax/check_softmax.py --impl mine --quick   # ~10 s, 13 row lengths
python experiments/day10_triton_softmax/check_softmax.py --impl mine           # ~20 s, must print SOFTMAX CHECK: PASSED
```

Until you delete the `tl.static_assert`, every script prints "not implemented yet" and exits with code 3.

Three details are easy to get wrong:

- **int64 row offsets.** `row * in_row_stride` is computed in int32 unless you convert first. It overflows above 2³¹ elements (8 GiB of fp32). No check case is that large, so this one is on you.
- **The dtype on store.** `y.to(out_ptr.dtype.element_ty)` rounds once, explicitly. `tl.store` would convert an fp32 value to the pointer's element type anyway (`semantic.py` casts `val` to the element type), so leaving the cast out is not an error. Writing it keeps the one rounding step visible in your code. The step that is not optional is the opposite one, `.to(tl.float32)` after the load (Block 3 Stop 2).
- **`axis=0`.** `x` is a 1-D block, so both reductions are over axis 0 and return a scalar that broadcasts back over the block.

### Stage 2 (15 min): read the correctness report

Open `check_softmax.py` and match each case to the bug it catches. The Stage 3 table answers the controls part.

- `row_strided`: the input is a column slice of a wider tensor (row stride N + 13), and the output is a view into a buffer pre-filled with 123.0. The 13 columns after each row must still read 123.0 afterwards.
- `single_row`: a grid of one program.
- `nd_rows`: a `[2, 3, 7, N]` tensor whose rows are the flattened leading dimensions, like attention scores `[B, H, T, T]`.
- Stability: `large_offset`, `large_spread`, `causal_mask` (masked entries must be exactly 0), and `all_neg_inf_row` (must be NaN, as in torch).
- The float64 columns in the summary: the error of **both** implementations against float64. Your kernel should not be noticeably worse than torch's.

### Stage 3 (15 min): the negative controls

Read `softmax_controls.py` (30 lines). Each control is the reference with one line changed:

| Control | Change | Passes | Caught by |
| --- | --- | --- | --- |
| `no_max` | no `x − max(x)` | random rows | rows near 1e4: values overflow |
| `zero_fill` | masked loads use `other=0.0` | N = 1,024 (no padding lanes) | N = 1,000: padding lanes add exp(0 − m) to the sum |
| `no_store_mask` | unmasked store | N = 1,024 | N = 1,000: zeros written over the sentinel padding |

```bash
python experiments/day10_triton_softmax/check_softmax.py --impl ref --controls   # ends with CONTROLS: ALL DETECTED
```

### Stage 4 (20 min): look inside your compiled kernel

```bash
python experiments/day10_triton_softmax/inspect_kernels.py --impl mine --run-id try1
```

Compare Part 1 with the reference. The same configuration should give the same registers and spills, unless your kernel keeps extra live values. Open `outputs/day10_softmax/inspect_mine_try1/softmax_N1024_fp32.ptx` and find these lines from your kernel:

- the predicated `@%p ld.global.v4.b32`, which is your masked load;
- `ex2.approx.f32`, which is `tl.exp` lowered to a base-2 exponential after multiplying by log₂e;
- `shfl.sync.bfly`, the warp-level part of each reduction;
- `bar.sync`, the barrier of the cross-warp part;
- `div.full.f32`, a fast approximate division (the reason the fp32 tolerance is not zero).

### Stage 5 (optional, stretch, 30 min): RMSNorm

`y = x / sqrt(mean(x²) + eps) · w`. Write `rmsnorm` in `my_softmax.py` (kernel + wrapper; copy the softmax wrapper's structure). Then run:

```bash
python experiments/day10_triton_softmax/check_softmax.py --impl mine --op rmsnorm   # RMSNORM CHECK: PASSED
```

Three things differ from softmax. The neutral padding value for a *sum* of squares is 0, not −inf. The mean divides by `n_cols`, not BLOCK_SIZE. And every row also reads `w`, which all rows share, so repeated weight reads can hit in L2. The reference is `rmsnorm_kernel` in `softmax_ref.py`. It passed 369 cases in the reference run.

### Pitfalls (symptom → cause)

| Symptom | Cause |
| --- | --- |
| Wrong only when N is not a power of two | masked load with the wrong `other`, or a missing mask on the load |
| Values correct, sentinel padding overwritten | store without the mask |
| NaN/inf on large inputs, fine on `randn` | no max subtraction |
| fp16/bf16 results off by more than the tolerance, or `max\|y−fp64\|` much larger than torch's | reductions done in the input dtype: `tl.sum` does not widen floats (no `.to(tl.float32)` after the load) |
| `CompilationError` with `arange's range must be a power of 2` | BLOCK_SIZE not a power of two (use `launch_config`) |
| Correct for small tensors, garbage or illegal address for huge ones | `row * stride` overflowed int32 |
| Every new N seems to recompile | it should not: `n_cols` is a runtime argument. A new BLOCK_SIZE, dtype, num_warps, or a change in "divisible by 16" of an integer argument or pointer does recompile (Block 3 Stop 3) |
| First benchmark call 0.1–2 s | JIT compilation (cold cache), not kernel time |

---

## Block 3: Read the source (1 h; Stops 1–4 required, 5–6 optional)

Run `python experiments/day10_triton_softmax/find_symbols.py` for line numbers in your install. The reference run's line numbers (Triton 3.8.0, torch 2.14.1) are given below. Each stop ends with something you can see in your own logs or dumps.

### Stop 1 (15 min, required): your kernel as Triton IR. `inspect_<impl>_<RID>/softmax_N1024_fp32.ttgir`

Open the TTGIR, the IR after Triton assigns the block to threads. The `#blocked` layout attribute near the top answers "how are the 1,024 lanes split across 128 threads". For N = 1,024 the reference TTGIR shows `sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4]`: each thread owns groups of 4 contiguous elements, 8 elements in total, in 2 groups. Then find `tt.load` with its mask operand, the two `tt.reduce` ops (max and sum), `math.exp`, and `tt.store`. Now open `softmax_N1000_fp32.ttgir`. It is the same kernel source with the same BLOCK, but its layout is `sizePerThread = [1]`.

*Q: why did the compiler choose groups of 1 for N = 1,000, which gives 8 scalar `ld.global.b32` in the PTX instead of 2 `ld.global.v4.b32`? The row starts are in fact 4,000 bytes apart, a multiple of 16. What does the compiler know about `in_row_stride = 1000` and `n_cols = 1000`, and what does it not know (Stop 3)? Does the padding stage show the scalar loads costing any bandwidth?*

### Stop 2 (10 min, required): the language. `triton/language/core.py`: `program_id` `:1945`, `arange` `:1981`, `load` `:1425`, `store` `:1165`, `reduce` `:1216`; `triton/language/standard.py`: `max` `:177`, `sum` `:287`, `softmax` `:56`

- `load`: read the docstring's rules for `mask` and `other`, and the boundary-check alternative for block pointers.
- `standard.py` `softmax`: Triton ships a one-line softmax. Compare it with yours. Does it subtract the max? Which axis does it reduce? Is there a flag that changes the rounding of the division?
- `max`/`sum`: both are `reduce` with a combine function. In `sum`, read `_pick_sum_dtype` (`standard.py:270`): which input dtypes get a wider accumulator by default, and which do not?

*Q: what would `tl.sum` do with bf16 values if you removed the `.to(tl.float32)`, and which check would show the difference?*

### Stop 3 (15 min, required): what gets compiled, and when. `triton/runtime/jit.py`: `JITFunction` `:628`, `run` `:365` (the launch path), `_do_compile` `:877`; `triton/compiler/compiler.py`: `compile` `:226`, `CompiledKernel` `:407`, `_init_handles` `:448`

- In `run`, follow the path from the Python call to the launch. Arguments are *specialized*: constexprs by value, pointers and integers by properties such as "equal to 1" and "divisible by 16" (look for `specialize_impl` and the `tt.divisibility` attribute). Together with the dtypes and options such as `num_warps`, those form the cache key.
- On a cache miss, `_do_compile` runs the compiler and writes to `TRITON_CACHE_DIR`. On a hit, `run` binds arguments and calls the launcher. That Python path is the 14 µs measured as `triton_raw`.
- `_init_handles` loads the cubin and fills `n_regs` and `n_spills`. `inspect_kernels.py` prints exactly those fields.

*Q: in the reference `rows` stage, the first FP32 Triton call at N = 512 took 68.5 ms, while N = 1,024 took 2.83 ms. The check had already used BLOCK 512 at N = 500. From the specialization rules, explain what was new about N = 512.*

### Stop 4 (15 min, required): what eager PyTorch does. `aten/src/ATen/native/cuda/SoftMax.cu` and `PersistentSoftmax.cuh` on GitHub at tag `v2.14.1`

- For eligible short rows, `host_softmax` dispatches to `softmax_warp_forward`. Each row's reductions stay within one warp, and multiple rows can share a block. At N = 128, one warp can process two independent rows. The reductions use warp shuffles.
- For longer rows, `host_softmax` can select several variants. Trace `cunn_SoftMaxForward`, which assigns one block per row. Count the global reads for max, exponent sum and final writeback. The shared-memory variant can reuse a staged input row.

*Q: at N = 128 in the reference run, eager torch reached about 1,344 GB/s versus 1,097 for Triton in FP32, and 1,093 versus 564 in BF16. The Day 10 kernel launches 262,144 programs of 128 threads. Name two differences in work per block that could explain the gap, and which controlled comparison would separate them. The causes are [not isolated].*

### Stop 5 (optional, 10 min): what `torch.compile` generates. `inspect_<impl>_<RID>/inductor_softmax_*.py`; `torch/_inductor/runtime/triton_heuristics.py`: `persistent_reduction` `:4816`, `reduction` `:4693`

- `[4096, 1000]`: `triton_per_fused__softmax_prepare_softmax_online_0`, a *persistent* reduction. The whole row sits in one block (`R0_BLOCK = 1024`), as in yours, but each program handles `XBLOCK` rows (`xoffset = tl.program_id(0) * XBLOCK`), and the autotuned launcher picks XBLOCK.
- `[256, 32768]`: `triton_red_..._online_0`, a *looped* reduction. Loop 1 keeps a running max and a rescaled running sum over `R0_BLOCK` chunks (the online softmax), and loop 2 reads the row **again** to normalize and store. Find the `eviction_policy='evict_last'` hint on the first read, which is meant to keep the row in L2 for the second.

*Q: at N = 16,384 and 32,768 (fp32), `compile` ran at about 1,050 GB/s against 1,300–1,330 for the reference kernel. Using the generated code, how many bytes per element does the looped kernel request from memory, and how many must actually come from DRAM if the `evict_last` hint works? Which part of that is [not isolated]?*

### Stop 6 (optional, 10 min): CUDA Graph trees. `torch/_inductor/cudagraph_trees.py`: `cudagraphify_impl` `:432`, `CUDAGraphNode` `:920`; `torch/compiler/__init__.py` `cudagraph_mark_step_begin` `:505`

- In `CUDAGraphNode.run` (`:1264`), the first call is `_copy_inputs_and_remove_from_src` (`:1206`): every input that is not static is copied into the buffer the graph was recorded with, before replay. That copy is the extra kernel in the `compile_ro` row, and `compile_ro_static` removes it. (`cudagraphify_impl` at `:432` is where the static input indices are decided.)
- `cudagraph_mark_step_begin` tells the tree that the previous outputs are dead. Without it, a loop that never consumes outputs can trigger re-recording or "overwritten output" errors.

*Q: map the three Day 03 rules (warm up before capture, keep addresses static, update inputs in place) onto what the graph tree does for you, and onto what it costs per call here.*

---

## Block 4: Run + summarize (40–60 min)

### Commands

Your kernel first, then the reference:

```bash
cd /workspace/AI-Infra-Learning && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
D10=experiments/day10_triton_softmax
RID=manual_$(date -u +%Y%m%dT%H%M%SZ)
python $D10/check_softmax.py --impl mine --run-id "$RID"                       # ~20 s, SOFTMAX CHECK: PASSED
tmux new -s d10 "bash $D10/run_server_once.sh ${RID}_mine --impl mine; bash $D10/run_server_once.sh ${RID}_ref --impl ref --full"
```

The runner writes `outputs/day10_softmax/session_<RID>/` with `commands.txt`, `environment.log` (versions, GPU, clocks, `nvidia-smi topo -m`, a project snapshot, the installed Triton/Inductor sources from Block 3), one log per stage, `SHA256SUMS`, `status.txt` and `CORE_SUCCESS.txt`/`SUCCESS.txt`. The checksummed archive is `outputs/day10_exports/day10_<RID>.tar.gz`. `--impl mine` runs your checks (no controls), `inspect` and `bench --stage core` (~2 min in the pre-test). `--impl ref --full` adds the controls, the `launch` stage and the RMSNorm check (~2.5 min). Neither run shuts anything down.

To run the stages by hand instead:

```bash
python $D10/check_softmax.py --impl ref --controls --run-id "$RID"
python $D10/inspect_kernels.py --impl mine --run-id "$RID"
python $D10/bench_softmax.py --impl mine --stage core --run-id "$RID"
python $D10/bench_softmax.py --impl mine --stage launch --run-id "$RID"     # optional
```

Hand runs reuse the Triton/Inductor disk caches, so their `first_ms` values are warm. The runner's are cold.

### Download to Windows

From PowerShell in the project root, once per run ID:

```powershell
experiments/day10_triton_softmax/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>_mine
experiments/day10_triton_softmax/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>_ref
```

The script checks the archive's SHA-256, extracts it to `outputs/day10_downloads/<RID>/` and re-hashes every file in the manifest. The structured results are `bench_*.json`/`.csv`, `check_*.json` and `inspect_*/inspect.json`. **Stop the instance in the Vast.ai console only after it prints `Manifest OK`**, then check that its status shows stopped.

For hand runs, there is no archive. Copy the folder:

```powershell
scp -P <port> -r "root@<host>:/workspace/AI-Infra-Learning/outputs/day10_softmax" outputs/day10_downloads/<RID>_hand
```

### How to read the `check` log

One summary line per dtype: cases, failures, `max|y−torch|`, `max|y−fp64|` for yours and for torch's, and `max|rowsum−1|`. Then the stability table, the sentinel count, the contract line, the controls (ref only) and the verdict. `max|y−fp64|` should be about the same for both implementations. If yours is much larger in fp16/bf16, you are not computing in fp32.

### How to read the `inspect` log

Part 1: for each N, BLOCK, warps, elements per thread, registers, spills, and the occupancy **estimate**. Part 1b: the same for the warps sweep. Part 2: PTX instruction counts for N = 1,000 and N = 1,024. Part 3: Inductor's kernel kind for a short and a long row.

### How to read the `bench` log

Each row is one implementation at one shape. The pivot tables at the end give `loop_us (GB/s)` side by side and `torch/triton` (> 1 means Triton is faster). Read them in this order:

1. **`count`, small M.** Every implementation is flat. Note the floors (Triton, torch, copy, naive, compile) and compare `loop_us` with `graph_us`. → launch overhead.
2. **`count`, growing M.** Find the M where `loop_us ≈ graph_us`, and the rows flagged `L2` with GB/s above copy. → under-filled vs bandwidth-bound, and L2.
3. **`rows`, middle N.** `triton` ≈ `copy` ≈ 1,350 GB/s. `naive` ≈ a quarter of that. → fusion, bandwidth-bound.
4. **`rows`, ends.** N = 128 (Triton below torch) and N = 65,536 (everyone below copy). Read them together with `inspect` Part 1 and the `warps` stage. → kernel-limited, **[not isolated]**.
5. **`padding`.** N = 1,000, 1,024 and 1,025 for Triton: does doubling BLOCK at 1,025 (half the lanes masked) cost bandwidth? Does `compile` behave the same?
6. **`first_ms`.** The runner's stages share fresh cache directories. The FP32 Triton rows sweep shows 68.5 ms at N = 512 and 2.83 ms at N = 1,024. Compare the earlier check and inspection specializations to interpret these first-call times. The default `compile` rows-stage first calls range from 0.339 to 2.203 s. [First-call measurements](notes.md#45-first-call-and-warmup)

### Write `day10/notes.md`

The notes record the retrospective predictions, verified run, correctness and benchmark tables, and conclusions. [Day 10 notes](notes.md)

---

## Core questions from the reference run

The results cover grid and warps, stability and masks, fusion and attention, and the performance sweep. The notes tie each conclusion to a measured result, a calculation, or a cause the experiment has yet to isolate. [Interpretation](notes.md#5-interpretation)

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. Your grid is (M,). Which line of the kernel decides which row a program reads, and what would change if the grid were (M // 2,) with two rows per program?</summary>

`row = tl.program_id(0)` and the pointer arithmetic `in_ptr + row * in_row_stride`. With two rows per program, each program would load a `[2, BLOCK]` block (or loop twice), reduce along axis 1, and the grid would halve. That is the standard fix for very short rows, where per-program overhead dominates. ATen's warp-per-row kernel and Inductor's `XBLOCK` both do this.
</details>

<details><summary>2. Why is −inf the right `other` for the load, and what is the right `other` for RMSNorm?</summary>

−inf is the identity of max, and exp(−inf − m) = 0 is the identity of the sum. For RMSNorm the reduction is a sum of squares, so 0.
</details>

<details><summary>3. A colleague reports their softmax at 2,500 GB/s on an A100-40GB. What do you ask first?</summary>

The byte count used and the working-set size. 2,500 GB/s exceeds the 1555 GB/s HBM peak, so the data was most likely L2-resident across back-to-back calls (bytes ≤ 40 MiB), or the byte count is wrong.
</details>

<details><summary>4. `loop_us` is 31 µs and `graph_us` is 2 µs. What is the GPU doing for the other 29 µs, and what would help?</summary>

Waiting: the CPU's launch path is slower than the kernel. Helpful: fewer launches per unit of work (bigger batches, several rows per call, fusing neighbouring ops), a cheaper launch path (thinner wrapper, a C++ launcher), or one graph replay that covers many kernels.
</details>

<details><summary>5. Why did `compile_ro` make a one-kernel softmax slower?</summary>

A graph replay saves the launch of the kernels it contains, here one. The graph tree's per-call work is larger: input checks, a copy of each non-static input into its static buffer, and bookkeeping. The static-input variant was 21.6 µs faster at 1 × 128 in the reference run.
</details>

<details><summary>6. Does a fused softmax kernel let MATH attention avoid saving [B, H, T, T] for backward?</summary>

No. P = softmax(S) is still produced in HBM and still needed by the PV matmul and by the backward pass. Only fusing the whole attention computation (Flash or memory-efficient attention) avoids materializing S and P. Their backward recomputes tiles from Q, K, V and the saved per-row statistics.
</details>

**Next (Day 11):** tensor-parallel MLP with custom autograd Functions.

---

## Appendix A: pre-test results (`pretest2`, assistant run, 2026-11-11) [pre-test]

The assistant ran the reference with `run_server_once.sh pretest2 --impl ref --full` on 2026-11-11 from 08:51:58 to 08:54:12 UTC. These are **pre-test** numbers, for comparison only: they are not your measurements.

**Host.** Vast.ai container 60186096 (hostname `ca5090a08e4d`), 2 × A100-SXM4-40GB with NV12, **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`, NUMA node 0; GPU 1 idle on NUMA 1). 108 SMs, 40 MiB L2, maximum clocks 1410 MHz SM and 1215 MHz memory, 400 W power limit, driver 595.71.05. Python 3.12.15, torch 2.14.1+cu130 (CUDA 13.0 build, cuDNN 9.24), **Triton 3.8.0**. Cold Triton and Inductor caches. Exit code 0, `OUTPUT CHECKS PASSED`.

Logs: [`outputs/day10_downloads/pretest2/session_pretest2/`](../outputs/day10_downloads/pretest2/session_pretest2/) ([environment](../outputs/day10_downloads/pretest2/session_pretest2/environment.log), [check](../outputs/day10_downloads/pretest2/session_pretest2/check.log), [inspect](../outputs/day10_downloads/pretest2/session_pretest2/inspect.log), [bench](../outputs/day10_downloads/pretest2/session_pretest2/bench.log), [launch](../outputs/day10_downloads/pretest2/session_pretest2/bench_launch.log), [RMSNorm](../outputs/day10_downloads/pretest2/session_pretest2/check_rmsnorm.log)). Structured results: `bench_ref_core_pretest2.{json,csv}`, `bench_ref_launch_pretest2.{json,csv}`, `check_softmax_ref_pretest2.json`, `check_rmsnorm_ref_pretest2.json` and `inspect_ref_pretest2/` (PTX, TTGIR, Inductor code). The archive SHA-256 is `dcb8d845a15f917c41c3dd8797dffec2d479ce18e2bc900d278b231e2ff7bf41`. All 51 manifest files were re-verified on Windows.

Two other runs were kept. **`pretest1`** is the same day and host, with an earlier `inspect_kernels.py` that had no Part 1b and a PTX vector-width parser that was wrong. Its timings agree with `pretest2` within about 1 %. **`pretest2_mineplumbing`** checks the `--impl mine` path end to end: the reference body was pasted into a *temporary copy* of `my_softmax.py` on the server, and `--core` passed with 381 cases. At the time of that pre-test, the repository's `my_softmax.py` was still the empty starter.

### A.1 Correctness ([check log](../outputs/day10_downloads/pretest2/session_pretest2/check.log))

| dtype | cases | failed | max \|y − torch\| | max \|y − fp64\|, Triton | max \|y − fp64\|, torch | max \|row sum − 1\| | rtol / atol |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 127 | 0 | 1.19e-7 | 7.88e-8 | 9.14e-8 | 1.78e-7 | 1e-5 / 1e-6 |
| fp16 | 127 | 0 | 7.63e-6 | 2.44e-4 | 2.44e-4 | 4.04e-4 | 1e-3 / 1e-5 |
| bf16 | 127 | 0 | 1.53e-5 | 1.95e-3 | 1.95e-3 | 2.93e-3 | 1.6e-2 / 1e-5 |

All 12 stability cases passed. Without the max subtraction, torch in fp32 turned 100 % of the `large_offset` outputs and 46–47 % of the `large_spread` outputs non-finite. The sentinel padding was intact in 99 of 99 `row_strided` cases. Every contract case raised `ValueError`. `CONTROLS: ALL DETECTED`, `SOFTMAX CHECK: PASSED (381 cases)`. The RMSNorm stretch reference printed `RMSNORM CHECK: PASSED (369 cases)`.

### A.2 Kernel inspection ([inspect log](../outputs/day10_downloads/pretest2/session_pretest2/inspect.log))

| N (fp32) | BLOCK | warps | elements/thread | registers | spills | programs/SM [estimate] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 4 | 1 | 14 | 0 | 16 |
| 1,024 | 1,024 | 4 | 8 | 21 | 0 | 16 |
| 4,096 | 4,096 | 16 | 8 | 23 | 0 | 4 |
| 16,384 | 16,384 | 16 | 32 | 58 | 0 | 2 |
| 32,768 | 32,768 | 16 | 64 | 97 | 0 | 1 |
| 65,536 | 65,536 | 16 | 128 | 128 | **78** | 1 |

In the warp sweep (Part 1b), N = 16,384 at 1 and 2 warps spilled 916 and 616 registers. N = 65,536 spilled at every setting: 1,338, 614, 78 and 34 at 4, 8, 16 and 32 warps. At 32 warps, 1,024 threads × 64 registers fill the register file, so the per-thread budget is capped at 64.

PTX for fp32, BLOCK 1,024: N = 1,024 has 2 `ld.global.v4.b32` and 2 `st.global.v4.b32`, with TTGIR `sizePerThread = [4]`. N = 1,000 has 8 scalar `ld.global.b32` and 8 `st.global.b32`, with `sizePerThread = [1]`. Both have 8 `ex2.approx.f32` (each preceded by a multiply by log₂e, `0f3FB8AA3B`), 8 `div.full.f32`, 14 `shfl.sync.bfly.b32`, 3 `bar.sync` and no local-memory spills.

Inductor emitted `triton_per_fused__softmax_prepare_softmax_online_0` for `[4096, 1000]` (persistent reduction, `XBLOCK` rows per program) and `triton_red_fused__softmax_prepare_softmax_online_0` for `[256, 32768]`. The second is a looped reduction with two passes over each row: an online max and sum with `evict_last`, then normalize and store with `evict_first`.

### A.3 Rows stage: 2²⁵ elements per call, `loop_us` (effective GB/s) ([bench log](../outputs/day10_downloads/pretest2/session_pretest2/bench.log))

| dtype | M | N | triton | torch | naive | compile | copy | torch / triton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 262,144 | 128 | 244.8 (1,097) | 199.8 (1,343) | 902.2 (298) | 206.4 (1,301) | 197.1 (1,362) | 0.82 |
| fp32 | 131,072 | 256 | 200.3 (1,340) | 199.5 (1,346) | 817.3 (328) | 206.9 (1,298) | 197.2 (1,361) | 1.00 |
| fp32 | 32,768 | 1,024 | 198.5 (1,352) | 202.9 (1,323) | 815.9 (329) | 207.3 (1,295) | 197.1 (1,362) | 1.02 |
| fp32 | 8,192 | 4,096 | 199.4 (1,346) | 204.9 (1,310) | 836.1 (321) | 199.5 (1,346) | 197.1 (1,362) | 1.03 |
| fp32 | 4,096 | 8,192 | 201.8 (1,331) | 236.4 (1,135) | 819.4 (328) | 201.6 (1,332) | 197.2 (1,361) | 1.17 |
| fp32 | 2,048 | 16,384 | 201.8 (1,330) | 217.0 (1,237) | 818.3 (328) | 256.6 (1,046) | 197.1 (1,362) | 1.08 |
| fp32 | 1,024 | 32,768 | 206.0 (1,303) | 355.9 (754) | 820.4 (327) | 255.3 (1,052) | 197.2 (1,361) | 1.73 |
| fp32 | 512 | 65,536 | 283.3 (948) | 394.0 (681) | 829.7 (324) | 313.2 (857) | 197.3 (1,361) | 1.39 |
| bf16 | 262,144 | 128 | 239.0 (562) | 122.9 (1,093) | 612.3 (219) | 105.6 (1,271) | 100.9 (1,331) | 0.51 |
| bf16 | 131,072 | 256 | 129.8 (1,034) | 123.5 (1,087) | 554.9 (242) | 105.3 (1,275) | 101.4 (1,324) | 0.95 |
| bf16 | 32,768 | 1,024 | 101.2 (1,326) | 122.7 (1,093) | 527.3 (255) | 105.6 (1,271) | 100.8 (1,332) | 1.21 |
| bf16 | 4,096 | 8,192 | 102.2 (1,313) | 233.4 (575) | 537.3 (250) | 132.0 (1,016) | 100.7 (1,333) | 2.28 |
| bf16 | 512 | 65,536 | 141.7 (947) | 196.9 (682) | 539.5 (249) | 169.0 (794) | 100.7 (1,333) | 1.39 |

The full table, including N = 512, 2,048 and 32,768 in both dtypes, is in the log. All of these rows exceed L2. In every row (all five implementations) `loop_us` exceeded `graph_us` by only 0.2–2.4 %, so this stage is GPU-bound throughout.

### A.4 Count stage: fixed N, growing M (fp32) ([bench log](../outputs/day10_downloads/pretest2/session_pretest2/bench.log))

| N | M | MiB moved | triton loop / graph (µs) | torch loop / graph (µs) | copy loop / graph (µs) | triton GB/s (graph) | L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1,024 | 1 | 0.008 | 32.0 / 1.8 | 8.1 / 5.0 | 6.7 / 1.4 | 5 | yes |
| 1,024 | 108 | 0.84 | 31.5 / 2.1 | 8.4 / 5.3 | 6.7 / 1.8 | 420 | yes |
| 1,024 | 1,728 | 13.5 | 31.8 / 5.6 | 9.3 / 8.2 | 6.7 / 4.8 | **2,551** | yes |
| 1,024 | 6,912 | 54 | 44.8 / 43.1 | 47.3 / 45.8 | 45.0 / 43.5 | 1,315 | no |
| 1,024 | 32,768 | 256 | 198.5 / 196.6 | 202.8 / 201.3 | 197.1 / 195.6 | 1,366 | no |
| 16,384 | 1 | 0.125 | 31.8 / 5.2 | 9.6 / 8.6 | 6.7 / 1.5 | 25 | yes |
| 16,384 | 108 | 13.5 | 32.0 / 7.6 | 11.1 / 9.8 | 6.7 / 4.8 | 1,856 | yes |
| 16,384 | 432 | 54 | 43.1 / 41.4 | 55.5 / 53.9 | 42.2 / 40.7 | 1,367 | no |
| 16,384 | 1,728 | 216 | 170.9 / 169.2 | 185.7 / 185.3 | 167.0 / 165.4 | 1,338 | no |

`naive` stayed near 49 µs per call and `compile` near 52 µs (N = 1,024) or 76 µs (N = 16,384) in the launch-bound rows.

### A.5 Padding stage: M = 16,384, fp32, `loop_us` (GB/s)

| N | BLOCK (Triton) | triton | torch | compile | copy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1,024 | 99.0 (1,324) | 101.7 (1,289) | 104.3 (1,257) | 98.3 (1,333) |
| 1,024 | 1,024 | 101.2 (1,326) | 104.8 (1,281) | 106.1 (1,265) | 100.6 (1,334) |
| 1,025 | 2,048 | 101.2 (1,328) | 105.0 (1,280) | **147.7 (910)** | 100.8 (1,333) |
| 2,048 | 2,048 | 198.8 (1,350) | 204.6 (1,312) | 199.0 (1,349) | 197.2 (1,361) |

For Triton, doubling BLOCK at N = 1,025 (half the lanes masked) and using scalar loads at N = 1,000 cost no measurable bandwidth at this size. `compile` lost about 30 % at N = 1,025. Its kernel for that shape was not dumped, so the cause is **[not isolated]**.

### A.6 Warp sweep: Triton only, fp32, `loop_us` (GB/s)

| N (M) | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 (32,768) | 201.7 (1,331) | 200.2 (1,341) | **198.5 (1,352)** | 197.1 (1,362) | 202.9 (1,323) | 309.5 (867) |
| 16,384 (2,048) | 1,157.7 (232) | 1,176.3 (228) | 201.5 (1,332) | 201.3 (1,334) | **201.7 (1,331)** | 201.7 (1,331) |
| 65,536 (512) | — | — | 1,285.5 (209) | 1,121.1 (239) | **283.0 (949)** | 284.5 (944) |

Bold marks the `launch_config` default. Note that 32 warps at N = 65,536 still spilled 34 registers and ran no faster than 16 warps.

### A.7 Launch stage (optional): fp32, per call ([launch log](../outputs/day10_downloads/pretest2/session_pretest2/bench_launch.log))

| M × N | triton | triton_raw | torch | compile | compile_ro | compile_ro_static | Triton `graph_us` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 × 128 | 31.1 | 13.5 | 8.1 | 48.6 | 100.6 | 79.2 | 2.1 |
| 16 × 1,024 | 31.5 | 13.6 | 8.3 | 49.4 | 101.3 | 79.1 | 2.1 |
| 108 × 1,024 | 31.5 | 13.9 | 8.3 | 49.6 | 100.5 | 79.4 | 2.3 |
| 1 × 16,384 | 31.4 | 13.5 | 9.7 | 73.2 | 100.5 | 79.5 | 5.2 |

`loop_us` in µs. The first call of the two graph-tree variants took 125–344 ms, and their slowest warmup call (graph recording) 1.9–35 ms. The difference between the two graph-tree rows, about 21–22 µs per call, is the copy of the non-static input.

### A.8 What the pre-test supports and what it does not isolate

- **Supported [pre-test]:** correctness across 33 row lengths × 3 dtypes × 4 layouts, the stability cases, the masks and the contract. All three controls were detected. At mid N the fused kernel ran at the copy's bandwidth (within 1–3 %), and the unfused composition at about a quarter of it. Small calls were launch-bound, with fixed floors per implementation (Triton about 31 µs, of which the bare launch is about 14 µs; eager torch about 8 µs; one replayed kernel about 2 µs). Data that fits in L2 gave effective bandwidth above the DRAM data sheet. `reduce-overhead` was slower than the default `torch.compile` for this one-kernel function.
- **Not isolated:** why Triton is slower than torch at N = 128 (candidates: per-program fixed cost with one row per 128-thread program, versus ATen's warp-per-row kernel); why everyone slows down at N = 65,536 (spills, one program per SM, no overlap between a program's load and reduction phases); why eager torch drops at bf16 N = 4,096–8,192 and fp32 N = 32,768 (a multi-pass block kernel is a candidate, Block 3 Stop 4); and why `compile` drops at N = 1,025. Each needs either a hardware-counter profile (Nsight Compute: DRAM bytes, local-memory traffic, achieved occupancy) or a controlled kernel variant.
