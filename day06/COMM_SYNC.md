# Day 06 follow-up: rank arrival skew vs communication, normal P2P vs `NCCL_P2P_DISABLE=1`

**Run `a100_sync_20261103T120618Z`** (2026-11-03, 12:06–12:18 UTC, 64 jobs, all exit code 0). Code: [`experiments/day06_comm_sync/`](../experiments/day06_comm_sync/README.md). Raw evidence: the verified archive listed at the end. The original Day 06 measurements (`a100_20261102T124935Z`, other host) are unchanged in [`notes.md`](notes.md) §8.

**Question.** In Day 06's non-overlapped collectives, how much of the time is one rank waiting for the other, and how much is the collective? Does synchronizing both processes after one iteration (`prev_step`) make the next iteration's traces show communication instead of waiting? Compared with a barrier immediately before the collective (`pre_collective`), a 10 ms injected-delay control, and a communication-only benchmark.

Evidence labels: **[measured]** this run; **[understanding]** interpretation consistent with it, not isolated; **[not tested]**.

## Answers

1. **Synchronizing after the previous iteration is not sufficient.** [measured] The iteration starts were already aligned *without* any barrier: the previous step ends with a blocking collective plus a local `synchronize` on both ranks, so the next iteration's start differs by a median 0.05 ms (normal P2P) / 0.18 ms (P2P disabled). Arrival skew at the target collective is created *during* the 104–206 ms of launch-bound forward/backward: the target arrival gap equals the gradient-ready gap to within 0.01–0.05 ms (median), and that equals the per-rank forward/backward duration difference. With `prev_step`, the per-job maximum arrival gap at the collective is still 3–19 ms unprofiled (median over jobs 8.19–14.06 ms). In the paired traces the NCCL kernel start gap is −6.73 / 1.29 / 37.59 / −9.81 ms (`flat_ar` / `zero1`, normal / disabled P2P).
2. **Immediate pre-collective alignment removes the waiting from the NCCL kernel.** [measured] Arrival gap 0.00 ms median, max |gap| ≤ 0.48 ms unprofiled; trace kernel start gaps −0.08 to +0.14 ms. Both ranks' kernels then last the same: normal P2P all-reduce 0.647 / 0.665 ms, reduce-scatter 0.573 / 0.430, all-gather 0.398 / 0.403; disabled P2P 37.22 / 37.14, 18.48 / 18.48, 18.46 / 18.59 ms. The two ranks leave the CPU Gloo barrier within 0.001–0.031 ms of each other (median).
3. **The waiting moved into the barrier; training did not get faster.** [measured] The early rank's barrier wait carries what its NCCL kernel used to carry: with P2P disabled, rank 1 waits a median 8.66 ms (`flat_ar`) in the barrier, matching rank 0's forward/backward being slower. Inclusive iteration time and loop time per step are unchanged within the repeat ranges (normal P2P `flat_ar` loop 122.91 [122.39–123.64] → 124.49 [112.35–126.28] ms; `zero1` 124.49 → 125.67; disabled P2P `flat_ar` 150.55 → 156.17, `zero1` 154.07 → 161.12, all with overlapping ranges). If anything, the barriers add a little. Injected 10 ms delay on rank 1: without alignment (and with `prev_step`) rank 0's reduce-scatter kernel lasts 10.93 ms vs 0.44 ms on rank 1; with `pre_collective` rank 0 waits in the barrier instead (median 8.62 ms; per step 0.4–11.0 ms, because the natural gap adds or subtracts), and kernels last 0.51 / 0.72 ms. Iteration time rises by ~10 ms in all three modes (134.0 / 135.0 / 134.4 ms vs 124.0–125.1 without injection).
4. **The slow-link "RS + AG > AR" residual is not a stable quantity on this host.** [measured] Aligned paired traces: all-reduce kernel 37.2 ms vs reduce-scatter + all-gather kernels 36.9–37.1 ms (no residual). Aligned unprofiled host phases: 37.9 vs 21.1 + 20.3 = 41.4 ms (+3.5 ms, wide ranges). Communication-only benchmark: 37.9 vs 22.2 + 22.6 = 44.8 ms (+6.9), RS→AG pair 46.0 ms, but each operation drifted by 25–35% within the 30-second benchmark (all-reduce 43.5 → 27.7 ms, reduce-scatter 24.7 → 15.3 ms). This host did not reproduce a stable excess. That does not explain away the earlier run's 5.28 ms: on that other host it was about 5.3 ms in both repeats (all-reduce 53.03 / 53.65 ms, reduce-scatter 28.80 / 29.30, all-gather 29.55 / 29.60). It remains a measured, host-specific result; it is not shown to be a fixed extra launch, and its cause is unresolved.

## Setup

| Item | Value |
| --- | --- |
| Host | Vast.ai `ssh2.vast.ai:12197`, container 58958418 (a **different host** from Day 06's `ssh4:17591`): 2 × A100-SXM4-40GB, NV12, both GPUs on **NUMA node 0** (CPU affinity 0–31, 64–95); driver 595.71.05 |
| Software | Python 3.12.15, torch 2.14.1+cu130 (git 5c48869), CUDA build 13.0, **runtime NCCL 2.30.7** (`torch.cuda.nccl.version()`, NCCL log "NCCL version 2.30.7+cuda13.3"; the image's `NCCL_VERSION` variable says 2.28.3-1 and is not what runs) |
| Clocks, load | SM 1410 MHz under load, no throttle reasons; no other GPU processes; GPU utilization ~16% average (launch-bound); runner affinity 0–127; cgroup quota 30.7 CPUs |
| Workload | `flat_ar` and `zero1` (FlatZeroAdamW), medium preset, W = 2, K = 4, m = 8, G = 64, FP32, deterministic settings, Day 01 AdamW, 12 optimizer steps; timing steps 4–11 |
| Alignment | `none`; `prev_step` (CPU Gloo barrier at the top of every iteration, after the previous `prof.step()`); `pre_collective` (barrier after gradients are GPU-ready, immediately before the reduction; for `zero1` also before the all-gather). Every barrier is preceded by a local `torch.cuda.synchronize()`; the Gloo group adds no GPU kernel and no counted collective |
| Transport | normal (override variables removed) and `NCCL_P2P_DISABLE=1`; a fresh `torchrun` per job |
| Matrix | 6 tiny correctness, 2 NCCL route-log, 2 communication-only, 36 unprofiled timing (12 configurations × 3 rounds, order rotated), 12 profiled paired-trace (CPU+CUDA, no shapes/stacks/memory; profiler warmup step 5, recorded steps 6–8), 6 injected-delay controls. 11.2 min |

Host timestamps are `time.perf_counter_ns()` (CLOCK_MONOTONIC, shared by both processes), so rank 0 and rank 1 values of one step are directly comparable. Kernels are paired through the trace: CPU range `## s<step> TARGET <op>` → the one `cuLaunchKernelEx` inside it → correlation id → the NCCL kernel. All 15 profiled jobs have identical `baseTimeNanoseconds` on both ranks. Their 72 target ranges per rank all mapped, and no NCCL kernel fell outside a target range.

## 1. Correctness [measured]

Tiny preset, same arguments: `sync_diag.py` `zero1` with `none` / `prev_step` / `pre_collective` and `flat_ar` with `pre_collective` vs the **unmodified** `train_zero1.py`. Final parameters and gathered AdamW `exp_avg` / `exp_avg_sq` are **bitwise identical**, and the Adam step counts are equal. Replicas were bitwise equal after every step. FlatZeroAdamW counters are identical (12 reduce-scatters + 12 all-gathers, or 12 all-reduces): the control barriers do not appear in them.

## 2. Where the skew is created [measured; unprofiled timing jobs, steps 4–11, 48 step samples per row]

| Transport | Align | median \|start gap\| at t0 | steps with \|t0 gap\| > 1 ms | median \|gradients-ready gap\| | median \|arrival gap − ready gap\| | median \|fwd/bwd r0 − r1\| |
| --- | --- | --- | --- | --- | --- | --- |
| normal | none | 0.050 | 1 | 2.05 | 0.010 | 2.06 |
| normal | prev_step | 0.079 | 0 | 3.09 | 0.043 | 3.15 |
| disabled | none | 0.179 | 2 | 9.30 | 0.046 | 9.42 |
| disabled | prev_step | 0.070 | 0 | 9.19 | 0.041 | 9.14 |

ms. t0 = after `zero_grad` + synchronize, i.e. when forward/backward starts. The arrival gap at the collective is the forward/backward duration difference; the start was already aligned. With P2P disabled, rank 0's forward/backward was slower in 17 of 18 jobs (per-job median +1.6 to +11.6 ms). With normal P2P the per-job medians scatter around 0, apart from 3 jobs with +16–18 ms. The cause of the disabled-P2P rank asymmetry was not isolated. [understanding] Both ranks run on separate cores with a 30.7-CPU quota on a shared host; the SHM transport's CPU-side proxy work is one candidate. Neither was tested.

A side observation, not isolated: with P2P disabled, the faster rank's forward/backward is ~16 ms *shorter* than with normal P2P (103.7 vs 119.7 ms median). The 2026-11-02 run on the other host shows the same direction (fwd+bwd phase 103.4–103.5 vs 119.0–119.2 ms).

## 3. Alignment modes without the profiler [measured; median [min–max] over 3 jobs of the per-job median over steps 4–11]

| Impl, op | Transport | Align | arrival gap r1 − r0 | max \|gap\| | barrier wait r0 / r1 | host phase r0 / r1 | loop ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `flat_ar` all-reduce | normal | none | 0.31 [0.01–0.36] | 11.22 [5.90–17.30] | — | 1.40 / 1.03 | 122.91 [122.39–123.64] |
| | normal | prev_step | −0.95 [−4.11–5.72] | 10.13 [3.21–11.92] | — | 1.20 / 2.03 | 126.81 [126.20–127.57] |
| | normal | pre_collective | 0.00 | 0.06 | 0.39 / 0.87 | 1.01 / 0.98 | 124.49 [112.35–126.28] |
| `zero1` reduce-scatter | normal | none | 0.84 [−18.49–2.60] | 4.06 [2.71–25.24] | — | 1.66 / 0.67 | 124.49 [124.27–125.97] |
| | normal | prev_step | 0.33 [−17.13–0.56] | 8.19 [5.28–18.89] | — | 1.46 / 1.30 | 125.27 [122.82–128.28] |
| | normal | pre_collective | −0.01 | 0.03 | 1.10 / 0.40 | 0.69 / 0.62 | 125.67 [125.33–129.62] |
| `flat_ar` all-reduce | disabled | none | −8.89 [−11.22–−7.69] | 13.39 | — | 37.82 / 47.20 | 150.55 [150.13–156.89] |
| | disabled | prev_step | −9.60 | 14.06 | — | 37.15 / 47.19 | 153.09 [151.26–155.84] |
| | disabled | pre_collective | 0.00 | 0.10 | 0.36 / 8.66 | 37.94 / 37.84 | 156.17 [150.57–157.30] |
| `zero1` reduce-scatter | disabled | none | −7.88 [−11.61–−4.48] | 13.65 | — | 20.36 / 28.19 | 154.07 [147.96–162.93] |
| | disabled | prev_step | −10.35 | 13.15 | — | 23.03 / 33.25 | 161.97 [156.18–163.33] |
| | disabled | pre_collective | 0.00 | 0.22 | 0.52 / 2.25 | 21.15 / 21.03 | 161.12 [154.99–162.25] |
| `zero1` all-gather | disabled | none / pre_collective | −0.08 / 0.02 | 0.28 / 0.06 | — / 0.37, 0.42 | 19.85, 19.93 / 20.32, 20.24 | (as above) |

ms. Host phase = `reduce_grads()` dispatch → return of the following `synchronize` (includes the `div_` and host dispatch; not isolated transfer time). Loop ms/step = wall time of steps 4–11 including every barrier, max over ranks. The all-gather needs no alignment: it follows a collective, a short identical optimizer step and a synchronize, so its arrival gap is already ≤ 0.3 ms. Full tables, including the steps-6–8 subset: `tables.md` in the archive.

## 4. Paired traces [measured; one CPU+CUDA profiled job per configuration, recorded steps 6–8]

| Impl, op | Transport | Align | kernel start gap r1 − r0 | NCCL kernel r0 / r1 | kernel end gap µs | barrier wait r0 / r1 |
| --- | --- | --- | --- | --- | --- | --- |
| `flat_ar` all-reduce | normal | none | 11.365 | 12.203 / 1.480 | −8.7 | — |
| | normal | prev_step | −6.730 | 0.742 / 7.468 | −2.0 | — |
| | normal | pre_collective | −0.014 | 0.647 / 0.665 | −0.5 | 0.97 / 0.74 |
| `zero1` reduce-scatter | normal | none | 7.915 | 8.329 / 0.388 | −27.3 | — |
| | normal | prev_step | 1.286 | 1.670 / 0.388 | 3.2 | — |
| | normal | pre_collective | 0.135 | 0.573 / 0.430 | −4.7 | 1.35 / 0.79 |
| `zero1` all-gather | normal | none / prev / pre | −0.118 / −0.023 / −0.025 | 0.387, 0.468 / 0.380, 0.399 / 0.398, 0.403 | | |
| `flat_ar` all-reduce | disabled | none | 21.186 | 52.245 / 31.118 | 56.6 | — |
| | disabled | prev_step | 37.589 | 73.274 / 37.985 | −316.6 | — |
| | disabled | pre_collective | −0.072 | 37.217 / 37.138 | −43.4 | 0.70 / 1.97 |
| `zero1` reduce-scatter | disabled | none | 9.943 | 28.108 / 18.246 | 60.3 | — |
| | disabled | prev_step | −9.813 | 18.477 / 28.268 | −24.4 | — |
| | disabled | pre_collective | −0.063 | 18.481 / 18.479 | −79.9 | 0.44 / 17.63 |
| `zero1` all-gather | disabled | none / pre_collective | 0.090 / −0.084 | 19.169, 18.448 / 18.455, 18.586 | | |

ms (median of the 3 recorded steps). With normal P2P the kernels end together (|end gap| ≤ 27 µs): the earlier rank's extra kernel time is waiting. With P2P disabled the end gaps are tens of µs to 0.3 ms. The later rank's disabled-P2P kernel (18.2–18.5 ms reduce-scatter, 31–38 ms all-reduce) is the transport time plus residual waiting.

**Profiler perturbation.** Recorded steps take 195–238 ms per iteration vs ~124–163 ms unprofiled. Forward/backward grows from ~104–120 to ~127–206 ms, and arrival gaps in `none`/`prev_step` traces (up to 65 ms) exceed the unprofiled ones (≤ 25.2 ms). The traces show the waiting mechanism; their absolute skews must not be read as unprofiled behaviour.

**Profiler transitions** (`prof.step()` host duration, max over ranks, median [min–max] over 15 jobs): entering profiler warmup at the end of step 4 costs 66 [43–101] ms and differs between ranks by up to ~25 ms; warmup → active at the end of step 5 costs 0.63 ms; stop + export at the end of step 8 costs 466 [442–501] ms; all others < 0.4 ms. The step-4 transition shifts step 5's start by up to 24 ms (`none`, `prev_step` jobs; `prev_step`'s barrier removes it). Step 5 is not recorded, and the next blocking collective re-aligns the ranks. In this schedule the transitions do not explain the recorded skew; the profiler's per-op CPU overhead during forward/backward does. [understanding] In the original Day 06 count stage (warmup step directly before 2 recorded steps) the same mechanisms apply; that run's 36–58 ms gaps were not re-measured here.

## 5. Injected 10 ms delay on rank 1 [measured; control, excluded from all performance rows]

| Align | rank 0 reduce-scatter host phase / kernel | rank 1 host phase / kernel | barrier wait r0 | iteration incl. (unprofiled) |
| --- | --- | --- | --- | --- |
| none | 11.08 ms / — (trace dominated by a 52 ms profiler skew in the other direction) | 0.66 / — | — | 134.03 ms |
| prev_step | 11.33 / 10.93 ms | 0.62 / 0.44 ms | — | 135.02 ms |
| pre_collective | 0.63 / 0.51 ms | 0.70 / 0.72 ms | 8.62 ms (per step 0.4–11.0) | 134.38 ms |

Without alignment, or with it one iteration earlier, the delay shows up as rank 0 waiting inside its NCCL kernel. With immediate alignment it shows up in rank 0's barrier. The iteration costs ~10 ms more in every mode (no-injection baseline 124.0–125.1 ms): the barrier relocates the wait without removing it.

## 6. Normal vs disabled P2P: route and communication-only cost [measured]

NCCL INFO logs (separate route jobs): normal → `P2P/CUMEM/read`, 24 collective channels, Ring/Simple for all three collectives. `NCCL_P2P_DISABLE=1` → logged as "set by environment to 1", `SHM/direct` in both directions (host shared memory, no GPU–GPU P2P over NVLink or PCIe), 4 collective channels, Ring/Simple. The NET/Socket transport was initialized but not selected for these channels.

Communication only (`comm_bench.py`: 104.6875 MiB FP32, FlatZeroAdamW's in-place layouts, Gloo-aligned, exact output validation, 20 timed iterations; per-rank median host time, max over ranks):

| Transport | all-reduce | reduce-scatter | all-gather | RS + AG | RS→AG pair (no optimizer between) |
| --- | --- | --- | --- | --- | --- |
| normal | 0.83 | 0.53 | 0.52 | 1.05 (+0.22 vs AR) | 0.95 (+0.11) |
| disabled | 37.87 [27.7–43.5] | 22.23 [15.3–24.7] | 22.55 [15.3–28.0] | 44.78 (+6.91) | 45.96 [30.8–52.1] (+8.10) |

The disabled-P2P times drift during the benchmark: the first 13 iterations sit near 38–43 / 22–25 / 22–24 ms, the last 7 near 28–33 / 15–18 / 15–19 ms. With normal P2P, splitting the all-reduce costs +0.1–0.2 ms on this host, consistent with the 2026-11-02 run's +0.17 ms. With P2P disabled, the aligned traces show no residual, while host phases and the benchmark show +3.5 to +8 ms inside ±25% drift. On this host, the slow-link all-reduce is ~45× the NVLink one (37.9 vs 0.83 ms) and faster than on the 2026-11-02 host (53.3 ms); slow-link numbers are host-specific.

## 7. What remains unresolved

- Why rank 0's forward/backward is ~9 ms slower with P2P disabled (17 of 18 jobs), and why the faster rank's forward/backward is ~16 ms shorter with P2P disabled than with it. CPU placement and proxy-thread interference were not tested.
- The cause and stability of the disabled-P2P RS + AG excess. Transport speed drifted 25–35% within 30 s; a longer, interleaved comm-only run (or `NCCL_DEBUG_SUBSYS=PROXY` timing) would be needed. No algorithm/protocol was forced.
- Three normal-P2P timing jobs with a 16–18 ms per-step forward/backward asymmetry (shared-host scheduling is plausible, not shown).
- Profiler effects were measured with the CPU+CUDA profiler only; a CUDA-only profile was not needed to answer the question and was not run.
- [not tested] An NCCL control barrier (adds a GPU kernel; not used), W > 2, overlapped/bucketed ZeRO collectives.

**Practical rule.** To read communication cost from a trace, look at the *later-arriving* rank's kernel, or align immediately before the collective and report the barrier wait next to it. Barriers one iteration earlier do not help, because skew accumulates in between. Never report time moved into a barrier as a saving: compare inclusive iteration times.

## Files and backup

Archive (downloaded and verified): `C:\Users\qly24\Downloads\day06_comm_sync_a100_sync_20261103T120618Z\day06_comm_sync_a100_sync_20261103T120618Z.tar.gz` with its `.sha256`, extracted next to it. Inside: `report.md` (this file), `tables.md`, `summary.json`, `rows.csv`, `commands.log`, `jobs/` (raw per-rank rows, metadata, 30 traces, tiny results), `comm/`, `route/` (NCCL INFO logs), `logs/`, `env/` (topology, versions, nvidia-smi before/after, GPU sampler), `source/` and `SHA256SUMS`.
