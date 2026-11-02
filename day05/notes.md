# Day 05 — CUDA Graphs: closed by the Day 03 follow-up

**Status: closed, no separate Day 05 exercise** (decided 2026-11-02). The plan's Day 05 objective was largely met during Day 03's optimizer-graph and bucket-view follow-up. The topics it did not cover are deferred to the days where they solve a concrete problem (§3). The next working session goes to Day 06 (ZeRO-1). Day numbers and directories are unchanged; this directory holds only this closure record.

Evidence labels: **[measured]** = a Day 03 run with saved artifacts; **[understanding]** = explained in the walkthrough, not separately tested; **[not tested]** = no run exercised it.

## 1. Original objective vs what exists

Plan row (README, Day 5): *CUDA Graph pitfalls — eager vs `torch.compile` vs manual graph capture — done when: parity over several steps; capture vs replay explained.*

| Objective | Status | Evidence |
| --- | --- | --- |
| Manual capture and replay | **Done for `optimizer.step()`**: AdamW with `capturable=True`, CUDA-tensor LR, warmup on a side stream, `torch.cuda.graph` capture, replay every step | [measured] [`optimizer_graph.py`](../experiments/day03_ddp_hooks/optimizer_graph.py), runs `a100_nvlink_20261101`, `a100_nvlink_view_20261102` |
| Parity over several steps | **Done**: matched eager vs replay bitwise identical over 24 tiny-model steps (gradients, first-update and final parameters, all losses) for MiniDDP and DDP; identical 60-step medium loss curves; Adam counters and the actual LR per step checked | [measured] [`day03/CUDA_GRAPH.md`](../day03/CUDA_GRAPH.md) |
| Pitfalls of static addresses | **Done**: stable `.grad` tensors (no `set_to_none`), gradient and LR addresses checked every step, capture only after DDP bucket rebuild once bucket-view addresses are stable, warmup's dummy updates/moments/counters/RNG restored before training, LR scheduler outside the graph writing the LR tensor in place | [measured] same |
| Capture vs replay explained | **Done**: capture records kernels with fixed pointers; replay re-launches them with one `cudaGraphLaunch` (1953 eager launches → 1 graph launch + 3 eager), kernels packed with ~0.1 ms of internal gaps; replay does not fuse (1950 kernels either way) and is not a cross-rank barrier | [measured] profiler counts; [understanding] the explanation |
| Costs of making code graph-safe | **Done**: graph-ready eager is *slower* than the original (capturable AdamW: 624 → 1950 kernels; +P of memory from persistent gradients), so the net replay gain is only 4.4% / 5.9%; bucket views recover P | [measured] |
| Profiler interaction | **Done, beyond the plan**: any torch.profiler setting adds 10–20 ms per step and 2–4 ms of rank offset on this launch-bound workload; alignment must be measured unprofiled | [measured] `a100_profiler_overhead_20261102` |
| `torch.compile` comparison | **Not done** | [not tested] |
| Capture beyond the optimizer | **Not done**: forward, backward and NCCL stayed eager | [not tested] |

The criterion as written ("parity over several steps; capture vs replay explained") is met. The missing pieces are the `torch.compile` column and capture of anything beyond the optimizer.

## 2. Capture vs replay in brief [understanding, backed by the Day 03 measurements]

1. **Warm up on a side stream.** Lazy state (Adam moments, `step` tensors) and allocator blocks must exist before capture, because capture cannot allocate in the normal way and must not see one-time initialization.
2. **Capture records, it does not run the math for real.** Every kernel launch is recorded with its arguments, i.e. the exact device addresses. Inputs that change (gradients, LR) must therefore live in tensors whose storage never moves; Python scalars are baked in.
3. **Replay re-submits the recorded kernels in one call.** It saves host submission time, not device time. It helps only where the GPU was waiting for the CPU, which on this workload was the optimizer's 1950 small launches.
4. **Anything outside the graph must update inputs in place.** The scheduler writes the LR tensor; backward/NCCL write the same gradient (or bucket-view) storage the graph reads.
5. **Graph-safe code is a different program.** Capturable optimizers and persistent buffers change the arithmetic and the memory, so the fair comparison is graph vs graph-ready eager, and the net comparison is against the original.

## 3. Remaining topics: why not now, and where each one becomes concrete

None of these is needed for Days 06–09. Each is attached to the day where it addresses a real bottleneck or a real requirement, not added as an exercise.

| Topic | Why it is not a Day 05 exercise now | Deferred to | Trigger to do it |
| --- | --- | --- | --- |
| **`torch.compile` and CUDA Graphs** (`mode="reduce-overhead"` uses CUDA Graphs, managed by Inductor's cudagraph trees; graphs are skipped or re-recorded on dynamic shapes, input mutation or CPU tensors) | Day 10 already compares a kernel against eager and `torch.compile`; the graph angle fits there, where small rows make launch overhead visible | Day 10 (optional column), Day 17 (how vLLM combines compile and graphs) | A Day 10 row length where kernel time ≈ launch time |
| **Dynamic inputs** (one graph per shape, padding to captured sizes, memory pools shared between graphs) | The training harness uses fixed shapes, so there is nothing dynamic to handle | Day 17: decode batch size changes every step; vLLM captures a set of batch sizes and pads | Reading vLLM's graph capture for decode |
| **RNG inside a graph** (Philox seed/offset advanced per replay; extra generators registered with the graph) | The model uses `dropout=0.0` and the optimizer draws no random numbers, so no RNG op is in any region worth capturing | Day 17 (sampling), Day 20 (stochastic policy actions) | A captured region that contains dropout or sampling |
| **Forward-only capture with static input buffers** | Training steps here are not limited by forward submission alone | Days 15/19/20: small-batch policy inference in rollout is the classic launch-bound case | A rollout profile with GPU idle gaps between small kernels |
| **Forward + backward capture** (`make_graphed_callables` or whole-step capture) | Day 04 shows ~9 ms of fixed cost per 8-row microbatch, but the cheaper fix measured there is a larger microbatch (1.41–1.52× faster when it fits); none of Days 06–09 is limited by launch overhead | Day 20, only if the chosen bottleneck is a launch-bound update step | A step that stays launch-bound at the largest microbatch memory allows |
| **Distributed capture** (NCCL collectives inside a graph; DDP needs extra conditions such as a side-stream construction and eager warmup iterations, per the PyTorch CUDA-semantics notes) | Highest complexity, and Day 03 showed the remaining rank skew is ~0.3–0.7 ms unprofiled, so there is no current problem it would solve | Day 20 at the earliest, same trigger as above | As above, with communication included in the captured region |

## 4. Decision and rationale

**Proceed directly to Day 06 (ZeRO-1).** Reasons:

- The plan's Day 05 criterion is already met by measured, bitwise-checked evidence. Repeating optimizer capture would not teach anything new.
- Every remaining topic is either inapplicable to the current workload (RNG: no dropout; dynamic inputs: fixed shapes) or addresses a launch-bound regime that the next five days do not have. ZeRO, FSDP2, DeepSpeed and mixed precision are memory- and communication-bound topics.
- The most interview-relevant remaining pieces, dynamic-shape capture and the `torch.compile`/graph relationship, are learned better on Day 17 against vLLM's real decode loop than on a fixed-shape toy.
- Day 06 connects directly to the Day 03/04 work: ZeRO-1 changes what happens at the accumulation boundary (a `reduce_scatter` of the accumulated gradient, a sharded optimizer step, an `all_gather` of parameters).

The freed day is not refilled. It is schedule slack, to absorb overruns on Days 07–08 (FSDP2, DeepSpeed), which have the most setup risk.

**Reopen this day** only if a later profile shows a step that stays launch-bound after the obvious fixes (larger microbatch, `foreach`/`fused` optimizer).
