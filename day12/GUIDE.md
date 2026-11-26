# Day 12: two-stage pipeline parallelism, GPipe and non-interleaved 1F1B (dual-A100 guide)

> **Run status, 2026-11-15.** The dual-A100 reference rerun (`manual_ref_20261115T113822Z`) and the completed `my_pp.py` run (`manual_mine_20261115T114621Z`) both passed the boundary check and all eight GPipe/1F1B cases. The reference rerun detected all three negative controls. Both archives were downloaded and their SHA-256 checks passed, including 45 manifest files per run. [Measured results](#measured-results-2026-11-15) · [Experiment notes](notes.md)

## Required to close Day 12 (about 4 h including the run)

| # | You can explain | Evidence | Where |
| --- | --- | --- | --- |
| 1 | **Who owns what and what crosses the boundary**: which blocks each stage holds, and which tensor goes each way for each microbatch | the hand-worked grids; the `P2P calls` and `labels` lines in your log | §1 |
| 2 | **Autograd across two processes**: what each stage must keep per microbatch, why the received activation must become a leaf that requires grad, how gradients find the right microbatch, and how the send/recv order avoids deadlock | `boundary_demo`, the Step 1 check, the `lifo_backward` and `detached_input` controls, `pp_sim.py --comm-style naive --send rendezvous` | §2 |
| 3 | **GPipe vs 1F1B**: warmup, steady state and cooldown; how to count idle slots and peak in-flight microbatches; why 1F1B lowers activation residency without lowering the idealized bubble | the simulator grids and sweep; measured order, in-flight trace and CUDA peak per rank | §3 |
| 4 | **Where the same steps live in Megatron Core** `core_v0.19.2`, from `forward_backward_pipelining_without_interleaving` down to `batch_isend_irecv` | Block 3 stops (verified file:line) | Block 3 |

**Completion record**

- [x] Record the P1–P8 reasoning and corrections in notes §1 as a discussion recap.
- [x] Complete `my_pp.py` and pass the dual-GPU boundary check and full GPipe/1F1B matrix with `--impl mine`.
- [x] Run the reference with controls, download and verify both archives, and fill notes §2–§5.
- [x] Review Block 3 Stops 1–5 in the walkthrough and fill the mapping table in notes §6.

**Optional (not needed to close the day):** §4 unequal stage times and communication latency in the simulator, Block 3 Stop 6, the `deallocate_output_tensor` question.

**Scope on purpose.** Two stages, one per GPU, a 4-block residual MLP, FP32, one forward/backward step **before** the optimizer step, no timing benchmark. No interleaved (virtual-stage) schedule, no TP inside a stage, no data parallelism. The four-GPU Megatron job (`TP=2 × PP=2`) is Day 13.

---

## What is new today (and what you already have)

You already have: microbatches and loss weighting (Day 04: weight microbatch j by n_j / N), point-to-point `send`/`recv` (Day 02's ring), custom communication inside autograd (Day 11's `f` and `g`). Day 11 split **one layer** across ranks that all hold the same tokens. Today you split **the model by depth**: each rank holds different layers, and the tokens travel through the ranks.

New today:

- **A graph that spans two processes.** Neither rank has the whole autograd graph. Stage 0's graph *ends* at a tensor it sent away; stage 1's graph *starts* at a tensor it received. You connect them by hand with two calls.
- **Several microbatches alive at once.** Accumulation (Day 04) finished one microbatch's backward before the next forward. A pipeline keeps forwards of later microbatches running while earlier ones wait for their backward, so each stage holds **several sets of saved activations**, and has to match each arriving gradient to the right one.
- **Schedules.** Same work, different order. The order decides how long the GPUs sit idle (the bubble) and how many microbatches' activations a stage must hold (in flight).
- **Deadlock as a design constraint.** In 1F1B's steady state both neighbours want to send and receive at the same moment. The call order must make that safe.

---

## Read this first: the host

The measured runs used Vast.ai container 60779373 with 2 × A100-SXM4-80GB, NV12, both GPUs on NUMA node 0, driver 580.178.04, torch 2.14.1+cu130 and NCCL 2.30.7. The earlier host is recorded in Appendix A. Day 12 also fits the 40 GB template. On a fresh rental, create the project directory from your Windows project root:

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/AI-Infra-Learning/experiments'
```

```bash
scp -P <port> README.md requirements.txt root@<host>:/workspace/AI-Infra-Learning/
```

```bash
scp -P <port> -r day12 root@<host>:/workspace/AI-Infra-Learning/
```

```bash
scp -P <port> -r experiments/day12_pipeline_parallel root@<host>:/workspace/AI-Infra-Learning/experiments/
```

Then, on the instance:

```bash
cd /workspace/AI-Infra-Learning
ln -sfn /venv/main .venv        # fresh rental only
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
nvidia-smi topo -m              # expect NV12 between GPU0 and GPU1
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.can_device_access_peer(0, 1))"
```

Expected: `2.14.1+cu130 2 True`. No new package is needed. The simulator (`pp_sim.py`) and `find_symbols.py` also run on Windows without torch.

---

## Files

All in `experiments/day12_pipeline_parallel/`:

| File | What it is | Read it when |
| --- | --- | --- |
| `pp_sim.py` | CPU schedule simulator: per-stage timelines, idle, bubble, in-flight counts, P2P programs, a deadlock model. No torch | §1 and §3, before anything else |
| `boundary_demo.py` | Autograd across a cut, one CPU process, scalar numbers you can check by hand, three mistakes | §2 |
| `pp_common.py` | Shapes, `make_problem`, the residual `Block`, `stage_blocks`, `microbatches`, `mb_loss`, `compare`, and the **provided** `P2P` helpers, `Stage` and `EventLog` | Stage 0 |
| `my_pp.py` | **Yours.** `forward_step`, `backward_step`, `run_gpipe`, `run_1f1b`, all **YOU WRITE** | Stages 1–3 |
| `pp_ref.py` | Reference for the same four functions | After yours passes, or when stuck |
| `baseline_1gpu.py` | The whole model on one GPU: accumulation over M = 1, 2, 4, 8 microbatches; saves y, loss and every gradient | Stage 0 |
| `check_pp.py` | Two ranks: `--test boundary` (Step 1) and `--schedules ... --microbatches ...` (parity, order, in-flight, labels, memory); `--controls` | Stage 1 onward (read its docstring) |
| `pp_controls.py` | `detached_input`, `lifo_backward`, `mean_per_microbatch` and the checks each must fail | Stage 4 |
| `find_symbols.py` | file:line of the Block 3 reading list in your Megatron checkout | Block 3 |
| `run_server_once.sh`, `download_results.ps1` | Everything once, logged and archived; download and verify on Windows | Block 4 |

---

## Block 1: Concepts (about 2 h). Predict first, then open the answer

This is the main block today. Write each prediction in notes §1 before you open the `<details>`.

### §1 The hand-worked example: two stages, four microbatches

**The model.** Four residual MLP blocks, $\text{out} = x + \mathrm{fc2}(\operatorname{gelu}(\mathrm{fc1}(x)))$, d = 512, ffn = 2048, activations `[seq, batch, d]` as in Megatron and Day 11. **Stage 0 (rank 0, GPU 0) owns blocks 0–1; stage 1 (rank 1, GPU 1) owns blocks 2–3.** The effective batch is 8 sequences of 256 tokens. With M = 4 each microbatch is `[256, 2, 512]`.

| | Stage 0 (rank 0) | Stage 1 (rank 1) |
| --- | --- | --- |
| Parameters | `blk0.*`, `blk1.*` (8 tensors) | `blk2.*`, `blk3.*` (8 tensors) |
| Data it reads | the input microbatches $x_m$ | the target microbatches $t_m$ |
| Computes | $a_m = \mathrm{blk1}(\mathrm{blk0}(x_m))$ | $y_m = \mathrm{blk3}(\mathrm{blk2}(a_m))$, $L_m = \sum (y_m - t_m)^2 / N$ |
| Sends | activation $a_m$ → stage 1 | gradient $g_m = \partial L/\partial a_m$ → stage 0 |
| Receives | $g_m$ | $a_m$ |

Per microbatch exactly **two tensors cross the boundary**, both `[256, 2, 512]` FP32 = 1 MiB: $a_m$ forward and $g_m$ backward. Parameters, targets, the loss and the parameter gradients never cross (with DP = 1 nothing reduces them).

**Timing assumptions** (the simulator's defaults; every number below depends on them): every stage takes t_F = 1 time unit for one microbatch's forward and t_B = 2 for its backward (dX and dW matmuls ≈ 2× forward), communication takes 0, and each stage runs its ops in a fixed order, starting an op as soon as the stage is free and its input has arrived.

**P1.** In both schedules, at what time can stage 1 start F0, and why can it not start at t = 0?

**P2 (GPipe).** GPipe runs F0 F1 F2 F3, then B0 B1 B2 B3, on both stages. Fill in this grid (one cell per time unit; `.` = idle). When does stage 0 start B0?

```text
t        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
stage 0 F0  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
stage 1  .  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?
```

**P3 (1F1B).** Megatron's non-interleaved 1F1B lets stage s run $\min(P - s - 1, M)$ forwards (warmup), then alternates one forward and one backward of the **oldest** microbatch in flight (steady state), then runs the remaining backwards (cooldown). Write each stage's op order, then fill the same grid. What does stage 0 do between t = 2 and t = 4?

**P4.** Count idle cells per stage in each grid, and the peak number of microbatches **in flight** on each stage (defined in §3.4: from the start of F(m) to the end of B(m) on that stage).

<details><summary>Answers P1–P4</summary>

**P1.** At t = 1. F0 on stage 1 needs $a_0$, which exists only after stage 0's F0 ends at t = 1. This start-up delay is the first part of the bubble.

**P2. GPipe.**

```text
t          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
stage 0   F0 F1 F2 F3  .  .  . B0 B0 B1 B1 B2 B2 B3 B3
  in flight 1  2  3  4  4  4  4  4  4  3  3  2  2  1  1
stage 1    . F0 F1 F2 F3 B0 B0 B1 B1 B2 B2 B3 B3  .  .
  in flight 0  1  2  3  4  4  4  3  3  2  2  1  1  0  0
```

Stage 1 starts B0 at t = 5, right after its last forward. Stage 0 needs $g_0$, which arrives when stage 1's B0 ends at t = 7. Transfers: $a_0 \ldots a_3$ at t = 1, 2, 3, 4; $g_0 \ldots g_3$ at t = 7, 9, 11, 13.

**P3. 1F1B.** Warmup: stage 0 runs $\min(2 - 0 - 1, 4) = 1$ forward, stage 1 runs 0. Orders: stage 0 `F0 F1 B0 F2 B1 F3 B2 B3`, stage 1 `F0 B0 F1 B1 F2 B2 F3 B3`.

```text
t          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
stage 0   F0 F1  .  . B0 B0 F2 B1 B1 F3 B2 B2  . B3 B3
  in flight 1  2  2  2  2  2  2  2  2  2  2  2  1  1  1
stage 1    . F0 B0 B0 F1 B1 B1 F2 B2 B2 F3 B3 B3  .  .
  in flight 0  1  1  1  1  1  1  1  1  1  1  1  1  0  0
```

Between t = 2 and t = 4 stage 0 has done its steady-state forward (F1) and **waits for $g_0$**: its next op is B0, and stage 1's B0 ends at t = 4. It may not run F2 instead: that would break the 1F1B order and add a third microbatch in flight. With Megatron's calls the exchanges are: $a_0$ at t = 1; $a_1$+$g_0$ at t = 4; $a_2$+$g_1$ at t = 7; $a_3$+$g_2$ at t = 10; $g_3$ at t = 13 (§2.4 explains why $a_1$ travels together with $g_0$).

**P4.** Both schedules: makespan 15, **3 idle cells per stage**, 6 of 30 stage-cells idle = 20 % bubble. GPipe idle sits at [4, 7) on stage 0 and at [0, 1) + [13, 15) on stage 1; 1F1B moves stage 0's idle to [2, 4) + [12, 13). **Peak in flight: GPipe 4 and 4; 1F1B 2 and 1.** Mean in flight over the step (area under the row / 15): GPipe 2.8 and 2.0, 1F1B 1.733 and 0.8.

`python experiments/day12_pipeline_parallel/pp_sim.py` prints exactly these grids, and `pp_sim.py --check` asserts every number above (pre-test: `SIMULATOR CHECK: PASSED`).
</details>

### §2 Autograd across the process boundary

#### §2.1 Cutting the graph

On one GPU, `loss.backward()` walks one graph from the loss back to `blk0`. With two processes the graph is cut at $a_m$:

```text
rank 0:  x_m ──blk0──blk1──▶ a_m (non-leaf, has grad_fn)        ── data of a_m ──▶  rank 1:  r_m (leaf, requires_grad) ──blk2──blk3──▶ L_m
rank 0:  torch.autograd.backward(a_m, grad_tensors=g_m)  ◀── g_m = r_m.grad ──   rank 1:  L_m.backward()
```

The chain rule does not care where the cut is: $\partial L/\partial \theta_0 = (\partial a/\partial \theta_0)^\top \cdot \partial L/\partial a$. Stage 1 computes the vector $\partial L/\partial a$ (it is `r_m.grad`) and sends it. Stage 0 calls `torch.autograd.backward(a_m, grad_tensors=g_m)`, which runs its part of the graph as a vector–Jacobian product with that vector. Only data crosses: `send` copies the numbers of $a_m$, never its `grad_fn`.

Run the scalar version (CPU, any machine with torch; pre-test output in Appendix A):

```bash
python experiments/day12_pipeline_parallel/boundary_demo.py
```

Stage 0 computes $h = w_0 \cdot x$ (w0 = 3), stage 1 computes $y = w_1 \cdot h$ (w1 = 2) and $L = \sum_m y_m^2/2$, with $x_0 = 1$, $x_1 = 2$.

**P5.** Work out by hand what crosses each way for both microbatches ($a_0, a_1, g_0, g_1$) and the final `w0.grad`, `w1.grad`. Then predict `w0.grad` for two mistakes: (a) stage 0 pairs $g_0$ with microbatch 1's saved graph and $g_1$ with microbatch 0's; (b) stage 1 forgets `requires_grad` on the received value.

<details><summary>Answer P5</summary>

$a_0 = 3$, $a_1 = 6$; $g_m = \partial L/\partial a_m = y_m \cdot w_1$: $g_0 = 6 \cdot 2 = 12$, $g_1 = 12 \cdot 2 = 24$. $\texttt{w1.grad} = y_0 a_0 + y_1 a_1 = 18 + 72 = 90$; $\texttt{w0.grad} = g_0 x_0 + g_1 x_1 = 12 + 48 = 60$. These equal the uncut graph.

(a) $12 \cdot x_1 + 24 \cdot x_0 = 48$. Same shapes, no error, wrong number: this is the silent bug the `lifo_backward` control plants in the real model. (b) `r.grad` is `None`: stage 1's own $\texttt{w1.grad} = 90$ is right, and stage 0 has nothing to backpropagate. The demo prints `w0.grad = None`.
</details>

#### §2.2 What each stage keeps for a microbatch in flight

| Stage | Kept by the schedule per microbatch m | Why | Kept inside autograd (saved tensors) |
| --- | --- | --- | --- |
| 0 | $a_m$, its output | the root of `backward(a_m, grad_tensors=g_m)`; its `grad_fn` is the graph | each block's fc1 input, gelu input, fc2 input |
| 1 | $r_m$, the received input (a leaf with `requires_grad=True`) | after backward, `r_m.grad` is the gradient to send back | the same per block, plus $y - t$ for the loss |
| 1 | $L_m$, the scalar loss | the root of `L_m.backward()` | — |

Why $r_m$ must be a leaf that requires grad: a freshly received buffer is a plain tensor. If it does not require grad, autograd never computes $\partial L/\partial r_m$. If it is not a leaf (for example `r_m * 1`), autograd does not keep its `.grad` unless you call `retain_grad()`. `received.requires_grad_()` makes it both. Megatron does the same in two places: the receive buffer is created with `requires_grad=True`, and `backward_step` still calls `retain_grad()` on its inputs, which handles a non-leaf input (Block 3 Stops 4–5).

Rough sizes (derived from the code, FP32, microbatch of b sequences): stage 0 keeps about **9b MiB** per microbatch (two blocks × (fc1 input 0.5b + gelu input 2b + fc2 input 2b) minus the data input, which is already on the GPU, plus the stage output), stage 1 about **9.5b MiB**. The full batch (b = 8) is ≈ 72 MiB on stage 0. Appendix A checks these against the CUDA allocator.

#### §2.3 Matching forward and backward microbatch IDs

Nothing on the wire says which microbatch a tensor belongs to. The match comes from three orders agreeing:

1. **The channel is FIFO.** The k-th tensor rank 0 sends to rank 1 is the k-th tensor rank 1 receives from rank 0 (per direction).
2. **Both stages run their backwards in the same microbatch order** (ascending in both schedules here), so the k-th gradient stage 1 sends is for the k-th microbatch stage 0 expects.
3. **Each stage keeps its saved microbatches in a FIFO queue** and takes the **oldest** for each backward (`deque.popleft()`; Megatron: `input_tensors.pop(0)`).

The provided `P2P` helpers take an `mb` label that is **logged, not sent**. The checker's `labels` check compares the sender's label sequence with the receiver's. A schedule that pops the newest entry instead (`lifo_backward`) passes every shape check, and the labels, order and stage-0 gradients fail.

**Loss weight.** Each microbatch's loss is its share of the full-batch mean, $\sum (y_m - t_m)^2 / N$ with N the elements of the whole batch (`mb_loss`, Day 04's n_m / N rule). Summed over the M microbatches it is exactly the full-batch loss, so the gradients do not depend on M. Megatron divides each microbatch's loss by `num_microbatches` (Block 3 Stop 3). The `mean_per_microbatch` control uses each microbatch's own mean and scales every gradient by M.

#### §2.4 The send/recv order and deadlock

A blocking point-to-point call may wait until the peer posts the matching call (rendezvous). Look at 1F1B's steady state on two stages after stage 0 runs F1 and stage 1 runs B0:

- stage 0 wants to **send $a_1$** and **receive $g_0$**;
- stage 1 wants to **send $g_0$** and **receive $a_1$**.

**P6.** If each rank posts its send first and its receive second, as separate blocking calls, what happens? Which two calls are the ranks stuck in? Why does GPipe never face this?

<details><summary>Answer P6</summary>

Both ranks block in their send, each waiting for the other to post a receive: a deadlock. The simulator shows it under the rendezvous model:

```bash
python experiments/day12_pipeline_parallel/pp_sim.py --schedule 1f1b --comm-style naive --send rendezvous --show-programs
```

```text
program stage 0: F0 send(a0) F1 send(a1) recv(g0) B0 F2 ...
program stage 1: recv(a0) F0 B0 send(g0) recv(a1) F1 B1 ...
DEADLOCK: stage 0 blocked in send_forward waiting for send(a1); stage 1 blocked in send_backward waiting for send(g0)
```

With buffered sends the same program completes (`--send buffered`, the default), so the bug can hide on one machine and hang on another. The self-check asserts both. GPipe's phases are one-way: all activations flow forward, then all gradients flow back, so a receive is always already waiting.

Two fixes:

1. **Post both ops together** and wait for both: `send_forward_recv_backward` / `send_backward_recv_forward`, implemented with one `dist.batch_isend_irecv([...])`. Megatron's default (`batch_p2p_comm=True`). The provided `P2P` helpers do this, and the simulator's Megatron program shows the pairs as `send(a1)+recv(g0)` and `send(g0)+recv(a1)`.
2. **Order by rank parity**: even ranks send then receive, odd ranks receive then send. Megatron's unbatched path `_p2p_ops` does this (Block 3 Stop 5).

A pairing that looks odd at first: $a_1$ is sent together with the receive of $g_0$, and stage 1 receives it together with sending $g_0$. In the two-stage grid that exchange completes at t = 4, when stage 1 finishes B0. Stage 1 does not need $a_1$ before t = 4, so nothing is lost.
</details>

### §3 GPipe vs 1F1B on the same timeline

Same P = 2, M = 4, t_F = 1, t_B = 2, comm 0, as in §1.

#### §3.1 GPipe: fill, all forwards, turn around, all backwards, drain

Stage s cannot start before $s \cdot t_F$ (fill). After its last forward it waits until the gradients start arriving from the stages below it. Then it runs all backwards. The stages finish at different times, and the step ends when stage 0 finishes its last backward.

#### §3.2 1F1B: warmup, steady state, cooldown

| Phase | Stage s does | P = 2, M = 4: stage 0 | stage 1 |
| --- | --- | --- | --- |
| warmup | $w = \min(P - s - 1, M)$ forwards | F0 | — |
| steady | $M - w$ pairs: forward of the newest, backward of the oldest | F1 B0, F2 B1, F3 B2 | F0 B0, F1 B1, F2 B2, F3 B3 |
| cooldown | the `w` backwards still owed | B3 | — |

The last stage has no warmup: it can backward a microbatch as soon as it has its loss. Each earlier stage runs one more warmup forward than the stage after it, which is exactly enough to keep the stages below it busy while the first gradient travels back.

#### §3.3 Counting idle slots

With equal stages and free communication both schedules take

**T = (M + P − 1)(t_F + t_B)**

Each stage does M (t_F + t_B) of work, so each stage idles **(P − 1)(t_F + t_B)**, and the bubble fraction is **(P − 1) / (M + P − 1)**. For GPipe you can read the derivation off the grid. Stage s idles $s \cdot t_F$ before its first forward, $(P - 1 - s)(t_F + t_B)$ between its last forward and its first backward, and $s \cdot t_B$ after its last backward while the stages above finish. The sum is $(P - 1)(t_F + t_B)$ for every s. 1F1B only moves the idle cells around (§1 P4). The simulator asserts the formula for P = 2..4 and M = 1..8.

**How to count on a grid.** Count `.` cells per stage row up to the makespan, multiply by the tick, and divide the sum by P × makespan. For §1 that is 3 + 3 = 6 of 30, or 20 %.

#### §3.4 What "in flight" means and how to count it

**Definition (used by the simulator and the checker).** Microbatch m is *in flight* on stage s from the **start of F(s, m)** to the **end of B(s, m)**. During that window the tensors autograd saved for B(s, m), plus (on stages > 0) the received input, live on stage s. Counting rule: +1 when a forward starts, −1 when a backward ends.

**P7.** Derive the peak in-flight count on stage s for both schedules for general P and M. Does the peak depend on t_F and t_B?

<details><summary>Answer P7</summary>

- **GPipe: M on every stage.** All forwards run before any backward.
- **1F1B: min(P − s, M).** Warmup leaves $w = \min(P - s - 1, M)$ in flight. The first steady forward adds one before the first backward removes one. After that each (F, B) pair keeps the count constant, and cooldown only lowers it. For P = 2 that is 2 on stage 0 and 1 on stage 1.
- **The peak depends on the order alone, not on the times.** The count changes only at the stage's own ops, which run in a fixed order. Times change *how long* each microbatch stays resident (mean in flight, max residency), not the peak. §4 shows this with a slow stage.

The checker reads the same count from your run. Each rank's `in flight after each op` line must reach the simulator's peak: `1 2 1 2 1 2 1 0` on rank 0 and `1 0 1 0 1 0 1 0` on rank 1 for 1F1B at M = 4.
</details>

#### §3.5 When 1F1B lowers residency without lowering the bubble

Both schedules do the same work under the same dependencies, and with equal stages and free communication their critical paths have the same length, so **the idealized bubble is identical**. 1F1B only changes *when* each backward runs: it runs a microbatch's backward as early as the order allows, which frees that microbatch's activations early. So:

- **M ≤ P − s:** the peak in-flight count on stage s is M for both schedules. At M = 2 and P = 2, stage 0 still holds 2 while stage 1's peak is already down to 1 with 1F1B.
- **M > P − s:** GPipe holds M microbatches on stage s and 1F1B holds at most P − s. Increasing M past this point leaves the 1F1B peak batch count fixed. At a fixed effective batch size, each microbatch also gets smaller, reducing the peak activation bytes.
- Reducing the bubble itself needs something else: more microbatches, or the **interleaved** schedule (several virtual stages per rank), which Megatron implements in `forward_backward_pipelining_with_interleaving`. Not today.

#### §3.6 Changing M while the effective batch stays fixed

**P8.** Keep 8 sequences per step and go from M = 1 to 2, 4, 8. Predict, for each M: the bubble fraction, the peak in-flight count per stage for both schedules, the **peak activation bytes** on stage 0 as a fraction of the full batch's (one microbatch is 1/M of the batch), and whether the loss and gradients change.

<details><summary>Answer P8</summary>

`pp_sim.py --sweep 1,2,4,8,16`. Microbatch time = full-batch time / M, an ideal assumption with no per-microbatch overhead.

```text
    M  step time   bubble  idle/stage | GPipe peak    act |  1F1B peak per stage    act per stage
    1     2.000x    0.500      1.000x |          1   1.00 |               [1, 1]       [1.0, 1.0]
    2     1.500x    0.333      0.500x |          2   1.00 |               [2, 1]       [1.0, 0.5]
    4     1.250x    0.200      0.250x |          4   1.00 |               [2, 1]      [0.5, 0.25]
    8     1.125x    0.111      0.125x |          8   1.00 |               [2, 1]    [0.25, 0.125]
   16     1.062x    0.059      0.062x |         16   1.00 |               [2, 1]   [0.125, 0.062]
```

- **Bubble** falls as (P − 1)/(M + P − 1). The step time, in units of one stage's full-batch forward + backward, falls toward 1.
- **GPipe's peak activation bytes stay at the whole batch's** (M microbatches × 1/M each). **1F1B's fall as min(P − s, M)/M.**
- **Numerics.** The $n_m / N$ weight preserves the mathematical loss and gradients for every M. In both measured runs the worst relative error against the full-batch baseline was `6.98e-7`. Against single-GPU accumulation at the same M, the output and all 16 gradients were bitwise equal. The reported scalar loss at M = 8 differed by `8.30e-8` relative due to the final FP32 summation.
- **What the ideal model leaves out.** Smaller microbatches run the GPUs less efficiently, and this small model is already launch-bound. The number of P2P transfers grows to 2M per boundary per step, each 1/M the size. In practice that limits how far M can usefully go.
</details>

### §4 Optional: a slow stage, and communication time (simulated only)

```bash
python experiments/day12_pipeline_parallel/pp_sim.py --stage-times 1:2,2:4
```

Stage 1 is twice as slow (t_F = 2, t_B = 4). Pre-test output, identical for both schedules in makespan:

```text
GPipe   stage 0  F0 F1 F2 F3  .  .  .  .  .  .  .  .  . B0 B0  .  . B1 B1  .  . B2 B2  .  . B3 B3
        stage 1   . F0 F0 F1 F1 F2 F2 F3 F3 B0 B0 B0 B0 B1 B1 B1 B1 B2 B2 B2 B2 B3 B3 B3 B3  .  .
1F1B    stage 0  F0 F1  .  .  .  .  . B0 B0 F2  .  .  . B1 B1 F3  .  .  . B2 B2  .  .  .  . B3 B3
        stage 1   . F0 F0 B0 B0 B0 B0 F1 F1 B1 B1 B1 B1 F2 F2 B2 B2 B2 B2 F3 F3 B3 B3 B3 B3  .  .
makespan 27; idle stage 0 = 15 (55.6 %), stage 1 = 3 (11.1 %); peak in flight GPipe [4, 4], 1F1B [2, 1]
max residency on stage 0: GPipe 24, 1F1B 14
```

- **The slow stage sets the pace.** Makespan = M·(t_F + t_B) of the slow stage + one pass through the fast one = 4·6 + 3 = 27. The fast stage idles 55.6 % of the step in both schedules. The "bubble fraction" (1/3) now mixes fill/drain with **imbalance**, which no schedule removes. Only re-partitioning the layers does.
- **Peaks are unchanged** (order-determined, §3.4), but GPipe's stage 0 keeps microbatch 0 resident for 24 time units, against 14 for 1F1B.
- With `--send rendezvous`, GPipe's stage 0 can no longer run ahead (F2 at t = 3, F3 at t = 5): each send waits for the slow receiver. The makespan stays 27.
- **Communication time.** With `--comm 1/2` (P = 2, M = 4): GPipe 16 vs 1F1B 17. In 1F1B's steady state each backward on stage 0 waits for a round trip through stage 1, so P2P latency lands on the critical path more often. **[simulated]** This model does not measure it on GPUs.

---

## Block 2: Code (about 1.5 h)

Write `my_pp.py` in three steps, testing each one. The P2P helpers, the event log and the checker are provided. You write the autograd glue and the two schedules. While a TODO remains, `check_pp.py` prints `INCOMPLETE (TODOs left): ...` before any communication, and each rank exits with code 3. torchrun reports `exitcode: 3` per rank and itself returns 1.

### Stage 0 (15 min): read the harness and make the baseline

In `pp_common.py`, read `Block`, `stage_blocks`, `microbatches` (why contiguous copies?), `mb_loss` (its docstring is §2.3), the `P2P` docstring (labels are logged, not sent; buffers do not require grad), `Stage.forward_step` / `Stage.backward_step` (the **logged** wrappers your schedules call) and `EventLog.compute` (where the in-flight count changes). Then:

```bash
D12=experiments/day12_pipeline_parallel
python $D12/pp_sim.py --check                                                  # SIMULATOR CHECK: PASSED
python $D12/boundary_demo.py                                                   # BOUNDARY DEMO: PASSED
CUDA_VISIBLE_DEVICES=0 python $D12/baseline_1gpu.py --run-id try1              # BASELINE: SAVED
```

The baseline table lists, per M, the loss, whether a repeat is bitwise identical, the difference from M = 1, and the CUDA peak of plain accumulation (one microbatch of the whole model at a time).

### Stage 1 (25 min): **YOU WRITE** `forward_step`, `backward_step` (TODOs 1a–1e)

```bash
torchrun --standalone --nproc_per_node=2 $D12/check_pp.py --impl mine --test boundary \
    --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt --run-id try1
```

A provided loop runs one microbatch (M = 1): F on stage 0, F on stage 1, B on stage 1, B on stage 0. **Check:** `order F0 B0` on both ranks. The `boundary:` line on rank 1 must read `stage input is_leaf=True requires_grad=True; ... input grad sent back: [[256, 8, 512], 'torch.float32']`, and on rank 0 `input grad sent back: None`. Every tensor must be `ok`. In the pre-test every tensor was also bitwise equal to the M = 1 baseline. Last line: `BOUNDARY (mine, float32, schedules boundary, M 1): PASSED`.

### Stage 2 (20 min): **YOU WRITE** `run_gpipe` (TODOs 2a–2b)

```bash
torchrun --standalone --nproc_per_node=2 $D12/check_pp.py --impl mine --schedules gpipe --microbatches 4 \
    --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt --run-id try1
```

**Check:** order `F0 F1 F2 F3 B0 B1 B2 B3` on both ranks. `in flight after each op: 1 2 3 4 3 2 1 0` with peak 4. `P2P calls: send_forward(a0) ... recv_backward(g3)` on rank 0. `labels ... -> ok`. `RESULT mine gpipe M=4: PASSED`.

### Stage 3 (30 min): **YOU WRITE** `run_1f1b` (TODOs 3a–3d)

```bash
torchrun --standalone --nproc_per_node=2 $D12/check_pp.py --impl mine --schedules 1f1b --microbatches 4 \
    --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt --run-id try1
```

**Check:** rank 0 `F0 F1 B0 F2 B1 F3 B2 B3`, in flight `1 2 1 2 1 2 1 0`, peak 2, and P2P calls `send_forward(a0) send_forward_recv_backward(a1,g0) send_forward_recv_backward(a2,g1) send_forward_recv_backward(a3,g2) recv_backward(g3)`. Rank 1 `F0 B0 F1 B1 F2 B2 F3 B3`, peak 1, calls `recv_forward(a0) send_backward_recv_forward(g0,a1) ... send_backward(g3)`. If your calls differ from the simulator's Megatron-style program, the checker prints the program as info. Order, peak and labels are the required checks. Then the whole matrix:

```bash
torchrun --standalone --nproc_per_node=2 $D12/check_pp.py --impl mine --schedules gpipe,1f1b --microbatches 1,2,4,8 \
    --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt --run-id try1
```

Expect `GPipe vs 1F1B, every tensor on both ranks bitwise equal: M=1: True, ...` and the summary table (§ Block 4).

### Stage 4 (15 min): the negative controls

Write your predictions in notes §5 first, then:

```bash
torchrun --standalone --nproc_per_node=2 $D12/check_pp.py --impl ref --schedules 1f1b --microbatches 4 --controls \
    --baseline outputs/day12_pp/tensors/baseline_float32_try1.pt --run-id try1
```

| Control | One change (in `pp_controls.py`) | Must fail |
| --- | --- | --- |
| `detached_input` | stage 1 skips `requires_grad_()` on the received activation (sends zeros instead of crashing on `None`) | the 8 stage-0 gradients, ratio 0 |
| `lifo_backward` | 1F1B takes the **newest** saved microbatch for backward | 7 stage-0 gradients, `order`, `labels` |
| `mean_per_microbatch` | each microbatch uses its own mean (no 1/M weight) | `loss` and all 16 gradients, ratio 4.0 |

<details><summary>Why does <code>lifo_backward</code> leave <code>blk1.fc2.bias</code> correct? (found in the pre-test)</summary>

`blk1.fc2.bias` is the last bias on stage 0. Its gradient is the sum over all tokens of the gradient arriving at stage 0's output, $\sum_m g_m$, and it uses no saved activation. Pairing $g_m$ with the wrong microbatch's graph permutes the terms of that sum without changing it. Every other stage-0 gradient multiplies $g_m$ by something saved for one particular microbatch, so it fails. The assistant first predicted all 8, and the pre-test said 7. The expected set was corrected after that pre-test.
</details>

### Pitfalls (symptom → cause)

| Symptom | Cause |
| --- | --- |
| Hang for 180 s, then an NCCL timeout | One rank raised (read the **first** traceback), or the two ranks posted calls that never match (a recv without a send, or different microbatch counts) |
| Rank 0 hangs in `recv_backward` until the timeout, rank 1 finished | The received activation never required grad, so `input.grad` is `None`, and `send_backward(None, mb)` posts nothing (§2.2). The `detached_input` control sends zeros instead, which gives stage-0 gradients exactly 0 |
| `grad can be implicitly created only for scalar outputs` on rank 0 | `output.backward()` without the received gradient; use `torch.autograd.backward(output, grad_tensors=g)` |
| `labels -> FAIL`, stage-0 gradients wrong, no error | Backward took the wrong saved microbatch (LIFO, or a wrong index) |
| All gradients exactly ×M | Per-microbatch mean instead of `mb_loss` |
| `order` FAIL with a correct result | Your schedule is valid but not the one the simulator models (for example 1F1B with an extra warmup forward). Compare with `pp_sim.op_order` |
| `RuntimeError: Trying to backward through the graph a second time` | Two backwards on one saved microbatch, or a microbatch popped twice |
| Exit code 3 / `INCOMPLETE` | A function still contains `raise NotImplementedError` |

---

## Block 3: Read Megatron Core (about 45 min; Stops 1–5 required, Stop 6 optional)

**Pinned version: Megatron-LM `core_v0.19.2`, commit `4b4acac9a1d28ea6829c8d4f566d75698a21249d`**, the candidate pin tested on Day 11. The pipeline files were re-read for this guide, and every line number below was produced by `find_symbols.py` on that checkout. On Windows, check out only `megatron/core` (some test paths exceed the path limit):

```bash
git clone --no-checkout --depth 1 --branch core_v0.19.2 https://github.com/NVIDIA/Megatron-LM.git
```

```bash
cd Megatron-LM && git restore --source=HEAD -- megatron/core
```

```bash
python ../experiments/day12_pipeline_parallel/find_symbols.py .
```

Paths are relative to `megatron/core/`. Read in this order; each stop names the step of yours it corresponds to.

### Stop 1 (5 min): which schedule runs. `pipeline_parallel/schedules.py`

- `get_forward_backward_func` `:48`. With `pp_size > 1` (`:156`) and no virtual stages it returns `forward_backward_pipelining_without_interleaving` (`:160`), with virtual stages the interleaved one (`:158`), and with PP = 1 `forward_backward_no_pipelining` (`:162`, defined `:696`, its microbatch loop at `:788`).
- **Maps to:** `forward_backward_no_pipelining` is your `baseline_1gpu.py` accumulation loop. **There is no GPipe schedule in Megatron Core.** Your `run_gpipe` is the comparison point, not a Megatron path.
- With PP = 2 and no virtual stages, the selected function corresponds to `run_1f1b`. A virtual stage is a logical model chunk. For example, with two chunks per GPU, four chunks in model order can be placed on ranks `0 → 1 → 0 → 1`, so each GPU serves two positions in the pipeline.

### Stop 2 (15 min): the schedule itself. `forward_backward_pipelining_without_interleaving` `:2120`

| Lines | What happens | Your step |
| --- | --- | --- |
| `:2169` | `P2PCommunicator(pp_group=..., config=config)` | the provided `P2P` |
| `:2243–2245` | `num_warmup_microbatches = total_stages - current_stage - 1`, capped at M; `remaining = M - warmup` | TODO 3a |
| `:2299–2334` | warmup: `recv_forward` `:2310` → `forward_step` `:2313` → `send_forward` `:2328` → `input_tensors.append` / `output_tensors.append` `:2332` → `deallocate_output_tensor` `:2334` | TODO 3a, saving `(mb, input, output)` |
| `:2336–2342` | "Before running 1F1B, need to receive first forward tensor" | TODO 3b |
| `:2344` | steady loop: `forward_step` `:2356` → `send_forward_recv_backward` `:2382` → append `:2387` → `input_tensors.pop(0)` / `output_tensors.pop(0)` `:2393–2394` → `backward_func` `:2402` → `send_backward` (last iteration, `:2408`) or `send_backward_recv_forward` `:2412` | TODO 3c; FIFO pop = §2.3 |
| `:2416–2440` | cooldown: `pop(0)` `:2429` → `recv_backward` `:2432` → backward `:2436` → `send_backward` `:2440` | TODO 3d |
| `:2240`, `:2400`, `:2427` | `disable_grad_sync()` at the start, `enable_grad_sync()` only before the last backward | Day 04's `no_sync()`: with DP > 1 the gradient reduction runs once per step |
| `:2459` | `finalize_model_grads_func(...)`: DP gradient reductions and the embedding all-reduce between the first and last stage | not needed at DP = 1 with untied embeddings |

*Q: Megatron's steady state pops the oldest entry while the newest was just appended (`:2387` then `:2393`). With P = 2, how many entries does stage 0's list hold right before the pop, and how does that equal the peak from §3.4?*

### Stop 3 (5 min): one forward and the loss weight. `forward_step` `:397`, `forward_step_calc_loss` `:290`

- `set_input_tensor(input_tensor)` `:501`: the received activation is handed to the model through a setter, not as a `forward` argument. The model's first layer on a non-first stage reads it instead of the embedding output. **Maps to:** the non-first-stage branch of your `forward_step` (TODO 1a).
- `output_tensor /= num_microbatches` `:335` (and the legacy path `:341`, after `*= cp_group_size` at `:340`). **Maps to:** `mb_loss` (§2.3), and the `mean_per_microbatch` control when it is missing.

### Stop 4 (10 min): one backward. `backward_step` `:532`

- `x.retain_grad()` `:554` on every input tensor.
- `torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])` `:574`. On the last stage `output_tensor_grad` is `None`, and backward starts from the scalar loss. **Maps to:** TODO 1d.
- `input_tensor_grad.append(x.grad)` `:584`: the gradient returned to the schedule and sent back. **Maps to:** TODO 1e.
- `deallocate_output_tensor` `:166`: after `send_forward`, `out.data = torch.empty((1,), ...)` `:196` frees the **data** of the output and keeps its `grad_fn`. `torch.autograd.backward` would then reject the shape mismatch, so with `deallocate_pipeline_outputs=True` Megatron calls `custom_backward` (`:572`, defined `:199`), which drives the C++ engine directly (`run_backward` `:220`).

*Q (optional): in your code, how many MiB per in-flight microbatch would deallocating stage 0's output save at M = 4, and why does stage 0 still need the output's `grad_fn`? (One `[256, 2, 512]` FP32 tensor; §2.2.)*

### Stop 5 (10 min): the P2P layer. `pipeline_parallel/p2p_communication.py`

- `P2PCommunicator` `:140`. `next_rank` / `prev_rank` `:158–159` are **global** ranks of the PP neighbours, from the PP group. On Day 13 they are ranks 0 ↔ 2 and 1 ↔ 3, not 0 ↔ 1.
- `recv_forward` `:424`, `recv_backward` `:455`, `send_forward` `:486`, `send_backward` `:507`, `send_forward_recv_backward` `:527`, `send_backward_recv_forward` `:560`: the same six helpers you call, each a no-op where there is no neighbour.
- `_communicate` `:275`. Receive buffers are created with `requires_grad=True` (`:333`, `:341`), which is your `received.requires_grad_()`. The buffer shape comes from `get_tensor_shapes` (`schedules.py:2088`): `(seq_length, micro_batch_size, hidden_size)`, the `[s, b, h]` you send, divided by CP and, with sequence parallelism, by TP.
- `elif config.batch_p2p_comm:` `:374` → `_batched_p2p_ops` `:17` → `torch.distributed.batch_isend_irecv(ops)` `:49`: all ops of one call posted together (§2.4, fix 1). With `batch_p2p_comm=False`, `_p2p_ops` `:55` orders by rank parity (`:79`, fix 2), and for a 2-rank group it puts one direction on the global group so the two directions can overlap.
- Defaults in `model_parallel_config.py` (`ModelParallelConfig` `:46`): `pipeline_dtype` `:376` (must be set for PP), `variable_seq_lengths` `:379`, `batch_p2p_comm = True` `:390`, `deallocate_pipeline_outputs = False` `:405`.

### Stop 6 (optional, 5 min): for Day 13. `parallel_state.py`

`initialize_model_parallel` `:601` builds the PP groups, and `get_pipeline_model_parallel_group` `:1749` returns the one the schedule passes to `P2PCommunicator` (`schedules.py:2169`). At `DP=1, TP=2, PP=2` the PP groups are `[0, 2]` and `[1, 3]`. On Day 13, look in the trace for a point-to-point NCCL kernel between those ranks, of shape `[s, b, h]`, issued under `send_forward_recv_backward` or `send_backward_recv_forward`. Its order on each stage should match the simulator's 1F1B order for P = 2 and Day 13's microbatch count.

---

## Block 4: Run + summarize (30 min)

### Commands (the manual run)

```bash
cd /workspace/AI-Infra-Learning && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
D12=experiments/day12_pipeline_parallel
RID=manual_$(date -u +%Y%m%dT%H%M%SZ)
bash $D12/run_server_once.sh "${RID}_ref" --impl ref          # ~30 s; reference, M = 1,2,4,8, + 3 controls
bash $D12/run_server_once.sh "${RID}_mine" --impl mine        # ~30 s; your code (after Steps 1-3 pass)
```

`--quick` limits the parity matrix to M = 4. Stages, in order (each logged to `outputs/day12_pp/session_<RID>/<stage>.log`): `environment` (versions, `nvidia-smi topo -m`, NVLink, a project snapshot), `sim_check`, `sim_example`, `sim_sweep`, `sim_unequal`, `sim_deadlock`, `boundary_demo`, `baseline` (GPU 0 only), `boundary`, `parity` (both GPUs; controls for `ref`), `verify_core`. Per-rank event logs go to `outputs/day12_pp/events_<RID>/<impl>_<schedule>_M<M>.jsonl` (one JSON object per event: `seq`, `rank`, `stage`, `type` compute/comm, `op`/`mb`/`in_flight`/`start_ms`/`end_ms` or `call`/`ops`). The archive is `outputs/day12_exports/day12_<RID>.tar.gz` with a `SHA256SUMS` manifest. The baseline tensors (`outputs/day12_pp/tensors/`, about 185 MB) stay on the server, and the baseline JSON records their seed hash. **The runner never stops the instance.**

**Expected:** `SIMULATOR CHECK: PASSED`, `DEADLOCK` in `sim_deadlock.log`, `BOUNDARY DEMO: PASSED`, `BASELINE: SAVED`, `BOUNDARY (...): PASSED`, `RESULT ... PASSED` for all 8 cases, `CONTROLS: ALL DETECTED AS PREDICTED` (ref), and `OUTPUT CHECKS PASSED: ...`. Exit code 0.

### Download to Windows, then stop the instance

```powershell
experiments/day12_pipeline_parallel/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>_ref
```

```powershell
experiments/day12_pipeline_parallel/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>_mine
```

Each call checks the archive's SHA-256, extracts it to `outputs/day12_downloads/<RID>/`, and re-hashes every manifest entry. Look for `Manifest OK` and `exit_code=0` before you stop the instance (from the Vast console, or from inside with `vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY`).

### How to read the logs

1. **`parity.log`, one block per case.** Read in this order:
   - `order` vs `simulator`: the schedule ran in the modelled order.
   - `in flight after each op` and `peak`: the measured count from your schedule's own ops. Compare with the simulator's in-flight row at the matching ops.
   - `P2P calls`: which calls were paired. Compare with `sim_example.log`'s `program stage s` lines.
   - `GPU ms since t0` gives the start and end of each F/B on that rank's compute stream (CUDA events, t0 right after a barrier). A gap before a backward on rank 0 can include waiting for `g`, communication and launch overhead. Use the timestamps to locate dependencies in the schedule. Isolating communication time or benchmarking throughput would require separate measurements.
   - `labels`: four lists that must pair up.
   - The tensor table: `rel` vs accumulation with the same M and vs the full batch, `bitwise` for each, `ratio` vs the full batch.
2. **The summary table** (end of the parity block) puts the simulated bubble and peak next to the measured order/peak match, the worst error, the bitwise flag and the CUDA peak per rank.
3. **CUDA peak, what exactly it is.** `torch.cuda.max_memory_allocated() − torch.cuda.memory_allocated()` measured on each rank from just before the schedule to just after it (`reset_peak_memory_stats()` first). The parameters, the pre-allocated `.grad` buffers, the input/target microbatches and the output buffer were allocated **before** the reading starts, so it counts saved activations, received buffers and backward temporaries. NCCL's internal buffers are not allocated by PyTorch's caching allocator and are not in it. Compare it with "count × per-microbatch activations" (§2.2) and read the remainder as one microbatch's backward temporaries.
4. **`baseline.log`**: the single-GPU accumulation numbers. Its peak column is one microbatch of the **whole** model at a time.

### What to write in `notes.md`

§1 predictions (yours) against the answers, §2 run record, §3 simulated vs measured order/peak/bubble, §4 parity and the M sweep, §5 controls against your predictions, §6 the Megatron mapping in your words, §7 explanations. Label values **[measured]**, **[pre-test]**, **[simulated]**, **[derived]**, and causes the data does not isolate **[not isolated]**.

### Measured results (2026-11-15)

Both the reference and the completed implementation passed the M = 1 boundary check and the full FP32 parity matrix. Rank 1's received activation was a gradient-tracked leaf and returned a gradient of shape `[256, 8, 512]`. Both runs matched the simulator's order, in-flight peak and P2P sequence, and all activation and gradient labels paired. The archived `mine` source matches the local implementation after normalizing line endings. [Run record](notes.md#2-run-record) · [Schedule and P2P evidence](notes.md#3-simulated-vs-measured-schedule-m--4)

| M = 4 | Bubble [simulated] | Peak in flight, rank 0 / 1 [measured] | Incremental CUDA peak MiB, rank 0 / 1 [measured] |
| --- | ---: | ---: | ---: |
| GPipe | 20% | 4 / 4 | 81 / 84 |
| 1F1B | 20% | 2 / 1 | 45 / 28 |

At M = 4, 1F1B reduced the measured incremental peak by about 44% on rank 0 and 67% on rank 1. Rank 1 finishes each microbatch's backward before starting the next forward, so it keeps just one microbatch in flight. The GPU event logs show the order and where the gaps occur. The 20% bubble comes from the idealized simulator. [Full M sweep and memory definition](notes.md#4-parity-and-the-m-sweep-effective-batch-8--256-tokens-fixed)

At M = 8, the scalar `loss` differs from the same-M baseline because the checker reduces all detached loss scalars with `stack().sum()`, while the baseline adds them sequentially. Both comparisons pass the `1e-5` tolerance. The reference negative controls reproduced zeroed rank-0 gradients, the LIFO label mismatch and the fourfold loss/gradient scale error, respectively. [Negative controls](notes.md#5-negative-controls-1f1b-m--4)

---

## Self-check (answer aloud, < 2 min each)

<details><summary>1. A colleague's 1F1B matches single-GPU outputs and all stage-1 gradients, but stage-0 gradients are off by 10–80 % with no error. What do you check first?</summary>

The microbatch pairing on stage 0: does each backward use the saved graph of the microbatch whose gradient just arrived? Look at the oldest-vs-newest pop and the label sequences (the `lifo_backward` control). A missing `requires_grad` gives zeros instead, and a loss-weight bug scales stage 1 too.
</details>

<details><summary>2. Why is the received activation made a leaf that requires grad, rather than kept as it arrives?</summary>

Autograd computes and stores `.grad` only for tensors that require grad, and keeps it only on leaves unless `retain_grad()` is called. Stage 1 needs `∂L/∂r` to send back. Without it, stage 0's backward has no input.
</details>

<details><summary>3. With P = 4 and M = 16, how many microbatches does stage 0 hold at peak under GPipe and under 1F1B, and what is the bubble fraction of each?</summary>

GPipe 16, 1F1B min(4 − 0, 16) = 4. Both bubbles are (P − 1)/(M + P − 1) = 3/19 ≈ 15.8 % with equal stages and free communication.
</details>

<details><summary>4. Why must the steady-state send of an activation and receive of a gradient be one call?</summary>

At that moment the neighbour does the mirror image. Two blocking sends posted first wait for each other forever. Posting both ops together (`batch_isend_irecv`), or ordering by rank parity, makes the exchange always matchable. GPipe never exchanges in both directions at once.
</details>

<details><summary>5. You double M at a fixed effective batch. What happens to the bubble, to GPipe's and 1F1B's peak activation memory, and to the gradients?</summary>

The bubble shrinks as (P − 1)/(M + P − 1). GPipe's activation bytes stay at about the whole batch. Under 1F1B each microbatch is half the size and the peak count stays fixed once the stage has reached its $P - s$ cap, so the estimated live activation bytes then halve. The gradients are mathematically unchanged with the $n_m / N$ weight. Both measured runs matched single-GPU accumulation bitwise in every parameter gradient at the same M.
</details>

**Next (Day 13):** the same 1F1B inside Megatron Core at `DP=1, TP=2, PP=2` on four GPUs, with Day 11's TP all-reduce inside each stage and today's P2P between stages.

---

## Appendix A: pre-test results (`pretest1`, assistant run, 2026-11-14) [pre-test]

`bash experiments/day12_pipeline_parallel/run_server_once.sh pretest1 --impl ref`, 2026-11-14 06:40:49–06:41:17 UTC (28 s). These are **pre-test** numbers for comparison, not your measurements.

**Host.** Vast.ai container 60604495 (hostname `9c70fe432bf3`), 2 × **A100-SXM4-80GB**, NV12, GPU 0 on NUMA node 0 and GPU 1 on NUMA node 1, peer access both ways, driver 620.70.06. Python 3.12.15, torch 2.14.1+cu130, CUDA build 13.0, NCCL 2.30.7. Exit code 0: `OUTPUT CHECKS PASSED: simulator check, deadlock demo, boundary demo, baseline, boundary check, parity GPipe and 1F1B at M=[1, 2, 4, 8], 3 controls detected as predicted`. Archive SHA-256 `7f8914decee77490b5beac8f9902fd969d50dd993e8080c4ca7afac1d1f3f27a`, 46 manifest files re-verified on Windows.

Logs: [`outputs/day12_downloads/pretest1/session_pretest1/`](../outputs/day12_downloads/pretest1/session_pretest1/) ([environment](../outputs/day12_downloads/pretest1/session_pretest1/environment.log), [sim check](../outputs/day12_downloads/pretest1/session_pretest1/sim_check.log), [sim example](../outputs/day12_downloads/pretest1/session_pretest1/sim_example.log), [sweep](../outputs/day12_downloads/pretest1/session_pretest1/sim_sweep.log), [unequal](../outputs/day12_downloads/pretest1/session_pretest1/sim_unequal.log), [deadlock](../outputs/day12_downloads/pretest1/session_pretest1/sim_deadlock.log), [boundary demo](../outputs/day12_downloads/pretest1/session_pretest1/boundary_demo.log), [baseline](../outputs/day12_downloads/pretest1/session_pretest1/baseline.log), [boundary](../outputs/day12_downloads/pretest1/session_pretest1/boundary.log), [parity](../outputs/day12_downloads/pretest1/session_pretest1/parity.log)); event logs in [`events_pretest1/`](../outputs/day12_downloads/pretest1/events_pretest1/).

**`pretest1_mineplumbing`** checked the `--impl mine` path end to end. The reference was copied over a *temporary* `my_pp.py` in a separate directory on the server (`/workspace/mineplumb`), and `run_server_once.sh ... --impl mine --quick` passed the boundary check and both schedules at M = 4 (40 files verified, SHA-256 `b191662c75696b2b270f3f5020789814783fa1a6951ec8d905024c7e4604bd4c`). The completed local implementation was validated separately in the full November 15 run.

A CPU (gloo) run of the checker at M = 4 also passed on the server during development. It was not archived.

### A.1 Order, in-flight count, parity (FP32, effective batch 8 × 256 tokens)

| Schedule | M | Measured order and peak = simulator | Worst rel vs full batch (tensor) | Bitwise vs 1-GPU accumulation, same M | CUDA peak MiB rank 0, rank 1 |
| --- | ---: | --- | --- | --- | --- |
| GPipe | 1 | yes, peaks [1, 1] | 0 | all | 120.3, 120.3 ¹ |
| GPipe | 2 | yes, [2, 2] | 5.49e-7 (`blk0.fc2.weight`) | all | 90.0, 92.0 |
| GPipe | 4 | yes, [4, 4] | 5.49e-7 (`blk0.fc2.weight`) | all | 81.0, 84.0 |
| GPipe | 8 | yes, [8, 8] | 6.98e-7 (`y`) | all except `loss` ² | 78.5, 82.0 |
| 1F1B | 1 | yes, [1, 1] | 0 | all | 104.0, 104.0 |
| 1F1B | 2 | yes, [2, 1] | 5.49e-7 (`blk0.fc2.weight`) | all | 90.0, 56.0 |
| 1F1B | 4 | yes, [2, 1] | 5.49e-7 (`blk0.fc2.weight`) | all | 45.0, 28.0 |
| 1F1B | 8 | yes, [2, 1] | 6.98e-7 (`y`) | all except `loss` ² | 24.5, 16.0 |

GPipe vs 1F1B: every tensor on both ranks bitwise equal at every M. Labels paired in every case. The P2P call sequences equalled the simulator's programs.

¹ GPipe M = 1 runs the same order and the same calls as 1F1B M = 1 but read 16.25 MiB more on both ranks. It was the first case in the process. The boundary check, also first in its process, read 120.3 MiB on rank 0 too. Probably a one-time allocation on the first backward **[not isolated]**. Compare M = 1 numbers from the second schedule.

² The M = 8 loss differs from the baseline's by 8.3e-8 relative because the two harnesses add the 8 microbatch losses differently: the checker uses `torch.stack(...).sum()`, the baseline `loss += l_m` one at a time **[derived from the code]**. The gradients and `y` are bitwise equal.

**Memory, derived vs measured (stage 0).** Per-microbatch activations ≈ 9b MiB (§2.2). The estimate for saved activations is 72 MiB at every M for GPipe and 72, 72, 36, 18 MiB at M = 1, 2, 4, 8 for 1F1B. For M = 2, 4, 8, the measured peak minus that estimate is 18.0, 9.0, 6.5 MiB for either schedule. At M = 1, the remainder is about 48.3 MiB for the first GPipe case and 32.0 MiB for 1F1B. The experiment does not isolate the components of these remainders.

### A.2 Negative controls (1F1B, M = 4)

| Control | Failed (ratio vs full batch) |
| --- | --- |
| `detached_input` | all 8 stage-0 gradients (0.0) |
| `lifo_backward` | `blk0.*` (0.213, 0.850, 0.922, 0.957), `blk1.fc1.weight` (0.778), `blk1.fc1.bias` (0.889), `blk1.fc2.weight` (0.969), `order` (rank 0 `F0 F1 B1 F2 B2 F3 B3 B0`), `labels` (rank 1 sent g `[0, 1, 2, 3]`, rank 0 expected `[1, 2, 3, 0]`) |
| `mean_per_microbatch` | `loss` and all 16 gradients (4.0) |

All three detected as predicted. The `lifo_backward` set was corrected after a development run showed `blk1.fc2.bias` passing (Block 2 Stage 4).

### A.3 Single-GPU baseline

| M | Microbatch | Loss | Repeat bitwise | Worst grad rel vs M = 1 | Accumulation peak MiB |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | [256, 8, 512] | 5.746177673 | yes | 0 | 192.3 |
| 2 | [256, 4, 512] | 5.746177673 | yes | 5.49e-7 | 92.0 |
| 4 | [256, 2, 512] | 5.746177673 | yes | 5.49e-7 | 48.0 |
| 8 | [256, 1, 512] | 5.74617815 | yes | 6.52e-7 | 28.0 |

### A.4 What the pre-test supports and what it does not

- **Supported [pre-test]:** the two-stage pipeline, under both schedules and M = 1..8, reproduces single-GPU accumulation with the same M **bitwise** in `y` and all 16 parameter gradients, and the full-batch step to ≤ 7.0e-7 relative. Each rank's measured F/B order and peak in-flight count equal the simulator's. GPipe and 1F1B give bitwise-identical numbers. The three planted bugs fail exactly where predicted. Stage 0's CUDA peak is about halved by 1F1B at M = 4 (45 vs 81 MiB) and cut to about a third at M = 8 (24.5 vs 78.5 MiB).
- **Not measured:** the bubble on GPUs (the timestamps are not a benchmark; the bubble numbers are simulated), P2P bandwidth or latency, BF16, more than two stages, the interleaved schedule, Megatron's own pipeline (Day 13).
