# Day 14: a top-1 MoE layer with expert parallelism, EP=4 on one four-GPU host

> **Run status, 2026-11-27 [measured].** Both full runs passed on four A100-SXM4-40GB GPUs (NV12, NCCL). `mine` passed dispatch and all four EP=4 parity cases against the one-GPU reference. Worst relative errors were 5.93e-7 for `ref` and 6.13e-7 for `mine`, with bitwise-equal logits in every case. All five negative controls behaved as predicted.
>
> Both archives and their manifests were re-verified locally. The optional EP=2 stage used `ref` and passed in both sessions. The local-aux diagnostic differed from the global baseline for learned and ragged routing, as expected.
>
> [GPU reruns](#appendix-c-gpu-reruns-2026-11-27) · [Completed notes](notes.md) · [Historical GPU pre-test](#appendix-b-gpu-pre-test-pretest2-assistant-run-2026-11-22-pre-test) · [CPU pre-test](#appendix-a-local-cpu-results-assistant-run-2026-11-22-pre-test-cpu)

## Required to close Day 14 (about 4.5 h including the run)

| # | You can explain | Evidence | Where |
| --- | --- | --- | --- |
| 1 | **Routing and capacity on one GPU**: why the router needs a gradient path, and how tokens per expert, max/mean load and drop rate change between balanced and skewed inputs at the same token count. Also, why per-source capacity drops more than global capacity. | `route_stats.log` (Exp 1) | §1, Block 4 |
| 2 | **The dispatch**: how every rank learns its receive sizes, why there is exactly one device-to-host sync, which split lists the backward all_to_all uses, and what happens with zero-size splits | your `my_moe.py` passes `--tests dispatch`. `ep4_dispatch.log` | §2, Block 2 Stage 2 |
| 3 | **EP=4 parity**: outputs, losses, the input gradient and the router, expert and dense gradients all match the one-GPU baseline within the stated tolerances, for learned, balanced, ragged and all-to-one routing | `PARITY (...): PASSED` in `ep4_parity.log` with `--impl mine` | §3.1, Block 2 Stage 3 |
| 4 | **The negative controls**: why each one fails where it does, and why the presorted wrong inverse fails nothing | `CONTROLS (...): PASSED` in `ep4_controls.log` | Block 2 Stage 4 |
| 5 | **Skew at EP=4**: for balanced, hot and all-to-one, tokens received per rank, the largest receive buffer, expert time, dispatch/combine time, and the slowest rank's step time. Also, which phase on the *other* ranks absorbs the slow rank's work | `bench_ep4.log` and `bench_ep1.log` (Exp 3), notes §5 | §3.2, Block 4 |
| 6 | **Megatron's MoE path**: router → aux loss → capacity → dispatcher preprocess → all_to_all → experts → combine, and how DDP scales expert gradients | notes §6, filled from Block 3 Stops 1–6 | Block 3 |

**Done means**

- [ ] P1–P8 predicted in notes §1 before opening the answers. Corrections recorded.
- [x] `my_moe.py` TODO 1–5 written. `check_local.py --impl mine` passed all 29 CPU checks.
- [ ] Confirm both four-process CPU `check_ep.py --impl mine` stages. The GPU reruns verified their NCCL counterparts.
- [x] `run_server_once.sh --impl mine` finished with exit code 0 and `CORE_SUCCESS.txt`. The archive was downloaded and its manifest verified.
- [ ] Instance stopped (not destroyed) after the download. Console check recorded in notes §2.
- [x] Notes §3–§5 filled from the logs with [measured] labels. Notes §6 and §7 contain the Megatron mapping and the explanations from the walkthrough.

**Optional (not needed to close the day):**

- P9–P12
- Block 3 Stop 7
- EP=2 × EDP=2 parity (`ep2_parity`)
- Megatron's local aux scope (`aux_local`)
- the aux-loss training run (`aux_train`)
- the profiled step (`bench_trace`)
- top-2 routing, grouped GEMM, and shared or fine-grained experts. These are reading only. Nothing here measures them.

**Scope on purpose.**

- **Hardware and model.** One four-GPU host. The Day 01 GPT at the small preset (d 128, 2 layers, 8 × 64 tokens) is used for parity. One MoE layer at the medium and large shapes is used for timing.
- **Routing.** E = 4 experts, top-1, dropless at EP=4.
- **Numerics and parallelism.** FP32 with TF32 off, and TP = PP = CP = ETP = 1.
- **Not included.** No Megatron run: Megatron is read, not executed. No grouped GEMM kernel, no communication/compute overlap, no training curve. The timings come from one layer and are not a benchmark of MoE training.

---

## Time budget and cut order

| Block | README budget | Required | Optional / stretch |
| --- | --- | --- | --- |
| 1 Design | 30 min | P1–P8, about 4 min each with the small numbers given | P9–P12 (+20 min) |
| 2 Code | 2 h | Stage 0 read (10 min), TODO 1–2 (35 min), TODO 3–4 (40 min), TODO 5 + parity (20 min), controls (15 min) | — |
| 3 Read | 1 h | Stops 1–6 (50 min) + 10 min to fill notes §6 | Stop 7 (+15 min) |
| 4 Run + summarize | 30–60 min | upload (5), run `--impl mine` (2 min 40 s in the full rerun), download + stop (5), notes §2–§5 (30) | optional stages run in the same script |

**Cut order if you run out of time:**

1. Drop the optional stages: run with `--core`.
2. Drop Stop 7.
3. Drop P9–P12.
4. Run the GPU stages with `--impl ref` and defer the `--impl mine` GPU rerun. Your code must still pass on CPU.

The required rows above never move.

---

## What is new today (and what you already have)

You already have these, and today does not re-teach them:

- **Day 03**: gradient all-reduce and the 1/DP average.
- **Day 11**: autograd Functions that wrap a collective, with forward = one collective and backward = its transpose.
- **Day 12**: point-to-point exchanges whose sizes both sides must agree on.
- **Day 13**: rank groups from a rank-numbering rule, and the "every rank calls `new_group` for every group" rule.

New today:

- **Data-dependent communication.** On Days 11–13 every message size was known before the step started. Here the router decides at run time how many rows each rank sends to each other rank, and the sizes differ per step and per rank. The receivers must learn them, and the host must know them before it can enqueue the call.
- **One collective that moves different amounts to different peers.** This is uneven `all_to_all_single` with split lists, including zeros.
- **Parameters that are not replicated.** Expert weights live on one rank each, so the Day 03 rule "all-reduce every gradient" is wrong for them. Two gradient groups now exist: dense DP and expert DP.
- **Load as a correctness-free performance problem.** A skewed router gives the *same* numbers on four GPUs as on one, but it makes one rank do the work while three wait. Today you see *where* the waiting shows up.

---

## Concept map

Every concept from the planning list, classified. "Required" items are taught *and* exercised. "Reading-only" items are taught at a Block 3 stop and not exercised.

| Concept | Class | Why this class | Taught at | Exercised by |
| --- | --- | --- | --- | --- |
| Top-1 softmax gating, gradient path | required | the first thing that is easy to get silently wrong (P1) | §1.1, Stop 1 | TODO 1, `check_local` gating |
| Capacity factor, drop policy, drop-and-pad | required | brief evidence #1 (drop rate) | §1.2, Stop 3 | `route_stats` (Exp 1), `check_local` capacity |
| Collapse / load metrics (max/mean, CV, entropy, idle) | required | brief evidence #1 | §1.2 | `route_stats` |
| Permutation, stable sort, inverse | required | core of dispatch; one control depends on it | §1.3 | TODO 2, control `wrong_inverse` |
| Counts exchange, uneven a2a, `.tolist()` host sync | required | brief evidence #2 | §2.1, Stop 4 | TODO 3–4, `--tests dispatch` |
| Backward a2a with swapped splits | required | gradients must reach the source | §2.1, Stop 5 | TODO 4, parity `dx` |
| sort_chunks (src, expert) → (expert, src) | optional | a no-op at EP=4 (one expert per rank). It is needed when E_local > 1. | §2.1, Stop 5 | `ep2_parity` (optional) |
| Worst-case receive buffer | required | brief evidence #3 | §2.2 | `route_stats`, `bench_skew` |
| Dense DP vs EP vs EDP, gradient scaling | required | parity of every expert gradient | §3.1, Stop 6 | TODO 5, controls `expert_*` |
| Megatron's separate expert rank order | optional | EP=4 has EDP=1. It matters only with EDP > 1. | P10, Stop 6 | `ep2_parity` |
| Skew → slowest rank, waiting inside collectives | required | brief evidence #3 | §3.2 | `bench_skew` (Exp 3) |
| Aux load-balancing loss, AutoScaler | required | the standard fix for skew, and its gradient | §3.3, Stop 2 | `check_local` aux, the model (global scope) |
| Aux scope: local vs global counts | optional | Megatron's default differs from the one-GPU math | P9, Stop 2 | `aux_local` (optional) |
| z-loss | reading-only | used in the model at a fixed coefficient. Nothing to predict beyond its formula. | Stop 2 | (included in parity) |
| Aux-loss-free expert bias | reading-only | an alternative balancer, no gradient | Stop 7 | `route_stats` "learned+bias" shows a bias-induced skew only |
| Sinkhorn routing | reading-only | an alternative router, not used | Stop 7 | — |
| SequentialMLP vs TEGroupedMLP (grouped GEMM) | optional | performance only, nothing changes numerically | P11, Stop 7 | — |
| Shared / fine-grained experts | reading-only | FLOP accounting only | P11, Stop 7 | — |
| Comm/compute overlap (shared-expert overlap, NCCL stream) | reading-only | not measurable with one expert per rank and a loop | Stop 4–5 | — |
| Determinism (index_add vs scatter_add, top-2 collisions) | optional | explains bitwise vs ≤ tolerance | P12 | baseline sha repeatability |

---

## Read this first: the host

You need one four-GPU instance: a Vast.ai 4 × A100 with NV12 between all pairs, as on Day 13. Day 14 runs at FP32 with one expert per GPU, so any four GPUs work. `environment.log` records `nvidia-smi topo -m` and the P2P matrix. Timings are only compared within that log. The one-GPU baseline runs on `cuda:0` of the same host.

The November 27 reruns used container `62787249`, with four A100-SXM4-40GB GPUs, NV12 between all GPU pairs and full P2P access. The reference finished in 2 min 38 s and the mine run in 2 min 40 s. Record your host's topology alongside its timings. [Rerun record](#appendix-c-gpu-reruns-2026-11-27)

If your Day 13 instance was stopped and not destroyed, restart it. `/workspace/Megatron-LM` is then still there for the symbols stage. On the pre-test host it found all of the Block 3 anchors. Otherwise the stage is skipped, and Block 3 works from any checkout of `core_v0.19.2` (`find_symbols.py PATH`). The assistant's output for that commit is in [`outputs/day14_moe/cpu_check/symbols.md`](../outputs/day14_moe/cpu_check/symbols.md).

From your Windows project root, upload with `scp`, not git:

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/AI-Infra-Learning/experiments && ln -sfn /venv/main /workspace/AI-Infra-Learning/.venv'
```

```bash
scp -P <port> README.md requirements.txt root@<host>:/workspace/AI-Infra-Learning/
```

```bash
scp -P <port> -r day14 root@<host>:/workspace/AI-Infra-Learning/
```

```bash
scp -P <port> -r experiments/day14_moe experiments/day01_single_gpu root@<host>:/workspace/AI-Infra-Learning/experiments/
```

`moe_common.py` loads Day 01's `train.py` by path to build the batches, which is why `day01_single_gpu` goes too. The disk is 60 GB. The run writes about 40 MB plus logs and never saves full tensors at timing shapes.

---

## Files

See [experiments/day14_moe/README.md](../experiments/day14_moe/README.md) for the full table, the commands, the exit codes and the deviations from Megatron. The file you write is `my_moe.py`. The one you read first is `moe_model.py` (`MoELayer.forward`).

---

## Block 1: Design (30 min). Predict first, then open the answer

Write each prediction in notes §1 *before* opening its answer. The numbers are small on purpose. Each section ends with one **Question** that you answer in notes §7 after the run. None of them have a single-word answer.

The layer you are designing, for one rank r at EP=4 (rank r owns expert r). The boxes are numbered in execution order. §1.1–§1.3 cover steps 2, 3 and 9, and §2 covers steps 4–8.

![One top-1 MoE layer at EP=4, seen from rank r. Top row, left to right: 1 input x_r [T_r, d] with T_r = T/4 tokens; 2 router, local: logits = x_r Wᵀ + b, probs = softmax(logits), idx = argmax(probs), w = probs[t, idx_t]; 3 permute, local: order = stable argsort(idx), counts = bincount(idx), x_r[order] grouped by expert; 4 count exchange, all_to_all with equal splits of counts.view(EP, E_loc): rank r sends row r and receives column r; 5 host sync: one .tolist() device-to-host copy gives input_splits = row r and output_splits = column r, and the host waits for steps 2–4. The split lists go down as Python ints to the bottom row, right to left: 6 dispatch, uneven all_to_all_single of x_r[order] with in = input_splits and out = output_splits, zero-size splits allowed; 7 expert e = r, fc1, GELU, fc2 on the sum of output_splits rows, run even on 0 rows, with sort_chunks first when E_loc > 1; 8 combine, all_to_all_single with the splits swapped (in = output_splits, out = input_splits); 9 unpermute, y[order[k]] = y_sorted[k] via zeros.index_add, then multiply by w on rank r; 10 output y_r, added to the residual h (a dropped token adds 0). A dashed arrow from step 2 to step 9 marks w staying on rank r as the router's only task-loss gradient path. Inset left: the ragged send matrix [[128,0,0,0],[86,42,0,0],[0,32,0,96],[64,0,0,64]], rows = source, columns = destination; row 1 [86,42,0,0] is rank 1's input_splits, column 1 [0,42,32,0] its output_splits from step 4, so rank 1's expert computes 74 rows and rank 2 receives 0 rows but still enters every collective. Inset right, backward with the same split lists and no new count exchange: combine backward all_to_all(in = input_splits, out = output_splits); expert backward sums every source's rows on the owner; dispatch backward all_to_all(in = output_splits, out = input_splits); router gradient through w and the aux loss; finish_grads: dense all-reduce ÷DP, experts ÷DP with no reduce at EDP = 1. Per layer per step: 3 all_to_alls forward and 2 backward.](assets/d14_ep_flow.svg)

The figure is generated by [`assets/make_figures.py`](assets/make_figures.py): `python day14/assets/make_figures.py day14/assets`.

### §1.1 Routing: the combine weight is the gradient path

The router is a linear layer $\text{logits} = x W_r^\top + b$, with $\text{probs} = \mathrm{softmax}(\text{logits})$ over all E experts. Top-1 sends each token to $\text{idx} = \mathrm{argmax}(\text{probs})$. `argmax` is not differentiable: no gradient flows through *which* expert was chosen. The only way the task loss can teach the router is through the **combine weight**: $y = w \cdot \text{expert}_{\text{idx}}(x)$, with $w = \text{probs}[\text{idx}]$.

For one token, $\partial w/\partial \text{logit}_j = w \cdot (\delta_{j,\text{idx}} - p_j)$. The four entries sum to zero.

**P1.** Three tokens with logits `[2,1,0,0]`, `[0,0,3,0]` and `[1,1,1,1]`:

- (a) Find idx and w for each token.
- (b) Find $\partial w_0/\partial \text{logits}_0$.
- (c) Some implementations renormalize the chosen weights to sum to 1, as top-k routers do for k ≥ 2. For k = 1, what is w then, and what task-loss gradient does the router get?

<details><summary>Answer P1</summary>

(a) probs are:

- `[.6103 .2245 .0826 .0826]`
- `[.0433 .0433 .8700 .0433]`
- `[.25 .25 .25 .25]`

idx is `[0, 2, 0]` and w is `[.6103, .8700, .25]`. Token 2 is a four-way tie, and `argmax` returns the *first* maximum, expert 0. A forced pattern or a tie-break rule matters for reproducibility, and P12 comes back to this.

(b) $w_0 \cdot (e_0 - p) = .6103 \cdot [1 - .6103,\ -.2245,\ -.0826,\ -.0826] = [0.2378,\ -0.1370,\ -0.0504,\ -0.0504]$. The entries sum to 0.

(c) w = 1 for every token, so $\partial w/\partial \text{logits} = 0$, and the task loss gives the router **no gradient at all**. `check_local` measures a router-gradient norm of 2.155e-1 with the full softmax weight and exactly 0.0 renormalized. The router then learns only from the auxiliary losses. Megatron's default `moe_router_pre_softmax=False` takes the softmax *after* top-k, over the k selected scores. For k = 1 that is the softmax of one number, which is 1.0. Block 3 Stop 1 shows the line. This harness uses the pre-softmax weight on purpose. With k = 2 the renormalized weights still depend on two logits, and `check_local` top2 measures a non-zero gradient.

</details>

**Question 1.** In both reruns, the router assigned layer 0's 4096 medium-preset tokens as `[921, 909, 1008, 1258]` at initialization. A bias of +1.5 on expert 0 changed that to `[2900, 330, 410, 456]`. Why can one per-expert constant move 1979 tokens? Which score gap controls whether a token switches to expert 0? [Routing results](notes.md#3-experiment-1-routing-and-capacity-on-one-gpu)

### §1.2 Capacity: fixed buffers buy predictability with dropped tokens

A capacity-limited layer gives every expert room for $C = \lceil \text{cf} \cdot T k / E \rceil$ rows. Tokens beyond C are **dropped**. Their MoE output is 0, so the block's output for them is just the residual `h`. With **drop-and-pad** every expert computes exactly C rows, padded, so every buffer and every all_to_all has a fixed size: $E \cdot C$ rows whatever the routing.

The **position** policy keeps the first C tokens in token order. The **probs** policy, Megatron's default, keeps the C highest-probability ones. Load metrics, for counts n_i:

| Metric | Definition | Balanced | Collapsed |
| --- | --- | --- | --- |
| max/mean | max n_i / mean n_i | 1 | E |
| CV | std / mean | 0 | — |
| load entropy | normalized, Σ q_i log q_i / log E with q = n/T | 1 | 0 |
| idle experts | count of n_i = 0 | — | — |

**P2.** 16 tokens, E = 4, position policy, routed `idx = [0,0,1,0,2,0,1,0,3,1,0,2,1,0,1,2]`:

- (a) For cf = 1.0, 1.25 and 2.0, find C, the kept tokens per expert, the dropped token indices and the rows drop-and-pad computes.
- (b) Megatron applies capacity **per source rank, before dispatch**, with each rank's local tokens. Suppose these 16 tokens are four sources of 4. At cf = 1.0, does per-source capacity drop more, fewer or the same tokens as global capacity? Decide from the formula before counting.

<details><summary>Answer P2</summary>

(a) The counts are `[7, 5, 3, 1]`.

| cf | C | Kept per expert | Dropped | Drop rate | Drop-and-pad rows |
| ---: | ---: | --- | --- | ---: | --- |
| 1.0 | 4 | `[4,4,3,1]` | tokens 7, 10, 13 (expert 0) and 14 (expert 1) | 25 % | 16 rows for 12 kept |
| 1.25 | 5 | — | 10, 13 | 12.5 % | 20 |
| 2.0 | 8 | — | nothing | 0 % | 32 rows for 16 |

In this example, padding wastes as much compute as dropping saves.

(b) More, or equal. Per source, $C_s = \lceil \text{cf} \cdot 4/4 \rceil = 1$ per expert per source. A source that sends 3 tokens to expert 0 keeps only 1, even though another source sent expert 0 nothing. The capacity is split into 4 fixed slices that cannot lend to each other. On medium learned routing, both GPU reruns dropped 234 tokens with global C and 255 with per-source C at cf 1.0. At cf 1.25 the counts were 0 and 6. With ragged routing at cf 1.0, the drop rates were 35.4% and 56.2%. [Measured routing and capacity](notes.md#3-experiment-1-routing-and-capacity-on-one-gpu)

</details>

**Question 2.** On medium learned routing at cf 1.0, the position and probs policies each dropped 234 tokens, with 43 in common in both reruns. Which tokens does each policy protect? Which policy makes the *output for a given token* depend on other tokens later in the batch? Why does that matter for the GUIDE's parity checks, which run dropless? [Routing results](notes.md#3-experiment-1-routing-and-capacity-on-one-gpu)

### §1.3 Permutation: grouping rows by expert, and putting them back

The experts want their rows contiguous: `order = stable_argsort(idx)` and `x_sorted = x[order]`, with `counts = bincount(idx, minlength=E)`. The inverse puts row k back at position `order[k]`: `y[order[k]] = y_sorted[k]`. It is a scatter, not a gather.

**P3.** `idx = [2,0,3,0,2,0,3,3]`:

- (a) Write order and counts.
- (b) Someone writes the inverse as `y_sorted[order]`. Does it give back `x` from `x[order]`?
- (c) On what kind of routing would that bug pass every check?

<details><summary>Answer P3</summary>

(a) order is `[1,3,5,0,4,2,6,7]` and counts are `[3,0,2,3]`. Expert 1 is idle and must still appear as 0, so `minlength` matters. Stable means tokens of one expert keep their original order. An unstable sort gives a valid permutation but a different row order inside each expert, so the layout and the floating-point summation order can change between runs.

(b) No. `x[order][order]` applies the permutation twice. That is `x` only if `order` is its own inverse, and `[1,3,5,0,4,2,6,7]` is not.

(c) When `order` is the identity, the bug is invisible. That happens when the tokens already come grouped by expert. The `presorted` pattern does exactly this, and the control `wrong_inverse@presorted` must fail **nothing**. A planted bug that passes on easy data is why the parity cases use `ragged` and `learned` routing. On ragged data at EP=4, the same bug breaks logits on ranks 1–3 but not on rank 0. Rank 0 sends every token to expert 0, so its local order is the identity (Appendix A).

</details>

**Question 3.** `unpermute` must be *out-of-place* (`zeros.index_add(0, order, y_sorted)`), and dropped tokens must come back as exact zeros. Name one failure you would get from writing `y = torch.empty(...)` and filling only the kept rows. Name one more from an in-place write into a tensor autograd saved.

### §2.1 Dispatch: every rank must learn its receive sizes before the call

At EP=4 each rank owns one expert: rank j owns expert j, and $E_\text{local} = E/\text{EP} = 1$. After permuting, rank r's rows for rank j are contiguous, and there are `send[r][j]` of them, rank r's counts for rank j's experts. `all_to_all_single(out, x, output_split_sizes, input_split_sizes)` needs on every rank:

- `input_split_sizes`: row r of the count matrix, the rows I send to each rank
- `output_split_sizes`: column r, the rows each rank sends me

Each rank knows only its own row. So first, one small **equal-split** `all_to_all` of the `[EP, E_local]` count matrix gives every rank its column. Then **one** `.tolist()` copies all the sizes to the host. NCCL needs Python ints to enqueue the call. That copy is a **device → host sync**: the host waits until the router, the counts and the count exchange have run on the GPU.

```text
                   send matrix (rows = source rank, cols = destination rank = expert)
                   dst0  dst1  dst2  dst3     ← each rank r knows only ROW r (its counts)
         src0  [    128     0     0     0 ]
         src1  [     86    42     0     0 ]
         src2  [      0    32     0    96 ]
         src3  [     64     0     0    64 ]
                    ↑ rank 1 must receive COLUMN 1 = [0, 42, 32, 0]
```

This is the `ragged` pattern on the small preset: 512 tokens, 128 per source.

**P4.**

- (a) For rank 1, write `input_split_sizes`, `output_split_sizes` and the number of rows its expert computes. Do the same for rank 2.
- (b) Which zero-size splits appear, and must the call still be made on a rank that receives nothing?
- (c) Write rank 1's split lists for the **combine** all_to_all, which returns expert outputs to their sources, and for the **backward** of the dispatch.
- (d) How many all_to_all calls does one MoE layer make per training step, forward plus backward?

<details><summary>Answer P4</summary>

(a) Rank 1 sends `[86,42,0,0]` (its row), receives `[0,42,32,0]` (column 1) and computes 74 rows. Rank 2 sends `[0,32,0,96]` and receives `[0,0,0,0]` (column 2), so its expert computes **0 rows**.

(b) There are many zero splits, for example rank 0 → 1, 2 and 3. Every rank must still enter every all_to_all, with zeros in its lists. A collective is called by every member of the group or it hangs. Rank 2's expert still runs on an empty `[0, d]` tensor, so its gradients exist and are exact zeros. Skipping the expert would give `None` gradients and a different optimizer state. `check_ep` checks `grad.expert.L*.e2` against the baseline's zeros.

(c) The combine sends back exactly what was received, so its splits are the dispatch's with `input` and `output` **swapped**. Rank 1 uses `input=[0,42,32,0]` and `output=[86,42,0,0]`. The dispatch's backward is also the transpose: the gradient of each received row goes back to its source. So the dispatch backward uses the same swapped lists, and the combine backward uses the original dispatch lists. No new count exchange is needed in backward. The sizes are saved from forward.

(d) Five per layer per step:

- forward: count exchange, dispatch and combine
- backward: combine-backward and dispatch-backward

`check_ep` counts them: 3L forward and 2L backward for L layers. With E_local > 1, received rows arrive grouped by (source, expert). The expert loop needs them grouped by (expert, source). That reorder is `sort_chunks`, Block 3 Stop 5, and it is exercised only by the optional EP=2 stage.

</details>

**Question 4.** Megatron exchanges counts with an **all-gather** of every rank's full count vector over the EP group, not an all_to_all of one row each. What does every rank then know that ours does not? Name one use for it: Megatron's capacity path, logging, or a fixed-size buffer. What does it cost at EP=64?

### §2.2 The worst-case buffer

Dropless routing has no capacity, so a rank must be ready to receive **every** token in the job. Its receive buffer and its expert's intermediate activations scale with the routing, not with the batch.

**P5.** The medium shapes are T = 4096 global tokens (16 × 256), d = 512, ffn = 2048, FP32 and EP=4. For balanced and all-to-one routing, find:

- (a) the largest receive buffer in rows and MiB
- (b) the largest expert intermediate `[rows, ffn]` in MiB
- (c) the same at EP=8 with E = 8. This part is [simulated], because there is no eight-GPU run.

<details><summary>Answer P5</summary>

(a) Balanced: 1024 rows × 512 × 4 B = 2 MiB. All-to-one: 4096 rows = 8 MiB, the whole batch on one rank. `route_stats` prints 8.0 vs 2.0 MiB.

(b) Balanced: 1024 × 2048 × 4 B = 8 MiB. All-to-one: 32 MiB, kept for backward.

(c) All-to-one at EP=8 is 8× the balanced share. The hot pattern (70 % to one expert) gives 5.6× at EP=8 vs 2.8× at EP=4 in `check_local`'s ep_sim [simulated]. The worst case grows with EP while the balanced share shrinks. That is why production systems use a capacity limit or a padded maximum. Megatron's dropless path allocates exactly the received size each step, which is why it needs the host sync from §2.1.

</details>

**Question 5.** With drop-and-pad at cf = 1.25, what is the receive buffer at EP=4 for both patterns? What is the extra compute for balanced routing? Which of the two costs, memory spikes or wasted compute, would you rather pay at EP=64, and why?

### §3.1 Three gradient groups: dense DP, EP, expert DP

On every rank, `loss_r` is the mean over its own tokens. The one-GPU loss is $L = (1/\text{DP}) \cdot \sum_r \text{loss}_r$, with DP = world = 4 at TP = PP = 1.

- **Dense parameters** (attention, LayerNorms, router, embeddings) are replicated on all four ranks. The Day 03 rule applies: all-reduce SUM, then divide by DP.
- **Expert parameters** live on one rank, the owner. In backward, the combine all_to_all brings each source rank's gradient rows to the owner, and the expert's own backward produces $\sum_r \partial \text{loss}_r / \partial W_e$ there. The rows were already summed across sources by the communication.

Megatron names three groups. With E experts, EP ranks per expert set and world W:

$$
\begin{aligned}
\text{dense DP} &= W && (\text{TP} = \text{PP} = \text{CP} = 1)\\
\text{EP} &= \text{ranks that split the experts between them}\\
\text{EDP} &= W / \text{EP} && \text{ranks holding the SAME expert (replicas)}\\
\text{EP}=4:&\quad \text{EP groups } [0,1,2,3];\quad \text{EDP groups } [0]\ [1]\ [2]\ [3] && \text{(no replicas)}\\
\text{EP}=2:&\quad \text{EP groups } [0,1]\ [2,3];\quad \text{EDP groups } [0,2]\ [1,3] && \text{(two replicas of each expert)}
\end{aligned}
$$

**P6.**

- (a) At EP=4, which factor is still missing on an expert gradient after backward, and over which group do you reduce?
- (b) A bug all-reduces expert gradients over the EP group, treating them like dense ones, then divides by 4. What does `check_ep` see?
- (c) A second bug forgets the factor from (a). What does it see?
- (d) At EP=2 × EDP=2, what is the full rule?

<details><summary>Answer P6</summary>

(a) The sum over sources is already there, so the missing factor is **1/DP**. At EDP=1 there is no reduction at all.

(b) Rank j's expert-j gradient gets averaged with ranks k ≠ j's gradients of *different* experts. The result is a mix, not a multiple. The projection ratio `<got,ref>/<ref,ref>` comes out as an unclean number. It is ≈0.25 in Appendix A, because the others' experts are nearly orthogonal. Only expert gradients fail. Control `expert_sum_over_ep`.

(c) Every expert gradient is exactly 4× too large: ratio 4.0000, rel 3.0. Dense gradients pass. Control `expert_no_dp_scale`.

(d) Divide by DP = 4, then all-reduce SUM over EDP. Each replica holds the sum over the two sources in its EP group, and the EDP all-reduce adds the other replica's two sources. Megatron's DDP does the same: `expert_gradient_scaling_factor = 1/dp_size` on buffers whose parameters have `allreduce=False`, reduced over the expert-DP group (Stop 6). Router gradients are dense, because every rank has the router.

</details>

**Question 6.** A token's gradient takes *two* routes back to the router: through the combine weight w, and through the aux loss's P. With aux scope "global", the counts f are all-reduced over DP before `L_aux` is formed. Why does that all-reduce need **no backward communication**? Hint: what is the gradient of an integer count?

### §3.2 Skew: one slow rank, four ranks waiting

The all_to_all is a synchronization point. No rank can finish the combine until the slowest expert has produced its rows. A rank's phase timer around the combine therefore measures **transfer + waiting for the slowest peer**. The job's step time is the slowest rank's.

`bench_skew.py` puts CUDA events at every phase boundary on every rank. It times 20 unprofiled steps after 5 warmup steps, each step starting after a barrier and a synchronize. It reports the **job step**, the max over ranks for each step, then the median over steps.

**P7.** One MoE layer at the large shape:

- d = 1024 and ffn = 4096
- T = 16384 global tokens, 4096 per rank
- FP32 with TF32 off
- A100: assume about 17 TFLOP/s for FP32 GEMMs. The peak is 19.5.

An expert does $4 \cdot d \cdot \text{ffn} \approx 16.8\ \text{MFLOP}$ per row forward, and backward is about 2× that.

- (a) Estimate the expert time per step for rank 3 under balanced (4096 rows) and all-to-one (16384 rows) routing.
- (b) Estimate the dispatch volume rank 3 receives under all-to-one, and its transfer time at roughly 200 GB/s.
- (c) Under all-to-one, which phases on ranks 0–2 grow? What is the predicted job-step ratio of all-to-one to balanced? Is the medium shape's ratio larger or smaller?

<details><summary>Answer P7 [derived]</summary>

(a) Balanced: 4096 × 16.8 MFLOP × 3 ≈ 206 GFLOP, about 12 ms. All-to-one: 4×, about 48 ms on rank 3. Ranks 0–2 compute 0 rows.

(b) 3 × 4096 rows × 4 KiB = 48 MiB, about 0.25 ms. The bytes are not the cost.

(c) Ranks 0–2 wait in the **combine** all_to_all in forward and in the **dispatch-backward** all_to_all in backward. Those are the first collectives that need rank 3's expert output and its expert's input gradient. Their "communication" time grows by roughly rank 3's expert time. Their expert time drops to almost nothing. All four ranks finish within a fraction of a millisecond of each other, because the final all_to_all lines them up.

The job step at all-to-one should be roughly 3–4× balanced. It is not exactly 4× because the router, the permutation and the dense parts do not grow. The medium shape has 16× fewer expert FLOPs per row × rows, about 0.8 ms balanced. Fixed costs (kernel launches, the host sync, small all_to_alls) are a larger share there, so its ratio should be **smaller**.

**GPU rerun, large shape** [measured]. The table uses the benchmark from the `mine` session, which runs the reference implementation.

| Routing | Rank 3 expert fwd + bwd | Combine on ranks 0–2 | Dispatch-backward on ranks 0–2 | Job step |
| --- | --- | --- | --- | --- |
| balanced | 4.04 + 7.77 ms | 0.11–0.23 ms | 0.11–0.36 ms | 13.89 ms |
| all-to-one | 14.93 + 30.10 ms | 15.34–15.41 ms | 30.58 ms | 47.44 ms (3.42×) |

The timing on ranks 0–2 tracks rank 3's expert work. Rank 3's receive buffer was 64 MiB, including 16 MiB of local rows, and its measured dispatch phase was 0.336 ms. The balanced expert forward ran at 4096 × 16.8 MFLOP / 4.036 ms ≈ 17 TFLOP/s.

The large all-to-one ratio reproduced across the two sessions at 3.28–3.42×. The medium ratio was smaller and varied more, at 1.05–1.29×. Report each session against its own balanced median. [Both timing tables](notes.md#5-experiment-3-skew-at-ep4)

In the `mine` session's benchmark, the large single-GPU and EP=4 all-to-one medians were close (47.75 and 47.44 ms). Balanced routing gave a 3.58× EP speedup (49.75 / 13.89 ms).

</details>

**Question 7.** Hot routing (~70 % to expert 0) gives rank 0 about 2.8× the balanced rows. Your per-rank table will show rank 0 with the largest expert time, but possibly *not* the largest step time. Why can every rank's step be about equal? What then is the right number to report as "the cost of skew"?

### §3.3 The auxiliary load-balancing loss

The Switch Transformer loss, Megatron's `switch_load_balancing_loss_func`, is:

$$
\begin{aligned}
L_\text{aux} &= \alpha \cdot E \cdot \sum_i f_i \cdot P_i\\
f_i &= \frac{\text{tokens routed to } i}{T \cdot k} && \text{fraction of assignments; integer counts, NO gradient}\\
P_i &= \mathrm{mean}_\text{tokens}\ \text{probs}[:, i] && \text{differentiable}
\end{aligned}
$$

It is 1 when perfectly balanced (f = P = 1/E) and E when collapsed (f = P = one-hot). Its gradient reaches the router only through P, with $\partial L_\text{aux}/\partial P_i = \alpha \cdot E \cdot f_i$. That pushes probability away from experts in proportion to how overloaded they already are.

**MoEAuxLossAutoScaler** attaches L_aux to the graph without adding it to the reported loss. It is identity in forward on the combine weight. In backward it hands L_aux a gradient of 1 × scale, so `L_aux.backward()` runs inside the main backward.

**P8.** E = 4 and 16 tokens with counts `[7,5,3,1]` and $P = [0.4,\ 0.3,\ 0.2,\ 0.1]$:

- (a) Find $L_\text{aux}/\alpha$.
- (b) Find $\partial (L_\text{aux}/\alpha)/\partial P$.
- (c) Find $L_\text{aux}/\alpha$ for balanced and for collapsed routing.
- (d) With the AutoScaler, does the router's gradient differ from that of $\text{task} + L_\text{aux}$ written explicitly?

<details><summary>Answer P8</summary>

(a) f is `[7,5,3,1]/16`. Σ f·P = (2.8 + 1.5 + 0.6 + 0.1)/16 = 0.3125, and × 4 = **1.25**.

(b) $E \cdot f = [1.75,\ 1.25,\ 0.75,\ 0.25]$. Expert 0's mean probability is pushed down 7× harder than expert 3's.

(c) 1.000 and 4.000.

(d) No. `check_local aux.autoscaler` measures a max difference of 0.0 in float64. The AutoScaler only moves *where* the aux term enters backward. It lets the reported loss stay the task loss.

In both optional GPU reruns, the small preset was trained for 200 steps on one GPU from a router biased +2 to expert 0. Layer 0's max/mean started at 3.46 for every coefficient [measured]:

| α | Step 100 | Step 200 | Final CE |
| --- | --- | --- | --- |
| 0.1 | 1.66 | 1.64 | 0.604 |
| 0.01 | 2.93 | 2.95 | 0.651 |
| 0 | 3.27 | 3.33 | 0.664 |

Both sessions used the same seed. In this toy run the most balanced setting also had the lowest final CE. Generalization beyond this task and seed remains untested.

</details>

**Question 8.** If f_i *had* a gradient, for example a soft count $\sum \text{probs}$, `L_aux` would become $E \cdot \sum P_i^2$ up to scale. What would its minimum be and what would it push toward? Why is the hard-count version still the one used?

### §4 Optional predictions (P9–P12, +20 min)

**P9 (aux scope).** Megatron's default `aux_loss` reduces the counts over `tp_cp_group`, which has size 1 here. So each DP rank uses its **own** f with its own P: $\frac{1}{\text{DP}} \sum_r E \cdot \sum_i f_{r,i} \cdot P_{r,i}$. The one-GPU loss is $E \cdot \sum_i \bar f_i \cdot \bar P_i$. When are they equal? Is the local version usually larger or smaller?

<details><summary>Answer P9</summary>

They are equal when every rank has the same f (balanced routing, or identical sources). Otherwise the difference is the covariance across ranks between f_{r,i} and P_{r,i}, summed over i. A rank whose tokens prefer expert i also sends more to i, so the covariance is usually positive and the local loss is **larger**.

Both four-GPU `aux_local` reruns reproduced these outcomes [measured]:

- learned: `loss.aux` ratios 1.0051 / 1.0243
- ragged: worst `grad.router.L1` rel 8.9e-2
- balanced: equal, PASS

Router, dense and dx gradients fail. Only **layer 0's** expert gradients fail, because layer 1's aux gradient enters through layer 1's router into `h`, and from there into earlier layers only. Megatron's `global_aux_loss` (`reduce_group=tp_dp_cp_group`) is the variant that matches one GPU.

</details>

**P10 (rank order).** Megatron builds a *separate* rank generator for experts, with order `tp-cp-ep-dp-pp`, cp = 1, and $\text{EDP} = \text{world} / (\text{ETP} \cdot \text{EP} \cdot \text{PP})$. At world 8 with EP=4, write the EP groups and EDP groups. Is dense DP the same group as EDP?

<details><summary>Answer P10 [derived]</summary>

- EP groups: `[0,1,2,3] [4,5,6,7]`
- EDP groups: `[0,4] [1,5] [2,6] [3,7]`
- Dense DP: all 8 ranks

Dense DP and EDP are different groups. Rank r holds expert-set index r mod 4 and replica index r // 4, which is `moe_common.layout_ranks` and the optional `--ep 2` stage at world 4. The separate generator lets ETP differ from TP. The dense layout and the expert layout share the world, not the group shapes.

</details>

**P11 (expert compute).**

- (a) With E_local = 4 experts on a rank, how many GEMM launches does a `SequentialMLP`-style loop make per forward, versus a grouped GEMM?
- (b) Fine-grained experts split each expert into m experts of ffn/m and route top-m. What happens to FLOPs per token and to the number of all_to_all rows?
- (c) A shared expert (always-on dense MLP) adds what per token, and what communication?

<details><summary>Answer P11</summary>

(a) 2 GEMMs per expert (fc1, fc2), so 8, plus activations, versus 2 grouped launches. At EP=4 with one expert per rank there is no difference. Megatron's `SequentialMLP` even skips the split when it has one local expert (Stop 7).

(b) FLOPs per token are the same, $m \cdot 4 \cdot d \cdot (\text{ffn}/m)$. Each token now goes to m experts, so up to m times the dispatched rows. Communication grows, and so does the number of (expert, source) chunks.

(c) $4 \cdot d \cdot \text{ffn}_\text{shared}$ FLOPs per token and no all_to_all. It runs on every rank's own tokens, which is why Megatron can overlap it with the dispatch (`shared_expert_overlap`).

</details>

**P12 (determinism).** On CPU, the baseline was written twice (reference and a reference-filled `my_moe.py`) with the same sha. EP parity is within 1e-5 but mostly not bitwise: on GPU, logits were bitwise equal but only 2–8 of 18 checks per case were. Which steps could make two identical runs differ? Which make EP=4 differ from one GPU without being bugs?

<details><summary>Answer P12</summary>

Run to run, nothing differs in this harness:

- The stable sort fixes the row order.
- Top-1 `index_add` has no index collisions.
- The all_to_alls copy without reducing.
- `use_deterministic_algorithms` is on.

Top-2 *would* collide, since two expert outputs land in one row. A non-deterministic `scatter_add` then sums them in a varying order. Megatron uses `index_add` when deterministic mode is on (Stop 5, `moe_utils.py:563`).

EP=4 vs one GPU differs without bugs because:

- the dense gradients are an all-reduce of 4 partial sums instead of one GEMM over all 512 rows
- the expert GEMMs see the same rows in a different number of calls

Floating-point addition is not associative.

</details>

---

## Block 2: Code (2 h)

### Stage 0 (10 min): read the harness

Open `moe_model.py`, `MoELayer.forward`. Find:

- where `stats["send"]` and `stats["recv"]` come from
- the single `.tolist()` for all split sizes
- the two `AllToAllV.apply` calls, and which split lists each gets
- where the combine weight multiplies, after the combine and on the source rank

Then make the baseline on CPU:

```powershell
python experiments/day14_moe/baseline_1gpu.py --cpu --run-id try1
```

### Stage 1 (35 min): **YOU WRITE** `top1_route`, `permute`, `unpermute` (TODO 1, 2a–2c)

```powershell
python experiments/day14_moe/check_local.py --impl mine
```

The gating, permute and capacity lines use your TODO 1–2. While a function still raises `NotImplementedError`, its lines print `INCOMPLETE` and the script exits 3. The aux, splits, ep_sim and top2 lines test the reference and pass from the start. Stage 1 is done at `CHECK_LOCAL (mine): 29 passed, 0 failed, 0 incomplete: PASSED`, exit 0.

### Stage 2 (40 min): **YOU WRITE** `exchange_counts` and `AllToAllV` (TODO 3, 4a–4d)

```powershell
python experiments/day14_moe/check_ep.py --impl mine --cpu --nproc 4 --tests dispatch
```

The dispatch test sends labelled rows: every row is `[source rank, destination rank, k]`, for the k-th row sent to that destination. It checks three things, for the ragged and all-to-one patterns:

- `exchange_counts` returned column j of the count matrix.
- Rank j received exactly those rows, grouped by source in rank order.
- In backward, every receiver multiplies its rows by (its rank + 1), so every source must get back, on each row it sent, the code of the rank it sent that row to.

A row sent to the wrong rank, or a backward with unswapped splits, fails here before any model runs.

### Stage 3 (20 min): **YOU WRITE** `finish_grads` (TODO 5), then parity

```powershell
python experiments/day14_moe/check_ep.py --impl mine --cpu --nproc 4 --baseline outputs/day14_moe/tensors/baseline_small_try1.pt
```

Each case prints one row per check:

| Column | Meaning |
| --- | --- |
| rel | max \|got − ref\| / max \|ref\| |
| ratio | `<got,ref>/<ref,ref>` |
| bitwise | yes / no |
| ranks | the ranks that failed |

The checks are:

- `routing.L*`, exact
- `counts.L*`, exact. The send row must match the global routing, and recv must match what the peers sent.
- `loss.aux.L*` and `loss.z.L*`
- `logits`, `loss.ce`, `dx` (gradient of the embedding output)
- `grad.router.L*`, `grad.expert.L*.e*`, `grad.dense` (with the worst parameter named)
- `replicated`: dense gradients identical on every rank
- `collectives`: the all_to_all and all_reduce counts per phase

**Pass rule:** rel ≤ 1e-5 in FP32 (1e-12 in FP64), and exact equality for routing and counts. Ratio is a diagnostic. 4.0000 is a scale bug. Anything not clean is a mixing bug.

### Stage 4 (15 min): the negative controls

```powershell
python experiments/day14_moe/check_ep.py --cpu --nproc 4 --impl controls --baseline outputs/day14_moe/tensors/baseline_small_try1.pt
```

Before running, write in notes §4 which checks each control should fail, then compare.

| Control | Planted bug | Must fail | Why there and only there |
| --- | --- | --- | --- |
| `wrong_inverse` (ragged) | `y_sorted[order]` | logits, losses, dx, dense, router, and the experts that receive rows from a non-identity order | rank 0's order is the identity (P3) |
| `wrong_inverse@presorted` | the same bug | **nothing** | order is the identity everywhere: the blind spot |
| `expert_sum_over_ep` | expert grads all-reduced over EP, ÷EP | every `grad.expert.*`, with an unclean ratio | mixes different experts (P6b) |
| `expert_no_dp_scale` | missing 1/DP | the experts that got tokens, ratio 4.0000 | scale bug (P6c). Expert 2 in ragged has zero grads, and 4 × 0 = 0. |
| `recv_equals_send` | `exchange_counts` returns its input | `raised:SplitMetadataError` on every rank, **before** NCCL | the split check (`validate_splits_collective`) all-gathers every rank's lists and finds rank 0 sending 0 rows to rank 1, which expects 86 |

The last control is about failing *safely*. Mismatched split sizes inside NCCL give a hang or garbage, not an error message. So the harness checks the metadata collectively first, and every rank raises together.

### Pitfalls (symptom → cause)

| Symptom | Likely cause |
| --- | --- |
| hang at the first all_to_all | one rank skipped the call (zero rows) or split lists disagree. Run with the split check on (default). |
| `INCOMPLETE (TODOs left)`, exit 3 | a function still has `raise NotImplementedError` |
| `grad.expert.L*.e2` is `None`/FAIL in ragged | the expert was not called on its empty chunk |
| every expert grad has ratio 4.0000 | 1/DP missing (P6c) |
| `dx` fails, everything forward passes | backward all_to_all with unswapped splits, or `AllToAllV.backward` returned too few values |
| logits fail only on ranks 1–3 | the wrong inverse (P3) |
| `RuntimeError: ... modified by an inplace operation` | writing into `x` or into a saved tensor in `unpermute` |
| routing.L0 FAIL on `learned` only | TF32 on, or a different dtype in the router. The learned top-1 margin can be as small as 4.6e-4. |

---

## Block 3: Read Megatron Core (1 h; Stops 1–6 required, Stop 7 optional)

All paths are under `megatron/core/` in `core_v0.19.2`, commit `4b4acac9`. The line numbers come from `find_symbols.py`, and the assistant's output is in [`symbols.md`](../outputs/day14_moe/cpu_check/symbols.md). Run it on your checkout first. If anything prints `(not found)`, find the stop by name.

The call chain of one MoE layer, `transformer/moe/moe_layer.py`, `MoELayer.forward` `:607`:

```text
route :438 ──► TopKRouter.forward router.py:840 ──► routing :741
preprocess ──► token_dispatcher.preprocess :495 (counts, splits, sync point)
dispatch :489 ──► dispatch_preprocess :618 (permute) ──► token_dispatch :675 (all_to_all) ──► dispatch_postprocess :721 (sort_chunks)
routed_experts_compute :529 ──► experts (SequentialMLP :1260 / TEGroupedMLP :182)
combine ──► combine_preprocess ──► token_combine :830 (all_to_all) ──► combine_postprocess (unpermute :884)
```

### Stop 1 (8 min): routing. `router.py` `TopKRouter.routing` `:741`, `moe_utils.py` `topk_routing_with_score_function` `:710`

`routing` calls `topk_routing_with_score_function` at `:774`. In it:

- `:840` (`if use_pre_softmax`) is the softmax over all E before top-k, which is this harness's choice.
- `:853` is the default: `probs = torch.softmax(scores)` over the **k selected** scores.
- `:868` is the sigmoid branch, which renormalizes only `if topk > 1`.

**Question:** with the defaults (`moe_router_pre_softmax=False`, `moe_router_score_function="softmax"`) and topk = 1, what is every entry of `probs`? Through which terms does the router weight then get any gradient? Connect this to P1(c).

### Stop 2 (10 min): aux loss, its scope, and the AutoScaler. `router.py` `:407`–`:676`, `moe_utils.py` `:56`, `:284`

- `switch_load_balancing_loss_func` `:56` forms $\sum P_\text{sum} \cdot \text{counts}$ at `:139`. Check it is $E \cdot \sum f_i P_i$ with Megatron's normalization.
- `_apply_aux_loss` `:407` reduces counts with `reduce_group=self.tp_cp_group` at `:422`. `_apply_global_aux_loss` `:504` uses `tp_dp_cp_group` at `:520`.
- `attach_and_log_load_balancing_loss` `:550` attaches with `MoEAuxLossAutoScaler.apply(activation, aux_loss * num_local_tokens * tp_cp_group.size())` at `:632`. The long comment above it explains the factor.
- `MoEAuxLossAutoScaler.backward` `:304` returns `ones_like(aux_loss) * main_loss_backward_scale`.
- The z-loss is `apply_z_loss` `:639` → `z_loss_func(logits, ...)` `:655`, defined at `moe_utils.py:146`.

**Question:** this harness's main loss is a *mean* over tokens. Megatron's (with `--calculate-per-token-loss`) is a *sum* later divided by the global token count. Why does that force Megatron to pre-multiply aux by `num_local_tokens` times the TP × CP group size, while this harness attaches aux unscaled? Which of P9's two scopes is Megatron's default?

### Stop 3 (7 min): capacity. `moe_utils.py` `get_capacity` `:241`, `apply_router_token_dropping` `:958`

- `:256`: `capacity = math.ceil((num_tokens / num_experts) * capacity_factor)`
- `:990`: called with the router's **local** tokens (×topk)
- `:1002`: the probs policy, `torch.topk(routing_probs, k=expert_capacity, dim=0)`, keeps the top-C tokens per expert column
- The router calls it at `router.py:789`. The dispatcher's own `get_capacity` in `preprocess` sizes the drop-and-pad buffers.

**Question:** your `route_stats.log` reports both "global C" and "per source C". Which one is Megatron's `apply_router_token_dropping`? At cf 1.0 on your learned routing, how many more tokens does it drop than a global capacity would?

### Stop 4 (10 min): counts, splits and the host sync. `token_dispatcher.py` `MoEAlltoAllTokenDispatcher.preprocess` `:495`

- `:550` dropless: `num_out_tokens = routing_map.size(0) * topk`. The *total* is static, so no sync is needed for it. With a capacity factor, `:545` makes it dynamic, with sync point `before_permutation_1`.
- `:557`: `input_splits` = local counts reshaped `[ep, E_local]` and summed per destination, which is your send row.
- `:565`: `gather_from_sequence_parallel_region(num_local_tokens_per_expert, group=tp_ep_group)`, an all-gather of **every rank's** count vector.
- `:580`: `output_splits = num_global_tokens_per_rank[tp_rank]`, which is your receive column.
- `:590`: `_maybe_update_cuda_sync_point("before_ep_alltoall")`.
- `_maybe_dtoh_and_synchronize` `:913`: the counts are copied on a side stream, and the host waits at `:948` only when it reaches the chosen sync point.

**Question:** in your `bench_ep4.log`, the `counts` phase includes the `.tolist()` sync, and `host@counts` is the host clock when it returns. Megatron copies early on a side stream and waits late. What work can its host enqueue between the copy and the wait that ours cannot? Why can neither version avoid the wait entirely in dropless mode?

### Stop 5 (10 min): permute, the two all_to_alls, sort_chunks, combine. `token_dispatcher.py` `:618`–`:884`, `mappings.py` `:424`

- `dispatch_preprocess` `:618` → `permute(` `:665` (`moe_utils.py:337`): the same grouping by expert as your TODO 2a.
- `token_dispatch` `:675`: `all_to_all(ep_group, tokens, self.output_splits, self.input_splits)` at `:698`, and a **second** all_to_all for the probs at `:711`.
- `dispatch_postprocess` `:721` → `sort_chunks_by_idxs(` `:775`: received chunks ordered (source, expert) → (expert, source). Inside, `split_sizes.tolist()` at `moe_utils.py:607` is another host copy, on the non-fused path.
- `token_combine` `:830`: `all_to_all(ep_group, hidden, self.input_splits, self.output_splits)` at `:857`, the same lists in swapped positions.
- `combine_postprocess` → `unpermute(` `:884`, with `index_add_` at `moe_utils.py:563` in deterministic mode.
- `_AllToAll.backward` `mappings.py:470` calls `_AllToAll.apply` with `input_split_sizes, output_split_sizes` swapped.

**Question:** Megatron sends the probs to the expert rank and multiplies them inside the expert (before fc2, `moe_apply_probs_on_input` aside). This harness keeps w at the source and multiplies after the combine. Count the extra all_to_all traffic of Megatron's choice per layer per step, forward and backward. Why might it still prefer it? Think about what the expert can fuse, and what happens with top-k > 1.

### Stop 6 (5 min): expert gradients and groups. `distributed/distributed_data_parallel.py`, `parallel_state.py`

- `distributed_data_parallel.py:624`: `is_expert_parallel = not getattr(param, 'allreduce', True)`. Expert parameters go to separate buffers.
- `:213`: `expert_gradient_scaling_factor = 1.0 / data_parallel_world_size`, the default. `:208` is `expt_dp_group.size() / dp_cp_group.size()` with `average_in_collective`.
- `parallel_state.py:880`: `expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size`.
- `:887`: the separate `expert_decoder_rank_generator`, with order `tp-cp-ep-dp-pp` and cp = 1.
- `:1357`: EP groups (`get_ranks('ep')`). `:1457`: EDP groups (`get_ranks("dp")`).

**Question:** match each line to your TODO 5. Why is the expert scaling factor 1/dp and not 1/edp, when the expert gradient is reduced over EDP only? Use P6(a).

### Stop 7 (optional, 15 min): expert compute and the alternatives

- `experts.py` `TEGroupedMLP` `:182`: `tokens_per_expert.tolist()` at `:769` feeds one grouped GEMM.
- `SequentialMLP` `:1260`: `tolist`, `torch.split` and a Python loop at `:1352`. With one local expert it skips the split entirely (just above).
- `shared_experts.py` `SharedExpertMLP` `:102`.
- `moe_utils.py` `get_updated_expert_bias` `:1136`: $\text{bias} + \mathrm{sign}(\text{avg} - \text{count}) \cdot \text{rate}$ at `:1165`. It is applied outside the optimizer, with no gradient.
- `sinkhorn` `:178`, used by `router.py:298`.

**Question:** the aux-loss-free bias changes *which* expert is selected but not the combine weight (`router.py` `_apply_expert_bias` `:728`, `moe_utils.py` sigmoid branch). Compare that with your `learned+bias` row in `route_stats.log`, where the bias was added to the logits and therefore also changed the weight. Which is the cleaner balancer, and why?

---

## Block 4: Run + summarize (30–60 min)

### Commands (the manual run)

On the instance, from `/workspace/AI-Infra-Learning`:

```bash
RID=manual_$(date -u +%Y%m%dT%H%M%SZ)_mine; bash experiments/day14_moe/run_server_once.sh "$RID" --impl mine
```

The full reruns took 2 min 38 s (`ref`) and 2 min 40 s (`mine`) on the 2026-11-27 host. Use `--core` to skip optional stages. In a `mine` session, local checks, the extra one-GPU baseline, dispatch and EP=4 parity exercise your implementation. Routing statistics, the negative controls, timing, and optional EP=2 and auxiliary-loss runs use the reference. The EP=4 `mine` parity oracle is the session's one-GPU `ref` baseline.

### Download to Windows, then stop the instance

```powershell
experiments/day14_moe/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>
```

The script checks the archive's SHA-256 and every manifest entry. Then **stop** the instance in the Vast console. Do not destroy it. An idle running instance is still billed. Record the console check in notes §2.

### Experiment matrix

| Exp | Stage (log) | Layout | Expected output | Pass / fail rule | Label |
| --- | --- | --- | --- | --- | --- |
| 0 | `check_local` | CPU | P1–P8 numbers as in Block 1; ep_sim receive rows | all 29 `PASS`, exit 0 | [derived] / [simulated] |
| 1 | `route_stats` | 1 GPU, medium, 4096 tokens | tokens/expert, max/mean, CV, entropy, idle for learned, learned+bias, balanced, hot, ragged, all-to-one; drops at cf 1/1.25/2 global vs per-source; EP=4 receive rows and largest buffer (8 MiB all-to-one vs 2 MiB balanced) | `ROUTE_STATS: done`. Counting is exact, so a CPU result must equal the GPU result for the forced patterns. | [measured] |
| — | `baseline` | 1 GPU, small | CE per pattern, aux/z per layer, tokens/expert, min top-1 margin, sha | `BASELINE written`. Read, not enforced: the learned min margin should be far above FP32 noise (≥ 1e-5). If it is not, a `routing.L*` failure in 2b may be a tie, not a bug. | [measured] |
| 2a | `ep4_dispatch` | EP=4 | `exchange_counts` gives column j; labelled rows `[src, dst, k]` arrive grouped by source; backward returns each row times its receiver's code | `DISPATCH (...): PASSED` | [measured] |
| 2b | `ep4_parity` | EP=4 vs 1 GPU | per check: rel, ratio, bitwise, ranks; send matrices | all four cases `PASSED`: rel ≤ 1e-5, routing/counts exact, collective counts 3L/2L/L | [measured] |
| 2c | `ep4_controls` | EP=4 | each control's failure set equals the prediction; presorted fails nothing; recv=send raises before NCCL | `CONTROLS (...): PASSED` | [measured] |
| 3 | `bench_ep1`, `bench_ep4` | 1 GPU / EP=4, medium + large × balanced/hot/all-to-one | per-rank receive rows and MiB, phase medians (route, counts, dispatch, experts, combine, unpermute; backward phases), job step (max over ranks per step: median, min, max), slowest rank, phase with the largest rank-to-rank spread, expert ms per 1000 rows | `BENCH_SKEW: done`. There is no numeric pass rule: compare with P7. | [measured] |
| opt | `ep2_parity` | EP=2 × EDP=2 | as 2b, with `replicated` equal across EDP replicas | `PARITY ... PASSED` (recorded, not fatal) | [measured] |
| opt | `aux_local` | EP=4, local aux | `loss.aux` ratios > 1 except balanced; layer 0 experts only | expected **FAILED** for learned/ragged (P9) | [measured] |
| opt | `aux_train` | 1 GPU, small, 200 steps | layer-0 max/mean every 20 steps for α = 0, 1e-2, 1e-1 | none: a hypothesis | [measured], one seed |
| opt | `bench_trace` | EP=4, large | one profiled step per traced case, after the timed steps | trace files exist. Never used for timing. | [measured] |

### How to read the logs

- **`environment.log`**: four GPUs, the P2P matrix of ones, and `topo -m`. Note the link type between the GPUs.
- **`route_stats.log`**: for each pattern, one load line, three capacity lines and one EP=4 line. Compare learned (at init) with learned+bias, then hot with all-to-one. The per-source/global drop gap is Stop 3's question.
- **`ep4_parity.log`**: one table per case. Read the `counts.L0` row's send matrix for ragged. It must be the §2.1 matrix.
- **`bench_ep4.log`**: per case, a per-rank table and one summary line.
  - Find the slowest rank and its expert time.
  - Then find the *other* ranks' `combine` and `bwd_dispatch_a2a`. P7 predicts they absorb the difference.
  - `expert ms per 1000 received rows` should be roughly constant across ranks and patterns. If it is not, the expert is launch-bound at that size.
- **`bench_ep1.log`**: the same layer on one GPU. Its step is the "no communication" reference for the matching EP=4 case. The expert time for balanced EP=1 (4096 rows on one GPU) should be about 4× one EP=4 rank's.

### What to write in `notes.md`

Fill §2–§5 from the logs with [measured], §6 from Block 3, and §7 from the eight Questions. Keep [derived] predictions and [measured] results in separate columns. Anything the run does not isolate goes in §8 with [not isolated].

---

## Self-check (answer aloud, < 2 min each)

1. Why does a top-1 router need the full-softmax weight, and what does Megatron's default do instead?
2. Walk one token from rank 2 to expert 3 and back in forward and backward. Name each split list.
3. Why exactly one `.tolist()` per layer, and what does the host wait for there?
4. A rank receives zero rows. What must it still do, and why?
5. Expert gradient at EP=4: which factor, which group? At EP=2 × EDP=2?
6. Under all-to-one routing, where in the per-rank table does the skew appear on the three idle ranks?
7. Why does per-source capacity drop more tokens than global capacity?
8. Which planted bug passes on presorted routing, and what does that teach about test data?

---

## Appendix A: local CPU results (assistant run, 2026-11-22) [pre-test, CPU]

These come from Windows, torch 2.13.0+cu130, CPU only, with gloo and four local processes standing in for GPUs. They show that the harness, the reference and the controls work end to end. **The timings are meaningless.** The counting results, such as drops and receive rows, are exact and should repeat on GPU for the forced patterns. Learned routing on GPU may differ in the last digits.

### A.1 check_local (29/29 PASS)

| Example | Result |
| --- | --- |
| P1 probs | `[.6103 .2245 .0826 .0826] [.0433 .0433 .8700 .0433] [.25×4]` |
| P1 idx, w | `[0,2,0]`, `[.6103,.87,.25]` |
| P1 ∂w₀/∂logits₀ | `[0.2378, −0.137, −0.0504, −0.0504]` |
| P1 router gradient | full softmax 2.155e-1, renormalized 0.0 |
| P3 order, counts | `[1,3,5,0,4,2,6,7]`, `[3,0,2,3]` |
| P2 cf 1.0 | C=4, kept `[4,4,3,1]`, dropped `[7,10,13,14]` |
| P2 cf 1.25 | C=5, dropped `[10,13]` |
| P2 cf 2.0 | C=8, no drops |
| P8 L_aux/α | balanced 1, collapsed 4, example 1.25 |
| P8 ∂/∂P | `[1.75,1.25,.75,.25]` |
| P8 AutoScaler vs explicit | diff 0.0 |
| ep_sim [simulated], EP4 hot | receive `[179,27,25,25]` (2.8×) |
| ep_sim [simulated], EP8 hot | max 5.6× balanced |
| ep_sim [simulated], all-to-one | 4× and 8× |
| top-2 renormalized router gradient | 6.142 (≠ 0) |

### A.2 One-device baseline (small preset, CPU)

The sha is `0d974b25fbcde3e1`, and the reference and a reference-filled `my_moe.py` give the same sha. The learned min top-1 margin is 4.56e-4.

| Pattern | CE |
| --- | --- |
| learned | 6.7032547 |
| balanced | 6.7257771 |
| hot | 6.7142148 |
| ragged | 6.7276931 |
| all_to_one | 6.7239494 |
| presorted | 6.7331486 |

### A.3 EP parity on CPU

- **Dispatch:** `DISPATCH PASSED`.
- **EP=4:** `PARITY PASSED` for learned, balanced, ragged and all-to-one.
  - The max rel error was about 7e-7, with ratios of 1.0000.
  - Expert 2 in ragged has exact-zero gradients on both sides.
  - The ragged send matrix is `[[128,0,0,0],[86,42,0,0],[0,32,0,96],[64,0,0,64]]`.
- **EP=2 × EDP=2:** `PASSED`, with the replicated gradients bitwise equal.
- **`--impl mine`** (with the reference copied into `my_moe.py`, then restored): check_local 29/29, dispatch and parity PASSED.

### A.4 Negative controls (CPU, EP=4): `CONTROLS PASSED`

| Control | Observed |
| --- | --- |
| `wrong_inverse` | logits fail on ranks 1–3 only, plus the predicted loss, dx, dense, router and expert failures |
| `wrong_inverse@presorted` | nothing fails |
| `expert_sum_over_ep` | every expert grad, ratios about 0.25 |
| `expert_no_dp_scale` | experts 0, 1 and 3 in both layers, ratio 4.0000, rel 3.0 |
| `recv_equals_send` | `SplitMetadataError` raised on all four ranks before any token all_to_all |

### A.5 Megatron's local aux scope on CPU (`--aux-scope local`, optional P9)

| Case | Result |
| --- | --- |
| learned | `loss.aux` ratio 1.0051 (L0) / 1.0243 (L1). dx, dense and router fail. Experts fail in layer 0 only. |
| ragged | ratios 1.0196 / 1.0363, same pattern of failures |
| balanced | PASSED |

### A.6 route_stats on CPU (medium preset, 4096 tokens)

**Learned at init.** Layer 0 counts `[921,909,1008,1258]`, max/mean 1.229, router entropy 0.810. Max/mean ranges 1.229–2.29 over the six layers.

**Learned + bias 1.5.** `[2900,330,410,456]`.

**Capacity at cf 1.0, learned:**

- global capacity drops 234, per-source drops 255
- at cf 1.25: 0 vs 6
- position and probs policies each drop 234, but only 43 of the same tokens

**Ragged:** global 35.4 % vs per-source 56.2 %.

**EP=4 largest receive buffer:** all-to-one 8.0 MiB, balanced 2.0 MiB.

**aux-train, 60 steps, small preset, bias +2.** Layer-0 max/mean starts at 3.46 and ends at:

| α | end |
| --- | --- |
| 0.1 | ≈ 1.64 |
| 0.01 | ≈ 2.94 |
| 0 | ≈ 3.26 |

This is a hypothesis from one run.

### A.7 What the CPU check supports and what it does not

**It supports:**

- the reference and the harness compute the same numbers on four processes as on one
- each control fails where predicted
- the split check stops a bad exchange before the collective
- the stubs exit 3

**It does not support:** any timing, any statement about NCCL (gloo implements all_to_all differently), or GPU-specific numerics such as TF32 and cuBLAS algorithm choice. The four-GPU pre-test in Appendix B covers them.

---

## Appendix B: GPU pre-test `pretest2` (assistant run, 2026-11-22) [pre-test]

### B.1 Run record

| Item | Value |
| --- | --- |
| Run ID | `pretest2` |
| Command | `run_server_once.sh pretest2 --impl ref --megatron /workspace/Megatron-LM` (full mode, with the optional stages) |
| Time | 10:15:01–10:17:04 UTC, about 2 min |
| Host | Vast.ai container 61910115 |
| GPUs | 4 × A100-SXM4-40GB, driver 610.57.04 |
| Links and NUMA | NV12 between all pairs, P2P all ones; GPU 0–1 NUMA 3, GPU 2–3 NUMA 1 |
| Software | torch 2.14.1+cu130, NCCL 2.30.7 |
| Exit code | 0, with `CORE_SUCCESS.txt` and `SUCCESS.txt` |
| Optional exit codes | symbols 0, ep2_parity 0, aux_local 1 (expected, P9), aux_train 0, bench_trace 0 |
| Archive | `day14_pretest2.tar.gz`, 32 MB, SHA-256 `3A93406F…C9734`. The 62 manifest files were re-verified on Windows. |
| Download location | [`outputs/day14_downloads/pretest2/`](../outputs/day14_downloads/pretest2/session_pretest2/) |
| Instance | stopped from inside with `vastai stop instance` ("stopping instance 61910115.", 10:19 UTC), after which SSH stopped answering. Confirm in the Vast console. |

**What the first attempt found.** The first attempt, `pretest1`, stopped at the baseline with `indices should be either on cpu or on the same device`. The `ragged` pattern built its lookup table on the CPU while the GPU model passed token ids on `cuda:0`. CPU-only runs cannot show this. The fix is `device=t.device` in `moe_common.forced_experts`. `pretest1`'s outputs were deleted, and `pretest2` is the same command after the fix.

### B.2 Correctness

**Fixed results:**

- `check_local`: 29/29 PASS, the same numbers as on CPU.
- `symbols`: 0 not found.
- `DISPATCH`: PASSED on NCCL.

**Baseline** (one GPU, small preset), sha `8fcdb6b6d9b532f3`. The CPU sha differs, as expected on different hardware. The min top-1 margin matters only for `learned`; the forced patterns ignore the argmax.

| Pattern | CE | aux/α per layer | Min top-1 margin |
| --- | --- | --- | --- |
| learned | 6.7032547 | 1.017 / 1.022 | 4.56e-4 |
| balanced | 6.7257767 | — | — |
| hot | 6.7142153 | — | — |
| ragged | 6.7276926 | — | — |
| all_to_one | 6.7239490 | — | — |
| presorted | 6.7331486 | — | — |

**EP=4 parity** (rel ≤ 1e-5, routing and counts exact). In every case, `replicated` was bitwise and the collectives were 6 forward a2a, 4 backward a2a, 2 forward all-reduce and 1 finish all-reduce, for L = 2.

| Case | Worst check | rel | Bitwise checks | Notes |
| --- | --- | --- | --- | --- |
| learned | `grad.router.L1` | 4.3e-7 | 3/18 | logits bitwise, ratios 1.0000 |
| balanced | `dx` | 4.4e-7 | 2/18 | |
| ragged | `grad.router.L1` | 4.9e-7 | 4/18 | |
| all_to_one | `grad.router.L0` | 5.9e-7 | 8/18 | |

**EP=2 × EDP=2 parity** (optional): PASSED in all four cases, worst rel 6.7e-7 (`grad.router.L0`, all_to_one).

**Controls:** `CONTROLS PASSED`.

| Control | Observed |
| --- | --- |
| `wrong_inverse` | the predicted 14 failing checks, ratios from 0.02 to 0.37 |
| `wrong_inverse@presorted` | nothing fails |
| `expert_sum_over_ep` | all eight `grad.expert` checks fail, ratios 0.250–0.270 |
| `expert_no_dp_scale` | experts 0, 1 and 3 fail, ratio 4.0000, rel 3.000 |
| `recv_equals_send` | `SplitMetadataError` on every rank before any token all_to_all |

**Local aux scope** (optional, expected to differ):

- learned and ragged FAILED as P9 predicts, with `loss.aux` ratios 1.0051 / 1.0243 on learned. Only layer 0's experts were among the failures.
- balanced PASSED.

### B.3 Experiment 1 on one GPU (medium, 4096 tokens, layer 0)

The counts are identical to the CPU run (Appendix A.6).

| Routing | Tokens per expert | max/mean | Drops at cf 1.0, global / per-source | Drops at cf 1.25 | EP=4 largest receive buffer |
| --- | --- | --- | --- | --- | --- |
| learned | `[921,909,1008,1258]` | 1.229 | 234 / 255 | 0 / 6 | 1258 rows, 2.5 MiB |
| learned + bias 1.5 | `[2900,330,410,456]` | 2.832 | 1876 / 1876 | — | 2900 rows, 5.7 MiB |
| hot | `[2867,412,409,408]` | 2.800 | 1843 / 1843 | — | 2867 rows, 5.6 MiB |
| ragged | `[2219,597,0,1280]` | 2.167 | 1451 (35.4 %) / 2304 (56.2 %) | — | 2219 rows, 4.3 MiB |
| all_to_one | `[0,0,0,4096]` | 4.000 | 3072 / 3072 | — | 4096 rows, 8.0 MiB |
| balanced | `[1024]×4` | 1.000 | 0 / 0 | — | 1024 rows, 2.0 MiB |

When every source routes the same way (hot, all_to_one, learned+bias), per-source and global capacity drop the same number of tokens. They differ when sources disagree (ragged, learned).

On learned routing at cf 1.0, the position and probs policies each drop 234 tokens, with 43 in common.

EP=8 [simulated]: hot gives 5.6× balanced and all-to-one 8.0×.

### B.4 Experiment 3: skew timing

Medians over 20 timed steps after 5 warmup steps, in ms. The EP=4 job step is the max over ranks per step.

| Shape | Routing | EP=1 step | EP=4 job step, median (min–max) | vs EP=4 balanced | Largest receive buffer |
| --- | --- | ---: | --- | ---: | --- |
| medium | balanced | 5.52 | 2.57 (2.53–3.69) | 1.00 | 1024 rows, 2.0 MiB |
| medium | hot | 5.51 | 4.22 (4.19–4.76) | 1.64 | 2867, 5.6 MiB |
| medium | all_to_one | 4.22 | 4.66 (4.63–4.85) | 1.81 | 4096, 8.0 MiB |
| large | balanced | 49.85 | 13.59 (13.51–13.77) | 1.00 | 4096, 16 MiB |
| large | hot | 49.73 | 34.62 (34.56–38.56) | 2.55 | 11471, 44.8 MiB |
| large | all_to_one | 47.92 | 47.61 (47.55–48.13) | 3.50 | 16384, 64 MiB |

**Per rank, large all_to_one:**

| Rank | Recv rows | experts | combine | bwd_experts | bwd_dispatch_a2a | dispatch | counts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.04 | 15.49 | 0.04 | 30.69 | 0.16 | 0.37 |
| 1 | 0 | 0.04 | 15.48 | 0.04 | 30.69 | 0.16 | 0.37 |
| 2 | 0 | 0.04 | 15.46 | 0.04 | 30.68 | 0.17 | 0.39 |
| 3 | 16384 | 14.98 | 0.16 | 30.19 | 0.16 | 0.32 | 0.38 |

**Large hot**: rank 0 receives 11471 rows (experts 11.03 + 21.45). Ranks 1–3 receive about 1640 rows each (1.72 + 3.21) and wait about 9.6 ms in combine and 18.6 ms in dispatch-backward.

**What it shows** [pre-test, one run on one host]:

- **Where the waiting goes.** Skew shows up on the idle ranks as time *inside* the combine and the dispatch-backward all_to_alls, almost exactly equal to the busy rank's expert forward and backward. All four ranks end the step together, so the per-rank step times are equal and the "slowest rank" label is a tie-break. The cost of skew is the job step against balanced.
- **Large shape.** Expert ms per 1000 received rows is 0.91–1.05 at every load, so it is compute-bound.
- **Medium shape.** The same number is 0.30–0.62, depending on the row count, so it is launch- and occupancy-bound. This is why its skew ratio is smaller.
- **On one GPU.** All-to-one is the *fastest* EP=1 case (one large GEMM instead of four), and hot costs the same as balanced.
- **The host sync.** The count exchange plus its `.tolist()` (`counts`) costs about 0.37–0.39 ms per layer on every rank and every pattern. The host passes that point about 0.6–0.75 ms after the step starts.
- **Traces.** `bench_trace` wrote eight 0.27–0.30 MB traces (`run_pretest2/traces/`), taken after the timed steps and never used for timing.

### B.5 Optional aux-loss training (one GPU, small preset, bias +2 on expert 0, Adam 1e-3, 200 steps)

Layer-0 max/mean, from 3.46 at step 0:

| α | Step 100 | Step 200 | CE at step 200 |
| --- | ---: | ---: | ---: |
| 0.1 | 1.66 | 1.64 | 0.604 |
| 0.01 | 2.93 | 2.95 | 0.651 |
| 0 | 3.27 | 3.33 | 0.664 |

This is one seed and a hypothesis only.

### B.6 What the pre-test supports and what it does not

**It supports:**

- The reference, the harness and the controls work on NCCL on four GPUs, with the same verdicts as on CPU.
- The P7 derivation holds on this host. The expert work came within 10 % of the prediction, the waiting moved to the next all_to_all, and the job-step ratio was 3.5× at the large shape.
- The runner, the manifest and the download script work end to end.

**It does not support:**

- anything about your `my_moe.py`
- timings on another rental or topology
- run-to-run timing variance, since there is one run and only within-run min/max
- any claim about real MoE training, since one layer was timed

---

## Appendix C: GPU reruns (2026-11-27)

[Reference session](../outputs/day14_downloads/day14_20261127T023509Z_ref/session_day14_20261127T023509Z_ref/) · [Mine session](../outputs/day14_downloads/day14_20261127T023509Z_mine/session_day14_20261127T023509Z_mine/) · [Reference terminal](../outputs/day14_downloads/day14_20261127T023509Z_ref/day14_20261127T023509Z_ref.log) · [Mine terminal](../outputs/day14_downloads/day14_20261127T023509Z_mine/day14_20261127T023509Z_mine.log)

### C.1 Run record and correctness [measured]

Both full sessions ran on container `62787249` with 4 × NVIDIA A100-SXM4-40GB, NV12 between all GPU pairs, driver 595.71.05, Python 3.12.16, PyTorch 2.14.1+cu130 and NCCL 2.30.7. GPU 0–1 were on NUMA 0 and GPU 2–3 on NUMA 1. The Megatron checkout was absent, so the symbols stage was skipped.

| Run ID | UTC | Result | Files re-verified |
| --- | --- | --- | ---: |
| `day14_20261127T023509Z_ref` | 02:35:09–02:37:47 | exit 0, both success markers | 65 |
| `day14_20261127T023509Z_mine` | 02:37:48–02:40:28 | exit 0, both success markers | 68 |

Both sessions passed all 29 local checks and the NCCL dispatch check. All three one-GPU baselines reported the same tensor digest, `8fcdb6b6d9b532f3`. Each EP=4 parity case had exact routing and counts and bitwise-equal logits.

| EP=4 routing case | Worst check | `ref` max rel | `mine` max rel | Bitwise floating-point checks, both |
| --- | --- | ---: | ---: | ---: |
| learned | `grad.router.L1` | 4.33e-7 | 4.33e-7 | 3/18 |
| balanced | `dx` | 4.44e-7 | 4.44e-7 | 2/18 |
| ragged | `grad.router.L1` | 4.86e-7 | 4.86e-7 | 4/18 |
| all_to_one | `grad.router.L0` | 5.93e-7 | 6.13e-7 | 8/18 |

The two-layer model used 6 forward and 4 backward all-to-alls, plus 2 forward all-reduces for auxiliary counts. For gradient finalization, `ref` used 1 dense bucket all-reduce and `mine` used 26 per-parameter dense all-reduces. The parity check reports the finish-stage counts and enforces the forward/backward counts. Ragged expert 2 had exact zero gradients in both layers. The five negative controls reproduced their expected outcomes, including the presorted blind spot and the metadata exception before token dispatch. [Correctness details](notes.md#4-experiment-2-ep4-parity-and-negative-controls)

### C.2 Routing, timing and optional stages [measured]

- Routing counts, capacity results and the auxiliary-training results matched across the two sessions. For learned routing at cf 1.0, per-source capacity dropped 21 more tokens than global capacity (255 vs 234). The position and probs policies each dropped 234 tokens, with 43 in common.
- Large balanced EP=4 took 14.477 ms in the `ref` session and 13.890 ms in the `mine` session. Large all-to-one took 47.455 and 47.443 ms, respectively. Both timing stages used `ref`. The two sessions show the shape dependence of skew and variation in medium timings. No per-implementation speed comparison was made.
- The optional reference EP=2 × EDP=2 test passed in both sessions, with worst rel 6.72e-7. The local auxiliary-loss test passed for balanced routing and differed from the global baseline for learned and ragged routing, as expected. Each session saved eight trace files from the optional trace stage. [Full tables and scope](notes.md#5-experiment-3-skew-at-ep4)
