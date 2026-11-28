# Day 15: actor–learner RL on Ray, CPU rollout actors + one GPU learner

> **Run status, 2026-11-28 [measured].** The same-host `ref` and corrected `mine` runs both passed all 27 core stages on 1 × A100-SXM4-40GB. The CPU quota was 15.36 and Ray counted 15. Both downloaded archives passed the SHA-256 and 86-file manifest checks.
>
> `mine` measured 187.79 ms sync and 119.26 ms async (1.57×). Async started at lag 0 and then stayed at lag 1. All four CartPole learning runs reached the target, and their final weight digests matched `ref`. The original `mine` run stopped at seed 2 with a return of 142.6. Keeping the GAE calculation in `float64` until the final cast restored parity on the rerun, where seed 2 reached 316.5.
>
> [Personal server results](#appendix-c-personal-server-runs-2026-11-28) · [Completed experiment notes](notes.md) · [2026-11-27 pre-test and optional results](#appendix-b-server-pre-test-pretest1-and-pretest2_full-assistant-run-2026-11-27-pre-test)

## Required to close Day 15 (about 4 h including the run)

| # | You can explain | Evidence | Where |
| --- | --- | --- | --- |
| 1 | **Ray's execution model**: what a remote call returns, why `ray.get` in a loop serializes the work, and what `ray.wait` changes. Also, why per-call overhead decides how long a fragment must be. | `probes.log` `get_vs_wait`, `overhead` | §1.1 |
| 2 | **Object store and placement**: put once vs by value, why numpy comes back zero-copy and read-only but torch is copied, how a cgroup quota becomes Ray's CPU count, and what a 0-GPU actor can still see | `probes.log` `put_once`, `zero_copy`, `resources`, `placement`, `pending` | §1.2, Stop 7 |
| 3 | **The loop and its versions**: every fragment carries `policy_version` and a digest. Sync gives lag 0. Your one-step pipeline gives `[0, 1, 1, …]` *by schedule*, not by luck. | `CHECK versions` / `CHECK digest` PASS with `--impl mine` in `exp2_async.log` | §1.3, TODO 3 |
| 4 | **Where the time goes**: the per-phase table sync vs async, which phase hides which, and why async is 1.57× and not 2× | `exp2_sync.log`, `exp2_async.log`, `summary.md`, the two `ray.timeline` traces | §1.4 |
| 5 | **Scaling**: your Exp 3 prediction (1/2/4 actors × two env costs) written *before* the run, then where it was right and wrong, and which side is the bottleneck in each cell | `summary.md` Exp 3, notes §6 | §1.4 P6 |
| 6 | **GAE at fragment boundaries**: terminated vs truncated vs the fragment cut, on the hand example and on 200 random fragments | `check_rl.py --impl mine --tests gae` PASSED, with all three GAE controls detected | §2, TODO 1 |
| 7 | **PPO's ratio as a staleness meter**: near 1 at lag 0 before any update, and how lag 1 changes its distribution. One control breaks the version bookkeeping and the other hides the probability difference. | `check_rl.py --impl mine --tests ppo` PASSED, `control_skip_sync.log`, `control_recompute_logp.log` | §3, TODO 2 |
| 8 | **RLlib's PPO on the same problem**: its `training_step` in three calls, its weight sync, its version counter, where its GAE values come from, and why its own timers mislead on a short run | `rllib.log`, `summary.md` Exp 4, notes §8, filled from Block 3 Stops 1–5 and 7 | Block 3 |

**Done means**

- [ ] P1–P9 predicted in notes §1 before opening the answers. Notes §1 reconstructs the guided discussion and includes corrections. A separate, complete pre-answer record is unverified.
- [x] `my_rl.py` TODO 1–3 written. `check_rl.py --impl mine` passes (6 checks, 4 controls detected).
- [x] `run_server_once.sh day15_20261128T125243Z_mine --impl mine --core` finished with exit code 0 and `CORE_SUCCESS.txt`. The archive was downloaded and its manifest verified.
- [ ] Exp 3 prediction table recorded with a pre-run timestamp. Notes §6 reconstructs the prediction from the discussion, where the exact timestamp was not recorded.
- [x] Instance stopped after the download, with a console check recorded in notes §2. The archive confirms process cleanup. Both containers (62799402 and 63018167) were confirmed stopped in the Vast console on 2026-11-28.
- [x] Notes §3–§8 filled from the logs with [measured] labels. The Questions are answered in notes §9.

**Optional (not needed to close the day):**

- P10–P12
- Block 3 Stop 6 (IMPALA/APPO and V-trace)
- The full-mode stages: `wait` mode with 2 and 4 actors, the reproducibility pairs, 8 actors over the quota, and the oversubscription probe
- Reading only: placement groups, fault tolerance (`max_restarts`), the dashboard, spilling, SEED RL

**Scope on purpose.**

- **Hardware.** One GPU and the CPU quota of one container. Ray runs as a single local node. Nothing here measures multi-node behaviour.
- **Workload.** CartPole-v1 (Exp 2, learning) and a synthetic env with a busy-wait step cost (Exp 3). Two-layer tanh MLPs with 64 hidden units, separate actor and critic, 36,620 bytes of weights.
- **Algorithm.** PPO with clipping, no KL penalty, no value clipping. Async means **one-step pipelined** (lag 1, deterministic schedule). IMPALA-style "first finished" is the optional `wait` mode.
- **Not included.** No V-trace implementation (reading only), no multi-GPU learner (Day 16 does weight sync at scale), no RLlib tuning. The learner is host-bound at this model size. The timings describe this harness on this host and are not a benchmark of Ray or RLlib.

---

## Time budget and cut order

| Block | README budget | Required | Optional / stretch |
| --- | --- | --- | --- |
| 1 Design | 30 min | P1–P9, about 3–4 min each with the small numbers given (35 min) | P10–P12 (+15 min) |
| 2 Code | 2 h | Stage 0 read (15 min), TODO 1 GAE (25), TODO 2 PPO (20), TODO 3 pipeline (30), controls (15): 1 h 45 | — |
| 3 Read | 1 h | Stops 1–5 and 7 (50 min) + 10 min to fill notes §8 | Stop 6 (+15 min) |
| 4 Run + summarize | 30–60 min | upload (5), run `--impl mine --core` (7 min on the pre-test host), download + stop (5), notes §2–§7 (30) | full mode (+5 min of run time) |

The required total is about 4 h 5 min.

**Cut order if you run out of time:**

1. Drop the full-mode optional stages: run with `--core`.
2. Drop Stop 6 and P10–P12.
3. Drop the RLlib *timer* discussion (Exp 4 column 3), but keep its three-call mapping.
4. Run the server stages with `--impl ref` and defer the `--impl mine` Ray rerun. TODO 1 and 2 must still pass `check_rl.py --impl mine` on your machine.

The required rows above never move.

---

## What is new today (and what you already have)

You already have these, and today does not re-teach them: collectives and their costs (Days 02–04), DDP and sharding (Days 03–08), host-bound steps and "time the GPU only after a sync" (Days 01, 10, 13), and the profiler's GC trap (Day 13 Appendix C).

New today:

- **A different program shape.** Days 1–14 ran one SPMD program on every GPU. RL has two roles that do different work at different speeds: **actors** step environments on CPUs and run the policy for inference, and a **learner** trains on a GPU. They exchange two things: fragments of experience (actors → learner) and weights (learner → actors).
- **Ray as the runtime.** Ray gives you remote objects and remote calls instead of `torch.distributed` collectives. Every call returns a *future* (`ObjectRef`). Data lives in a shared-memory object store, and the scheduler places work by declared resources, not by rank.
- **Staleness as a first-class quantity.** The moment actors and learner overlap, the learner trains on data from older weights. Today you tag every fragment with the version that produced it, measure the lag, and watch PPO's importance ratio report it.
- **Episodes do not align with batches.** Fragments cut episodes in the middle, and environments end episodes in two different ways. GAE must treat the three cases differently.

---

## Concept map

Every concept from the planning list, classified. "Required" items are taught *and* exercised. "Reading-only" items are taught at a Block 3 stop or in a paragraph and not exercised.

| Concept | Class | Why this class | Taught at | Exercised by |
| --- | --- | --- | --- | --- |
| Tasks vs actors, ObjectRef | required | the harness is built from actors, and the probe compares the two call costs | §1.1 | `probes` overhead, every stage |
| `ray.get` vs `ray.wait` | required | the serializing-`get` bug is the most common Ray performance mistake, and TODO 3 depends on not calling `get` too early | §1.1 P1 | `probes` get_vs_wait, TODO 3 |
| Object store: put once vs by value | required | the learner publishes weights once for N actors | §1.2 P2 | `probes` put_once, `Learner.publish` |
| numpy zero-copy vs torch copy | required | decides what `set_weights` costs, and it is measured | §1.2 P2, Stop 7 | `probes` zero_copy |
| `/dev/shm`, object-store size, spilling | reading-only | 36 KB of weights never comes close. The run only records `df -h /dev/shm` and the store size. | §1.2 | `disk_before.txt`, `probes` resources |
| `num_cpus` / `num_gpus` | required | they decide placement and `OMP_NUM_THREADS` | §1.2 P3, Stop 7 | `probes` placement |
| Fractional GPUs | optional | the probe shows two 0.5-GPU actors on one device. The harness does not share the GPU. | §1.2 | `probes` placement |
| `CUDA_VISIBLE_DEVICES` | required | a 0-GPU actor can still see and use the GPU, and that is a surprise worth one probe | §1.2 P3, Stop 7 | `probes` placement |
| Placement groups | reading-only | one node, nothing to gang-schedule | §1.2 | — |
| cgroup quota vs `os.cpu_count` | required | the host shows 64 CPUs and the quota allows 7.68. Every scaling statement depends on it. | §1.2 P3, Stop 7 | `probes` resources, every run's `cgroup` line |
| OMP / torch-thread oversubscription | optional | real, but a second-order effect once Ray sets `OMP_NUM_THREADS` | P12 | `probes_oversub`, `exp3n8_*` (full mode) |
| Driver, raylet, worker processes | reading-only | the picture is needed to read a timeline. There is nothing to measure. | §1.1 | `ray_stop.txt` |
| Fault tolerance (`max_restarts`) | reading-only | RLlib's warning in `rllib.log` mentions it. Nothing fails on purpose today. | Stop 2 | — |
| `ray.timeline` | required | the only direct view of overlap between actors and learner | §1.4, Block 4 | `trace_sync`, `trace_async` |
| Dashboard | reading-only | started with `include_dashboard=False`. It shows the same events as the timeline. | Block 4 | — |
| Actor–learner loop | required | the day's subject | §1.3 | every stage |
| `policy_version`, digest, lag | required | the evidence for every staleness statement | §1.3 P4 | `CHECK versions`, `CHECK digest` |
| Sync vs async (pipelined) PPO | required | brief evidence #3–#5 | §1.3–§1.4 | Exp 2, Exp 3, TODO 3 |
| IMPALA / APPO / V-trace | optional | the correction for lag > 1. Reading plus the `wait`-mode experiment. | §3.2, P10, Stop 6 | `wait_n2`, `wait_n4` (full mode) |
| Fragments, truncation vs termination | required | the easiest GAE bug to make silently | §2 P7 | TODO 1, GAE controls |
| GAE | required | reused on Day 19 | §2 P7 | TODO 1, `check_rl` |
| Episode bookkeeping | required | the learning check needs correct returns across fragment cuts | §2 | `CHECK learn` |
| Vector envs (E per actor) | required | the inference cost is per *vector* step. The Exp 3 model needs it. | §1.4 | Exp 3 |
| SEED RL (central inference) | reading-only | the alternative when actor inference dominates. Today it does: 25 of 35 ms. | P12 | — |
| Seeding and reproducibility | optional | sync and async are deterministic schedules; `wait` is not | P11 | `repro_*` (full mode) |
| RLlib new API stack | required | names verified against Ray 2.60.0 | Block 3, Exp 4 | `rllib` |

---

## Read this first: the host

You need **one GPU**. Any GPU works. The learner trains a 36 KB model and is host-bound. What matters is the container's CPU quota, because the actors, the learner and Ray's own processes share it. `environment.log` and every run print `os.cpu_count`, the affinity mask, cgroup `cpu.max` and Ray's CPU count. Compare timings only within one host.

The pre-test used container `62799402`: 1 × A100-PCIE-40GB, a 64-CPU host with `cpu.max = 768000 100000` (7.68 CPUs), 60 GB disk, 23 GiB `/dev/shm`. The core run needs 2 actors + 1 learner, and Exp 3 needs up to 4 actors + 1 learner, so 7 CPUs are enough. [Pre-test record](#b1-run-record)

The 2026-11-28 personal comparison used container `63018167` with 1 × A100-SXM4-40GB, `cpu.max = 1536000 100000` (15.36 CPUs), and 31 GiB `/dev/shm`. Ray exposed 15 logical CPUs. The personal `ref` and `mine` timings form a same-host comparison. [Personal run record](#c1-run-record)

**Install Ray once.** It must not touch torch. Pin torch and its CUDA wheels with a constraints file:

```bash
/venv/main/bin/python -m pip freeze | grep -E "^(torch|numpy|nvidia-nccl|triton|torchvision|torchcodec)==" > /tmp/cons.txt
```

```bash
/venv/main/bin/python -m pip install -c /tmp/cons.txt "ray[default,rllib]==2.60.0" "gymnasium==1.2.2"
```

Ray 2.60.0 (commit `b7f4259e`) was released more than two weeks before this guide, installs on Python 3.12 and leaves torch 2.14.1+cu130 unchanged. Check `df -h /workspace /dev/shm` afterwards.

**Upload with `scp`, not git.** From your Windows project root, build one tar and send it:

```powershell
tar -cf day15_upload.tar README.md day15 experiments/day15_ray
```

```powershell
scp -P <port> day15_upload.tar root@<host>:/workspace/
```

```bash
mkdir -p /workspace/AI-Infra-Learning && cd /workspace/AI-Infra-Learning && tar -xf /workspace/day15_upload.tar && ln -sfn /venv/main .venv
```

The run writes about 1 MB plus logs.

---

## Files

See [experiments/day15_ray/README.md](../experiments/day15_ray/README.md) for the full table, the commands, the exit codes and the deviations from RLlib. The file you write is `my_rl.py`. The ones you read first are `workers.py` (`RolloutWorker.sample`, `Learner.update`) and `train.py` (the three driver loops).

---

## Block 1: Design (30 min). Predict first, then open the answer

Write each prediction in notes §1 *before* opening its answer. The numbers are small on purpose. Each section ends with one **Question** that you answer in notes §9 after the run. None of them have a single-word answer.

The system you are designing:

![One sync iteration of the Day 15 loop on Ray. The driver (train.py) calls ray.init, wraps RolloutWorker with ray.remote(num_cpus=1) and Learner with ray.remote(num_cpus=1, num_gpus=1), and creates the actor handles. Step 1: refs = a.sample.remote(128) on each RolloutWorker; each returns a 21.5 KB fragment (obs, actions, logp, values, next_values, rewards, terminated, truncated, meta with policy_version and digest) inline to the driver's in-process memory store. Step 2: ray.get(learner.update.remote(*refs)); Ray copies the fragment values into the call, and the Learner computes lag, checks the digest, runs GAE on CPU, one H2D copy and 16 PPO steps on the GPU, increments version and calls publish(). Step 3: the Learner calls ray.put(weights) once per version, writing a 36,620 B dict of numpy arrays into the shared-memory plasma store, and returns stats inline. Step 4: ray.get of a.set_weights.remote(stats['pub']) on every actor; each actor calls ray.get(pub['ref'][0]) for a zero-copy, read-only view. Ray's GCS and raylet start with ray.init, schedule actors by num_cpus and num_gpus, and set OMP_NUM_THREADS and CUDA_VISIBLE_DEVICES. Returns of 100 KiB or more go to plasma instead of inline. Async mode submits the next sample before the update, giving lag 1. ray.wait, ray.timeline and ray.shutdown are also shown.](assets/d15_ray_system.svg)

The figure is generated by [`assets/make_figures.py`](assets/make_figures.py): `python day15/assets/make_figures.py day15/assets`.

The driver passes **refs**, and the learner receives values; nobody touches the bytes by hand. Where the bytes sit depends on their size, not on who passes the ref:

- **Small return values** (< 100 KiB, `max_direct_call_object_size`) go back **inline** to the caller's in-process memory store. A CartPole fragment is about 21.5 KB, so its bytes do reach the driver process. When the driver passes `F0_ref` to `update.remote`, Ray copies those bytes into the call. The driver never deserializes them.
- **`ray.put` objects, and returns of 100 KiB or more,** live in the node's shared-memory store (plasma). The weights are `ray.put` once per version by the learner. Every actor's `ray.get` on that ref maps the same pages: a zero-copy, read-only numpy view.

### §1.1 Ray's execution model: futures, and who waits

`f.remote(x)` returns at once with an `ObjectRef`, a future for the result. On an **actor** (a class instance living in its own worker process), method calls from one caller run **one at a time, in submission order**. A **task** is a stateless function. Each call can land on any worker, so it pays for scheduling every time.

`ray.get(ref)` blocks until that one result exists. `ray.wait(refs, num_returns=1)` returns as soon as *any* one is ready. The trap: `ray.get` inside the submission loop turns parallel work into a sequence.

**P1.** Four actors, one call each, taking 0.4, 0.1, 0.3 and 0.2 s. After each result the driver spends 0.05 s processing it. Find the time to the first processed result and the total for:

- (a) submit one call, `ray.get` it, process it, then submit the next
- (b) submit all four, then `ray.get` them in submission order
- (c) submit all four, then loop on `ray.wait(..., num_returns=1)`

<details><summary>Answer P1</summary>

(a) Everything is serial: 0.4 + 0.1 + 0.3 + 0.2 + 4 × 0.05 = **1.2 s**. The first result is processed at 0.45 s.

(b) All four run at once. The first `get` waits for the 0.4 s call. By then all four are done, so the rest is processing. First result at **0.4 s** (processed by 0.45), total 0.4 + 4 × 0.05 = **0.6 s**.

(c) Results are processed in finishing order: ready at 0.1, done 0.15; ready 0.2, done 0.25; ready 0.3, done 0.35; ready 0.4, done **0.45 s**. First result at **0.1 s**. When processing is shorter than the gaps between arrivals, `wait` hides it completely.

Pre-test: 1.213 s; first 0.403 / total 0.604 s; first 0.103 / total 0.453 s, order `[0.1, 0.2, 0.3, 0.4]`. [Probes](#b2-exp-0-ray-probes)

</details>

**Question 1.** The probe measured a round trip of 765 µs for an actor method and 1425 µs for a task. One CartPole `sample()` of 128 vector steps took about 35 ms. Below what fragment length would the per-call overhead exceed 10 % of the rollout time? Why does the harness use actors, and not a task per fragment, even though both would compute the same thing?

### §1.2 The object store and placement

**Objects.** `ray.put(x)` serializes `x` once into the node's shared-memory object store (on Linux, `/dev/shm`) and returns a ref. Passing a ref to N calls costs one serialization. Passing the *value* to N calls serializes it N times. Objects are **immutable**: a ref always means the bytes that were put.

**Zero copy.** A numpy array is stored as a raw buffer. `ray.get` returns a numpy view onto shared memory, **read-only**, with no copy. A torch tensor is pickled and copied on every `get`, unless you opt in with `RAY_ENABLE_ZERO_COPY_TORCH_TENSORS` (default off in 2.60.0, Stop 7). The learner therefore publishes **numpy** arrays, and each actor copies them into its own parameters.

**Placement.** An actor declared with `num_cpus=k` gets `k` logical CPUs out of the node's total, and Ray sets `OMP_NUM_THREADS=floor(k)` in it. `num_gpus=g` sets `CUDA_VISIBLE_DEVICES` to the assigned GPU ids. An actor whose request no node can satisfy stays **pending**, with no error, until the cluster changes. Ray reads the cgroup quota to decide the node's CPU total.

**P2.** The learner publishes a 64 MiB weight dict to 4 actors.

- (a) How many bytes does the driver serialize with put-once? By value? Predict the ratio of the two submit times.
- (b) Inside an actor, `ray.get` returns the 64 MiB object. Which of numpy and torch is near-instant? Can the actor write into the returned numpy array?
- (c) For today's 36,620 bytes of weights, does any of this matter?

<details><summary>Answer P2</summary>

(a) 64 MiB vs 4 × 64 = 256 MiB. The ratio should be near 4 if serialization dominates. Pre-test: submit **26.6 ms vs 95.2 ms** (3.6×), and all four loaded at 28.5 vs 97.4 ms.

(b) numpy: **0.18 ms**, a read-only view (`writeable False`, `owns_data False`). Writing into it raises an error. Torch: **77.4 ms**, a full copy of 64 MiB on every `get`. Two `get`s of the numpy ref in the driver share memory, and two of the torch ref do not.

(c) Barely. The pre-test's `sync.put` took 0.75 ms and the actors' `sync.get` 0.24 ms. Most of that is fixed per-call cost, not bytes. This matters on Day 16, where the weights are gigabytes and the actors are GPUs.

</details>

**P3.** The host has `os.cpu_count() = 64`, and the container's cgroup has `cpu.max = 768000 100000`. Predict:

- (a) Ray's CPU total
- (b) `OMP_NUM_THREADS` inside a `num_cpus=1` actor, and inside a `num_cpus=2` actor
- (c) `CUDA_VISIBLE_DEVICES` and `torch.cuda.device_count()` inside a `num_gpus=0` actor
- (d) what happens to an actor that asks for `num_cpus = total + 1`

<details><summary>Answer P3</summary>

(a) The quota is 768000/100000 = 7.68 CPUs. Ray truncates it to **7** and warns that it "does not support initializing Ray with fractional cpus" (`_private/utils.py:488`). `os.cpu_count()` still says 64, and a thread pool sized from it would oversubscribe the quota about 8×.

(b) **1** and **2** (`floor(num_cpus)`, `utils.py:270`). torch picks up the same thread count.

(c) `CUDA_VISIBLE_DEVICES` is **unset** for a 0-GPU actor (`utils.py:292` skips it), so `torch.cuda.device_count()` is **1**. A CPU actor can create CUDA tensors on a GPU that the scheduler gave to someone else. Ray does not stop it. Set `CUDA_VISIBLE_DEVICES=""` yourself (for example in `runtime_env`) if that matters.

(d) It stays **pending**: no error, only an autoscaler message ("No available node types can fulfill resource request {'CPU': 8.0}"). The probe gives up after 5 s. A forgotten `num_cpus=8` looks like a hang.

</details>

**Question 2.** `RolloutWorker.set_weights` copies the read-only numpy views into its torch parameters (`workers.py:63`). Suppose instead it kept the views and made the parameters point at them. What would go wrong at the next `sample()`, and why can the actor never see the learner's *next* version through this view? What does the answer say about how Ray objects are versioned?

### §1.3 The loop: versions, lag and the schedule

Each `update` advances the learner's version from $v$ to $v+1$. Each fragment records the version $b$ of the weights that produced it. The **lag** of a fragment is

$$\ell = v_{\mathrm{learner}} - b,$$

where $v_{\mathrm{learner}}$ is the version the learner holds when it starts training on the fragment. At $\ell = 0$ the data is on-policy for the weights being trained.

Two schedules, both fully determined by submission order:

- **sync**: `set_weights(v)` → `sample` on every actor → `update` → repeat. The actors idle while the learner trains.
- **async (one-step pipelined)**: at iteration $k$, first submit the *next* `sample` on every actor, then run `update` on the fragments already in flight, then queue `set_weights(v+1)` on every actor. Actor calls run in submission order, so sample $k+1$ runs with the weights queued *before* it, never the ones being computed.

The pre-test traces of the last two iterations, drawn to scale from the `ray.timeline` spans (one character = 5 ms; actor 1 looks the same as actor 0):

- `L` is `set_weights`.
- `[B2 v1]` is a `sample` producing batch 2 with weights v1.
- `[B1(v0) v1→v2]` is an `update` on batch 1, which was sampled with v0, taking the learner from v1 to v2. Its lag is 1 − 0 = 1.

```text
sync  (trace_sync)     one iteration = 110 ms traced (103 ms untraced)
time ms   0         50        100       150       200
actor 0   ..............L[B1 v1]..............L[B2 v2]
learner   [B0(v0) v0→v1]........[B1(v1) v1→v2]........
lag                 0                     0

async (trace_async)    one iteration = 71 ms traced (66 ms untraced)
time ms   0         50        100       150
actor 0   L[B2 v1]......L[B3 v2]......L[B4 v3]
learner   [B1(v0) v1→v2][B2(v1) v2→v3][B3(v2) v3→v4]
lag             1             1             1
```

In the async picture, while the learner turns v1 into v2 using batch 1 (sampled with v0), the actors are already sampling batch 2 with v1. The learner is idle only for the 2–3 ms between updates, the actors idle for $U + o - R$, and every batch after the first is one version old.

**P4.** Use R = 35 ms for one `sample` on both actors (in parallel), U = 62 ms for GAE + H2D + PPO + publish, and o ≈ 6 ms of other per-iteration overhead (transfer, return, set_weights).

- (a) Write the lag of the first four batches in sync and in async.
- (b) Predict the iteration time of each schedule and the actors' idle time per iteration.
- (c) TODO 3 submits three kinds of call in a fixed order. What happens to the lags and to the iteration time if the next `sample` is submitted only *after* the update's stats have come back? What does it cost to `ray.get` the in-flight fragments and pass arrays to `update` instead of refs?

<details><summary>Answer P4</summary>

(a) Sync: `0, 0, 0, 0`. Async: `0, 1, 1, 1`. The first batch was sampled with $v_0$ and trained at $v_0$. From then on, the next batch was launched before the update that would have refreshed its weights. The pre-test `exp2_async` histogram was `{0: 2, 1: 44}` (2 actors × 1 batch at lag 0).

(b) Sync: $T = R + U + o \approx 35 + 62 + 6 = 103$ ms, and the actors idle for $U + o \approx 68$ ms. Async: $T = \max(R,\ U + o) \approx 65$–68 ms, and the actors idle for $U + o - R \approx 31$ ms. The learner is now the bottleneck. Pre-test: **103.25 ms and 65.93 ms**, with `idle.actor` **68.6 and 30.9 ms**.

(c) Submitting the next `sample` after the update means it queues behind `set_weights(v+1)`, so it samples with the newest weights: every lag becomes **0** and the actors idle during every update. The schedule has collapsed back to sync. `CHECK versions` **fails** (it expects `[0, 1, 1, …]`), and e2e goes back to about 103 ms. The same happens if you try to queue `set_weights` first: $v+1$ only exists after the update.

Calling `ray.get` on the fragments still passes every check, but now the driver **deserializes** every fragment into numpy arrays and **serializes** it again for `update.remote`, on the critical path, and only `transfer` grows. With refs, Ray forwards the small fragments' bytes (they already sit inline in the driver's memory store, see the Block 1 figure) without unpacking them; large ones stay in plasma and the learner reads them there. That is why `update` takes refs.

</details>

**Question 3.** RLlib's `sync_weights` returns without waiting for the actors (`timeout_seconds=0.0`, Stop 2). Suppose this harness did the same and an actor could start its next `sample` *before* its `set_weights` ran. Which lags could then appear in async mode? Which check would catch a fragment whose tag says $v$ while its weights are still $v-1$, and which check would *not*? Use the `skip_sync` control's log to support the answer.

### §1.4 Measuring the phases, and the bottleneck model

Every stamp in this harness is `time.monotonic()`, one clock for every process on the host, so differences between processes are meaningful ([derived]). For each iteration:

| Phase | Measured where | Meaning |
| --- | --- | --- |
| `rollout` (= `env` + `infer` + bookkeeping) | actor | `sample()` wall time, slowest actor |
| `transfer` | learner start − last fragment end | dispatch + object transfer, **plus queueing behind a busy learner** |
| `learn.gae`, `learn.h2d`, `learn.update` | learner, GPU synchronized | GAE on CPU, one H2D copy, all PPO epochs |
| `sync.d2h`, `sync.put` | learner | parameters to numpy, one `ray.put` |
| `sync.get`, `sync.load`, `sync.digest` | actor | fetch, copy into parameters, SHA-256 of the real parameters |
| `return` | driver − learner end | stats back to the driver |
| `idle.actor`, `idle.learner` | e2e − busy | waiting |

The first `--warmup` iterations (CUDA initialization, about 0.8 s) are excluded, and the table reports medians.

**The model.** For N actors with E = 4 envs each, a global batch of B steps, an env cost c per `env.step`, i = inference per *vector* step, $e_0$ = the env's own step cost, S optimizer steps of u each, g = learner CPU work per sample and o = fixed overhead:

$$R = \frac{B}{4N}\left(i + 4(e_0 + c)\right), \qquad U = S u + B g$$

$$T_{\mathrm{sync}} = R + U + o, \qquad T_{\mathrm{async}} = \max(R,\ U + o).$$

Async can never beat the slower side. Its best speedup over sync is $(R + U + o)/\max(R,\ U + o) \le 2$, reached only when the two sides are equal.

**P5.** In the Exp 2 async run, predict the `transfer` phase. Is the object store slow? What is the largest speedup async could give at these R and U?

<details><summary>Answer P5</summary>

`transfer` is learner start minus fragment end. In async, the fragment for iteration $k+1$ finishes about 35 ms into the 62 ms update of iteration $k$, so it waits about $U + o - R \approx 30$ ms in the queue. Pre-test: **31.2 ms async vs 1.85 ms sync**. Same bytes, same store: the difference is queueing. A phase computed from two processes' stamps includes all the waiting between them, which is why the table labels it [derived].

Largest speedup: $103/\max(35,\ 68) \approx 1.5$. Pre-test: 103.25/65.93 = **1.57×**. Making the actors faster would not help async at all. Only a faster learner would.

</details>

**P6 (Exp 3, write it before the run).** The synthetic env has `obs_dim` 16 and a dialed busy-wait step cost c. The global batch per iteration is fixed at B = 4096 steps (fragment length 4096/(4N) per actor), minibatch 512, 4 epochs, so S = 32 optimizer steps. From Exp 2 sync, the pre-test derived i = 194 µs per vector step, $e_0$ = 14 µs, u = 3.75 ms, g = 1.45 µs per sample and o = 4.94 ms. Predict e2e and the bottleneck side for N ∈ {1, 2, 4}, c ∈ {0, 400 µs}, sync and async: 12 cells. Where will adding actors stop helping?

<details><summary>Answer P6</summary>

$U = 32 \times 3.75 + 4096 \times 0.00145 \approx 126$ ms. For c = 0, $R = (1024/N)(0.194 + 0.056) \approx 256/N$ ms. For c = 400 µs, $R \approx (1024/N)(0.194 + 1.656) \approx 1894/N$ ms.

| c | N | R | sync predicted / pre-test | async predicted / pre-test | bottleneck |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | 1 | 256 | 387 / 423 | 256 / 275 | rollout |
| 0 | 2 | 128 | 259 / 276 | 131 / 141 | about equal |
| 0 | 4 | 64 | 195 / 216 | 131 / 141 | learner |
| 400 | 1 | 1894 | 2025 / 2064 | 1894 / 1918 | rollout |
| 400 | 2 | 947 | 1078 / 1109 | 947 / 970 | rollout |
| 400 | 4 | 474 | 605 / 628 | 474 / 485 | rollout |

With a free env, async stops improving at N = 2: 2 and 4 actors give the same 141 ms, because the learner is the bottleneck. With an expensive env, rollout dominates and the time halves with each doubling of N. The model was 2–10 % optimistic in every cell. It leaves out per-call costs that grow with N, the extra actor's inference, and noise. [Exp 3 pre-test](#b5-exp-3-scaling)

</details>

**Question 4.** In Exp 3 with c = 0, async gave 141 ms with both 2 and 4 actors. Which of these would make it faster, and by roughly how much: (a) 8 actors, (b) minibatch 1024 instead of 512 (half the optimizer steps), (c) a second learner GPU with gradient all-reduce? Use the measured u = 3.75 ms per optimizer step and the fact that this learner is host-bound (the GPU kernels for a 64-unit MLP are microseconds).

### §2 Fragments, termination, truncation and GAE

A fragment is T steps × E envs, time-major. Episodes run across fragments. Row $t$ is the transition $s_t \to s_{t+1}$, and three flags can end the recursion there:

| Case at row t | Bootstrap from $V(s_{t+1})$? | Recursion continues into t+1? | Example |
| --- | --- | --- | --- |
| **terminated** | no: the episode really ended, the future value is 0 | no | the pole fell |
| **truncated** | **yes**, from V(final observation) | no: the next row belongs to a new episode | CartPole's 500-step time limit |
| **fragment cut** (last row) | **yes**, from $V(s_T)$ | no: the next row is in the next fragment | every fragment's last row |

With $d_t$ = terminated and $e_t$ = (terminated or truncated or last row):

$$\delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)$$

$$A_t = \delta_t + \gamma \lambda (1 - e_t) A_{t+1}, \qquad \text{return target} = A_t + V(s_t).$$

The actor fills `next_values[t]` = $V(s_{t+1})$ for every row: from the next forward pass, from an extra forward on the final observation at a truncation (the env has already reset, so the policy never sees that observation otherwise), and from one extra forward at the cut (`workers.py:133` and `:143`).

**P7.** One env, six rows, γ = λ = 0.5 (so γλ = 0.25):

| t | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reward | 1 | 0 | 1 | 1 | 0 | 1 |
| V(s_t) | 1 | 2 | 2.5 | 0.5 | 2 | 2 |
| V(s_t+1) = `next_values` | 2 | 2.5 | 4 | 9 | 2 | 4 |
| flag | | | truncated | terminated | | (cut) |

- (a) Compute δ and A for every row, then the return targets.
- (b) Three planted bugs: treat truncated as terminated, do not bootstrap at the cut, and let the recursion run across episode ends. Which entries of A does each one change?

<details><summary>Answer P7</summary>

(a) δ = `[1, −0.75, 0.5, 0.5, −1, 1]`. Row 2 bootstraps from 4 (truncated). Row 3 ignores the 9 (terminated: whatever is in `next_values` there must not matter). Row 5 bootstraps from 4 (the cut).

Backward: $A_5 = 1$, $A_4 = -1 + 0.25 \times 1 = -0.75$, $A_3 = 0.5$ (end), $A_2 = 0.5$ (end), $A_1 = -0.75 + 0.25 \times 0.5 = -0.625$, $A_0 = 1 + 0.25 \times (-0.625) = 0.84375$.

A = `[0.84375, −0.625, 0.5, 0.5, −0.75, 1.0]`, returns = `[1.84375, 1.375, 3.0, 1.0, 1.25, 3.0]`.

(b)

| Control | δ changed | A becomes | Entries changed |
| --- | --- | --- | --- |
| `gae_trunc_as_term` | $\delta_2 = 1 - 2.5 = -1.5$ | `[0.71875, −1.125, −1.5, 0.5, −0.75, 1.0]` | 0, 1, 2: the truncated row and everything before it in its episode |
| `gae_no_cut_bootstrap` | $\delta_5 = 1 - 2 = -1$ | `[0.84375, −0.625, 0.5, 0.5, −1.25, −1.0]` | 4, 5: the cut row and its episode's earlier rows |
| `gae_cross_boundary` | none | `[0.8486, −0.6055, 0.578125, 0.3125, −0.75, 1.0]` | 0–3: the next episode's advantages leak backwards |

Each control fails both the hand example and the 200-fragment comparison against the O(T²) definition. Max |diff| was 3.86, 3.29 and 18.2 in the pre-test. [check_rl](#b3-correctness)

</details>

**Question 5.** CartPole gives reward 1 per step and truncates at 500. Near the time limit, the true $V(s) \approx 1/(1-\gamma) = 100$ at γ = 0.99. What return target does `gae_trunc_as_term` teach the critic for the last state before truncation, and what does that do to the advantages of the good actions that kept the pole up for 500 steps? Why would this bug make a *good* policy look worse to itself, and only once the policy is good?

### §3.1 PPO's clipped surrogate

For each sample, the importance ratio between the policy being trained and the behaviour policy that acted is

$$\rho = \exp\left(\log \pi_\theta(a \mid s) - \log \mu(a \mid s)\right),$$

where $\log \mu$ is the `logp` the actor stored at sampling time. With advantage A and clip ε:

$$L = -\frac{1}{B} \sum_i \min\left(\rho_i A_i,\ \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right).$$

The statistics this harness reports, all computed under `no_grad` and returned as tensors (no `.item()` per minibatch):

- clip fraction = mean of $\lvert \rho - 1 \rvert > \epsilon$
- max ratio deviation = $\max \lvert \rho - 1 \rvert$
- approximate KL (the k3 estimator) = $\frac{1}{B}\sum_i \left((\rho_i - 1) - \log \rho_i\right)$

**P8.** Five samples, ε = 0.2: ratios `[0.5, 0.9, 1.0, 1.1, 1.5]`, advantages `[1, −1, 1, 1, −1]`.

- (a) Find the loss, the clip fraction, max |ρ − 1| and the approximate KL.
- (b) Find $\partial L / \partial \log \pi_\theta$ for each sample. Which samples get zero gradient?

<details><summary>Answer P8</summary>

(a) The min terms are 0.5, −0.9, 1.0, 1.1 and −1.5. Sample 0: 0.5 < 0.8, the unclipped term. Sample 4: −1.5 < −1.2, again the unclipped one. Their mean is 0.04, so **L = −0.04**. Clip fraction **0.4** (samples 0 and 4), max deviation **0.5**, approximate KL **0.05955**.

(b) In every sample the min picks the *unclipped* term, so $\partial L/\partial \log\pi_i = -A_i \rho_i / 5$ = **`[−0.1, 0.18, −0.2, −0.22, 0.3]`**. **No sample has zero gradient.** Samples 0 and 4 are outside the clip range on the *pessimistic* side: the policy made a good action less likely (or a bad one more likely), and the min keeps the full gradient that undoes it. The clip only removes the gradient when the ratio has moved past $1 \pm \epsilon$ in the direction the advantage wants. "Clip fraction" counts samples outside the band. It is not the fraction with no gradient.

`check_rl` checks the loss, the three statistics and this gradient. [check_rl](#b3-correctness)

</details>

**Question 6.** Sync runs have clip fraction exactly 0 on the first minibatch but a median of 0.032 over all minibatches. Async lag-1 runs had first-minibatch clip fractions from 0 to 0.28 and a median over all minibatches of 0.054. Why is the all-minibatch value non-zero even at lag 0? What fraction of the async clipping is "free" staleness that the learner did not cause itself, and how would you measure it from the logs?

### §3.2 Staleness, and two ways to get it wrong

At lag 0, the learner's weights at the start of `update` are bitwise the weights the actor used, up to the device: the actor computed `logp` on CPU and the learner recomputes it on GPU. So the first minibatch has $\rho = 1$ up to float32 rounding. At lag 1, the learner already took one full update (16 optimizer steps) since the behaviour weights, so $\rho \neq 1$ from the first minibatch on. The ratio is a **staleness meter**, and the harness's `CHECK ratio` reads it.

Two planted bugs, both run as negative controls:

- **`skip_sync`**: actor 1 ignores the version-2 weights but tags its fragments as version 2. Its digest is the SHA-256 of the parameters it really holds.
- **`recompute_logp`**: the learner throws away the stored behaviour `logp` and recomputes it with its own current weights before PPO. This is the "use the learner's logp as old logp" mistake.

**P9.**

- (a) At lag 0, what is max |ρ − 1| on the first minibatch? At lag 1?
- (b) Under `skip_sync`, which of the version, digest and ratio checks fails? Is PPO's math for that fragment wrong?
- (c) Under `recompute_logp`, which check fails, and why is this bug worse than (b) for learning?

<details><summary>Answer P9</summary>

(a) Lag 0: float32 noise, **≤ 4.77e-7** in the pre-test (the CPU and GPU forward passes round differently). Lag 1: median first-minibatch **0.245** (`exp2_async`) and **0.287** (`learn_async_s0`), with first-minibatch clip fractions up to 0.28.

(b) Only **digest** fails: `(iteration 3, actor 1, claimed version 2)`, 11/12 fragments match. Versions pass, because the tag is the claimed version. Ratio passes too, and that is the instructive part: the stored `logp` came from the weights that really acted, so the importance ratio is still correct. Only the *bookkeeping* is wrong (the true lag is 2, the claimed lag 1). Lag statistics, staleness limits and anything that drops "too old" data would be silently wrong.

(c) **ratio** fails: every lag-1 batch has ρ = 1 exactly, `HIDDEN 5`. The learner now treats stale data as on-policy. PPO's clip never engages for staleness, and the gradient is biased by the policy mismatch it can no longer see. Nothing crashes, and the loss curve looks normal. This is why the check flags "lag ≥ 1 but ratio = 1" as a failure.

V-trace (IMPALA, APPO, Stop 6) is the correction for larger lag: it truncates the importance weights at $\bar\rho$ (default 1) so that a very stale sample cannot dominate the value targets, trading bias for bounded variance. With lag ≤ 1, PPO's clip is enough.

</details>

**Question 7.** `skip_sync` was caught only by the digest, while the ratio check passed. Give one production situation where the digest check is the *only* thing that can catch a broken weight sync (think of what happens to a fragment whose behaviour policy is two versions old but whose tag says one). Why is a digest of the *loaded* parameters a better check than comparing version numbers?

### §4 Optional predictions (P10–P12, +15 min)

**P10 (`wait` mode, IMPALA-style).** The driver waits for the *first* finished fragment, updates on it alone, refreshes only that actor and relaunches it. With N equal-speed actors, what lag do you expect in steady state? How does the lag histogram for N = 4 compare with N = 2? Hint: a one-fragment update (512 steps, 8 optimizer steps) takes about as long as one `sample`.

<details><summary>Answer P10</summary>

If the learner keeps up, each fragment arrives after the other N − 1 actors' fragments have each caused one update, so the lag is **N − 1**: 1 for N = 2, 3 for N = 4.

That reasoning assumes the fragments are consumed in arrival order. Here the learner is the bottleneck: one update takes about 37 ms, as long as one `sample`. With 4 actors, several fragments are already finished at every `ray.wait`. When more than one ref is ready, `ray.wait(num_returns=1)` returns an **arbitrary** one, not the oldest.

Pre-test:

- N = 2: `{0: 1, 1: 39}`, which is N − 1.
- N = 4: `{0: 1, 1: 33, 2: 3, 3: 1, 8: 1, 23: 1}`, mostly lag 1. Two actors kept winning while the others' finished fragments waited in the store. One fragment waited 1.49 s, a lag of 23.
- A separate probe reproduced the pattern. With all four refs ready at each call, the consumed actors were `[3, 2, 3, 0, 3, 0, 3, 0, …]`, and actors 1 and 2 starved.

The lag distribution is a property of the **consumer's policy**, not of N. A fair queue (oldest first) is the driver's job. And lag 23 is exactly where V-trace's $\bar\rho$ clip, or a "drop if older than k" rule, has to step in. [B.7](#b7-optional-stages-pretest2_full)

</details>

**P11 (reproducibility).** Each mode runs twice with the same seed. For which modes will the final weight digests be identical?

<details><summary>Answer P11</summary>

**sync and async**: their schedules are fixed by submission order, the actors sample with seeded torch generators and seeded envs, and the learner runs with TF32 off and deterministic algorithms. **wait**: not guaranteed. Arrival order and `ray.wait`'s pick depend on timing, and a different order means different batches per update.

Pre-test: sync `3dac3cf17bce9b88` twice, async `3bfa23e723801a95` twice, wait `de539c319ec69701` vs `820fbbeeea3d29e6`. The learning and Exp 2 runs also ended with identical digests in both pre-test sessions (`e22feb74…` for `learn_sync_s0`), while their timings differed by up to 6 %. Determinism here is a property of the schedule, not of the timing. [B.7](#b7-optional-stages-pretest2_full)

</details>

**P12 (oversubscription and central inference).**

- (a) With `--ray-num-cpus 10` on a 7.68-CPU quota, 8 actors and a learner start. What happens to rollout time, and where does the cgroup record it?
- (b) Four actors with 8 torch threads each instead of 1, on a forward pass of this 64-unit MLP. Faster or slower per forward?
- (c) Actor inference was 25 of 35 ms per fragment. What would SEED RL change?

<details><summary>Answer P12</summary>

(a) Ray schedules 10 logical CPUs, but the kernel enforces 7.68. When all 8 actors run, the cgroup is throttled: `nr_throttled` and `throttled_usec` grow in the run's `cgroup` line, and rollout time stretches beyond the 4-actor scaling.

Pre-test (c = 400 µs, B = 4096): the quota-free model gives R = 237 ms. Measured rollout was **253 ms sync** (4.4 s throttled) and **281 ms async** (6.6 s throttled; the learner competes for the same quota while the actors sample). Async e2e went from 485 ms at 4 actors to 284 ms at 8: 1.71×, not 2×.

(b) Slower. The threads' synchronization costs more than the tiny matmuls save, and 32 threads on a 7.68-CPU quota lose whole scheduling periods. Pre-test: **1028 µs vs 1929 µs** per forward, CPU used 2.2 vs 7.0 s, and **20.5 s throttled** for the 8-thread version. Ray's `OMP_NUM_THREADS=floor(num_cpus)` exists to prevent exactly this. An actor that calls `torch.set_num_threads` itself overrides it.

(c) SEED RL moves inference to the learner's accelerator: actors only step envs and send observations, and a batched GPU forward serves all of them. Here that would remove most of the 25 ms but add a round trip per step (765 µs per actor call), so it pays only with many envs per actor or a much bigger model. Reading only.

</details>

---

## Block 2: Code (1 h 45 min)

### Stage 0 (15 min): read the harness

Open `workers.py` and `train.py`. Find:

- in `RolloutWorker.sample` (`workers.py:88`), the three places that fill `next_values`: the next forward, the truncation branch `:133` and the cut `:143`
- in `Learner.update` (`workers.py:207`), where lag and `digest_ok` are computed (`:217`–`:218`) and where your `compute_gae` and `ppo_surrogate` are called (`:223`, `:254`)
- the single `.cpu().tolist()` for all minibatch statistics (`:266`) and the version increment (`:269`)
- `Learner.publish` (`:187`): one `ray.put` (`:197`) of numpy arrays for all actors
- the three driver loops in `train.py`: sync `:152`, async `:164` (it calls your `pipelined_iteration` at `:169`), wait `:176`

### Stage 1 (25 min): **YOU WRITE** `compute_gae` (TODO 1a–1c)

```powershell
python experiments/day15_ray/check_rl.py --impl mine --tests gae
```

This needs only numpy, so it runs on Windows. The hand example is P7. The random test compares 200 fragments (T = 37, E = 3, random flags) against `gae_slow_reference`, the O(T²) definition. Your function must work on `[T]` and `[T, E]` inputs. Vectorize over envs and loop over time. Keep the recurrence and `advantages + values` in `float64`, then cast both outputs to `float32`. The first personal run passed the unit-test tolerance with an earlier cast and later diverged during learning. [Precision fix and rerun](#c2-correctness-and-the-gae-precision-fix)

Done at `gae.hand_example`, `gae.vs_slow_reference` and `gae.shape_TxE` PASS.

### Stage 2 (20 min): **YOU WRITE** `ppo_surrogate` (TODO 2a–2c)

```powershell
python experiments/day15_ray/check_rl.py --impl mine --tests ppo
```

This needs torch on CPU. The hand example is P8, including the gradient. `ppo.lag0_ratio_one` builds a policy, stores its `logp`, and checks ρ = 1 exactly. `ppo.lag1_stale` takes four Adam steps first and checks that the stored `logp` gives a ratio ≠ 1 while a recomputed one gives exactly 1. Return the statistics as detached 0-dim tensors: the learner stacks them and copies them to the host once per update. Done at `CHECK_RL (mine): 6 passed, 0 failed, 0 incomplete, controls 4 detected / 0 missed: PASSED`, exit 0.

### Stage 3 (30 min): **YOU WRITE** `pipelined_iteration` (TODO 3a–3c)

On the server, or on any machine with Ray (add `--learner-gpus 0` for a CPU learner):

```bash
python experiments/day15_ray/train.py --impl mine --mode async --iters 8 --warmup 2
```

Three lines and the order is the whole exercise (P4c). Done at `CHECK versions: lag histogram {0: 2, 1: 14}` PASS, `CHECK digest` PASS, `CHECK ratio` PASS. Then compare `e2e` with `--mode sync`: if async is not clearly faster, the next `sample` is not overlapping the update.

### Stage 4 (15 min): the negative controls

```bash
python experiments/day15_ray/check_rl.py --impl ref
```

```bash
python experiments/day15_ray/train.py --mode async --control skip_sync --iters 6 --warmup 1
```

```bash
python experiments/day15_ray/train.py --mode async --control recompute_logp --iters 6 --warmup 1
```

A control run *passes* when its planted bug is detected. Each must fail exactly the check named here:

| Control | Planted bug | Must fail | Must still pass | Pre-test |
| --- | --- | --- | --- | --- |
| `gae_trunc_as_term` | truncated treated as terminated | `gae.hand_example`, `gae.vs_slow_reference` | — | detected, max diff 3.86 |
| `gae_no_cut_bootstrap` | the last row not bootstrapped | both GAE tests | — | detected, 3.29 |
| `gae_cross_boundary` | recursion not reset at episode ends | both GAE tests | — | detected, 18.2 |
| `ppo_recompute_logp` (unit) | old logp recomputed after updates | ratio = 1 at lag 1 | — | detected |
| `skip_sync` (Ray) | actor 1 keeps v1, tags v2 | `CHECK digest` | versions, ratio | detected `(3, 1, 2)` |
| `recompute_logp` (Ray) | learner recomputes old logp | `CHECK ratio` (`HIDDEN 5`) | versions, digest | detected |

### Pitfalls (symptom → cause)

| Symptom | Likely cause |
| --- | --- |
| `INCOMPLETE (TODOs left in my_rl.py: …)`, exit 3 | a function still raises `NotImplementedError` |
| `gae.hand_example` wrong only at rows 0–2 | truncation treated as termination (P7 table) |
| `gae.vs_slow_reference` fails, hand example passes | `end_t` not forced True on the last row, or flags mixed up across the env axis |
| `ppo.hand_example` loss right, gradient wrong | `ratio` computed under `no_grad`, or `clamp` applied to the advantage |
| async `CHECK versions` sees all lags 0, and e2e equals sync | the next `sample` was submitted after the update |
| async run hangs at iteration 0 | `ray.get` on a list that contains an unsubmitted ref, or the learner waiting for a fragment that was never launched |
| `CHECK ratio` lag-0 max ≈ 1e-3 | TF32 on in the learner, or a different policy build between actor and learner |
| a CPU actor allocates GPU memory | `num_gpus=0` does not hide the GPU (P3c) |
| `ray.init` waits about 30 s, then "The current node timed out during startup", with no `raylet.out` in `/tmp/ray/session_latest/logs` | `num_cpus` passed as a float (`10.0`). In Ray 2.60.0 the raylet never starts. Pass an int. This harness's `--ray-num-cpus` had this bug in the pre-test (B.7). |
| `wait` mode shows lags far above N − 1 | `ray.wait` returns an arbitrary ready ref when several are ready (P10). Keep your own oldest-first order. |
| async run "faster" but learning stalls | `recompute_logp`-style bug: compare the lag-1 ratio median with the pre-test's 0.25 |

---

## Block 3: Read RLlib and Ray core (1 h; Stops 1–5 and 7 required, Stop 6 optional)

All paths are under the installed `ray/` package, version **2.60.0**, commit `b7f4259e`. The line numbers come from `find_symbols.py` on the pre-test host, which found all 47 anchors. Run it on your install first. If anything prints `(not found)`, find the stop by name.

```bash
python experiments/day15_ray/find_symbols.py
```

One PPO iteration on RLlib's new API stack, `rllib/algorithms/ppo/ppo.py` `PPO.training_step` `:391`:

```text
synchronous_parallel_sample :400 ──► every EnvRunner.sample() single_agent_env_runner.py:152 ──► episodes (refs)
learner_group.update :428 ──► Learner.update learner.py:1028 ──► learner connector (GAE) ──► _update per minibatch :1124
env_runner_group.sync_weights :465 ──► get_state from the learner, ONE ray.put :802, set_state(ref) on every runner :808
```

### Stop 1 (8 min): one iteration, and its timers. `ppo.py` `training_step` `:391`; `metrics_logger.py` `log_time` `:583`

- `:400`: `synchronous_parallel_sample` blocks until every EnvRunner returns its share of `total_train_batch_size`.
- `:428`: `learner_group.update(episodes=...)` runs the learner connector (GAE) and all epochs.
- `:465`: `env_runner_group.sync_weights(from_worker_or_learner_group=self.learner_group, ...)`.
- Each block is wrapped in `self.metrics.log_time(...)`. In `log_time`, `reduce: str = "ema"` at `:587`: timers are exponential moving averages with coefficient 0.01 by default.

**Question:** map the three calls onto `train.py --mode sync`. Which mode of this harness is RLlib's PPO? In the pre-test, the raw wrapped `learner_group.update` took a median of 157 ms, while RLlib's `learner_update_timer` reported 614 ms after 23 iterations. Explain the gap from the EMA coefficient and the first iteration's cost. What does that mean for any RLlib timing you read after a short run?

### Stop 2 (8 min): how weights reach the EnvRunners. `env_runner_group.py` `sync_weights` `:695`

- `:701`: `timeout_seconds: Optional[float] = 0.0`, documented as fire-and-forget
- `:757`: `rl_module_state = weights_src.get_state(...)` from the LearnerGroup. This is a blocking call to the learner actor.
- `:802`: `rl_module_state_ref = ray.put(rl_module_state)`, one put for all runners
- `:808`: `foreach_env_runner_async_fetch_ready(func="set_state", ...)` (`:662`) sends `set_state(ref)` to every runner without waiting

**Question:** RLlib's raw `sync_weights` took 4.3 ms, against this harness's 1.5 ms for d2h + put + get + load. Which part of `sync_weights` is still blocking even with `timeout_seconds=0.0`? Given Ray's per-caller ordering of actor calls (§1.3), can the next `sample()` on a remote EnvRunner run *before* its `set_state`? What would have to change for that to happen?

### Stop 3 (10 min): sampling, the version tag, truncation and the cut. `single_agent_env_runner.py` `sample` `:152`, `_sample` `:302`, `set_state` `:596`

- `:390`: `extra_model_output[WEIGHTS_SEQ_NO] = self._weights_seq_no`, a version tag on **every timestep**
- `:412`: the gymnasium `truncated` flag is stored with the episode
- `:492`: `eps.cut(...)`: ongoing episodes are cut at the end of `sample()` and continued in the next call
- `:610`: `if weights_seq_no == 0 or self._weights_seq_no < weights_seq_no:` before loading new weights

**Question:** this harness tags per *fragment*, RLlib per *timestep*. When could a single `sample()` call contain more than one version? (Think of an EnvRunner that receives `set_state` from another thread, or of `sample()` calls that span a weight update.) What does the guard at `:610` protect against, and what does `weights_seq_no == 0` force?

### Stop 4 (10 min): the learner update, the version counter, H2D and the ratio. `learner.py` `update` `:1028`; `torch_learner.py` `_convert_batch_type` `:379`; `ppo_torch_learner.py` `compute_loss_for_module` `:35`

- `learner.py:1097`: `self._weights_seq_no += 1`, once per `update()` call, before the minibatch loop
- `:1124`: `fwd_out, loss_per_module, _ = self._update(...)`, once per minibatch
- `torch_learner.py:386`: `batch_dict = convert_to_torch_tensor(...)`, the host-to-device copy
- `ppo_torch_learner.py:74`: `logp_ratio = torch.exp(...)` against the stored `Columns.ACTION_LOGP`, and `:91`: `torch.clamp(logp_ratio, ...)`

**Question:** at the pre-test settings, how many optimizer steps share one `weights_seq_no` (batch 1024, minibatch 256, 4 epochs)? This harness also counts one version per update. At the end of the RLlib run, the EnvRunners reported `weights_seq_no` 22 and the learner 23. Is that a lag of 1 in this harness's sense? Which value would RLlib's data carry when the learner trains on it next?

### Stop 5 (10 min): GAE, terminated vs truncated. `general_advantage_estimation.py` `GeneralAdvantageEstimation` `:21`; `value_predictions.py` `compute_value_targets` `:7`; `add_one_ts_to_episodes_and_truncate.py` `:178`

- `general_advantage_estimation.py:94`: `vf_preds = rl_module.foreach_module(...)`: the values come from the **learner's** module, recomputed at update time
- `:123`: `module_value_targets = compute_value_targets(...)`
- `value_predictions.py:28`: `non_terminal = 1.0 - terminateds`, no bootstrap after termination
- `:31`: `propagate = non_terminal * (1.0 - truncateds)`, so a truncation stops the recursion
- `:41`: `td_residuals = rewards + gamma * non_terminal * next_state_values - values`
- `add_one_ts_to_episodes_and_truncate.py:178`: each episode chunk gets one extra timestep, and its last real step is marked truncated when it was cut. That extra step supplies the bootstrap value.

**Question.** Compare RLlib's three cases with P7's table. Where does each system get $V(s_{t+1})$ at a truncation and at a fragment cut? This harness uses the *behaviour* critic's values. RLlib recomputes them with the learner's critic. At lag 1, which advantages can differ, and why? RLlib's personal GAE connector reported 97.1 ms (EMA). The harness's GAE phase took 3.26 ms (post-warmup median). Where would you add timestamps to separate the value prediction and the GAE recursion, and how would you compare the two timings?

### Stop 6 (optional, 15 min): the asynchronous family. `impala.py` `training_step` `:643`; `appo_torch_learner.py` `:46`; `impala/torch/vtrace_torch_v2.py` `vtrace_torch` `:73`

- `impala.py:783`: weights are returned for broadcast only every `broadcast_interval` updates
- `:820`: `learner_group.update(..., async_update=do_async_updates)` does not wait for the learner
- `appo_torch_learner.py:89`: `behaviour_actions_logp = batch[Columns.ACTION_LOGP]`
- `:155`: V-trace targets from the *old target* policy's log-probs and the behaviour log-probs
- `:168`: `is_ratio = torch.clip(exp(behaviour − old_target), 0.0, 2.0)`, and the surrogate uses `is_ratio × exp(target − behaviour)`
- `vtrace_torch_v2.py:81`: `clip_rho_threshold`, the $\bar\rho$ clip

**Question:** APPO keeps three policies apart: behaviour μ (the EnvRunner's weights), old target (a periodically updated copy) and the current target. Which one plays the role of this harness's stored `logp`? With `broadcast_interval` = 4 and `wait`-style arrival, what lag would you expect, and what does $\bar\rho = 1$ do to a sample whose ratio is 3?

### Stop 7 (8 min): the Ray core facts behind the probes. `_private/utils.py`, `accelerators/nvidia_gpu.py`, `serialization.py`, `ray_constants.py`, `worker.py`

- `utils.py:235` `set_omp_num_threads_if_unset`; `:270` `omp_num_threads = max(math.floor(num_assigned_cpus), 1)`
- `utils.py:275` `set_visible_accelerator_ids`; `:292` `if not override_on_zero and len(accelerator_ids) == 0:`, so 0 GPUs leaves `CUDA_VISIBLE_DEVICES` alone
- `nvidia_gpu.py:112`: `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`, the opt-out
- `utils.py:365` `_get_docker_cpus`; `:369` reads `/sys/fs/cgroup/cpu.max`; `:432` `get_num_cpus`; `:488` "truncated from"
- `serialization.py:164`: `self._zero_copy_tensors_enabled = (...)`, and `ray_constants.py:785`: `RAY_ENABLE_ZERO_COPY_TORCH_TENSORS = env_bool(..., False)`
- `worker.py:3047` `put`, `:3103` `wait`

**Question:** for each P3 answer, name the line that causes it. Then: your actor sets `torch.set_num_threads(8)` in `__init__`. Which of Ray's settings does that override, and what does the cgroup then do (P12b)?

---

## Block 4: Run + summarize (30–60 min)

### Commands (the manual run)

On the instance, from `/workspace/AI-Infra-Learning`:

```bash
RID=day15_$(date -u +%Y%m%dT%H%M%SZ)_mine; nohup bash experiments/day15_ray/run_server_once.sh "$RID" --impl mine --core > /workspace/$RID.log 2>&1 < /dev/null &
```

```bash
tail -f /workspace/$RID.log
```

The pre-test core run took 7 min. Full mode (drop `--core`) adds about 5 min. In a `mine` session, `check_rl`, the learning runs and Exp 2 use your `my_rl.py`. Controls, traces, Exp 3, RLlib and the optional stages use the reference. The runner ends with `ray stop --force` and records `nvidia-smi --query-compute-apps` in `gpu_processes_after.txt`.

### Download to Windows, then stop the instance

```powershell
experiments/day15_ray/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>
```

The script checks the archive's SHA-256 and every manifest entry. Then **stop** the instance in the Vast console, or from inside it with `vastai stop instance $CONTAINER_ID --api-key $CONTAINER_API_KEY`. Do not destroy it. An idle running instance is still billed. Record the console check in notes §2.

### Experiment matrix

| Exp | Stage (log) | Layout | Expected output | Pass / fail rule | Label |
| --- | --- | --- | --- | --- | --- |
| 0 | `probes` | Ray, 1 node | resources vs quota, placement env, round trips, get vs wait, put once vs by value, zero copy, a pending actor | `PROBES: done`. No numeric rule: compare with P1–P3. | [measured] |
| 1 | `check_rl` | CPU, no Ray | P7 and P8 numbers, 200 random GAE fragments, lag-0 and lag-1 ratio tests, 4 controls | 6 PASS, 4 DETECTED, exit 0 | [derived] (hand values) / [measured] |
| 1 | `learn_sync_s0..2`, `learn_async_s0` | 2 actors + GPU learner, CartPole, 60 iterations | lag histogram, digests, ratio, the 20-episode mean return | every `CHECK` PASS and return ≥ 150 | [measured] |
| 2 | `exp2_sync`, `exp2_async` | same, 23 iterations, 3 warmup | per-phase medians, steps/s, idle | `TRAIN … PASSED`. Compare the phases with P4–P5. | [measured]; `transfer`, `return`, `idle` [derived] |
| 2 | `control_skip_sync`, `control_recompute_logp` | async, 6 iterations | the named check fails | `CONTROL …: DETECTED … -> PASSED` | [measured] |
| 2 | `trace_sync`, `trace_async` | 6 iterations, traced | `ray.timeline` text: sample/update overlap | the timeline files exist. Never used for timing. | [measured] |
| 3 | `exp3_{sync,async}_c{0,400}_n{1,2,4}` | synthetic env, B = 4096 | e2e, rollout, learner busy, bottleneck side, throttling | each run `PASSED`. Compare with your P6 table. | [measured] vs [derived] prediction |
| 4 | `rllib` | RLlib PPO, 2 EnvRunners × 4 envs, 1 GPU Learner | raw wrapped call times, RLlib's EMA timers, return | `RLLIB_PPO: done … PASSED` (return improved) | [measured] |
| opt | `wait_n2`, `wait_n4` | `ray.wait` first finished | lag histogram | none: compare with P10 | [measured] |
| opt | `repro_{sync,async,wait}_{1,2}` | same seed twice | final weight digests | none: compare with P11 | [measured] |
| opt | `exp3n8_{sync,async}` | 8 actors, `--ray-num-cpus 10`, c = 400 | rollout and throttling over the quota | none: compare with P12a | [measured] |
| opt | `probes_oversub` | 4 actors, threads 1 vs 8 per actor | per-forward time, throttled seconds | none: compare with P12b | [measured] |

### How to read the logs

- **`environment.log`**: GPU, quota, Ray version, `/dev/shm`. Note `cpu.max`. Every scaling claim depends on it.
- **`probes.log`**: one section per P1–P3 item. The `pending` section ends with `started within 5 s = False` and an autoscaler error. That is the expected result.
- **`exp2_*.log`**: one line per iteration (`learner v`, `behaviour`, `lag`, `digest ok`, first-minibatch ratio and clip, all-minibatch clip, KL, return, e2e), then the phase table, then the four `CHECK` lines.
  - In sync, look for `idle.actor` ≈ U + o.
  - In async, `transfer` jumps to ≈ U + o − R and `idle.learner` drops to a few ms.
- **`trace_*.log`**: per worker process, the actor method spans of the last two iterations. In async, an actor's `sample` must lie *inside* a learner `update` span.
- **`summary.md`**: Exp 2 side by side, the Exp 3 model vs measured table, and the Exp 4 RLlib mapping.
- **`rllib.log`**: per iteration, the raw wrapped call times and `return_mean`, then every RLlib timer key at the last iteration (EMA). Compare `time_this_iter_s` (the last iteration) with `timers/training_iteration` (EMA).

### What to write in `notes.md`

Fill §2–§7 from the logs with [measured], §8 from Block 3, and §9 from the seven Questions. Keep [derived] predictions and [measured] results in separate columns. Anything the run does not isolate goes in §10 with [not isolated].

---

## Self-check (answer aloud, < 2 min each)

1. Why does `ray.get` inside a submission loop serialize the work, and what is the fix when results can be processed in any order?
2. What does put-once save for N actors, and why is a numpy weight dict cheaper to fetch than a torch one?
3. A container shows 64 CPUs. How many will Ray use, and where does that number come from?
4. Draw the async timeline for two iterations and write each batch's version and lag.
5. Async was 1.57× faster than sync, not 2×. Which phase is now the bottleneck, and what would raise the speedup?
6. Truncated vs terminated vs fragment cut: which bootstrap, which stops the recursion?
7. Why is PPO's first-minibatch ratio exactly 1 at lag 0, and what does a ratio of exactly 1 at lag 1 tell you?
8. Name RLlib PPO's three calls per iteration, and why its timers overstate a short run.

---

## Appendix A: local results (assistant run, 2026-11-27) [pre-test, local]

The local Windows machine has no torch and no Ray, so only the numpy part runs there. It ran in a scratch virtual environment with numpy only.

| Check | Result |
| --- | --- |
| `check_rl.py --impl ref --tests gae` | `gae.hand_example`, `gae.vs_slow_reference` (200 fragments, max diff 0) and `gae.shape_TxE` PASS. The 3 GAE controls were DETECTED. The PPO tests were SKIPPED (no torch). |
| `check_rl.py --impl mine` with the stubs | `INCOMPLETE`, exit 3 |
| `train.py --impl mine` with the stubs | `INCOMPLETE (TODOs left in my_rl.py: compute_gae, ppo_surrogate, pipelined_iteration); exit 3` before Ray starts |
| `py_compile` on every `.py`, `bash -n run_server_once.sh` | clean. The `.sh` file has LF line endings. |

Everything that needs torch, Ray or a GPU ran on the server (Appendix B).

---

## Appendix B: server pre-test `pretest1` and `pretest2_full` (assistant run, 2026-11-27) [pre-test]

These are the assistant's runs of the **reference** implementation. They show that the harness, the checks and the controls work end to end on the target setup. They are not your measurements. Your numbers go in the "your run" columns of `notes.md`.

### B.1 Run record

| Item | `pretest1` (required) |
| --- | --- |
| Command | `run_server_once.sh pretest1 --core --impl ref` (nohup, stdin from `/dev/null`) |
| UTC on 2026-11-27 | 04:52:36–04:59:28 (6 min 52 s) |
| Exit and markers | 0, `CORE_SUCCESS.txt`, `OUTPUT CHECKS PASSED (core): 27 stages` |
| Archive | `day15_pretest1.tar.gz` (224 KB), SHA-256 `2b07ba18…7c96bb81f` verified locally, manifest 79/79 |
| After the run | `ray stop`: "Did not find any active Ray processes". `gpu_processes_after.txt`: header only, no GPU process. |
| Host | container `62799402`, 1 × NVIDIA A100-PCIE-40GB (driver 595.91.07), 64-CPU host, `cpu.max 768000 100000` = 7.68 CPUs, Ray CPU 7, `/dev/shm` 23 GiB, disk 60 GB (1.1 GB used) |
| Software | Python 3.12.16, torch 2.14.1+cu130, numpy 2.5.5, Ray 2.60.0 (`b7f4259e`), gymnasium 1.2.2 |

A separate plumbing check ran `run_server_once.sh --quick --impl mine` on a copy of the project whose `my_rl.py` held the reference code. It passed: `check_rl (mine)` 6/6 with 4 controls detected, and the sync learning, Exp 2 sync/async and both controls all passed. The untouched stubs gave exit 3 for both `check_rl.py` and `train.py`.

### B.2 Exp 0: Ray probes

| Probe | Result |
| --- | --- |
| resources | `os.cpu_count` 64, quota 7.68, **Ray CPU 7.0** (warning: "truncated from 7.68 to 7"), GPU 1.0, object store 4.0 GiB (set by the probe) |
| `num_cpus=1` / `=2` actor | `OMP_NUM_THREADS` '1' / '2', torch threads 1 / 2 |
| `num_gpus=1` actor | `CUDA_VISIBLE_DEVICES='0'` |
| `num_gpus=0` actor | `CUDA_VISIBLE_DEVICES` unset, **`torch.cuda.device_count()` 1** |
| two `num_gpus=0.5` actors | both on device '0' |
| round trip | actor method 765 µs, task 1425 µs (median of 300) |
| get vs wait | serial 1.213 s; get in order first 0.403 / total 0.604 s; `wait` first 0.103 / total 0.453 s |
| put once vs by value, 1 MiB | driver submit 1.71 vs 4.17 ms |
| put once vs by value, 64 MiB | driver submit 26.62 vs 95.24 ms; driver serialized 64 vs 256 MiB |
| zero copy, 64 MiB in an actor | numpy `get` 0.18 ms, read-only view; torch 77.4 ms (a copy) |
| pending | `num_cpus=8.0` not started within 5 s; autoscaler: "No available node types can fulfill resource request {'CPU': 8.0}" |

### B.3 Correctness

`CHECK_RL (ref): 6 passed, 0 failed, 0 incomplete, controls 4 detected / 0 missed, 0 skipped: PASSED`

| Check | Result |
| --- | --- |
| `gae.hand_example` | `[0.84375, -0.625, 0.5, 0.5, -0.75, 1.0]` |
| `gae.vs_slow_reference` | 200 fragments, max diff 0.00e+00 |
| GAE controls | trunc_as_term 3.86, no_cut_bootstrap 3.29, cross_boundary 18.2 (max diff on the random set): all detected |
| `ppo.hand_example` | loss −0.0400, clip 0.40, KL 0.05955, gradient `[-0.1, 0.18, -0.2, -0.22, 0.3]` |
| `ppo.lag0_ratio_one` | max \|r−1\| 0, clip 0, KL 0 |
| `ppo.lag1_stale` | stored logp: max \|r−1\| 0.1662; recomputed: 0, so the staleness is hidden |

Learning on CartPole (2 actors × 4 envs × 128 steps, 60 iterations = 61,440 steps, target 150):

| Run | Lags | Digests | Lag-0 first-mb max \|r−1\| | Lag-1 first-mb median | Return (last 20 episodes) | steps/s (whole run) |
| --- | --- | --- | --- | --- | --- | --- |
| `learn_sync_s0` | `{0: 120}` | 120/120 | 4.77e-7 | — | 274.0 | 8,704 |
| `learn_sync_s1` | `{0: 120}` | 120/120 | 4.77e-7 | — | 454.8 | 8,344 |
| `learn_sync_s2` | `{0: 120}` | 120/120 | 4.77e-7 | — | 316.5 | 8,765 |
| `learn_async_s0` | `{0: 2, 1: 118}` | 120/120 | 1.19e-7 | 0.287 | 354.8 | 12,171 |

One seed per mode is not enough to compare sync with async learning speed. The learning check only shows that both schedules learn.

### B.4 Exp 2: phase breakdown (CartPole, 2 actors, 23 iterations, 3 warmup, medians in ms)

| Phase | sync | async |
| --- | ---: | ---: |
| rollout | 34.56 | 34.99 |
| rollout.env | 7.19 | 7.15 |
| rollout.infer | 24.79 | 25.28 |
| transfer [derived] | 1.85 | 31.17 |
| learn.gae | 1.20 | 1.20 |
| learn.h2d | 0.28 | 0.24 |
| learn.update | 60.07 | 59.76 |
| sync.d2h | 0.38 | 0.37 |
| sync.put | 0.75 | 0.65 |
| sync.get | 0.24 | 0.23 |
| sync.load | 0.14 | 0.14 |
| sync.digest | 0.15 | 0.15 |
| return [derived] | 1.43 | 1.73 |
| idle.actor [derived] | 68.64 | 30.94 |
| idle.learner [derived] | 39.78 | 3.19 |
| **e2e** | **103.25** (102.0–110.3) | **65.93** (64.3–67.7) |
| steady-state env steps/s | 9,917 | 15,532 |

- Async lag histogram `{0: 2, 1: 44}`. First-minibatch max |r−1| at lag 1: median 0.245. First-minibatch clip fraction at lag 1: median 0.016, max 0.281. All-minibatch clip fraction median: sync 0.032, async 0.054.
- `learn.update` is 16 optimizer steps (1024/256 × 4 epochs), so 3.75 ms per step. For a 64-unit MLP that is launch and Python cost, not GPU work. The learner is host-bound.
- Inference was 72 % of `rollout`: 25 of 35 ms for 128 vector steps of 4 envs, against 7 ms of CartPole stepping.
- Weights: 36,620 bytes. The whole weight path (d2h + put + get + load + digest) took about 1.7 ms per iteration.
- The traced runs (`trace_*.log`, the §1.3 timelines) took about 110 ms sync and 71 ms async per iteration. In the async trace, each actor's `sample` (0.4 → 35.6 ms) lies inside the learner's `update` (0 → 69.1 ms).

Controls: `skip_sync` digest 11/12, mismatch `(3, 1, 2)`, actors reported skipped `[2]`, ratio PASS, giving `DETECTED by the digest check`. `recompute_logp`: lag-1 first-minibatch max |r−1| 0 in 5/5 batches (`HIDDEN 5`), giving `DETECTED by the ratio check`.

### B.5 Exp 3: scaling

Model inputs from Exp 2 sync [derived]: i 193.7 µs per vector step, $e_0$ 14.0 µs per env step, u 3.75 ms per optimizer step, g 1.45 µs per sample, o 4.94 ms per iteration. B = 4096, minibatch 512, 4 epochs, 10 iterations, 2 warmup.

| Mode | c (µs) | N | Predicted e2e (ms) | Measured e2e (ms) | Rollout | Learner busy | steps/s | Bottleneck | Throttled (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| async | 0 | 1 | 256 | 275 | 274 | 127 | 14,876 | rollout | 0.00 |
| async | 0 | 2 | 131 | 141 | 138 | 126 | 29,102 | rollout (barely) | 0.04 |
| async | 0 | 4 | 131 | 141 | 73 | 135 | 29,004 | learner | 0.00 |
| sync | 0 | 1 | 387 | 423 | 275 | 139 | 9,679 | rollout | 0.00 |
| sync | 0 | 2 | 259 | 276 | 137 | 129 | 14,852 | rollout | 0.00 |
| sync | 0 | 4 | 195 | 216 | 73 | 134 | 18,945 | learner | 0.03 |
| async | 400 | 1 | 1894 | 1918 | 1914 | 142 | 2,135 | rollout | 0.00 |
| async | 400 | 2 | 947 | 970 | 966 | 129 | 4,223 | rollout | 0.00 |
| async | 400 | 4 | 474 | 485 | 483 | 122 | 8,448 | rollout | 0.00 |
| sync | 400 | 1 | 2025 | 2064 | 1923 | 128 | 1,985 | rollout | 0.00 |
| sync | 400 | 2 | 1078 | 1109 | 965 | 133 | 3,692 | rollout | 0.00 |
| sync | 400 | 4 | 605 | 628 | 482 | 136 | 6,522 | rollout | 0.00 |

The model was 1–10 % optimistic in every cell: about 1–3 % when the env is expensive, 7–10 % when it is free. With c = 0, async stops scaling at 2 actors (141 ms at both N = 2 and N = 4) because the learner, at about 130 ms, becomes the bottleneck. With c = 400 µs, rollout halves with each doubling of N, and async approaches the pure rollout time. The quota was never hit (≤ 0.04 s throttled), since 4 actors + learner + driver fit in 7.68 CPUs.

### B.6 Exp 4: RLlib PPO (23 iterations; train.py sync at the same batch for comparison)

| Phase (ms) | train.py sync | RLlib raw (wrapped call, median of iterations 3–22) | RLlib's own timer (EMA, last iteration) |
| --- | ---: | ---: | ---: |
| sampling | 34.6 (+ transfer 1.9) | 134.8 | 149.1 |
| learner update | 61.6 (GAE + H2D + PPO) | 157.5 | 614.3 |
| weight sync | 1.51 (d2h + put + get + load) | 4.34 (fire-and-forget) | 6.09 |
| iteration | 103.3 | 306.0 | 770.3 |
| GAE (learner connector) | 1.20 | — | 59.2 |
| per vector step: inference / env step | 0.194 / 0.056 | — | 0.181 / 0.091 |

- The RLlib run was **3,346 env steps/s**, against this harness's 9,917 for the same batch, model size and actor count. `return_mean` went from 23.2 to 167.4 in 23 iterations, and is noisy per iteration (137–402 in the last 11).
- Every EMA timer is dominated by the first iteration: `time_this_iter_s` was 0.303 s at the last iteration, while `timers/training_iteration` said 0.770 s.
- `weights_seq_no`: EnvRunners 22, learner 23 at the end.
- In 2.60.0, the config used a local EnvRunner in the driver process, and the single learner (`num_learners=1`) ran as a remote actor.
- The difference in raw sampling (135 vs 35 ms) is mostly RLlib's per-step connector pipeline: `module_to_env` 0.44 ms + `env_to_module` 0.14 ms per vector step × 128. This is [derived] from its timers and not isolated by a profile.

### B.7 Optional stages (`pretest2_full`)

`pretest2_full` ran `run_server_once.sh pretest2_full --impl ref` (core + optional stages) from 05:01:28 to 05:11:27 UTC. It exited 0, with `OUTPUT CHECKS PASSED (full): 27 stages`. The archive (280 KB, SHA-256 `53596b5a…1def7202`) and its manifest (101 files) were verified locally.

Its core stages repeated `pretest1`. Every learning and Exp 2 run ended with the same final weight digest as in `pretest1`. The steady-state e2e was 109.05 ms sync and 66.82 ms async (pretest1: 103.25 and 65.93).

| Stage | Result |
| --- | --- |
| `wait_n2` | lags `{0: 1, 1: 39}`; e2e 35.5 ms per one-fragment update; lag ≥ 1 first-minibatch max \|r−1\| median 0.200 |
| `wait_n4` | lags `{0: 1, 1: 33, 2: 3, 3: 1, 8: 1, 23: 1}`; e2e 37.8 ms; `transfer` median 6.0 ms, **max 1489.6 ms**; ratio median 0.210 |
| `repro_sync_1/2` | `3dac3cf17bce9b88` / `3dac3cf17bce9b88`: identical |
| `repro_async_1/2` | `3bfa23e723801a95` / `3bfa23e723801a95`: identical |
| `repro_wait_1/2` | `de539c319ec69701` / `820fbbeeea3d29e6`: different |
| `probes_oversub` (4 actors × CPU inference) | threads 1: 1027.6 µs per forward, wall 0.44 s, CPU 2.18 s, 0 s throttled. Threads 8: 1928.5 µs, wall 0.88 s, CPU 6.97 s, **20.49 s throttled** |
| `exp3_n8_sync`, `exp3_n8_async` | **failed (exit 1) in `pretest2_full`**: `ray.init` timed out ("The current node timed out during startup"). Fixed and rerun, see below. |

**The `exp3_n8` failure, and the fix.** Both 8-actor stages failed in `ray.init` after 30 s. No raylet log was written. Bisected by hand on the same host:

- a plain `ray.init(num_cpus=10)` worked, from any directory, with or without the harness's imports
- `train.py` failed with *any* `--ray-num-cpus`, including 7
- `ray.init(num_cpus=7.0)` failed after 31.3 s, while `ray.init(num_cpus=7)` started in 2.7 s

The cause: `train.py` parsed `--ray-num-cpus` with `type=float`. The fix is `type=int` (`train.py:72`). The core stages never pass this flag, so `pretest1` and `pretest2_full`'s core stages are unaffected.

Both stages were rerun with the fixed file (`session_pretest2_n8fix/`, SHA256SUMS verified locally, `train_py.sha256` matches the local file). Both passed:

| 8 actors, c = 400 µs, `--ray-num-cpus 10` | sync | async |
| --- | ---: | ---: |
| rollout (ms) | 253.2 | 280.8 |
| learn.update (ms) | 121.5 | 142.5 |
| e2e (ms) | 392.2 | 284.4 |
| throttled | 4.45 s in 42/123 periods | 6.58 s in 50/111 periods |
| 4 actors, same cost (B.5) | e2e 628 | e2e 485 |

**The `ray.wait` pick.** A probe on the same host (`waitorder_probe/waitorder.py`) made four actors finish in the order 3, 2, 1, 0 before calling `ray.wait(num_returns=1)` repeatedly. It returned `[1, 0, 2, 3]`, `[0, 1, 2, 3]` and `[3, 1, 2, 0]` in three trials: neither list order nor finish order. In the wait-mode pattern (all four ready at every call, the relaunched actor appended last) it consumed `[3, 2, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0]`, so actors 1 and 2 starved. This explains `wait_n4`'s histogram. *Why* Ray picks this way is in its C++ core and was not read [not isolated].

After all server work, `ray stop` found no Ray processes, and `nvidia-smi --query-compute-apps` listed no process. The instance was then stopped from inside (`vastai stop instance`, 05:24:10 UTC: "stopping instance 62799402"). The container closed SSH right after, so `vastai show instance --raw` could not be run from inside. Confirm in the Vast console.

### B.8 What the pre-test supports and what it does not

- It supports: the harness runs end to end on a 1-GPU host with a fractional quota; all checks pass on the reference; every control is caught by the check it targets; the Exp 3 model, fed only by Exp 2, predicts every cell within 10 %; the async speedup is bounded by the learner, as the model says.
- It does not support: any claim about sync vs async *learning efficiency* (one seed per mode, 61,440 steps); any multi-node statement; RLlib's performance with tuned settings (default connectors, a local EnvRunner, short run); the cause of RLlib's slower sampling beyond its own per-connector timers [not isolated].
- The learner's 3.75 ms per optimizer step is host-bound. The assistant also tried minibatch 1024 and hidden 1024 on the same host while debugging (16 ms and 69 ms per update). Those runs were not archived and are not evidence here.

---

## Appendix C: personal server runs (2026-11-28)

### C.1 Run record

Both successful sessions used `--core` on container `63018167` with one A100-SXM4-40GB (driver 595.71.05). The host exposed 128 logical CPUs. The container quota was 15.36 and Ray scheduled with 15. Python 3.12.16, NumPy 2.5.5, torch 2.14.1+cu130, Ray 2.60.0 and Gymnasium 1.2.2 matched the pre-test versions. The hardware and quota differed from Appendix B's host.

| Run ID | UTC | Exit | Core stages | Local manifest |
| --- | --- | ---: | --- | ---: |
| `day15_20261128T123501Z_ref` | 12:35:01–12:42:58 | 0 | 27 passed | 86/86 |
| `day15_20261128T123501Z_mine` | 12:42:59–12:44:50 | 1 | Stopped at `learn_sync_s2` | 39/39 |
| `day15_20261128T125243Z_mine` | 12:52:43–13:00:39 | 0 | 27 passed | 86/86 |

All three archive hashes and manifests were verified locally. Both successful sessions contain `CORE_SUCCESS.txt`. The successful `mine` snapshot matches the local `my_rl.py` by SHA-256. Its cleanup logs report no remaining Ray processes or GPU compute processes. The learner confirmed in the Vast console on 2026-11-28 that the container was stopped. [Reference terminal](../outputs/day15_downloads/day15_20261128T123501Z_ref/day15_20261128T123501Z_ref.log) · [Initial mine terminal](../outputs/day15_downloads/day15_20261128T123501Z_mine/day15_20261128T123501Z_mine.log) · [Successful mine terminal](../outputs/day15_downloads/day15_20261128T125243Z_mine/day15_20261128T125243Z_mine.log)

### C.2 Correctness and the GAE precision fix

The first `mine` session passed all six unit checks with a maximum GAE difference of `1.91e-6`, below the test tolerance of `1e-5`. On GPU, `learn_sync_s2` reached 142.6 mean return against the 150 threshold. Version, digest and ratio checks passed. Those checks compare versions, published parameters and action probabilities within a run. The final digest comparison across implementations is a separate test. [Initial GAE check](../outputs/day15_downloads/day15_20261128T123501Z_mine/session_day15_20261128T123501Z_mine/check_rl.log) · [Initial seed-2 failure](../outputs/day15_downloads/day15_20261128T123501Z_mine/session_day15_20261128T123501Z_mine/learn_sync_s2.log)

The fragments are `float32`, so the original code computed the TD errors, the backward recursion and the value targets all in `float32`. The reference computes them in `float64` and casts once. Both are valid GAE, about 1e-6 apart. The rerun keeps the GAE arithmetic and value targets in `float64` and casts both outputs once.

The seed-2 failure was therefore not a correctness bug. PPO on CartPole amplifies 1e-6 differences into a different training run. With the `float32` version, seed 0 finished at 266.1 instead of 274.0, and seed 2 at 142.6 instead of 316.5. The 150-return threshold on three seeds is a sanity check with that much spread. Identical final weight digests against `ref` are the stronger evidence of a correct implementation. A local CPU replay of eight updates on identical synthetic fragments reproduced the parameter divergence with the original code and matched `ref` after the change. The GPU rerun gave a maximum GAE difference of `0.00e+00`, passed all six unit checks and detected all four unit controls. The two Ray controls, run with `ref` in the same session, also caught their intended errors. [Implementation](../experiments/day15_ray/my_rl.py) · [Mine unit checks](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/check_rl.log) · [Core checks](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/verify_core.log)

| CartPole stage | Rerun `mine` return (last 20 episodes) | Final digest, same-host `ref` and `mine` |
| --- | ---: | --- |
| `learn_sync_s0` | 274.0 | `e22feb745f643ec4` |
| `learn_sync_s1` | 454.8 | `4b1efadb52be38ef` |
| `learn_sync_s2` | 316.5 | `84e19b5f1acdc8b1` |
| `learn_async_s0` | 354.8 | `5bf8b2c72793d1d3` |

All four passed after 61,440 steps. Exp 2 also matched the same-host reference's final weight digests in sync and async. [Measured notes §§4–5](notes.md#4-correctness-and-learning)

### C.3 Personal timing and scaling

| Exp 2 (steady-state medians) | Same-host `ref` sync | Same-host `ref` async | `mine` sync | `mine` async |
| --- | ---: | ---: | ---: | ---: |
| Rollout (ms) | 65.83 | 61.96 | 65.31 | 63.06 |
| `transfer` (ms) [derived] | 2.24 | 55.98 | 2.27 | 56.66 |
| PPO update (ms) | 108.04 | 107.79 | 108.56 | 108.83 |
| Learner idle (ms) [derived] | 72.59 | 4.27 | 72.79 | 4.41 |
| End-to-end (ms) | 185.55 | 117.37 | 187.79 | 119.26 |
| Env steps/s | 5,519 | 8,725 | 5,453 | 8,586 |

`mine` async was 1.57× faster than sync. Sync had 46 lag-0 fragments. Async had `{0: 2, 1: 44}` with 46/46 digest matches and a lag-1 first-minibatch median max ratio deviation of 0.245. The `ref` timelines archived with the personal run show the serial and overlapping schedules. The `mine` measurements supply the personal timing comparison. [Reference summary](../outputs/day15_downloads/day15_20261128T123501Z_ref/session_day15_20261128T123501Z_ref/summary.md) · [Mine summary](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/summary.md)

- **Exp 3 [measured].** The core runner uses `ref` to hold the implementation fixed. From 1 to 2 to 4 actors, cheap-env async went **254 → 240 → 235 ms**. The learner was the slower phase at 2 and 4 actors. With 400 µs extra per env step, it went **1890 → 949 → 479 ms**, with rollout still setting the pace. The retrospective model overestimated cheap rollout at one actor most strongly (483 ms predicted end-to-end versus 254 ms measured). [Prediction and full scaling table](notes.md#6-exp-3-scaling-your-prediction-first)
- **Exp 4 [measured].** RLlib's raw learner update took 241.71 ms after warmup, while its last-iteration EMA reported 740.13 ms after a first update of 862.1 ms. Raw median iteration time was 420.41 ms. RLlib reached a mean return of 167.4 at iteration 22. [RLlib log](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/rllib.log)
