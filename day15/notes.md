# Day 15: actor–learner RL on Ray, CPU rollout actors + one GPU learner (notes)

> **Personal run, 2026-11-28 [measured].** `day15_20261128T125243Z_mine` passed all 27 core stages. The same-host `ref` run also passed. Both archives were downloaded and verified. The 2026-11-27 assistant runs remain labelled [pre-test]. [GUIDE](GUIDE.md) · [Personal results](GUIDE.md#appendix-c-personal-server-runs-2026-11-28) · [Pre-test appendix](GUIDE.md#appendix-b-server-pre-test-pretest1-and-pretest2_full-assistant-run-2026-11-27-pre-test)

**Labels.**

- [measured] comes from your archived run.
- [pre-test] comes from the assistant's earlier reference runs on a separate host.
- [derived] is arithmetic from stated inputs. `transfer`, `return` and `idle` are differences of stamps from two processes.
- [simulated] marks results from a model, not a run.
- [understanding] records your reasoning, including corrections after reading the answers.
- [not isolated] marks questions these runs leave open.

## 1. Predictions and corrections [understanding]

Reconstructed from the guided discussion before the server runs. The discussion covered some questions in parts. The table marks values or subquestions supplied in the review.

| Question | Prediction or reasoning from the discussion | Correction or measured result |
| --- | --- | --- |
| P1 · get vs wait | With all four calls submitted, ordered `get` starts consuming at 0.4 s and finishes at 0.60 s. `wait` consumes B, D, C, A and finishes at 0.45 s. | Processing finishes for the first result at 0.45 s vs 0.15 s. One-submit-at-a-time costs 1.20 s [derived]. The probe records the first arrival before the 0.05 s processing. |
| P2 · put once, zero copy | The fourth actor's shared ref still identifies the original snapshot. | `ray.put` serializes 64 MiB once vs 256 MiB by value. NumPy's shared view is read-only. A change to the actor's private parameter copy does not change what another actor reads from that ref. At the real weight size of 36,620 bytes, the sync phase's GPU-to-CPU, put, get, load and digest medians add to 2.56 ms [derived]. |
| P3 · quota, threads, CUDA, pending | On the guide's 7-slot node, a learner claiming one slot leaves six. An 8-slot request cannot fit. A zero-GPU actor still sees the exposed GPU in the probe. | Personal host had 15 slots from a quota of 15.36. A 16-slot request stayed pending. Ray set `OMP_NUM_THREADS` to 1 and 2 for the respective CPU declarations. |
| P4 · lags and iteration time | `103 ms` sync, `68 ms` async. Faster sampling would not lower the async time while the learner is slower. After comparing the versions actually used, the steady async lag stays at 1. | Initial lag 0, then lag 1 by schedule. `mine` measured 187.79 and 119.26 ms on the personal host. |
| P5 · async `transfer` | `62 − 35 = 27 ms` of queueing before accounting for other overhead. Extra timestamps can distinguish readiness from queueing. | With overhead the exercise gives about 33 ms. Personal `transfer` was 56.66 ms async and 2.27 ms sync, and the speedup was 1.57×. |
| P6 · Exp 3 | At `c=0`, predicted async 256 → 131 → 131 ms. At `c=400`, predicted 947 → about 474 ms from 2 to 4 actors, with the learner waiting. | Both scaling patterns held. §6 records the completed model and the numerical errors. |
| P7 · GAE and boundary controls | Correctly kept the next-state value at a time limit, used zero after a true termination, and stopped the recursion at both. | The GAE recurrence uses `γλ = 0.25` here. Treating the time limit as terminal makes row 2's target 1.0 and changes rows 0–2. Dropping cut bootstrap changes rows 4–5. Crossing a boundary mixes the next episode's evidence into earlier advantages. |
| P8 · PPO objective and gradient | Computed the sample terms and noticed they equal the original products. Raising the probability on sample 5 increases the loss, so its gradient pushes the other way. | Two of five ratios are out of band, though clipping changes zero sample scores. Mean negative score is −0.04. Gradient with respect to `new_logp` is `[-0.1, 0.18, -0.2, -0.22, 0.3]`. |
| P9 · lag and the two controls | Same starting weights give ratio 1 before the first update. In the example, `skip_sync` gives 1.5 and `recompute_logp` gives 1. | Only the digest failed for `skip_sync`. Only the ratio check failed for `recompute_logp`. Later minibatches can depart from 1 within a lag-0 batch. |
| P10–P12 · optional | Discussed the thread budget and throttling. | Optional stages have [pre-test] evidence in §10. |

## 2. Run record [measured]

| Item | Pre-test `pretest1` [pre-test] | Successful personal run [measured] |
| --- | --- | --- |
| Run ID | `pretest1` | `day15_20261128T125243Z_mine` |
| Command | `run_server_once.sh pretest1 --core --impl ref` | `run_server_once.sh day15_20261128T125243Z_mine --impl mine --core` |
| Start and finish (UTC) | Nov 27, 04:52:36–04:59:28 (6 min 52 s) | Nov 28, 12:52:43–13:00:39 (7 min 56 s) |
| Exit and markers | 0, `CORE_SUCCESS.txt`, 27 core stages | 0, `CORE_SUCCESS.txt`, `SUCCESS.txt`, 27 core stages |
| Archive SHA-256 and manifest | `2b07ba18…7c96bb81f`, verified 79/79 | `bcf30b95…634e63` (full hash in sidecar), verified 86/86 |
| Host | container `62799402`, 1 × A100-PCIE-40GB, driver 595.91.07 | container `63018167`, 1 × A100-SXM4-40GB, driver 595.71.05 |
| CPUs (`os.cpu_count` / `cpu.max` / Ray) | 64 / `768000 100000` = 7.68 / 7 | 128 / `1536000 100000` = 15.36 / 15 |
| `/dev/shm`, disk | 23 GiB, 60 GB (1.1 GB used) | 31 GiB, 60 GB (807 MiB used at start) |
| Software | Python 3.12.16, torch 2.14.1+cu130, NumPy 2.5.5, Ray 2.60.0 (`b7f4259e`), Gymnasium 1.2.2 | Same versions as the pre-test |
| Process cleanup | No Ray processes or GPU compute processes in the logs | `ray_stop.txt` found no active Ray process. `gpu_processes_after.txt` has only its header. |
| Instance stop (console check) | Stop requested at 05:24:10 UTC. Confirmed stopped in the Vast console on 2026-11-28. | Confirmed stopped in the Vast console on 2026-11-28. |

The same-host `day15_20261128T123501Z_ref` passed 27/27 from 12:35:01 to 12:42:58 UTC. Its archive SHA-256 (`8c87d0a0…10c99`) and 86/86 manifest entries were verified. The first `mine` run (`day15_20261128T123501Z_mine`, 12:42:59–12:44:50 UTC) exited 1 at `learn_sync_s2`. Its 39/39 archived files were verified. Seed 2 reached 142.6 against a target of 150. The version, digest and ratio checks all passed. The rerun kept GAE advantages and value targets in `float64` until the final `float32` cast. [ref session](../outputs/day15_downloads/day15_20261128T123501Z_ref/session_day15_20261128T123501Z_ref/) · [Initial mine failure](../outputs/day15_downloads/day15_20261128T123501Z_mine/session_day15_20261128T123501Z_mine/learn_sync_s2.log) · [Successful mine session](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/) · [Successful mine terminal log](../outputs/day15_downloads/day15_20261128T125243Z_mine/day15_20261128T125243Z_mine.log)

## 3. Exp 0: Ray probes (`probes.log`)

| Probe | Pre-test [pre-test] | Personal run [measured] | Explanation |
| --- | --- | --- | --- |
| Ray CPU vs quota | 7.0 from 7.68 | 15.0 from 15.36 | Ray rounds the quota down for scheduling. Linux enforces the actual fractional CPU-time quota. |
| `OMP_NUM_THREADS` at `num_cpus` 1 / 2 | '1' / '2' | '1' / '2' | The default intra-op thread count follows the resource declaration here. |
| 0-GPU actor: `CUDA_VISIBLE_DEVICES`, `device_count` | unset, 1 | unset, 1 | The actor claims zero GPU resources. CUDA visibility still comes from the container. |
| two 0.5-GPU actors | both on '0' | both on '0' | The fractional reservations can share the same device. |
| round trip: actor / task | 765 / 1425 µs | 999.0 / 2208.5 µs | Short fragments spend a larger fraction of their time on each call. |
| serial get / get in order / `wait` (total) | 1.213 / 0.604 / 0.453 s | 1.211 / 0.603 / 0.452 s | Submit first for parallel work. Consume ready results to use the gaps while other calls run. |
| first result: get in order / `wait` | 0.403 / 0.103 s | 0.403 / 0.103 s | The ordered consumer waits for the slow first submission. |
| put once vs by value, 64 MiB, driver submit | 26.6 vs 95.2 ms (64 vs 256 MiB) | 32.36 vs 72.35 ms (64 vs 256 MiB) | Bytes serialized fell 4×. Observed submit latency fell 2.24× [derived]. |
| zero copy 64 MiB: NumPy / torch `get` | 0.18 ms read-only / 77.4 ms copy | 0.247 ms read-only / 88.43 ms copy | The NumPy bytes are shared. The default torch path copies them on `get`. |
| `num_cpus = total + 1` | pending, not started in 5 s | 16.0 requested, pending after 5 s | This single node has 15 slots even when empty. |

[Personal probe log](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/probes.log)

## 4. Correctness and learning

### 4.1 `check_rl.py --impl mine`

| Check | Pre-test (ref) [pre-test] | Your `mine` [measured] |
| --- | --- | --- |
| `gae.hand_example` | `[0.84375, -0.625, 0.5, 0.5, -0.75, 1.0]` | Same vector, PASS |
| `gae.vs_slow_reference` | max diff 0.00e+00 over 200 fragments | 0.00e+00 on the rerun. The initial `mine` run had 1.91e-6 within the 1e-5 tolerance. |
| `gae.shape_TxE` | `(6, 2)` | `(6, 2)`, PASS |
| `ppo.hand_example` | loss −0.0400, clip 0.40, KL 0.05955, grad `[-0.1, 0.18, -0.2, -0.22, 0.3]` | Same loss, statistics and gradient, PASS |
| `ppo.lag0_ratio_one` | max \|r−1\| 0 | max \|r−1\| 0, clip 0 and KL 0 |
| `ppo.lag1_stale` | stored 1.662e-1, recomputed 0 | stored 1.662e-1, recomputed 0 |
| verdict | 6 passed, 4 controls detected | 6 passed, 4 controls detected, exit 0 |

P7 hand calculation [derived] gives `δ = [1, −0.75, 0.5, 0.5, −1, 1]` and returns `[1.84375, 1.375, 3.0, 1.0, 1.25, 3.0]`. The full cross-boundary control changes advantages in rows 0–3, because it propagates through both the termination and the truncation. [Personal check log](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/check_rl.log)

### 4.2 Versions, digests and ratios (`exp2_*.log`, your `pipelined_iteration` in async)

Both Exp 2 stages in the successful `mine` session passed the three checks and reproduced the pre-test figures below. The lag-1 column is the median across batches of first-minibatch max \|r−1\|.

| Run [measured] | Lag histogram | Digests | Lag-0 first-mb max \|r−1\| | Lag-1 first-mb median |
| --- | --- | --- | --- | --- |
| `exp2_sync` | `{0: 46}` | 46/46 | 4.77e-7 | — |
| `exp2_async` | `{0: 2, 1: 44}` | 46/46 | 1.19e-7 | 0.245 |

[Sync log](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/exp2_sync.log) · [Async log](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/exp2_async.log)

### 4.3 Learning on CartPole (60 iterations, 61,440 steps, target 150)

| Run | Pre-test return (last 20 episodes) | Pre-test steps/s (whole run) | Mine return [measured] | Mine steps/s [measured] | Final digest, `mine` and same-host `ref` |
| --- | ---: | ---: | ---: | ---: | --- |
| `learn_sync_s0` | 274.0 | 8,704 | 274.0 | 5,095 | `e22feb745f643ec4` |
| `learn_sync_s1` | 454.8 | 8,344 | 454.8 | 5,136 | `4b1efadb52be38ef` |
| `learn_sync_s2` | 316.5 | 8,765 | 316.5 | 5,120 | `84e19b5f1acdc8b1` |
| `learn_async_s0` | 354.8 | 12,171 | 354.8 | 7,592 | `5bf8b2c72793d1d3` |

All four reached the target. Each matched the final weight digest of the corresponding same-host `ref` run. Sync fragments had lag `{0: 120}` per seed. Async had `{0: 2, 1: 118}` and a lag-1 first-minibatch median max deviation of 0.287. Each run matched 120/120 published digests. The unit-test tolerance admitted the original rounding difference. The original computed the whole GAE recursion in `float32`, because the fragments are `float32`. That is valid GAE, but the 1e-6 difference produced a different training run: seed 0 finished at 266.1 and seed 2 at 142.6. The seed-2 failure came from run-to-run spread, not a bug [measured]. Matching `ref`'s final digests is the stronger correctness evidence. In a local CPU replay, the initial implementation diverged from `ref` after each of eight updates on identical synthetic fragments. The corrected implementation matched in all eight. The GPU rerun confirmed parity on the four CartPole learning runs and both Exp 2 runs. Learning-efficiency comparisons remain [not isolated]. [Learning verdicts and core verification](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/verify_core.log)

## 5. Exp 2: where the time goes (CartPole, 2 actors × 4 envs × 128 steps, medians in ms)

| Phase | Pre-test sync | Pre-test async | Mine sync [measured] | Mine async [measured] |
| --- | ---: | ---: | ---: | ---: |
| rollout | 34.56 | 34.99 | 65.31 | 63.06 |
| rollout.env | 7.19 | 7.15 | 14.05 | 13.38 |
| rollout.infer | 24.79 | 25.28 | 46.37 | 44.88 |
| transfer [derived] | 1.85 | 31.17 | 2.27 | 56.66 |
| learn.gae | 1.20 | 1.20 | 3.26 | 3.17 |
| learn.h2d | 0.28 | 0.24 | 0.39 | 0.39 |
| learn.update | 60.07 | 59.76 | 108.56 | 108.83 |
| sync.d2h + put [derived] | 1.13 | 1.02 | 1.54 | 1.61 |
| sync.get + load + digest [derived] | 0.53 | 0.52 | 1.02 | 0.96 |
| return [derived] | 1.43 | 1.73 | 1.52 | 2.18 |
| idle.actor [derived] | 68.64 | 30.94 | 121.81 | 55.84 |
| idle.learner [derived] | 39.78 | 3.19 | 72.79 | 4.41 |
| **e2e** | **103.25** | **65.93** | **187.79** | **119.26** |
| steady-state steps/s | 9,917 | 15,532 | 5,453 | 8,586 |

- PPO updates were the largest work phase in both modes. Sync pays for rollout and update serially. Async hides most of the rollout under the learner's work and cuts median learner idle time from 72.79 to 4.41 ms.
- `transfer` includes the time a ready fragment waits for the learner. From the sync medians, `U + o ≈ 187.79 − 65.31 = 122.48 ms`, so `U + o − R ≈ 57.17 ms`, close to the observed 56.66 ms [derived]. The same-host `ref` run was 185.55 ms sync and 117.37 ms async.
- `187.79 ÷ 119.26 = 1.57×` [derived]. The learner side sets the steady async rate here. Cutting rollout time further mainly makes fragments arrive earlier.
- The runner's `ref` traces show both actor samples between updates in sync (about 119–184 ms, between updates ending at 114 ms and starting at 185 ms). In async, the samples at about 118–182 ms run inside the learner's 118–233 ms update. The `mine` Exp 2 timing and idle data support the same schedule. A direct `mine` trace would require running the tracer with `--impl mine`. [Sync trace](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/trace_sync.log) · [Async trace](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/trace_async.log) · [Measured phase table](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/summary.md)

## 6. Exp 3: scaling. Your prediction first

The pre-run discussion predicted the cheap-env plateau and near-halving for the costly env. It did not record an exact UTC timestamp or all 12 cells. The first prediction column completes the model from that discussion using the pre-test inputs. The second is the retrospective model printed by the personal archive. Its inputs came from the personal Exp 2 sync run: `i = 362.3 µs`, `e₀ = 27.4 µs`, `u = 6.78 ms`, `g = 3.56 µs`, `o = 6.35 ms`. Exp 3 runs the reference implementation in both archives.

| Mode | c (µs) | N | Pre-run model reconstructed (ms) [derived] | Personal Exp 2 model (ms) [derived] | Pre-test actual (ms) | Personal e2e (ms) [measured] | Personal bottleneck (rollout / learner busy, ms) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| sync | 0 | 1 | 387, rollout | 721 | 423 | 525 | rollout (286 / 227) |
| sync | 0 | 2 | 259, about equal | 480 | 276 | 402 | learner (164 / 227) |
| sync | 0 | 4 | 195, learner | 359 | 216 | 343 | learner (102 / 230) |
| async | 0 | 1 | 256, rollout | 483 | 275 | 254 | rollout (251 / 228) |
| async | 0 | 2 | 131, about equal | 242 | 141 | 240 | learner (158 / 233) |
| async | 0 | 4 | 131, learner | 238 | 141 | 235 | learner (102 / 227) |
| sync | 400 | 1 | 2025, rollout | 2360 | 2064 | 2133 | rollout (1893 / 230) |
| sync | 400 | 2 | 1078, rollout | 1299 | 1109 | 1199 | rollout (957 / 230) |
| sync | 400 | 4 | 605, rollout | 768 | 628 | 724 | rollout (482 / 230) |
| async | 400 | 1 | 1894, rollout | 2122 | 1918 | 1890 | rollout (1887 / 229) |
| async | 400 | 2 | 947, rollout | 1061 | 970 | 949 | rollout (946 / 225) |
| async | 400 | 4 | 474, rollout | 530 | 485 | 479 | rollout (476 / 226) |

- The pre-run model correctly put the cheap-env async plateau near two actors. Measured `e2e` was 254 → 240 → 235 ms. At two actors, measured rollout was already 75 ms shorter than learner busy time. The predicted plateau of 131 ms used a faster learner from the earlier host.
- With 400 µs of extra work per env step, async went from 1890 → 949 → 479 ms and stayed rollout-bound in every cell. Increasing the actor count from two to four saved 470 ms, close to the 473 ms estimated in the discussion.
- The retrospective model's largest error was cheap async with one actor. The CartPole-calibrated rollout cost implied about 483 ms, while the synthetic rollout took 251 ms. The model priced each vector step at the CartPole Exp 2 cost: `i = 362 µs` of inference plus `4 × 27.4 µs` of env work, so `1024 × 472 µs ≈ 483 ms`. The one-actor synthetic run, with 1024-step fragments, measured 181.65 ms of inference (177 µs per vector step) and 51.62 ms of env work (12.6 µs per env step) [derived from `exp3_async_c0_n1.log`]. Measured inference per vector step fell with fragment length: 362 µs at 128 steps (CartPole), 290 µs at 256 (four actors) and 177 µs at 1024. The model assumes a constant cost. Why the per-step cost falls is [not isolated]. The model was close for cheap async with two or four actors once the measured learner work set the rate. Exp 3 reported 0.00 s throttled across all 12 cells. [Personal Exp 3 table](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/summary.md) · [Same-host reference table](../outputs/day15_downloads/day15_20261128T123501Z_ref/session_day15_20261128T123501Z_ref/summary.md)

## 7. Negative controls

| Control | Must fail | Pre-test [pre-test] | Personal run [measured] |
| --- | --- | --- | --- |
| `gae_trunc_as_term` | both GAE tests | detected, max diff 3.86 | Hand case failed, random max diff 3.86 |
| `gae_no_cut_bootstrap` | both GAE tests | detected, 3.29 | Hand case failed, random max diff 3.29 |
| `gae_cross_boundary` | both GAE tests | detected, 18.2 | Hand case failed, random max diff 18.2 |
| `ppo_recompute_logp` (unit) | lag-1 ratio = 1 | detected | `max \|r−1\|` 0.1662 with stored log probabilities and 0 with recomputed ones |
| `skip_sync` | `CHECK digest` only | 11/12, mismatch `(3, 1, 2)`, ratio PASS | Same mismatch. Actor 1 reported skipped `v2`. Versions and ratio passed. |
| `recompute_logp` | `CHECK ratio` only | `HIDDEN 5`, digest PASS | All five lag-1 batches were hidden. Versions passed and digest matched 12/12. |

The four unit controls ran with `check_rl.py --impl mine`. The two trainer controls ran with `--impl ref` as specified by the core runner. [Skip-sync control](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/control_skip_sync.log) · [Recompute-logp control](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/control_recompute_logp.log)

## 8. This harness ↔ RLlib 2.60.0 [understanding]

### 8.1 Exp 4 numbers (`rllib.log`, `summary.md`)

| Phase (ms) | Pre-test train.py sync | Pre-test RLlib raw | Pre-test RLlib EMA | Personal train.py sync [measured] | Personal RLlib raw [measured] | Personal RLlib EMA [measured] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sampling | 34.6 | 134.8 | 149.1 | 65.31 (+ 2.27 transfer) | 160.55 | 174.0 |
| learner update | 61.6 | 157.5 | 614.3 | 112.21 | 241.71 | 740.13 |
| weight sync | 1.51 | 4.34 | 6.09 | 2.23 | 5.63 | 6.35 |
| iteration | 103.3 | 306.0 | 770.3 | 187.79 | 420.41 | 921.27 |
| steady-state steps/s [derived] | 9,917 | 3,346 | — | 5,453 | 2,436 | — |

The learner update took 862.1 ms on the first iteration, then 244.8 ms on the last. At 23 iterations, `0.99²² ≈ 0.80` of the initial sample still contributes to the EMA. The RLlib GAE connector reported 97.1 ms as an EMA. A steady-state comparison of value prediction and GAE recursion separately would require timing those stages directly. RLlib's return rose from 23.2 to 167.4 over the 23 iterations. [Personal RLlib log](../outputs/day15_downloads/day15_20261128T125243Z_mine/session_day15_20261128T125243Z_mine/rllib.log)

### 8.2 Code map, filled from Block 3

| This harness | RLlib / Ray anchor | Connection, in your words |
| --- | --- | --- |
| `train.py --mode sync` loop | `ppo.py:391` `training_step`, `:400`, `:428`, `:465` | Collect the batch from all EnvRunners, update the learner, then sync to EnvRunners. |
| `Learner.publish` + `set_weights` | `env_runner_group.py:695`, `:757`, `:802`, `:808` | Fetch learner weights with a blocking call, put them in the store once, and submit `set_state` to each remote runner. |
| `policy_version` per fragment | `single_agent_env_runner.py:390`, `:610` and `learner.py:1097` | RLlib tags each timestep. The counter increments once per `update()`. The runner rejects older positive sequence numbers and accepts `0` as a forced load. |
| truncation / cut in `sample()` | `single_agent_env_runner.py:412`, `:492` and `add_one_ts_to_episodes_and_truncate.py:178` | Record time limits and continue unfinished episodes across calls. The learner gets an extra observation for the boundary value. |
| `compute_gae` (behaviour values) | `general_advantage_estimation.py:94`, `:123` and `value_predictions.py:28`, `:31`, `:41` | Both compute GAE at the learner. The harness supplies actor values. RLlib predicts values on the learner. Termination blocks bootstrap and propagation. A time limit or cut only blocks propagation. |
| `ppo_surrogate` | `ppo_torch_learner.py:74`, `:91` | Both compute the current-over-behaviour action ratio from log probabilities and clamp the ratio for the clipped objective. |
| phase timers (medians, warmup excluded) | `metrics_logger.py:587` (EMA) | EMA retains cold-start cost on short runs. The wrapped per-call medians after warmup give a comparable steady-state window. |
| probe results (P3) | `_private/utils.py:270`, `:292`, `:369`, `:488` and `serialization.py:164` | CPU claims set the default OMP thread count, zero GPU claims leave visibility alone, the cgroup quota supplies Ray's integer capacity, and torch zero-copy is opt-in in this version. |
| (optional) `wait` mode, staleness correction | `impala.py:783`, `:820`, `appo_torch_learner.py:155`, `:168` and `vtrace_torch_v2.py:81` | Reference reading. V-trace limits the influence of stale samples when estimating value targets. |

## 9. Explanations [understanding]

| Prompt | Your explanation |
| --- | --- |
| Q1 · per-call overhead vs fragment length and actors vs tasks | At 35 ms per 128 vector steps, overhead exceeds 10% below 28 steps for an actor method and below 53 for a task. The personal probe and 65.31 ms rollout give thresholds of 20 and 44 steps under the same linear model [derived]. Actors retain env state, RNG state and a local policy across calls. |
| Q2 · keeping the read-only views as parameters | Inference can read from an immutable view. An in-place load into parameters backed by that view would attempt to write to read-only shared memory. The actor copies into writable parameters. Each publication has a fresh ref, and a view of `v0` always reads that snapshot. |
| Q3 · fire-and-forget weight sync | Submission without waiting is compatible with per-actor ordering. If sample calls can overtake a pending weight load, the normal steady lag of 1 can grow to 2 or more. `skip_sync` claimed `v2` while acting with `v1`. At learner `v3`, the reported lag was 1 and the actual lag was 2. The digest caught the mismatch. The version tag and the ratio based on actual behaviour logp both passed their checks. If instead the overtaking actor tagged the fragment with the version it really had loaded, the tag would show lag 2 and `CHECK versions` would catch it directly. The digest is needed only when the tag and the weights disagree. |
| Q4 · faster cheap-env async | On the pre-test host, 2 and 4 actors both took 141 ms. At 4 actors, 8 actors would shorten rollout while the learner still controls throughput. A minibatch of 1024 halves the 32 optimizer steps and saves roughly `16 × 3.75 = 60 ms` of learner work. With a ~73 ms rollout and ~135 ms learner busy time, a 4-actor estimate is roughly 75–85 ms [derived], assuming little change to other costs. At 2 actors, ~138 ms rollout would still dominate. A second GPU does not directly reduce the host overhead and adds all-reduce. No speedup from a second GPU was measured. |
| Q5 · `gae_trunc_as_term` near CartPole's time limit | The last target becomes `1` instead of `1 + 0.99 × 100 ≈ 100`. The last advantage drops by about 99. Earlier advantages in that segment inherit the error with powers of `γλ`. A policy that frequently survives to 500 steps is punished most often by this bug. |
| Q6 · clip fraction and the staleness share | At lag 0 the first minibatch starts with the same policy, then optimizer steps move later ratios out of band. In personal Exp 2 the lag-1 first-minibatch median clip fraction was about 0.014 (range 0–0.281 from printed rows), with no learner steps on that batch yet. The overall medians across iterations were 0.032 sync and 0.054 async. Use each lag-1 batch's `mb0 clip` to measure what is already out of band at entry. An exact attribution over all minibatches needs per-sample ratios at entry and after updates. |
| Q7 · when only the digest catches broken sync | A delayed copy, skipped load or wrong weight payload can leave `v1` loaded while a bookkeeping field says `v2`. The lag inferred from the label looks fine and the stored logp can still give the correct ratio. Hashing the loaded parameters compares the bytes that drove sampling with the bytes published for the label. |
| Stop 1 · three calls and EMA timers | RLlib collects the whole batch, updates the learner, then sends weights. If runners finish at 20 and 80 ms, the update starts only after the second is done. A coefficient of 0.01 leaves about 80% of the first timing sample after 22 more iterations. Use post-warmup raw medians for steady-state comparisons. |
| Stop 2 · what blocks in `sync_weights` | `get_state` blocks on the learner. The driver publishes once and returns after submitting remote `set_state` calls. For a normal synchronous Ray actor, a sample submitted later by the same caller waits behind its own queued load. Other runners can start earlier. Overtaking would require different ordering, concurrency or a missing load submission. |
| Stop 3 · per-timestep tags and the guard | An EnvRunner could mix sequence numbers if a load were allowed to interleave with sampling. A tag on every timestep preserves which weights selected each action. With `v6` loaded, the guard skips late `v5`. Incoming sequence `0` forces a load. |
| Stop 4 · optimizer steps per version and 22 vs 23 | `1024 / 256 × 4 = 16` optimizer steps share one sequence increment. The sampled version 22 and the learner counter 23 reported after update describe different times. If both started with the `v22` parameters, lag at the start of training was 0. After loading `v23`, the next samples carry 23. |
| Stop 5 · learner vs behaviour values and GAE cost | Our actor records the critic's next-state estimate using the final observation at a time limit and the continuing observation at a cut. RLlib adds the boundary observation and predicts it with the learner critic. Both compute GAE in the learner. At lag 1, changed critic weights can alter advantages even at terminal rows through the current-state baseline. The learner critic gives a baseline aligned with the training weights, while the actor critic gives the estimates aligned with the policy that generated the trajectory. PPO still uses stored behaviour logp. Compare post-warmup timings for the same operations and separately time value prediction and recursion. |
| Stop 6 (optional) · behaviour, old target, current target and $\bar\rho$ | The behaviour policy supplies the stored logp. The old target and current target have their own probabilities. Broadcasting every four updates can leave actors several versions behind, and queueing can add more delay. For V-trace, $\bar\rho = 1$ caps a ratio of 3 at 1 for the value correction. Reference reading. |
| Stop 7 · Ray scheduling and thread limits | `utils.py:270` sets the default threads, `:292` preserves visibility for zero assigned GPUs, and `:369` and `:488` read and round the CPU quota. `torch.set_num_threads(8)` changes PyTorch's CPU parallelism. Four one-slot actors still claim four Ray slots. Linux throttles their combined execution once the cgroup exhausts its CPU-time budget. |

## 10. Optional stages, and what these runs do not isolate

### 10.1 Optional stages (full mode)

The personal runs used `--core`. Optional-stage evidence comes from the earlier full-mode pre-test. [Guide B.7](GUIDE.md#b7-optional-stages-pretest2_full)

| Stage | Pre-test `pretest2_full` [pre-test] |
| --- | --- |
| `wait_n2` lag histogram | `{0: 1, 1: 39}` |
| `wait_n4` lag histogram | `{0: 1, 1: 33, 2: 3, 3: 1, 8: 1, 23: 1}`. One fragment waited 1.49 s. |
| `repro_sync` digests | `3dac3cf17bce9b88` twice |
| `repro_async` digests | `3bfa23e723801a95` twice |
| `repro_wait` digests | `de539c319ec69701` and `820fbbeeea3d29e6` |
| `exp3n8` (8 actors over the pre-test quota) | Initial run failed on a float `--ray-num-cpus`. After the integer parsing fix, rollout was 253 / 281 ms, e2e 392 / 284 ms, and throttling 4.4 / 6.6 s (sync / async). |
| `probes_oversub` (4 actors, 1 vs 8 threads each) | 1028 vs 1929 µs per forward and 0 vs 20.5 s throttled |

### 10.2 Limits [not isolated]

- Both schedules learned in 61,440 steps on CartPole. One async seed and three sync seeds leave relative learning efficiency unresolved.
- Personal raw RLlib sampling took 160.55 ms versus 65.31 ms for the harness's rollout. Connector timers and Ray overhead would need matching instrumentation or a profile before assigning the gap to individual stages. The 97.1 ms GAE connector metric is an EMA.
- The personal learner averaged roughly 6.78 ms per optimizer step [derived] from the steady sync median of 108.56 ms over 16 steps. A profile would be needed to apportion it among Python, kernel launches and optimizer work.
- All runs were single-node. Multi-node transfer, placement groups and actor failure recovery remain open.
- The optional pre-test measured long lag tails in `wait`. V-trace was covered in reading, and the personal core run exercised lag 0 and lag 1.
- The ready-ref tie-breaking mechanism in Ray's C++ core remains uninspected. The optional pre-test observed that repeated calls can pick ready refs in different orders. [Guide B.7](GUIDE.md#b7-optional-stages-pretest2_full)
