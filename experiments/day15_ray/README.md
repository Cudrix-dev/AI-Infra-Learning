# Day 15: actor–learner RL on Ray (CPU rollout actors + one GPU learner)

Guide: [day15/GUIDE.md](../../day15/GUIDE.md). Notes template: [day15/notes.md](../../day15/notes.md).

Ray core only (actors, ObjectRefs, `ray.put`, `ray.wait`): no RLlib in the hand-written path. RLlib PPO runs on the same environment, for comparison only (`rllib_ppo.py`). Tested with Ray **2.60.0** (commit `b7f4259e`), gymnasium 1.2.2, torch 2.14.1+cu130 and Python 3.12. The Ray install leaves torch unchanged.

## Install (server, once)

```bash
/venv/main/bin/python -m pip freeze | grep -E "^(torch|numpy|nvidia-nccl|triton|torchvision|torchcodec)==" > /tmp/cons.txt
/venv/main/bin/python -m pip install -c /tmp/cons.txt "ray[default,rllib]==2.60.0" "gymnasium==1.2.2"
```

Run `pip install --dry-run` with the same arguments first if you want to see what it would change. On the pre-test host the dry run kept torch/numpy unchanged.

## Files

| File | What it is |
| --- | --- |
| `my_rl.py` | **YOU WRITE**: TODO 1 `compute_gae`, TODO 2 `ppo_surrogate`, TODO 3 `pipelined_iteration`. Stubs make the tools exit 3. |
| `rl_ref.py` | the reference for the three TODOs, plus `gae_slow_reference` (the O(T²) definition) |
| `rl_common.py` | envs (CartPole, `SyntheticEnv` with a busy-wait step cost), the policy MLP, weight digests, host/cgroup info, the phase-table format (`PHASES`, `SYNC_PHASES`, reused on Day 16) |
| `workers.py` | `RolloutWorker` (CPU actor: E envs, fragments with `policy_version` + digest) and `Learner` (GPU actor: GAE, H2D, PPO, publish once with `ray.put`) |
| `train.py` | the driver: `--mode sync | async | wait`, per-phase records, the version/digest/ratio/learn checks, `--control`, `--trace` |
| `check_rl.py` | local checks without Ray: GAE hand example + slow reference + 3 GAE controls (numpy); PPO surrogate tests (torch CPU) |
| `rl_controls.py` | the planted bugs (GAE variants; `skip_sync` and `recompute_logp` live in `workers.py`) |
| `ray_probes.py` | Exp 0: resources vs quota, what an actor sees, call overhead, get vs wait, put once vs by value, zero copy, a pending actor, (optional) oversubscription |
| `rllib_ppo.py` | Exp 4: RLlib PPO, 2 EnvRunners + 1 GPU Learner, with its timers and raw wrapped call times |
| `summarize.py` | Exp 2–4 tables (and the Exp 3 prediction from Exp 2) into `summary.md` |
| `read_timeline.py` | text view of a `ray.timeline()` Chrome trace |
| `find_symbols.py` | file:line of the Block 3 stops in the installed Ray |
| `run_server_once.sh` | every stage, checked, archived with a SHA256SUMS manifest |
| `download_results.ps1` | download + verify the archive on Windows |

## Commands

From the project root on the server:

```bash
python experiments/day15_ray/check_rl.py --impl mine --tests gae          # TODO 1 (numpy only; works on Windows too)
python experiments/day15_ray/check_rl.py --impl mine --tests ppo          # TODO 2 (torch, CPU)
python experiments/day15_ray/train.py --impl mine --mode async --iters 8 --warmup 2      # TODO 3 (Ray + GPU)
python experiments/day15_ray/train.py --mode sync --iters 23 --warmup 3                 # Exp 2, sync
python experiments/day15_ray/train.py --mode async --control skip_sync --iters 6        # a negative control
RID=day15_$(date -u +%Y%m%dT%H%M%SZ)_mine; bash experiments/day15_ray/run_server_once.sh "$RID" --impl mine --core
```

From Windows, after the run:

```powershell
experiments/day15_ray/download_results.ps1 -Remote root@<host> -Port <port> -RunId <RID>
```

## Runner modes and exit codes

| Mode | Stages |
| --- | --- |
| `--quick` | environment, symbols, check_rl, learn (seed 0), exp2 sync/async (10 iterations), the two controls |
| `--core` | everything required: + probes, learn seeds 1–2 and async, exp2 at 23 iterations, traces, Exp 3 (12 runs), RLlib, summary |
| (default) | `--core` + optional: `wait` mode at 2/4 actors, reproducibility pairs, 8 actors over the quota, the oversubscription probe |

`--impl mine` makes `check_rl`, the learning runs and Exp 2 use `my_rl.py`. Controls, traces, Exp 3, RLlib and the optional stages always use the reference.

Exit codes: `train.py` and `check_rl.py` return 0 when every check passed (a control run passes when its planted bug is **detected**), 1 when a check failed, and 3 when `--impl mine` still has TODOs. The runner returns the first failing stage's code and always writes the archive.

## Measurement rules used here

- All stamps use `time.monotonic()` (CLOCK_MONOTONIC). It is one system-wide clock, so actor, learner and driver stamps on one host line up. `transfer` and `return` are differences of stamps from two processes ([derived]).
- The GPU is synchronized before every learner timer is read. Statistics come back in one device-to-host copy per update, not per minibatch.
- Warmup iterations (`--warmup`, default 3) are excluded. The tables report the median, min and max of the steady state.
- Traced runs (`--trace`) are separate, short runs, and their timings are never used.
- Every run prints the host's `os.cpu_count`, the affinity mask, cgroup `cpu.max`, Ray's detected CPUs and the cgroup throttling during the run.

## Deviations from RLlib (on purpose)

- **Where GAE gets its values.** GAE uses the values the behaviour critic computed at sampling time, including V(final observation) at a truncation. RLlib recomputes them with the learner's current critic (GUIDE Block 3 Stop 5).
- **Version counter.** One version per update, like RLlib's `weights_seq_no`, but tagged per fragment rather than per timestep. The weights cannot change inside one `sample()` call here.
- **Weight sync.** Weights travel as one numpy dict put once by the learner. The actors' `set_weights` are queued in submission order and are not fire-and-forget. RLlib's `sync_weights` returns without waiting (`timeout_seconds=0.0`).
- **Model.** Separate actor and critic MLPs, minibatch-normalized advantages, no value clipping, no KL penalty.
