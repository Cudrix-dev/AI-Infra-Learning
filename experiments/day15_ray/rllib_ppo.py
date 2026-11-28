"""Day 15 - Exp 4: the same job in RLlib (new API stack): PPO on CartPole-v1, 2 EnvRunners x 4 envs, 1 GPU Learner.

    python experiments/day15_ray/rllib_ppo.py --iters 23 --warmup 3 --json rllib.json

Batch settings match train.py's defaults: 2 x 4 envs x 128 steps = 1024 env steps per iteration, minibatch 256,
4 epochs, lr 1e-3, gamma 0.99, lambda 0.95, clip 0.2, vf 0.5, entropy 0.01, grad clip 0.5 (by global norm),
64-64 tanh MLPs, separate value network. No numerical parity with train.py is expected (different init, connectors,
loss details); compare the STRUCTURE and the time split.

Two kinds of timing are printed, on purpose:
  * RLlib's own timers from algo.train()'s result dict. MetricsLogger.log_time reduces with an EMA
    (coefficient 0.01 in this version), so after 20 iterations the timer still carries ~83 % of iteration 0,
    which includes start-up. They are printed as RLlib reports them.
  * raw per-iteration times of the three phases of PPO.training_step, measured by wrapping the three calls:
    synchronous_parallel_sample, learner_group.update and env_runner_group.sync_weights.
Exit code 0 when it ran and the mean episode return rose above its first value.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rl_common import dump_json, host_info  # noqa: E402

RAW: dict[str, list[float]] = {"sample": [], "update": [], "sync_weights": []}


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            RAW[name].append(time.monotonic() - t0)

    return wrapper


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=23)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    import ray
    import ray.rllib.algorithms.ppo.ppo as ppo_mod
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

    info = host_info()
    ray.init(include_dashboard=False, object_store_memory=2 * 1024**3)
    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=2, num_envs_per_env_runner=4, rollout_fragment_length=128,
                     num_cpus_per_env_runner=1)
        .learners(num_learners=1, num_gpus_per_learner=1, num_cpus_per_learner=1)
        .training(train_batch_size_per_learner=1024, minibatch_size=256, num_epochs=4, lr=1e-3, gamma=0.99,
                  lambda_=0.95, clip_param=0.2, vf_loss_coeff=0.5, entropy_coeff=0.01, grad_clip=0.5,
                  grad_clip_by="global_norm", use_kl_loss=False)
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[64, 64], fcnet_activation="tanh",
                                                   vf_share_layers=False))
        .debugging(seed=0)
    )
    ppo_mod.synchronous_parallel_sample = timed("sample", ppo_mod.synchronous_parallel_sample)
    algo = config.build_algo()
    algo.learner_group.update = timed("update", algo.learner_group.update)
    algo.env_runner_group.sync_weights = timed("sync_weights", algo.env_runner_group.sync_weights)
    print(f"RLlib PPO: env runners {algo.env_runner_group.num_remote_env_runners()} remote, local env runner "
          f"{algo.env_runner_group.local_env_runner is not None}; learners {len(algo.learner_group)} "
          f"remote ({'local' if algo.learner_group.is_local else 'actor'}); cluster {ray.cluster_resources()}")
    rows, walls = [], []
    for it in range(a.iters):
        t0 = time.monotonic()
        res = algo.train()
        walls.append(time.monotonic() - t0)
        flat = flatten(res)
        rows.append(flat)
        ret = flat.get("env_runners/episode_return_mean", float("nan"))
        print(f"it {it:3d} wall {walls[-1] * 1e3:8.1f} ms  raw sample {RAW['sample'][-1] * 1e3:7.1f}  update "
              f"{RAW['update'][-1] * 1e3:7.1f}  sync_weights {RAW['sync_weights'][-1] * 1e3:6.2f} ms  "
              f"return_mean {ret:7.1f}  steps {flat.get('env_runners/num_env_steps_sampled', float('nan')):.0f}")
    timer_keys = sorted(k for k in rows[-1] if "timer" in k or k.startswith("timers/") or "weights_seq_no" in k
                        or k.endswith("time_this_iter_s"))
    print("\nRLlib's own timers at the last iteration (EMA-reduced, seconds) - every key that names a timer:")
    for k in timer_keys:
        print(f"  {k:78s} {rows[-1][k]:.6f}")
    st = slice(a.warmup, None)
    raw = {k: statistics.median(v[st]) for k, v in RAW.items()}
    raw["wall"] = statistics.median(walls[st])
    print(f"\nRaw medians over iterations {a.warmup}..{a.iters - 1} (ms): " +
          ", ".join(f"{k} {v * 1e3:.2f}" for k, v in raw.items()))
    print(f"  env steps/s: {1024 / raw['wall']:,.0f}")
    first = next((r["env_runners/episode_return_mean"] for r in rows if "env_runners/episode_return_mean" in r), None)
    last = rows[-1].get("env_runners/episode_return_mean")
    ok = first is not None and last is not None and last > first
    print(f"RLLIB_PPO: done, return_mean {first} -> {last}: {'PASSED' if ok else 'FAILED'}")
    algo.stop()
    ray.shutdown()
    dump_json(a.json, {"host": info, "raw_s": {k: v for k, v in RAW.items()}, "wall_s": walls,
                       "raw_median_s": raw, "timer_keys_last": {k: rows[-1][k] for k in timer_keys},
                       "rows": rows})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
