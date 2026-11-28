"""Day 15 - shared pieces: environments, the policy network, weight digests, host info, the phase-table format.

Nothing here needs Ray. `torch` is imported lazily so that the GAE checks in check_rl.py run with numpy alone.

Reused later:
  * Day 16 reuses `phase_summary` / `print_phase_table` and the weight-sync phase names in SYNC_PHASES.
  * Day 19 reuses `PolicyNet` and `ppo_update` (in learner.py) through rl_ref.ppo_surrogate.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ timing format
# One iteration record is a flat dict {phase_name: seconds}. Phase names are dotted: the prefix is the side that
# does the work. These names are the contract with Day 16 (weight sync) and the GUIDE's Block 4 tables.
SYNC_PHASES = {
    "sync.d2h": "learner: parameters GPU -> host numpy (synchronized)",
    "sync.put": "learner: ray.put of the weight dict (serialize into the object store, once)",
    "sync.get": "actor: ray.get of that object (slowest actor)",
    "sync.load": "actor: copy into the actor's parameters (slowest actor)",
    "sync.digest": "actor: SHA-256 of the actor's real parameters (check only, slowest actor)",
}
PHASES = {
    "rollout": "actor sample() wall time (slowest actor)",
    "rollout.env": "env.step + env.reset inside sample() (slowest actor)",
    "rollout.infer": "policy forward + action sampling inside sample() (slowest actor)",
    "transfer": "learner start - last fragment done: dispatch + transfer (+ queue behind the learner) [derived]",
    "learn.gae": "GAE on the learner's CPU",
    "learn.h2d": "batch host -> GPU (synchronized)",
    "learn.update": "all PPO epochs and minibatches (synchronized)",
    **SYNC_PHASES,
    "return": "learner end -> driver has the stats back [derived]",
    "idle.actor": "e2e - mean actor busy time [derived]",
    "idle.learner": "e2e - learner busy time [derived]",
    "e2e": "driver wall time of one iteration",
}


def now() -> float:
    """CLOCK_MONOTONIC: one system-wide clock, so stamps from different processes on ONE host compare directly."""
    return time.monotonic()


def phase_summary(records: list[dict], warmup: int) -> dict:
    """Median / min / max per phase over the steady-state records (records[warmup:]), in seconds."""
    steady = records[warmup:]
    out = {}
    for name in PHASES:
        vals = [r[name] for r in steady if name in r]
        if vals:
            out[name] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def print_phase_table(summary: dict, title: str, steps_per_iter: int | None = None) -> None:
    print(f"\n{title}")
    print(f"  {'phase':14s} {'median ms':>10s} {'min':>9s} {'max':>9s}  meaning")
    for name, desc in PHASES.items():
        if name in summary:
            s = summary[name]
            print(f"  {name:14s} {s['median'] * 1e3:10.2f} {s['min'] * 1e3:9.2f} {s['max'] * 1e3:9.2f}  {desc}")
    if steps_per_iter and "e2e" in summary:
        print(f"  env steps/s (steps per iteration / median e2e): {steps_per_iter / summary['e2e']['median']:,.0f}")


# ------------------------------------------------------------------------------------------------ host info
def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        return f"unreadable ({e.__class__.__name__})"


def cgroup_cpu() -> dict:
    """The CPU quota this container may use (cgroup v2 cpu.max = 'quota period') and the throttling counters."""
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    quota = None
    parts = cpu_max.split()
    if len(parts) == 2 and parts[0].isdigit():
        quota = int(parts[0]) / int(parts[1])
    stat = {}
    for line in _read("/sys/fs/cgroup/cpu.stat").splitlines():
        k, _, v = line.partition(" ")
        if v.strip().isdigit():
            stat[k] = int(v)
    return {"cpu.max": cpu_max, "quota_cpus": quota, "cpu.stat": stat}


def host_info() -> dict:
    """Recorded with every run: the host's CPUs are shared, and the quota differs from the CPU count."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "container_id": os.environ.get("CONTAINER_ID", "?"),
        "os.cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup": cgroup_cpu(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for mod in ("numpy", "torch", "ray", "gymnasium"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
            if mod == "ray":
                info["ray.commit"] = getattr(m, "__commit__", "?")
        except ImportError:
            info[mod] = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        info["gpus"] = r.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = []
    return info


def dump_json(path: str | None, obj) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ environments
class SyntheticEnv:
    """A gymnasium-style env whose step cost and observation size are dials (rollout-bound vs learner-bound).

    * step cost: a BUSY wait of `step_cost_us` (it burns CPU quota like a real simulator; sleep would not).
    * observations: `obs_dim` float32 features of a hidden 1-D random walk x.
    * reward 1 when the action equals (x > 0), else 0, so a policy CAN learn it (not needed for timing).
    * terminated with probability `p_term` per step; truncated at `max_steps` (a time limit, like CartPole's 500).
    """

    def __init__(self, obs_dim: int = 4, step_cost_us: float = 0.0, max_steps: int = 200, p_term: float = 0.005,
                 seed: int = 0):
        self.obs_dim, self.step_cost_us, self.max_steps, self.p_term = obs_dim, step_cost_us, max_steps, p_term
        self.rng = np.random.default_rng(seed)
        self.proj = self.rng.standard_normal(obs_dim).astype(np.float32)
        self.x, self.t = 0.0, 0

    def _obs(self):
        return (self.proj * self.x + 0.1 * self.rng.standard_normal(self.obs_dim)).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x, self.t = float(self.rng.standard_normal()), 0
        return self._obs(), {}

    def step(self, action: int):
        if self.step_cost_us > 0:
            end = time.perf_counter() + self.step_cost_us * 1e-6
            while time.perf_counter() < end:
                pass
        reward = 1.0 if int(action) == int(self.x > 0) else 0.0
        self.x = 0.9 * self.x + 0.5 * float(self.rng.standard_normal())
        self.t += 1
        terminated = bool(self.rng.random() < self.p_term)
        truncated = (not terminated) and self.t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}


def make_env(name: str, seed: int, obs_dim: int = 4, step_cost_us: float = 0.0):
    if name == "cartpole":
        import gymnasium as gym

        return gym.make("CartPole-v1")  # TimeLimit 500 -> truncated=True at step 500
    if name == "synthetic":
        return SyntheticEnv(obs_dim=obs_dim, step_cost_us=step_cost_us, seed=seed)
    raise ValueError(name)


def env_dims(name: str, obs_dim: int) -> tuple[int, int]:
    return (4, 2) if name == "cartpole" else (obs_dim, 2)


# ------------------------------------------------------------------------------------------------ policy
def build_policy(obs_dim: int, n_actions: int, hidden: int, seed: int):
    """Separate actor and critic MLPs (tanh), the CleanRL CartPole shape. Same seed -> same initial weights."""
    import torch
    from torch import nn

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                       nn.Linear(hidden, n_actions))
            self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                        nn.Linear(hidden, 1))

        def forward(self, obs):
            return self.actor(obs), self.critic(obs).squeeze(-1)

    torch.manual_seed(seed)
    return PolicyNet()


def weights_numpy(net) -> dict:
    """name -> contiguous float32 numpy array on the host. This dict is what travels through the object store."""
    return {k: v.detach().to("cpu").contiguous().numpy() for k, v in net.state_dict().items()}


def digest_arrays(arrays: dict) -> str:
    """SHA-256 over names, shapes and bytes, in sorted name order. 16 hex digits are plenty for a check."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def digest_module(net) -> str:
    """Digest of the parameters a module REALLY holds (not of what it was sent)."""
    return digest_arrays(weights_numpy(net))
