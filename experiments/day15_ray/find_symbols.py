"""Day 15 - file:line of the Block 3 reading stops in the INSTALLED Ray (no Ray import needed if you pass a path).

    python experiments/day15_ray/find_symbols.py                       # the ray package this python imports
    python experiments/day15_ray/find_symbols.py /venv/main/lib/python3.12/site-packages/ray

A text search: `def`/`class` lines, then the first line matching a pattern after a given def. A navigation aid, not
proof of a call chain. The GUIDE's line numbers come from this script on ray 2.60.0 (commit b7f4259e).
Exit code 1 if anything is not found (another Ray version: read the GUIDE's stops by name).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PPO = "rllib/algorithms/ppo/ppo.py"
PPO_L = "rllib/algorithms/ppo/torch/ppo_torch_learner.py"
ERG = "rllib/env/env_runner_group.py"
SAER = "rllib/env/single_agent_env_runner.py"
LRN = "rllib/core/learner/learner.py"
TLRN = "rllib/core/learner/torch/torch_learner.py"
GAE = "rllib/connectors/learner/general_advantage_estimation.py"
VP = "rllib/utils/postprocessing/value_predictions.py"
ADD1 = "rllib/connectors/learner/add_one_ts_to_episodes_and_truncate.py"
IMP = "rllib/algorithms/impala/impala.py"
APPO_L = "rllib/algorithms/appo/torch/appo_torch_learner.py"
VT2 = "rllib/algorithms/impala/torch/vtrace_torch_v2.py"
SER = "_private/serialization.py"
CONST = "_private/ray_constants.py"
UTL = "_private/utils.py"
NV = "_private/accelerators/nvidia_gpu.py"
WRK = "_private/worker.py"
MLOG = "rllib/utils/metrics/metrics_logger.py"

DEFS = [
    (PPO, ["training_step"]),
    (ERG, ["sync_weights", "foreach_env_runner_async_fetch_ready"]),
    (SAER, ["sample", "_sample", "set_state"]),
    (LRN, ["update", "_update"]),
    (TLRN, ["_convert_batch_type", "_update"]),
    (PPO_L, ["compute_loss_for_module"]),
    (GAE, ["GeneralAdvantageEstimation"]),
    (VP, ["compute_value_targets"]),
    (IMP, ["training_step"]),
    (APPO_L, ["compute_loss_for_module"]),
    (VT2, ["vtrace_torch"]),
    (UTL, ["set_omp_num_threads_if_unset", "set_visible_accelerator_ids", "_get_docker_cpus", "get_num_cpus"]),
    (WRK, ["put", "wait"]),
    (MLOG, ["log_time"]),
]
INSIDE = [
    # Stop 1: one PPO iteration
    (PPO, "training_step", r"synchronous_parallel_sample\(", "1 sample on every EnvRunner (blocks until all done)"),
    (PPO, "training_step", r"learner_results = self\.learner_group\.update\(", "2 learner update"),
    (PPO, "training_step", r"self\.env_runner_group\.sync_weights\(", "3 weight sync learner -> EnvRunners"),
    (MLOG, "log_time", r'reduce: str = "ema"', "timers are EMA-reduced by default"),
    # Stop 2: how weights reach the EnvRunners
    (ERG, "sync_weights", r"timeout_seconds: Optional\[float\] = 0\.0", "default: fire-and-forget"),
    (ERG, "sync_weights", r"rl_module_state = weights_src\.get_state\(", "get_state from the LearnerGroup"),
    (ERG, "sync_weights", r"rl_module_state_ref = ray\.put\(rl_module_state\)", "ONE ray.put"),
    (ERG, "sync_weights", r'func="set_state"', "set_state(ref) on every EnvRunner, async"),
    # Stop 3: sampling, the version tag, truncation and the fragment cut
    (SAER, "_sample", r"extra_model_output\[WEIGHTS_SEQ_NO\] = self\._weights_seq_no", "version tag per timestep"),
    (SAER, "_sample", r"truncated=truncateds\[env_index\]", "gymnasium truncated flag stored"),
    (SAER, "_sample", r"eps\.cut\(", "ongoing episodes cut at the end of sample()"),
    (SAER, "set_state", r"if weights_seq_no == 0 or self\._weights_seq_no < weights_seq_no", "skip older weights"),
    # Stop 4: the learner update, the version counter, H2D, the PPO ratio
    (LRN, "update", r"self\._weights_seq_no \+= 1", "one version per update() call"),
    (LRN, "update", r"fwd_out, loss_per_module, _ = self\._update\(", "per-minibatch _update"),
    (TLRN, "_convert_batch_type", r"batch_dict = convert_to_torch_tensor\(", "host -> device"),
    (PPO_L, "compute_loss_for_module", r"logp_ratio = torch\.exp\(", "ratio vs stored ACTION_LOGP"),
    (PPO_L, "compute_loss_for_module", r"torch\.clamp\(logp_ratio", "clipped surrogate"),
    # Stop 5: GAE, terminated vs truncated
    (GAE, "GeneralAdvantageEstimation", r"vf_preds = rl_module\.foreach_module\(", "values from the LEARNER's module"),
    (GAE, "GeneralAdvantageEstimation", r"module_value_targets = compute_value_targets\(", "GAE call"),
    (VP, "compute_value_targets", r"non_terminal = 1\.0 - terminateds", "terminated: no bootstrap"),
    (VP, "compute_value_targets", r"propagate = non_terminal \* \(1\.0 - truncateds\)", "truncated: stop recursion"),
    (VP, "compute_value_targets", r"td_residuals = rewards \+ gamma \* non_terminal", "delta_t"),
    (ADD1, "AddOneTsToEpisodesAndTruncate", r"\+ \[bool\(sa_episode\.is_truncated\)\]", "chunk end = truncated"),
    # Stop 6 (optional): the asynchronous family
    (IMP, "training_step", r"self\.config\.broadcast_interval", "weights broadcast every k updates"),
    (IMP, "training_step", r"async_update=do_async_updates", "learner updates without waiting"),
    (APPO_L, "compute_loss_for_module", r"behaviour_actions_logp = batch\[Columns\.ACTION_LOGP\]", "behaviour logp"),
    (APPO_L, "compute_loss_for_module", r"= vtrace_torch\(", "V-trace targets"),
    (APPO_L, "compute_loss_for_module", r"is_ratio = torch\.clip\(", "behaviour -> old-target ratio"),
    (VT2, "vtrace_torch", r"clip_rho_threshold", "rho clipping"),
    # Stop 7: Ray core facts behind the probes
    (UTL, "get_num_cpus", r"truncated from", "fractional cgroup quota truncated"),
    (UTL, "_get_docker_cpus", r'cpu_max_file_name="/sys/fs/cgroup/cpu\.max"', "reads cgroup v2 cpu.max"),
    (UTL, "set_omp_num_threads_if_unset", r"omp_num_threads = max\(math\.floor\(num_assigned_cpus\), 1\)",
     "OMP_NUM_THREADS = floor(num_cpus)"),
    (UTL, "set_visible_accelerator_ids", r"if not override_on_zero and len\(accelerator_ids\) == 0:",
     "0 GPUs: CUDA_VISIBLE_DEVICES left alone"),
    (NV, "set_current_process_visible_accelerator_ids", r"NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR",
     "opt-out env var"),
    (SER, "SerializationContext", r"self\._zero_copy_tensors_enabled = \(", "torch zero-copy is opt-in"),
    (CONST, None, r"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS = env_bool\(", "default False"),
]


def find_def(lines, name):
    pat = re.compile(rf"^\s*(def|class)\s+{re.escape(name)}\b")
    for i, l in enumerate(lines):
        if pat.match(l):
            return i
    return None


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        import ray

        root = Path(ray.__file__).parent
    version = "?"
    init = root / "_version.py"
    if init.exists():
        txt = init.read_text()
        v = re.search(r'^version = "([^"]+)"', txt, re.M)
        c = re.search(r'^commit = "([^"]+)"', txt, re.M)
        version = f"ray {v.group(1) if v else '?'} commit {c.group(1) if c else '?'}"
    print(f"# Day 15 Block 3 symbols in {root}\n# {version}\n")
    missing = 0
    cache = {}

    def lines_of(f):
        if f not in cache:
            p = root / f
            cache[f] = p.read_text().splitlines() if p.exists() else None
        return cache[f]

    for f, names in DEFS:
        ls = lines_of(f)
        for n in names:
            i = find_def(ls, n) if ls is not None else None
            print(f"{f}:{i + 1 if i is not None else '(not found)'}  def/class {n}")
            missing += i is None
    print()
    for f, owner, pat, what in INSIDE:
        ls = lines_of(f)
        hit = None
        if ls is not None:
            start = 0 if owner is None else find_def(ls, owner)
            if start is not None:
                rx = re.compile(pat)
                for j in range(start, len(ls)):
                    if rx.search(ls[j]):
                        hit = j
                        break
        print(f"{f}:{hit + 1 if hit is not None else '(not found)'}  {what}")
        missing += hit is None
    print(f"\nSYMBOLS: {missing} not found")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
