"""Day 15 - YOUR pieces. Fill in the TODOs in order (GUIDE Block 2). Keep the names and signatures.

  Stage 1  TODO 1 compute_gae      -> python experiments/day15_ray/check_rl.py --impl mine --tests gae      (numpy only)
  Stage 2  TODO 2 ppo_surrogate    -> python experiments/day15_ray/check_rl.py --impl mine --tests ppo      (torch, CPU)
  Stage 3  TODO 3 pipelined_iteration
           -> python experiments/day15_ray/train.py --impl mine --mode async --iters 8 --warmup 2   (Ray, 1 GPU)

While a function still contains `raise NotImplementedError`, check_rl.py prints INCOMPLETE for its checks and exits 3;
train.py --impl mine exits 3 before it starts Ray. The reference is rl_ref.py: read it after your version runs.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------------------------- TODO 1 (25 min)
def compute_gae(
    rewards, values, next_values, terminated, truncated, gamma: float, lam: float
):
    """GAE over one fragment, time-major. Inputs [T] or [T, E]; returns (advantages, returns) as float32.

    Row t is the transition s_t -> s_{t+1}.  next_values[t] = V(s_{t+1}) (already filled in by the actor, also at a
    truncation and at the fragment's last row).  The fragment's last row is a CUT: bootstrap there unless terminated.
      delta_t = r_t + gamma * (1 - terminated_t) * next_values_t - values_t
      A_t     = delta_t + gamma * lam * (1 - end_t) * A_{t+1},    end_t = terminated_t or truncated_t or t == T-1
      returns = A + values
    GUIDE §2 / P7: which of the two flags stops the bootstrap, and which only stops the recursion?
    """
    # TODO(1a): delta for every row (vectorized over rows and envs).
    # TODO(1b): the backward recursion from t = T-1 down to 0, reset where end_t is True.
    # TODO(1c): returns = advantages + values; cast both to float32.
    # Keep the recurrence and value targets in float64 until returning them.
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    use_nextvalue = ~terminated
    delta = rewards + gamma * use_nextvalue * next_values - values
    end = terminated | truncated
    end[-1] = True  # last row is always a cut
    advantages = np.empty_like(delta)
    running = np.zeros_like(delta[0])  # one entry per environment
    for t in range(len(delta) - 1, -1, -1):
        running = delta[t] + gamma * lam * (1 - end[t]) * running
        advantages[t] = running
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


# ---------------------------------------------------------------------------------------------- TODO 2 (20 min)
def ppo_surrogate(new_logp, old_logp, adv, clip: float):
    """Clipped PPO policy loss on one minibatch of torch tensors [B]. Returns (loss, stats).

      ratio = exp(new_logp - old_logp)       old_logp: the BEHAVIOUR log-prob stored by the actor at sampling time
      loss  = -mean(min(ratio * adv, clamp(ratio, 1 - clip, 1 + clip) * adv))
    stats (detached 0-dim tensors, no .item()): clip_frac = mean(|ratio - 1| > clip), max_ratio_dev = max |ratio - 1|,
                           approx_kl = mean((ratio - 1) - log ratio), ratio_mean = mean(ratio).
    Compute the stats under torch.no_grad(). GUIDE §3.1 / P8: at lag 0, what are they on the first minibatch?
    """
    # TODO(2a): log_ratio and ratio.
    # TODO(2b): the clipped objective (keep it differentiable).
    # TODO(2c): the four stats.
    import torch

    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clamp(ratio, 1 - clip, 1 + clip)
    loss = -torch.mean(torch.min(ratio * adv, clipped_ratio * adv))
    with torch.no_grad():
        clip_frac = torch.mean((torch.abs(ratio - 1) > clip).float())
        max_ratio_dev = torch.max(torch.abs(ratio - 1))
        approx_kl = torch.mean((ratio - 1) - torch.log(ratio))
        ratio_mean = torch.mean(ratio)
    stats = {
        "clip_frac": clip_frac,
        "max_ratio_dev": max_ratio_dev,
        "approx_kl": approx_kl,
        "ratio_mean": ratio_mean,
    }
    return loss, stats


# ---------------------------------------------------------------------------------------------- TODO 3 (30 min)
def pipelined_iteration(actors, learner, in_flight, frag_len: int, update_kwargs: dict):
    """One iteration of the one-step-pipelined (async) driver. Returns (next_in_flight, stats, load_refs).

    in_flight: one ObjectRef per actor from `actor.sample.remote(frag_len)`, launched in the previous iteration.
    Available calls:  actor.sample.remote(frag_len) -> ref to a fragment
                      learner.update.remote(*fragment_refs, **update_kwargs) -> ref to stats (stats["pub"] = weights)
                      actor.set_weights.remote(stats["pub"]) -> ref to a timing dict (collect them as load_refs)
    The actors must be sampling while the learner updates. GUIDE §1.3 / P4: in which order must you submit the
    three kinds of call so that the steady-state lag is exactly 1, and what breaks if you ray.get the fragments?
    """
    import ray  # noqa: F401  (you will need ray.get)

    # TODO(3a): launch the next fragment on every actor.
    # TODO(3b): give the in-flight refs to the learner; wait for its stats.
    # TODO(3c): queue the new weights on every actor; keep the refs.
    next_in_flight = [actor.sample.remote(frag_len) for actor in actors]
    stats_ref = learner.update.remote(*in_flight, **update_kwargs)
    stats = ray.get(stats_ref)
    load_refs = [actor.set_weights.remote(stats["pub"]) for actor in actors]
    return next_in_flight, stats, load_refs
