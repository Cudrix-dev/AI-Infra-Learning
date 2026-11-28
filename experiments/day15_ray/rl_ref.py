"""Day 15 - the reference for the three YOU WRITE pieces in my_rl.py (same names, same signatures).

  TODO 1  compute_gae           GAE with terminated / truncated / fragment-cut handling   (GUIDE §2)
  TODO 2  ppo_surrogate         the clipped PPO policy loss and its staleness statistics    (GUIDE §3.1)
  TODO 3  pipelined_iteration   one iteration of the one-step-pipelined (async) driver      (GUIDE §1.3)

Read it after your version runs, or when stuck. `gae_slow_reference` is a second, deliberately different GAE
(the textbook sum, O(T^2)); the checks compare both against the hand-worked example.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------------ TODO 1
def compute_gae(rewards, values, next_values, terminated, truncated, gamma: float, lam: float):
    """Generalized advantage estimation over one rollout fragment, time-major.

    All inputs have shape [T] or [T, E] (E envs in one actor). Row t is the transition s_t --a_t--> s_{t+1}.
      rewards[t]      r_t
      values[t]       V(s_t) from the behaviour policy's critic
      next_values[t]  V(s_{t+1}): the next row's value inside an episode; V(final observation) at a truncation;
                      V(current observation) at the fragment's last row. Unused (any value) where terminated.
      terminated[t]   the episode really ended at t (no s_{t+1}): do NOT bootstrap
      truncated[t]    the episode was cut by a time limit at t: s_{t+1} exists, DO bootstrap
    The fragment's last row (t = T-1) is a cut: bootstrap from next_values[T-1] unless terminated there.

      delta_t = r_t + gamma * (1 - terminated_t) * next_values_t - values_t
      A_t     = delta_t + gamma * lam * (1 - end_t) * A_{t+1}       end_t = terminated | truncated | (t == T-1)
      R_t     = A_t + values_t                                      (the value target)
    Returns (advantages, returns), float32, same shape as rewards.
    """
    r = np.asarray(rewards, np.float64)
    v = np.asarray(values, np.float64)
    nv = np.asarray(next_values, np.float64)
    term = np.asarray(terminated, bool)
    end = term | np.asarray(truncated, bool)
    end[-1] = True  # the fragment cut
    delta = r + gamma * (~term) * nv - v
    adv = np.zeros_like(delta)
    running = np.zeros_like(delta[0])
    for t in range(len(r) - 1, -1, -1):
        running = delta[t] + gamma * lam * (~end[t]) * running
        adv[t] = running
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def gae_slow_reference(rewards, values, next_values, terminated, truncated, gamma: float, lam: float):
    """The definition, written out: A_t = sum_k (gamma*lam)^k delta_{t+k}, summed until the end of t's segment.

    One column at a time, no recursion, O(T^2). Independent of compute_gae on purpose."""
    r = np.asarray(rewards, np.float64)
    squeeze = r.ndim == 1
    cols = [np.asarray(a, np.float64) for a in (rewards, values, next_values)]
    term_all = np.asarray(terminated, bool)
    trunc_all = np.asarray(truncated, bool)
    if squeeze:
        cols = [c[:, None] for c in cols]
        term_all, trunc_all = term_all[:, None], trunc_all[:, None]
    rr, vv, nn = cols
    T, E = rr.shape
    adv = np.zeros((T, E))
    for e in range(E):
        for t in range(T):
            total, k = 0.0, 0
            while True:
                s = t + k
                bootstrap = 0.0 if term_all[s, e] else nn[s, e]
                total += (gamma * lam) ** k * (rr[s, e] + gamma * bootstrap - vv[s, e])
                if term_all[s, e] or trunc_all[s, e] or s == T - 1:
                    break
                k += 1
            adv[t, e] = total
    ret = adv + vv
    if squeeze:
        adv, ret = adv[:, 0], ret[:, 0]
    return adv.astype(np.float32), ret.astype(np.float32)


# ------------------------------------------------------------------------------------------------ TODO 2
def ppo_surrogate(new_logp, old_logp, adv, clip: float):
    """Clipped PPO policy loss on one minibatch (torch tensors, same shape [B]).

      ratio = exp(new_logp - old_logp)          old_logp = the BEHAVIOUR policy's log-prob, stored at sampling time
      loss  = -mean( min(ratio * adv, clip(ratio, 1 - clip, 1 + clip) * adv) )
    Returns (loss, stats) with stats as detached 0-dim tensors (no .item(): that would sync the GPU every minibatch):
      clip_frac      fraction of samples with |ratio - 1| > clip
      max_ratio_dev  max |ratio - 1|            (exactly 0 up to float noise before the first step at lag 0)
      approx_kl      mean((ratio - 1) - log ratio)   (k3 estimator of KL(behaviour || current))
      ratio_mean
    """
    import torch

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        dev = (ratio - 1.0).abs()
        stats = {
            "clip_frac": (dev > clip).float().mean(),
            "max_ratio_dev": dev.max(),
            "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
            "ratio_mean": ratio.mean(),
        }
    return loss, stats


# ------------------------------------------------------------------------------------------------ TODO 3
def pipelined_iteration(actors, learner, in_flight, frag_len: int, update_kwargs: dict):
    """One iteration of the one-step-pipelined driver (lag 1 in steady state).

    in_flight: one ObjectRef per actor, `actor.sample.remote(frag_len)` launched in the previous iteration.
    Does, in this order:
      1. launch the NEXT fragment on every actor right away, so the actors keep sampling while the learner works;
      2. hand the in-flight refs to the learner (pass the ObjectRefs; do not ray.get the fragments in the driver)
         and wait for its stats;
      3. queue the new weights on every actor. An actor runs its calls in submission order, so these weights
         apply AFTER the fragment launched in step 1: that fragment was produced by the previous weights.
    Returns (next_in_flight, stats, load_refs): load_refs are the set_weights refs (their timing dicts).
    """
    import ray

    next_in_flight = [a.sample.remote(frag_len) for a in actors]
    stats = ray.get(learner.update.remote(*in_flight, **update_kwargs))
    load_refs = [a.set_weights.remote(stats["pub"]) for a in actors]
    return next_in_flight, stats, load_refs
