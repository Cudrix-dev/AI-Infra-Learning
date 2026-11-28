"""Day 15 - planted bugs for the negative controls. Each must FAIL the check it targets (GUIDE Block 2 Stage 4).

GAE controls (check_rl.py --tests gae): wrappers around the reference with one classic mistake each.
  gae_trunc_as_term    a time-limit truncation treated as a real termination: no bootstrap from V(final obs)
  gae_no_cut_bootstrap the fragment's last row treated as terminal: no bootstrap at the cut
  gae_cross_boundary   the recursion runs across episode ends (the deltas are still right)
The two Ray controls live where the bug would live:
  skip_sync        workers.RolloutWorker.set_weights: actor 1 claims version 2 but keeps version 1's weights
                   -> must be caught by the digest check (train.py --control skip_sync)
  recompute_logp   workers.Learner.update: "behaviour" logp recomputed with the learner's current weights
                   -> ratio == 1 although lag >= 1: the ratio check must flag HIDDEN staleness
"""

from __future__ import annotations

import numpy as np

import rl_ref


def gae_trunc_as_term(rewards, values, next_values, terminated, truncated, gamma, lam):
    term = np.asarray(terminated, bool) | np.asarray(truncated, bool)
    return rl_ref.compute_gae(rewards, values, next_values, term, np.zeros_like(term), gamma, lam)


def gae_no_cut_bootstrap(rewards, values, next_values, terminated, truncated, gamma, lam):
    term = np.array(terminated, bool, copy=True)
    term[-1] = True
    return rl_ref.compute_gae(rewards, values, next_values, term, truncated, gamma, lam)


def gae_cross_boundary(rewards, values, next_values, terminated, truncated, gamma, lam):
    r, v, nv = (np.asarray(a, np.float64) for a in (rewards, values, next_values))
    term = np.asarray(terminated, bool)
    delta = r + gamma * (~term) * nv - v
    adv = np.zeros_like(delta)
    running = np.zeros_like(delta[0])
    for t in range(len(r) - 1, -1, -1):
        running = delta[t] + gamma * lam * running  # BUG: never reset at an episode end
        adv[t] = running
    return adv.astype(np.float32), (adv + v).astype(np.float32)


GAE_CONTROLS = {"gae_trunc_as_term": gae_trunc_as_term, "gae_no_cut_bootstrap": gae_no_cut_bootstrap,
                "gae_cross_boundary": gae_cross_boundary}
