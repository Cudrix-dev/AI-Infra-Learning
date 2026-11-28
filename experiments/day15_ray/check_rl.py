"""Day 15 - local checks without Ray: GAE (numpy only) and the PPO surrogate (torch on CPU).

    python experiments/day15_ray/check_rl.py                     # reference, all tests, plus the controls
    python experiments/day15_ray/check_rl.py --impl mine --tests gae

gae   1. the hand-worked example of GUIDE §2, P7 (exact: every number is a binary fraction)
      2. 200 random fragments [T=37, E=3] with terminations, truncations and the cut vs gae_slow_reference
      controls: gae_trunc_as_term, gae_no_cut_bootstrap, gae_cross_boundary must each FAIL test 1 and test 2
ppo   3. the hand-worked surrogate of GUIDE §3.1 (P8): loss, clip fraction, approx KL and d loss / d new_logp
      4. lag 0: logp recomputed with the SAME weights -> ratio exactly 1, clip fraction 0
      5. lag 1: after one optimizer step the stored behaviour logp gives max|ratio-1| > 1e-5, while logp
         recomputed with the new weights (the recompute_logp control) gives exactly 1: staleness hidden
Exit 0 = all passed and every control detected; 1 = a failure; 3 = --impl mine still has TODOs in a tested function.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rl_controls  # noqa: E402
import rl_ref  # noqa: E402

G, L = 0.5, 0.5  # the hand example's gamma and lambda (gamma * lambda = 1/4)
HAND = {  # GUIDE §2 (P7): rows 0-2 episode A (truncated at 2), row 3 episode B (terminated), rows 4-5 episode C (cut)
    "rewards": [1, 0, 1, 1, 0, 1],
    "values": [1, 2, 2.5, 0.5, 2, 2],
    "next_values": [2, 2.5, 4, 9, 2, 4],  # row 2: V(final obs) = 4; row 3: unused (terminated); row 5: V(s_6) = 4
    "terminated": [False, False, False, True, False, False],
    "truncated": [False, False, True, False, False, False],
}
HAND_ADV = [0.84375, -0.625, 0.5, 0.5, -0.75, 1.0]
HAND_RET = [1.84375, 1.375, 3.0, 1.0, 1.25, 3.0]


class Results:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append({"name": name, "status": status, "detail": detail})
        print(f"{status:10s} {name:34s} {detail}")


def has_todo(fn) -> bool:
    return "raise NotImplementedError" in inspect.getsource(fn)


def hand_ok(fn):
    a, r = fn(*(np.array(HAND[k]) for k in ("rewards", "values", "next_values", "terminated", "truncated")), G, L)
    return bool(np.array_equal(a, np.float32(HAND_ADV)) and np.array_equal(r, np.float32(HAND_RET))), a


def random_frags(n=200, T=37, E=3, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        term = rng.random((T, E)) < 0.06
        trunc = (~term) & (rng.random((T, E)) < 0.06)
        yield (rng.standard_normal((T, E)), rng.standard_normal((T, E)), rng.standard_normal((T, E)), term, trunc)


def random_ok(fn, gamma=0.99, lam=0.95):
    worst = 0.0
    for fr in random_frags():
        a, r = fn(*fr, gamma, lam)
        a0, r0 = rl_ref.gae_slow_reference(*fr, gamma, lam)
        worst = max(worst, float(np.abs(a - a0).max()), float(np.abs(r - r0).max()))
    return worst <= 1e-5, worst


def test_gae(impl, res: Results):
    fn = impl.compute_gae
    if has_todo(fn):
        res.add("gae.hand_example", "INCOMPLETE", "TODO 1 left")
        res.add("gae.vs_slow_reference", "INCOMPLETE", "TODO 1 left")
    else:
        ok, a = hand_ok(fn)
        res.add("gae.hand_example", "PASS" if ok else "FAIL", f"advantages {np.round(a, 6).tolist()}")
        ok, worst = random_ok(fn)
        res.add("gae.vs_slow_reference", "PASS" if ok else "FAIL", f"200 fragments, max |diff| {worst:.2e} (tol 1e-5)")
        a1, _ = fn(*(np.array(HAND[k])[:, None].repeat(2, 1) for k in
                     ("rewards", "values", "next_values", "terminated", "truncated")), G, L)
        ok = a1.shape == (6, 2) and np.array_equal(a1[:, 0], np.float32(HAND_ADV))
        res.add("gae.shape_TxE", "PASS" if ok else "FAIL", f"[T, E] input -> {a1.shape}")
    for name, ctl in rl_controls.GAE_CONTROLS.items():
        ok_h, a = hand_ok(ctl)
        ok_r, worst = random_ok(ctl)
        detected = (not ok_h) and (not ok_r)
        res.add(f"control.{name}", "DETECTED" if detected else "MISSED",
                f"hand example {'passes' if ok_h else 'fails'} (adv {np.round(a, 6).tolist()}), "
                f"random max |diff| {worst:.2e}")


def test_ppo(impl, res: Results):
    try:
        import torch
    except ImportError:
        res.add("ppo.*", "SKIPPED", "no torch here: run on the server")
        return
    fn = impl.ppo_surrogate
    if has_todo(fn):
        for n in ("ppo.hand_example", "ppo.lag0_ratio_one", "ppo.lag1_stale"):
            res.add(n, "INCOMPLETE", "TODO 2 left")
        return
    torch.set_num_threads(1)
    # 3. hand example: ratios [0.5, 0.9, 1.0, 1.1, 1.5], advantages [1, -1, 1, 1, -1], clip 0.2
    ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5], dtype=torch.float64)
    adv = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
    old = torch.zeros(5, dtype=torch.float64)
    new = torch.log(ratio).requires_grad_(True)
    loss, st = fn(new, old, adv, 0.2)
    st = {k: float(v) for k, v in st.items()}
    loss.backward()
    want_grad = [-0.1, 0.18, -0.2, -0.22, 0.3]
    kl = float(((ratio - 1) - torch.log(ratio)).mean())
    ok = (abs(loss.item() + 0.04) < 1e-12 and abs(st["clip_frac"] - 0.4) < 1e-6  # clip_frac is a float32 mean
          and abs(st["max_ratio_dev"] - 0.5) < 1e-12 and abs(st["approx_kl"] - kl) < 1e-12
          and torch.allclose(new.grad, torch.tensor(want_grad, dtype=torch.float64), atol=1e-12))
    res.add("ppo.hand_example", "PASS" if ok else "FAIL",
            f"loss {loss.item():.4f} (want -0.0400), clip_frac {st['clip_frac']:.2f} (0.40), approx_kl "
            f"{st['approx_kl']:.5f} ({kl:.5f}), grad {[round(g, 4) for g in new.grad.tolist()]}")
    # 4./5. a real policy on CPU: lag 0, then lag 1 after one optimizer step
    from rl_common import build_policy

    net = build_policy(4, 2, 64, seed=0)
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(512, 4, generator=g)
    with torch.no_grad():
        logits, _ = net(obs)
        act = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(1)
        behaviour = torch.log_softmax(logits, -1).gather(1, act[:, None]).squeeze(1)
    advs = torch.randn(512, generator=g)

    def logp(n):
        lg, _ = n(obs)
        return torch.log_softmax(lg, -1).gather(1, act[:, None]).squeeze(1)

    _, st0 = fn(logp(net), behaviour, advs, 0.2)
    st0 = {k: float(v) for k, v in st0.items()}
    ok = st0["max_ratio_dev"] == 0.0 and st0["clip_frac"] == 0.0
    res.add("ppo.lag0_ratio_one", "PASS" if ok else "FAIL",
            f"max|r-1| {st0['max_ratio_dev']:.1e}, clip_frac {st0['clip_frac']}, approx_kl {st0['approx_kl']:.1e}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(4):  # one 'version' = several optimizer steps, as in an update with 4 minibatches
        loss, _ = fn(logp(net), behaviour, advs, 0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, st1 = fn(logp(net), behaviour, advs, 0.2)
        _, hid = fn(logp(net), logp(net), advs, 0.2)  # the recompute_logp control
        st1, hid = ({k: float(v) for k, v in d.items()} for d in (st1, hid))
    ok = st1["max_ratio_dev"] > 1e-5 and hid["max_ratio_dev"] == 0.0
    res.add("ppo.lag1_stale", "PASS" if ok else "FAIL",
            f"stored behaviour logp: max|r-1| {st1['max_ratio_dev']:.3e}, clip_frac {st1['clip_frac']:.3f}; "
            f"recomputed: max|r-1| {hid['max_ratio_dev']:.1e} -> staleness hidden")
    res.add("control.ppo_recompute_logp", "DETECTED" if hid["max_ratio_dev"] <= 1e-5 < st1["max_ratio_dev"]
            else "MISSED", "lag 1 with ratio == 1 is what the train.py ratio check flags as HIDDEN")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["ref", "mine"], default="ref")
    p.add_argument("--tests", default="gae,ppo")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    impl = rl_ref if a.impl == "ref" else __import__("my_rl")
    res = Results()
    tests = a.tests.split(",")
    if "gae" in tests:
        test_gae(impl, res)
    if "ppo" in tests:
        test_ppo(impl, res)
    counts = {s: sum(r["status"] == s for r in res.rows) for s in
              ("PASS", "FAIL", "INCOMPLETE", "DETECTED", "MISSED", "SKIPPED")}
    ok = counts["FAIL"] == 0 and counts["MISSED"] == 0 and counts["INCOMPLETE"] == 0
    verdict = "INCOMPLETE" if counts["INCOMPLETE"] and not counts["FAIL"] else ("PASSED" if ok else "FAILED")
    print(f"CHECK_RL ({a.impl}): {counts['PASS']} passed, {counts['FAIL']} failed, {counts['INCOMPLETE']} incomplete, "
          f"controls {counts['DETECTED']} detected / {counts['MISSED']} missed, {counts['SKIPPED']} skipped: {verdict}")
    if a.json:
        Path(a.json).write_text(json.dumps(res.rows, indent=1))
    sys.exit(3 if verdict == "INCOMPLETE" else (0 if ok else 1))


if __name__ == "__main__":
    main()
