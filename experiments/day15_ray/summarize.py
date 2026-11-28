"""Day 15 - tables from one session's JSON files: Exp 2 (sync vs async), Exp 3 (scaling + prediction), Exp 4 (RLlib).

    python experiments/day15_ray/summarize.py outputs/day15_ray/session_<RID>

The Exp 3 prediction uses ONLY Exp 2's sync medians (CartPole, 2 actors) plus the dialed env cost:
  i  = rollout.infer / frag_len                    actor inference per vector step (4 envs per actor)
  e0 = rollout.env / (frag_len * 4)                one CartPole env.step; stands in for the synthetic env's own cost
  u  = learn.update / optimizer steps              one PPO minibatch step on the GPU (host-bound at this size)
  g  = (learn.gae + learn.h2d) / steps             per-sample learner CPU work
  o  = transfer + return + max sync.get+load+digest + sync.d2h + sync.put   fixed per-iteration overhead
For N actors, B steps per iteration, cost c per env.step:  R = (B / 4N) * (i + 4 (e0 + c)),  U = S u + B g
  sync:  T = R + U + o              async (pipelined):  T = max(R, U + o)     [derived]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def med(js, k):
    return js["summary"][k]["median"] if k in js["summary"] else 0.0


def main():
    s = Path(sys.argv[1])
    out = []

    def say(line=""):
        print(line)
        out.append(line)

    e2 = {m: load(s / f"exp2_{m}.json") for m in ("sync", "async")}
    if all(e2.values()):
        say("## Exp 2: CartPole, 2 CPU actors x 4 envs x 128 steps + 1 GPU learner (steady-state medians, ms)")
        keys = [k for k in e2["sync"]["summary"]]
        say(f"| phase | sync | async |")
        say("| --- | ---: | ---: |")
        for k in keys:
            say(f"| {k} | {med(e2['sync'], k) * 1e3:.2f} | {med(e2['async'], k) * 1e3:.2f} |")
        for m in ("sync", "async"):
            js = e2[m]
            say(f"{m}: {js['steps_per_iter'] / med(js, 'e2e'):,.0f} env steps/s steady state; lag histogram "
                f"{hist(js['lags'])}; checks {js['checks']}")
    sync = e2.get("sync")
    rows = []
    for p in sorted(s.glob("exp3_*.json")):
        m = re.match(r"exp3_(sync|async)_c(\d+)_n(\d+)\.json", p.name)
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)), load(p)))
    if rows and sync:
        fl = sync["args"]["frag_len"]
        steps2 = sync["steps_per_iter"]
        opt2 = steps2 // sync["args"]["minibatch"] * sync["args"]["epochs"]
        i = med(sync, "rollout.infer") / fl
        e0 = med(sync, "rollout.env") / (fl * 4)
        u = med(sync, "learn.update") / opt2
        g = (med(sync, "learn.gae") + med(sync, "learn.h2d")) / steps2
        o = sum(med(sync, k) for k in ("transfer", "return", "sync.get", "sync.load", "sync.digest", "sync.d2h",
                                       "sync.put"))
        say("\n## Exp 3: synthetic env, fixed global batch per iteration, 4 envs per actor")
        say(f"model inputs from Exp 2 sync [derived]: i {i * 1e6:.1f} us/vector step, e0 {e0 * 1e6:.1f} us/env step, "
            f"u {u * 1e3:.2f} ms/optimizer step, g {g * 1e6:.2f} us/sample, o {o * 1e3:.2f} ms/iteration")
        say("| mode | env cost us | actors | predicted e2e ms | measured e2e ms | measured rollout | learner busy | "
            "steps/s | bottleneck | throttled s |")
        say("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        for mode, cost, n, js in sorted(rows, key=lambda r: (r[1], r[0], r[2])):
            B = js["steps_per_iter"]
            S = B // js["args"]["minibatch"] * js["args"]["epochs"]
            R = (B / (4 * n)) * (i + 4 * (e0 + cost * 1e-6))
            U = S * u + B * g
            pred = R + U + o if mode == "sync" else max(R, U + o)
            e2e = med(js, "e2e")
            roll = med(js, "rollout")
            learn = sum(med(js, k) for k in ("learn.gae", "learn.h2d", "learn.update", "sync.d2h", "sync.put"))
            side = "rollout" if roll > learn else "learner"
            say(f"| {mode} | {cost} | {n} | {pred * 1e3:.0f} | {e2e * 1e3:.0f} | {roll * 1e3:.0f} | {learn * 1e3:.0f} | "
                f"{B / e2e:,.0f} | {side} | {js['cgroup_delta']['throttled_usec'] / 1e6:.2f} |")
    rl = load(s / "rllib.json")
    if rl and sync:
        r = rl["raw_median_s"]
        say("\n## Exp 4: RLlib PPO (2 EnvRunners x 4 envs, 1 GPU Learner actor) vs train.py sync, CartPole, ms")
        say("| phase | train.py sync | RLlib raw (wrapped call) | RLlib's own timer (EMA, last iteration) |")
        say("| --- | ---: | ---: | ---: |")
        tk = rl["timer_keys_last"]
        say(f"| sampling | {med(sync, 'rollout') * 1e3:.1f} (+ transfer {med(sync, 'transfer') * 1e3:.1f}) | "
            f"{r['sample'] * 1e3:.1f} | {tk.get('timers/env_runner_sampling_timer', float('nan')) * 1e3:.1f} |")
        lu = sum(med(sync, k) for k in ("learn.gae", "learn.h2d", "learn.update"))
        say(f"| learner update | {lu * 1e3:.1f} | {r['update'] * 1e3:.1f} | "
            f"{tk.get('timers/learner_update_timer', float('nan')) * 1e3:.1f} |")
        ws = sum(med(sync, k) for k in ("sync.d2h", "sync.put", "sync.get", "sync.load"))
        say(f"| weight sync | {ws * 1e3:.2f} | {r['sync_weights'] * 1e3:.2f} (fire-and-forget) | "
            f"{tk.get('timers/synch_weights', float('nan')) * 1e3:.2f} |")
        say(f"| iteration | {med(sync, 'e2e') * 1e3:.1f} | {r['wall'] * 1e3:.1f} | "
            f"{tk.get('timers/training_iteration', float('nan')) * 1e3:.1f} |")
        say(f"| GAE (learner connector) | {med(sync, 'learn.gae') * 1e3:.2f} | - | "
            f"{tk.get('learners/__all_modules__/learner_connector/timers/connectors/general_advantage_estimation', float('nan')) * 1e3:.1f} |")
        say(f"| per vector step: inference / env step | {med(sync, 'rollout.infer') / 128 * 1e3:.3f} / "
            f"{med(sync, 'rollout.env') / 128 * 1e3:.3f} | - | "
            f"{tk.get('env_runners/rlmodule_inference_timer', float('nan')) * 1e3:.3f} / "
            f"{tk.get('env_runners/env_step_timer', float('nan')) * 1e3:.3f} |")
    reps = {}
    for p in sorted(s.glob("repro_*.json")):
        m = re.match(r"repro_(\w+)_(\d)\.json", p.name)
        if m:
            reps.setdefault(m.group(1), []).append(load(p)["final_digest"])
    if reps:
        say("\n## Reproducibility (optional): final weight digest of two identical runs")
        for k, v in reps.items():
            say(f"{k}: {v} -> {'identical' if len(set(v)) == 1 else 'DIFFERENT'}")
    for p in sorted(s.glob("exp3n8_*.json")):
        js = load(p)
        say(f"\nExp 3 optional, 8 actors, cost 400 us, --ray-num-cpus 10 ({p.stem}): e2e {med(js, 'e2e') * 1e3:.0f} ms, "
            f"rollout {med(js, 'rollout') * 1e3:.0f} ms, {js['steps_per_iter'] / med(js, 'e2e'):,.0f} steps/s, "
            f"throttled {js['cgroup_delta']['throttled_usec'] / 1e6:.2f} s in {js['cgroup_delta']['nr_throttled']} periods")
    for p in sorted(s.glob("wait_n*.json")):
        js = load(p)
        say(f"\nwait mode {p.stem}: lag histogram {hist(js['lags'])}, e2e median {med(js, 'e2e') * 1e3:.1f} ms per "
            f"one-fragment update")
    (s / "summary.md").write_text("\n".join(out) + "\n")
    print("\nSUMMARY: done")


def hist(lags):
    flat = [x for l in lags for x in l]
    return {k: flat.count(k) for k in sorted(set(flat))}


if __name__ == "__main__":
    main()
