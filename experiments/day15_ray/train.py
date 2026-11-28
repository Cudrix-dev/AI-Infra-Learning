"""Day 15 - the actor-learner driver on Ray core: N CPU rollout actors + 1 GPU learner, PPO, versioned weights.

    python experiments/day15_ray/train.py --mode sync  --iters 23 --warmup 3 --json out.json
    python experiments/day15_ray/train.py --mode async --iters 23 --warmup 3
    python experiments/day15_ray/train.py --mode wait  --actors 4 --iters 40          (optional, IMPALA-style)
    python experiments/day15_ray/train.py --mode async --control skip_sync           (negative control)
    python experiments/day15_ray/train.py --mode async --control recompute_logp      (negative control)

Modes (GUIDE §1.3):
  sync   set_weights -> sample on every actor -> update -> publish. Lag 0. Actors idle while the learner works.
  async  one-step pipelined: the next fragment is launched BEFORE the update (pipelined_iteration, TODO 3).
         Lag 0 for the first batch, 1 afterwards. Deterministic schedule.
  wait   ray.wait for the FIRST finished fragment, update on it alone, re-launch that actor with the newest
         weights. Lag depends on arrival order; not bitwise reproducible.

Checks printed at the end (each a line starting with CHECK):
  versions  sync: every lag 0; async: [0, 1, 1, ...]; wait: reported only
  digest    every fragment's weight digest == the learner's digest for the version it claims
  ratio     lag 0 batches: first-minibatch max|ratio-1| <= --ratio-tol and clip fraction 0;
            lag >= 1 batches: max|ratio-1| > --ratio-tol (else the staleness is HIDDEN)
  learn     (only with --learn-target) mean return of the last 20 finished episodes >= target
--control turns one check into an expected failure: the run PASSES when that check catches the planted bug.
Exit codes: 0 passed, 1 a check failed, 3 --impl mine still has TODOs.
"""

from __future__ import annotations

import argparse
import inspect
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rl_common import dump_json, host_info, now, phase_summary, print_phase_table  # noqa: E402


def missing_todos(module) -> list[str]:
    names = ["compute_gae", "ppo_surrogate", "pipelined_iteration"]
    return [n for n in names if "raise NotImplementedError" in inspect.getsource(getattr(module, n))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["ref", "mine"], default="ref")
    p.add_argument("--mode", choices=["sync", "async", "wait"], default="sync")
    p.add_argument("--control", choices=["none", "skip_sync", "recompute_logp"], default="none")
    p.add_argument("--env", choices=["cartpole", "synthetic"], default="cartpole")
    p.add_argument("--obs-dim", type=int, default=4, help="synthetic env only")
    p.add_argument("--step-cost-us", type=float, default=0.0, help="synthetic env only: busy wait per env.step")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--actors", type=int, default=2)
    p.add_argument("--envs-per-actor", type=int, default=4)
    p.add_argument("--frag-len", type=int, default=128, help="steps per env per fragment")
    p.add_argument("--iters", type=int, default=23)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=1, help="torch.set_num_threads in each actor; 0 = leave as is")
    p.add_argument("--learner-gpus", type=float, default=1.0, help="0 = CPU learner (local smoke test)")
    p.add_argument("--ray-num-cpus", type=int, default=None,
                   help="override the CPUs Ray detects (an int: ray.init(num_cpus=10.0) never starts its raylet in 2.60.0)")
    p.add_argument("--ratio-tol", type=float, default=1e-5)
    p.add_argument("--learn-target", type=float, default=None)
    p.add_argument("--trace", default=None, help="write ray.timeline() here at the end (do not time this run)")
    p.add_argument("--json", default=None)
    p.add_argument("--label", default="")
    p.add_argument("--quiet", action="store_true", help="no per-iteration lines")
    return p.parse_args(argv)


def build_record(t0, t1, loads, stats) -> dict:
    frags = stats["frags"]
    slow = max(frags, key=lambda m: m["t_end"] - m["t_start"])
    rec = {
        "rollout": slow["t_end"] - slow["t_start"],
        "rollout.env": slow["env_s"],
        "rollout.infer": slow["infer_s"],
        "transfer": stats["t_start"] - max(m["t_end"] for m in frags),
        **stats["timing"],
        "return": t1 - stats["t_end"],
        "e2e": t1 - t0,
    }
    if loads:
        for k in ("sync.get", "sync.load", "sync.digest"):
            rec[k] = max(l[k] for l in loads)
    busy = statistics.mean([m["t_end"] - m["t_start"] for m in frags]) + (
        statistics.mean([l["sync.get"] + l["sync.load"] for l in loads]) if loads else 0.0)
    rec["idle.actor"] = rec["e2e"] - busy
    rec["idle.learner"] = rec["e2e"] - (stats["t_end"] - stats["t_start"])
    return rec


def run(args) -> int:
    impl_name = "rl_ref" if args.impl == "ref" else "my_rl"
    impl = __import__(impl_name)
    todo = missing_todos(impl)
    if todo:
        print(f"INCOMPLETE (TODOs left in {impl_name}.py: {', '.join(todo)}); exit 3")
        return 3

    # Workers inherit the environment of the raylet, which ray.init starts from this process.
    os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import ray

    from workers import Learner, RolloutWorker

    info = host_info()
    ray.init(num_cpus=args.ray_num_cpus, include_dashboard=False, object_store_memory=2 * 1024**3,
             log_to_driver=True)
    res = ray.cluster_resources()
    cfg = {k: getattr(args, k) for k in ("env", "obs_dim", "step_cost_us", "hidden", "envs_per_actor", "seed",
                                         "torch_threads", "lr", "gamma", "lam", "clip", "vf_coef", "ent_coef",
                                         "max_grad_norm", "epochs", "minibatch", "learner_gpus", "impl")}
    cfg["control"] = args.control
    print(f"=== Day 15 train: mode={args.mode} impl={args.impl} control={args.control} env={args.env} "
          f"actors={args.actors}x{args.envs_per_actor} envs frag={args.frag_len} label={args.label}")
    print(f"host: os.cpu_count={info['os.cpu_count']} affinity={info['affinity_cpus']} "
          f"cgroup cpu.max='{info['cgroup']['cpu.max']}' (quota {info['cgroup']['quota_cpus']} CPUs) "
          f"ray={info['ray']} torch={info['torch']} gpus={info['gpus']}")
    print(f"ray.cluster_resources: CPU={res.get('CPU')} GPU={res.get('GPU')} "
          f"object_store={res.get('object_store_memory', 0) / 2**30:.1f} GiB")
    actor_cls = ray.remote(num_cpus=1, num_gpus=0)(RolloutWorker)
    learner_cls = ray.remote(num_cpus=1, num_gpus=args.learner_gpus)(Learner)
    learner = learner_cls.remote(cfg)
    actors = [actor_cls.remote(i, cfg) for i in range(args.actors)]
    linfo = ray.get(learner.info.remote())
    ainfo = ray.get([a.info.remote() for a in actors])
    print(f"learner: {linfo}")
    for a in ainfo:
        print(f"actor {a['actor']}: OMP_NUM_THREADS={a['OMP_NUM_THREADS']} default torch threads="
              f"{a['default_torch_threads']} -> {a['torch.get_num_threads']} CUDA_VISIBLE_DEVICES="
              f"{a['CUDA_VISIBLE_DEVICES']!r} assigned={a['assigned_resources']}")
    cg0 = info["cgroup"]["cpu.stat"]

    upd_kw = {"control": args.control if args.control == "recompute_logp" else None}
    records, rows, all_eps = [], [], []
    pub = ray.get(learner.publish.remote())
    load_refs = [a.set_weights.remote(pub) for a in actors]
    t_run = now()
    if args.mode == "sync":
        ray.get(load_refs)
        for it in range(args.iters):
            t0 = now()
            load_refs = [a.set_weights.remote(pub) for a in actors] if it > 0 else []
            samples = [a.sample.remote(args.frag_len) for a in actors]
            stats = ray.get(learner.update.remote(*samples, **upd_kw))
            loads = ray.get(load_refs)
            t1 = now()
            pub = stats["pub"]
            records.append(build_record(t0, t1, loads, stats))
            rows.append(stats)
    elif args.mode == "async":
        in_flight = [a.sample.remote(args.frag_len) for a in actors]
        prev_loads = load_refs
        for it in range(args.iters):
            t0 = now()
            in_flight, stats, new_loads = impl.pipelined_iteration(actors, learner, in_flight, args.frag_len, upd_kw)
            t1 = now()
            loads = ray.get(prev_loads)  # the loads queued one iteration earlier have run by now
            prev_loads = new_loads
            records.append(build_record(t0, t1, loads, stats))
            rows.append(stats)
        ray.get(in_flight + prev_loads)  # drain (not timed)
    else:  # wait
        ray.get(load_refs)
        pending = {a.sample.remote(args.frag_len): i for i, a in enumerate(actors)}
        for it in range(args.iters):
            t0 = now()
            ready, _ = ray.wait(list(pending), num_returns=1)
            i = pending.pop(ready[0])
            stats = ray.get(learner.update.remote(ready[0], **upd_kw))
            lref = actors[i].set_weights.remote(stats["pub"])
            pending[actors[i].sample.remote(args.frag_len)] = i
            t1 = now()
            loads = ray.get([lref])
            records.append(build_record(t0, t1, loads, stats))
            rows.append(stats)
        ray.get(list(pending))
    t_total = now() - t_run
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        ray.timeline(filename=args.trace)
        print(f"ray.timeline written to {args.trace} (this run is traced: do not use its timings)")
    skipped = sum(ray.get([a.get_skipped.remote() for a in actors]), [])
    cg1 = __import__("rl_common").cgroup_cpu()["cpu.stat"]
    ray.shutdown()

    # ---------------------------------------------------------------- per-iteration lines
    for it, (st, rec) in enumerate(zip(rows, records)):
        all_eps += st["episodes"]
        st["return20"] = statistics.mean([e[0] for e in all_eps[-20:]]) if all_eps else float("nan")
        if not args.quiet:
            lags = [m["lag"] for m in st["frags"]]
            bvs = [m["policy_version"] for m in st["frags"]]
            print(f"it {it:3d} learner v{st['version_at_update']:<3d} behaviour {bvs} lag {lags} "
                  f"digest {'ok' if all(m['digest_ok'] for m in st['frags']) else 'MISMATCH'} "
                  f"mb0 max|r-1| {st['mb0']['max_ratio_dev']:.2e} mb0 clip {st['mb0']['clip_frac']:.3f} "
                  f"clip(all) {st['clip_frac_mean']:.3f} kl {st['approx_kl_mean']:.2e} "
                  f"ret20 {st['return20']:7.1f} e2e {rec['e2e'] * 1e3:7.1f} ms")
    steps_per_iter = rows[0]["steps"]
    summary = phase_summary(records, args.warmup)
    print_phase_table(summary, f"Phase breakdown, steady state (iterations {args.warmup}..{args.iters - 1}), "
                               f"{steps_per_iter} env steps per iteration, weights {rows[0]['pub']['bytes']} B")
    total_steps = sum(r["steps"] for r in rows)
    print(f"  whole run: {total_steps} env steps in {t_total:.2f} s = {total_steps / t_total:,.0f} steps/s "
          f"(includes warmup)")
    thr = {k: cg1.get(k, 0) - cg0.get(k, 0) for k in ("nr_periods", "nr_throttled", "throttled_usec", "usage_usec")}
    print(f"  cgroup during the run: usage {thr['usage_usec'] / 1e6:.2f} CPU-s, throttled in {thr['nr_throttled']}"
          f"/{thr['nr_periods']} periods, {thr['throttled_usec'] / 1e6:.3f} s throttled")

    # ---------------------------------------------------------------- checks
    lags = [[m["lag"] for m in st["frags"]] for st in rows]
    flat = [x for l in lags for x in l]
    hist = {k: flat.count(k) for k in sorted(set(flat))}
    checks = {}
    if args.mode == "sync":
        checks["versions"] = all(x == 0 for x in flat)
        want = "every lag 0"
    elif args.mode == "async":
        checks["versions"] = all(x == 0 for x in lags[0]) and all(x == 1 for l in lags[1:] for x in l)
        want = "first batch lag 0, then lag 1"
    else:
        want = "reported only"
    print(f"CHECK versions: lag histogram {hist} (expected: {want}) -> "
          f"{'PASS' if checks.get('versions', True) else 'FAIL'}")
    bad = [(i, m["actor"], m["policy_version"]) for i, st in enumerate(rows) for m in st["frags"] if not m["digest_ok"]]
    checks["digest"] = not bad
    print(f"CHECK digest: {sum(len(l) for l in lags) - len(bad)}/{sum(len(l) for l in lags)} fragments match the "
          f"published digest of their version; mismatches (iteration, actor, claimed version): {bad[:6]} -> "
          f"{'PASS' if not bad else 'FAIL'}; skipped syncs reported by actors: {skipped}")
    lag0 = [st["mb0"] for st, l in zip(rows, lags) if max(l) == 0]
    stale = [st["mb0"] for st, l in zip(rows, lags) if min(l) >= 1]
    hidden = [s for s in stale if s["max_ratio_dev"] <= args.ratio_tol]
    ok0 = all(s["max_ratio_dev"] <= args.ratio_tol and s["clip_frac"] == 0 for s in lag0)
    checks["ratio"] = ok0 and not hidden
    dev0 = max((s["max_ratio_dev"] for s in lag0), default=float("nan"))
    devs = [s["max_ratio_dev"] for s in stale]
    print(f"CHECK ratio: lag-0 batches {len(lag0)}, worst first-minibatch max|r-1| {dev0:.2e} (tol "
          f"{args.ratio_tol:g}), clip fractions {sorted({s['clip_frac'] for s in lag0})[:4]}; lag>=1 batches "
          f"{len(stale)}, first-minibatch max|r-1| median {statistics.median(devs) if devs else float('nan'):.2e}, "
          f"HIDDEN (lag>=1 but ratio==1) {len(hidden)} -> {'PASS' if checks['ratio'] else 'FAIL'}")
    if args.learn_target is not None:
        final = rows[-1]["return20"]
        checks["learn"] = final >= args.learn_target
        print(f"CHECK learn: mean return of the last 20 episodes {final:.1f} after {total_steps} env steps "
              f"(target {args.learn_target:g}) -> {'PASS' if checks['learn'] else 'FAIL'}")

    expected_fail = {"skip_sync": "digest", "recompute_logp": "ratio"}.get(args.control)
    if expected_fail:
        detected = not checks[expected_fail]
        others = all(v for k, v in checks.items() if k != expected_fail)
        ok = detected and others
        verdict = f"CONTROL {args.control}: {'DETECTED' if detected else 'NOT DETECTED'} by the {expected_fail} check"
    else:
        ok = all(checks.values())
        verdict = "all checks"
    print(f"TRAIN ({args.mode}, {args.impl}, control={args.control}, {args.label or '-'}): {verdict} -> "
          f"{'PASSED' if ok else 'FAILED'}")
    dump_json(args.json, {"args": vars(args), "host": info, "ray_resources": res, "learner": linfo, "actors": ainfo,
                          "summary": summary, "records": records, "steps_per_iter": steps_per_iter,
                          "lags": lags, "checks": checks, "passed": ok, "cgroup_delta": thr,
                          "total_steps": total_steps, "total_s": t_total, "weights_bytes": rows[0]["pub"]["bytes"],
                          "final_digest": rows[-1]["pub"]["digest"],
                          "iters": [{k: st[k] for k in ("version_at_update", "mb0", "clip_frac_mean",
                                                        "approx_kl_mean", "return20", "steps")} |
                                    {"lags": l} for st, l in zip(rows, lags)]})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run(parse_args()))
