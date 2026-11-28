"""Day 15 - Exp 0: small measurements of Ray's execution and object model (GUIDE §1.1-§1.2, Block 4).

    python experiments/day15_ray/ray_probes.py --json probes.json [--oversub]

  resources   what Ray detected (CPU, GPU, object store) vs os.cpu_count, the affinity mask and the cgroup quota
  placement   what an actor sees: OMP_NUM_THREADS, torch threads, CUDA_VISIBLE_DEVICES, assigned resources,
              for num_cpus=1, num_cpus=2, num_gpus=1 and two num_gpus=0.5 actors; and a plain task
  overhead    round trip of a no-op actor call and of a no-op task (median of 300)
  get_vs_wait four actors finishing after 0.4/0.1/0.3/0.2 s, 50 ms of driver work per result:
              submit-and-get per actor  vs  ray.get in submission order  vs  ray.wait first-finished
  put_once    a weight dict of S bytes sent to 4 actors: ray.put once + pass the ref  vs  pass by value
  zero_copy   ray.get of a numpy array vs a torch tensor of the same size: copied or a read-only view?
  pending     an actor asking for more CPUs than Ray has: it never starts (checked for 5 s, then killed)
  oversub     (optional) 4 actors doing CPU inference with 1 vs 8 torch threads each, under the cgroup quota
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rl_common import cgroup_cpu, dump_json, host_info  # noqa: E402

OUT: dict = {}


def section(name):
    print(f"\n--- {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=None)
    p.add_argument("--oversub", action="store_true")
    p.add_argument("--sizes-mib", default="1,64")
    a = p.parse_args()
    os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
    import ray

    info = host_info()
    print(f"driver env: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    ray.init(include_dashboard=False, object_store_memory=4 * 1024**3)

    section("resources")
    res = ray.cluster_resources()
    shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").exists() else None
    OUT["resources"] = {"ray": res, "os.cpu_count": info["os.cpu_count"], "affinity": info["affinity_cpus"],
                        "cgroup": info["cgroup"]["cpu.max"], "quota_cpus": info["cgroup"]["quota_cpus"],
                        "dev_shm_GiB": shm.total / 2**30 if shm else None}
    print(f"os.cpu_count {info['os.cpu_count']}, affinity {info['affinity_cpus']}, cgroup cpu.max "
          f"'{info['cgroup']['cpu.max']}' = {info['cgroup']['quota_cpus']} CPUs")
    print(f"ray.cluster_resources: CPU {res.get('CPU')}, GPU {res.get('GPU')}, memory "
          f"{res.get('memory', 0) / 2**30:.1f} GiB, object_store_memory {res.get('object_store_memory', 0) / 2**30:.1f}"
          f" GiB; /dev/shm {OUT['resources']['dev_shm_GiB']:.1f} GiB")

    section("placement")

    class Probe:
        def look(self):
            import torch

            ctx = ray.get_runtime_context()
            return {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cuda_devices": torch.cuda.device_count(), "assigned": ctx.get_assigned_resources(),
                    "pid": os.getpid()}

        def noop(self):
            return None

        def work(self, d):
            time.sleep(d)
            return d

        def load(self, w):
            t0 = time.perf_counter()
            if isinstance(w, list):  # a ref passed inside a list is not resolved automatically
                w = ray.get(w[0])
            t1 = time.perf_counter()
            return {"get_s": t1 - t0, "nbytes": int(sum(v.nbytes for v in w.values()))}

        def inspect(self, wrapped):
            import torch

            t0 = time.perf_counter()
            x = ray.get(wrapped[0])
            t1 = time.perf_counter()
            if isinstance(x, np.ndarray):
                return {"type": "numpy", "get_s": t1 - t0, "writeable": bool(x.flags.writeable),
                        "owns_data": bool(x.flags.owndata)}
            return {"type": "torch", "get_s": t1 - t0, "is_tensor": isinstance(x, torch.Tensor)}

    variants = {"num_cpus=1": dict(num_cpus=1), "num_cpus=2": dict(num_cpus=2),
                "num_gpus=1": dict(num_cpus=1, num_gpus=1), "num_gpus=0.5 (a)": dict(num_cpus=1, num_gpus=0.5),
                "num_gpus=0.5 (b)": dict(num_cpus=1, num_gpus=0.5)}
    OUT["placement"] = {}
    probes = {}
    for name, opt in variants.items():
        if name == "num_gpus=1":
            h = ray.remote(**opt)(Probe).remote()
            OUT["placement"][name] = ray.get(h.look.remote())
            ray.kill(h)
            time.sleep(0.5)  # release the GPU before the fractional pair asks for it
        else:
            probes[name] = ray.remote(**opt)(Probe).remote()
            OUT["placement"][name] = ray.get(probes[name].look.remote())

    @ray.remote
    def task_look():
        ctx = ray.get_runtime_context()
        return {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "assigned": ctx.get_assigned_resources(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}

    OUT["placement"]["task (defaults)"] = ray.get(task_look.remote())
    for name, d in OUT["placement"].items():
        print(f"{name:18s} {d}")
    for name in ("num_cpus=2", "num_gpus=0.5 (a)", "num_gpus=0.5 (b)"):
        ray.kill(probes.pop(name))

    section("overhead")
    h = probes["num_cpus=1"]
    ray.get(h.noop.remote())

    @ray.remote
    def noop_task():
        return None

    ray.get(noop_task.remote())
    lat = {}
    for label, call in (("actor method", lambda: h.noop.remote()), ("task", lambda: noop_task.remote())):
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            ray.get(call())
            ts.append(time.perf_counter() - t0)
        lat[label] = statistics.median(ts)
        print(f"{label:13s} round trip median {lat[label] * 1e6:8.1f} us (300 sequential calls)")
    OUT["overhead_s"] = lat

    section("get_vs_wait")
    workers = [ray.remote(num_cpus=0)(Probe).remote() for _ in range(4)]  # num_cpus=0: placement only, no quota
    ray.get([w.noop.remote() for w in workers])
    durs, proc = [0.4, 0.1, 0.3, 0.2], 0.05
    gw = {}
    t0 = time.perf_counter()
    for w, d in zip(workers, durs):  # the serializing bug: submit one, wait for it, then the next
        ray.get(w.work.remote(d))
        time.sleep(proc)
    gw["submit_and_get_each"] = {"total": time.perf_counter() - t0}
    t0 = time.perf_counter()
    refs = [w.work.remote(d) for w, d in zip(workers, durs)]
    first = None
    for r in refs:  # all launched; consumed in submission order
        ray.get(r)
        first = first or time.perf_counter() - t0
        time.sleep(proc)
    gw["get_in_order"] = {"first": first, "total": time.perf_counter() - t0}
    t0 = time.perf_counter()
    pending, first, order = [w.work.remote(d) for w, d in zip(workers, durs)], None, []
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        order.append(ray.get(ready[0]))
        first = first or time.perf_counter() - t0
        time.sleep(proc)
    gw["wait_first_finished"] = {"first": first, "total": time.perf_counter() - t0, "order": order}
    for k, v in gw.items():
        print(f"{k:22s} " + ", ".join(f"{kk} {vv:.3f} s" if isinstance(vv, float) else f"{kk} {vv}"
                                       for kk, vv in v.items()))
    OUT["get_vs_wait"] = gw

    section("put_once vs by value (4 actors)")
    OUT["put_once"] = []
    for mib in [int(x) for x in a.sizes_mib.split(",")]:
        n = mib * 2**20 // 4
        w = {f"layer{i}": np.random.default_rng(i).standard_normal(n // 4).astype(np.float32) for i in range(4)}
        for mode in ("put_once", "by_value", "put_once", "by_value"):  # twice each; report the second
            t0 = time.perf_counter()
            if mode == "put_once":
                ref = ray.put(w)
                refs = [x.load.remote([ref]) for x in workers]
            else:
                refs = [x.load.remote(w) for x in workers]  # serialized once per call
            t_submit = time.perf_counter() - t0
            got = ray.get(refs)
            t_all = time.perf_counter() - t0
            row = {"MiB": mib, "mode": mode, "driver_submit_s": t_submit, "all_loaded_s": t_all,
                   "actor_get_s_max": max(g["get_s"] for g in got), "bytes_serialized_by_driver":
                   (1 if mode == "put_once" else len(workers)) * mib * 2**20}
            # keep the second (warm) measurement of each mode
            OUT["put_once"] = [r for r in OUT["put_once"] if not (r["MiB"] == mib and r["mode"] == mode)] + [row]
        for r in [r for r in OUT["put_once"] if r["MiB"] == mib]:
            print(f"{mib:4d} MiB {r['mode']:9s} driver submit {r['driver_submit_s'] * 1e3:8.2f} ms, all 4 loaded "
                  f"{r['all_loaded_s'] * 1e3:8.2f} ms, slowest actor-side get {r['actor_get_s_max'] * 1e3:7.2f} ms, "
                  f"driver serialized {r['bytes_serialized_by_driver'] / 2**20:.0f} MiB")

    section("zero_copy: numpy vs torch, 64 MiB, ray.get inside an actor (median of 5)")
    import torch

    arr = np.random.default_rng(0).standard_normal(16 * 2**20).astype(np.float32)
    ten = torch.from_numpy(arr.copy())
    zc = {}
    for label, obj in (("numpy", arr), ("torch", ten)):
        ref = ray.put(obj)
        rows = [ray.get(h.inspect.remote([ref])) for _ in range(6)][1:]
        zc[label] = {**rows[-1], "get_s": statistics.median(r["get_s"] for r in rows)}
    ref = ray.put(arr)
    d1, d2 = ray.get(ref), ray.get(ref)
    tref = ray.put(ten)
    t1, t2 = ray.get(tref), ray.get(tref)
    zc["driver_two_gets_share_memory"] = {"numpy": bool(np.shares_memory(d1, d2)),
                                          "torch": t1.data_ptr() == t2.data_ptr()}
    zc["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] = os.environ.get("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "unset")
    for k, v in zc.items():
        print(f"{k:30s} {v}")
    OUT["zero_copy"] = zc

    section("pending: an actor asking for more CPUs than the cluster has")
    want = float(res.get("CPU", 0)) + 1
    big = ray.remote(num_cpus=want)(Probe).remote()
    ready, _ = ray.wait([big.noop.remote()], timeout=5.0)
    OUT["pending"] = {"num_cpus_requested": want, "started_within_5s": bool(ready)}
    print(f"requested num_cpus={want}: started within 5 s = {bool(ready)}")
    ray.kill(big)

    if a.oversub:
        section("oversub (optional): 4 actors x CPU inference, torch threads 1 vs 8")

        class Infer:
            def __init__(self, threads):
                import torch as T

                T.set_num_threads(threads)
                self.T = T
                self.net = T.nn.Sequential(T.nn.Linear(256, 512), T.nn.Tanh(), T.nn.Linear(512, 512), T.nn.Tanh(),
                                           T.nn.Linear(512, 2))
                self.x = T.randn(64, 256)

            def run(self, n):
                T = self.T
                with T.no_grad():
                    for _ in range(20):
                        self.net(self.x)
                    t0 = time.perf_counter()
                    for _ in range(n):
                        self.net(self.x)
                    return (time.perf_counter() - t0) / n, T.get_num_threads()

        OUT["oversub"] = {}
        for threads in (1, 8):
            acts = [ray.remote(num_cpus=1, runtime_env={"env_vars": {"OMP_NUM_THREADS": str(threads)}})(Infer)
                    .remote(threads) for _ in range(4)]
            ray.get([x.run.remote(1) for x in acts])
            c0 = cgroup_cpu()["cpu.stat"]
            t0 = time.perf_counter()
            got = ray.get([x.run.remote(400) for x in acts])
            wall = time.perf_counter() - t0
            c1 = cgroup_cpu()["cpu.stat"]
            OUT["oversub"][threads] = {"per_forward_s": [g[0] for g in got], "threads": [g[1] for g in got],
                                       "wall_s": wall, "throttled_s": (c1.get("throttled_usec", 0) -
                                                                       c0.get("throttled_usec", 0)) / 1e6,
                                       "usage_cpu_s": (c1.get("usage_usec", 0) - c0.get("usage_usec", 0)) / 1e6}
            o = OUT["oversub"][threads]
            print(f"threads {threads}: per forward {statistics.median(o['per_forward_s']) * 1e6:8.1f} us (median of "
                  f"4 actors), wall {wall:.2f} s, CPU used {o['usage_cpu_s']:.2f} s, throttled {o['throttled_s']:.2f} s")
            for x in acts:
                ray.kill(x)

    ray.shutdown()
    dump_json(a.json, {"host": info, **OUT})
    print("\nPROBES: done")


if __name__ == "__main__":
    main()
