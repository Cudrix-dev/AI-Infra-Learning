"""Day 15 - the two actor classes. Plain Python classes; train.py turns them into Ray actors with ray.remote(...).

RolloutWorker (CPU, num_gpus=0): E environments stepped in one process, a CPU copy of the policy, one fragment
    per sample() call. Every fragment carries the version and the digest of the weights that produced its actions.
Learner (1 GPU): GAE on its CPU, one H2D copy, PPO epochs x minibatches on the GPU, then it publishes the new
    weights ONCE into the object store (ray.put) and returns {version, digest, [ref]} for the actors to fetch.

Both stamp every phase with rl_common.now() (CLOCK_MONOTONIC), so the driver can line up the processes of one host.
"""

from __future__ import annotations

import importlib
import os
import warnings

import numpy as np

from rl_common import build_policy, digest_arrays, digest_module, env_dims, make_env, now, weights_numpy


def _proc_info() -> dict:
    import ray
    import torch

    ctx = ray.get_runtime_context()
    return {
        "pid": os.getpid(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "torch.get_num_threads": torch.get_num_threads(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assigned_resources": ctx.get_assigned_resources(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


class RolloutWorker:
    def __init__(self, idx: int, cfg: dict):
        import torch

        self.default_threads = torch.get_num_threads()  # what Ray's environment gave this process
        if cfg["torch_threads"] > 0:
            torch.set_num_threads(cfg["torch_threads"])
        self.idx, self.cfg = idx, cfg
        self.n_envs = cfg["envs_per_actor"]
        base = cfg["seed"] * 100_003 + idx * 1009
        self.env_seeds = [base + e for e in range(self.n_envs)]
        self.envs = [make_env(cfg["env"], s, cfg["obs_dim"], cfg["step_cost_us"]) for s in self.env_seeds]
        obs_dim, n_act = env_dims(cfg["env"], cfg["obs_dim"])
        self.net = build_policy(obs_dim, n_act, cfg["hidden"], cfg["seed"])
        self.gen = torch.Generator().manual_seed(base)
        self.version, self.digest = -1, None
        self.obs = None
        self.ep_ret = np.zeros(self.n_envs)
        self.ep_len = np.zeros(self.n_envs, np.int64)
        self.skipped = []

    def info(self) -> dict:
        return {"actor": self.idx, "default_torch_threads": self.default_threads, **_proc_info()}

    def get_skipped(self) -> list:
        return self.skipped

    def set_weights(self, pub: dict) -> dict:
        """Fetch the published weights (one object, shared by all actors), load them, digest what we really hold."""
        import ray
        import torch

        t0 = now()
        arrays = ray.get(pub["ref"][0])  # numpy arrays: zero-copy, read-only views of the object store
        t1 = now()
        skip = self.cfg.get("control") == "skip_sync" and self.idx == 1 and pub["version"] == 2
        if skip:  # NEGATIVE CONTROL: claim the new version, keep the old weights
            self.skipped.append(pub["version"])
        else:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # torch warns that the read-only numpy views are not writable
                for k, p in self.net.state_dict().items():
                    p.copy_(torch.from_numpy(arrays[k]))
        t2 = now()
        self.version = pub["version"]
        self.digest = digest_module(self.net)
        t3 = now()
        return {"actor": self.idx, "version": pub["version"], "skipped": skip, "t0": t0, "t3": t3,
                "sync.get": t1 - t0, "sync.load": t2 - t1, "sync.digest": t3 - t2}

    def sample(self, T: int) -> dict:
        """One fragment of T steps x E envs. Episodes continue across calls; the last row is a fragment cut."""
        import torch

        E = self.n_envs
        t_start = now()
        env_s = infer_s = 0.0
        if self.obs is None:
            te = now()
            self.obs = np.stack([env.reset(seed=s)[0] for env, s in zip(self.envs, self.env_seeds)]).astype(np.float32)
            env_s += now() - te
        D = self.obs.shape[1]
        obs = np.zeros((T, E, D), np.float32)
        act = np.zeros((T, E), np.int64)
        logp, val, nval, rew = (np.zeros((T, E), np.float32) for _ in range(4))
        term, trunc = np.zeros((T, E), bool), np.zeros((T, E), bool)
        pending = np.zeros(E, bool)  # envs whose previous row still needs next_values from this step's forward
        episodes = []
        net = self.net
        for t in range(T):
            ti = now()
            with torch.no_grad():
                logits, v = net(torch.from_numpy(self.obs))
                a = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.gen).squeeze(1)
                lp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
            a_np, v_np = a.numpy(), v.numpy()
            infer_s += now() - ti
            obs[t], act[t], logp[t], val[t] = self.obs, a_np, lp.numpy(), v_np
            if t > 0:
                nval[t - 1, pending] = v_np[pending]
            te = now()
            next_obs = np.empty_like(self.obs)
            finals = {}
            for e, env in enumerate(self.envs):
                o, r, te_, tr_, _ = env.step(int(a_np[e]))
                rew[t, e], term[t, e], trunc[t, e] = r, te_, tr_
                self.ep_ret[e] += r
                self.ep_len[e] += 1
                if te_ or tr_:
                    episodes.append((float(self.ep_ret[e]), int(self.ep_len[e]), bool(tr_)))
                    self.ep_ret[e], self.ep_len[e] = 0.0, 0
                    if tr_:
                        finals[e] = o
                    o, _ = env.reset()
                next_obs[e] = o
            env_s += now() - te
            if finals:  # truncation: bootstrap from V(final observation), which the policy never sees otherwise
                ti = now()
                idx = sorted(finals)
                with torch.no_grad():
                    _, vf = net(torch.from_numpy(np.stack([finals[e] for e in idx]).astype(np.float32)))
                nval[t, idx] = vf.numpy()
                infer_s += now() - ti
            self.obs = next_obs.astype(np.float32, copy=False)
            pending = ~(term[t] | trunc[t])
        ti = now()
        with torch.no_grad():  # the fragment cut: V(s_T) for every env still running
            _, v_last = net(torch.from_numpy(self.obs))
        nval[T - 1, pending] = v_last.numpy()[pending]
        infer_s += now() - ti
        t_end = now()
        return {
            "obs": obs, "actions": act, "logp": logp, "values": val, "next_values": nval, "rewards": rew,
            "terminated": term, "truncated": trunc,
            "meta": {"actor": self.idx, "policy_version": self.version, "digest": self.digest, "steps": T * E,
                     "t_start": t_start, "t_end": t_end, "env_s": env_s, "infer_s": infer_s, "episodes": episodes},
        }


class Learner:
    def __init__(self, cfg: dict):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.cfg = cfg
        self.impl = importlib.import_module("rl_ref" if cfg["impl"] == "ref" else "my_rl")
        self.device = torch.device("cuda" if cfg["learner_gpus"] > 0 else "cpu")
        obs_dim, n_act = env_dims(cfg["env"], cfg["obs_dim"])
        self.net = build_policy(obs_dim, n_act, cfg["hidden"], cfg["seed"]).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"], eps=1e-5)
        self.gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 17)
        self.version = 0
        self.published = {}  # version -> digest of the weights published under that version
        self._refs = {}  # keep the last few published objects alive

    def info(self) -> dict:
        import torch

        return {"device": str(self.device),
                "device_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
                "cuda_device_count": torch.cuda.device_count(), **_proc_info()}

    def _sync(self):
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def publish(self) -> dict:
        """Weights of the current version -> host numpy -> ONE object in the object store. Returns the handle."""
        import ray

        self._sync()
        t0 = now()
        arrays = weights_numpy(self.net)  # .to("cpu") waits for the GPU
        t1 = now()
        digest = digest_arrays(arrays)
        t2 = now()
        ref = ray.put(arrays)
        t3 = now()
        self.published[self.version] = digest
        self._refs[self.version] = ref
        for old in [v for v in self._refs if v < self.version - 3]:
            del self._refs[old]
        nbytes = sum(a.nbytes for a in arrays.values())
        return {"version": self.version, "digest": digest, "ref": [ref], "bytes": nbytes,
                "timing": {"sync.d2h": t1 - t0, "sync.put": t3 - t2, "learner.digest": t2 - t1}}

    def update(self, *frags, control: str | None = None) -> dict:
        """One PPO update on the given fragments (Ray resolved the refs before this method started)."""
        import torch

        cfg, dev = self.cfg, self.device
        t_start = now()
        version_at_update = self.version
        metas = []
        for f in frags:
            m = dict(f["meta"])
            m["lag"] = version_at_update - m["policy_version"]
            m["digest_ok"] = self.published.get(m["policy_version"]) == m["digest"]
            metas.append(m)
        # ---- GAE per fragment, on the learner's CPU (the values came from the behaviour critic)
        advs, rets = [], []
        for f in frags:
            a, r = self.impl.compute_gae(f["rewards"], f["values"], f["next_values"], f["terminated"],
                                         f["truncated"], cfg["gamma"], cfg["lam"])
            advs.append(a.reshape(-1))
            rets.append(r.reshape(-1))
        t_gae = now()
        # ---- one host -> GPU copy of the whole batch
        obs_np = np.concatenate([f["obs"].reshape(-1, f["obs"].shape[-1]) for f in frags])
        cols = {"obs": obs_np,
                "actions": np.concatenate([f["actions"].reshape(-1) for f in frags]),
                "old_logp": np.concatenate([f["logp"].reshape(-1) for f in frags]),
                "adv": np.concatenate(advs), "ret": np.concatenate(rets)}
        b = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in cols.items()}
        self._sync()
        t_h2d = now()
        n = b["obs"].shape[0]
        if control == "recompute_logp":  # NEGATIVE CONTROL: "behaviour" logp recomputed with the learner's weights
            with torch.no_grad():
                logits, _ = self.net(b["obs"])
                b["old_logp"] = torch.log_softmax(logits, -1).gather(1, b["actions"][:, None]).squeeze(1)
        # ---- PPO epochs x minibatches
        mb = min(cfg["minibatch"], n)
        per_mb = []
        for epoch in range(cfg["epochs"]):
            perm = torch.randperm(n, generator=self.gen).to(dev)
            for s in range(0, n - mb + 1, mb):
                i = perm[s:s + mb]
                logits, v = self.net(b["obs"][i])
                logp_all = torch.log_softmax(logits, -1)
                new_logp = logp_all.gather(1, b["actions"][i][:, None]).squeeze(1)
                adv = b["adv"][i]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss, st = self.impl.ppo_surrogate(new_logp, b["old_logp"][i], adv, cfg["clip"])
                v_loss = 0.5 * ((v - b["ret"][i]) ** 2).mean()
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()
                per_mb.append(st)
        self._sync()
        t_upd = now()
        keys = list(per_mb[0])  # one device -> host copy for all minibatch stats
        vals = torch.stack([torch.stack([s[k].float() for k in keys]) for s in per_mb]).cpu().tolist()
        per_mb = [dict(zip(keys, v)) for v in vals]
        first = per_mb[0]
        self.version += 1
        pub = self.publish()
        t_end = now()
        episodes = [e for m in metas for e in m.pop("episodes")]
        return {
            "version_at_update": version_at_update, "frags": metas, "episodes": episodes, "pub": pub,
            "steps": int(n), "optimizer_steps": len(per_mb),
            "mb0": first,
            "clip_frac_mean": float(np.mean([s["clip_frac"] for s in per_mb])),
            "approx_kl_mean": float(np.mean([s["approx_kl"] for s in per_mb])),
            "t_start": t_start, "t_end": t_end,
            "timing": {"learn.gae": t_gae - t_start, "learn.h2d": t_h2d - t_gae, "learn.update": t_upd - t_h2d,
                       **pub["timing"]},
        }
