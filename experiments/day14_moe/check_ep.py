"""Day 14 - expert parallelism on four ranks vs the one-GPU baseline: dispatch tests, parity, negative controls.

On the four-GPU host (one process per GPU, NCCL):
    torchrun --standalone --nproc_per_node=4 experiments/day14_moe/check_ep.py --tests dispatch --impl ref
    torchrun --standalone --nproc_per_node=4 experiments/day14_moe/check_ep.py --impl ref \
        --baseline outputs/day14_moe/tensors/baseline_small_<rid>.pt
    torchrun ... check_ep.py --impl controls --baseline ...          # every control must fail as predicted
    torchrun ... check_ep.py --impl ref --ep 2 --baseline ...         # optional: EP=2 x EDP=2
On a laptop (4 CPU processes, gloo; same checks, labelled CPU in the output):
    python experiments/day14_moe/check_ep.py --cpu --nproc 4 --impl ref --baseline <file made with --cpu>

--tests dispatch  TODO 3-4 only: the count exchange and AllToAllV (forward and backward), with zero-size splits.
--tests parity    the full model: one forward + backward per routing case on this rank's quarter of the global
                  batch, then finish_grads, then every check below on every rank, merged on rank 0.

Checks (moe_controls.py lists the names): routing and send/recv counts equal the baseline's routing; logits,
losses (task CE, aux and z per layer) and the input gradient match; the gradients of the dense params, the routers
and every expert match the baseline's full-batch gradients; replicated grads are bitwise equal across ranks; the
number of all_to_all / all_reduce calls per phase equals the derived count. Pass rule: moe_common.REL_TOL.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

import moe_common as C
import moe_controls
import moe_ref
from moe_model import RouteOpts, build_model

DEFAULT_CASES = ("learned", "balanced", "ragged", "all_to_one")


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="ref", help="ref | mine | controls | one control name")
    ap.add_argument("--tests", default="parity", choices=["parity", "dispatch"])
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--ep", type=int, default=4, help="4 = core (EDP=1); 2 = optional EP=2 x EDP=2")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--dtype", default="float32", choices=list(C.DTYPES))
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--aux-scope", default="global", choices=["global", "local"])
    ap.add_argument("--no-validate", action="store_true", help="skip the split-metadata check before dispatch")
    ap.add_argument("--cpu", action="store_true", help="gloo on CPU instead of NCCL on GPUs")
    ap.add_argument("--nproc", type=int, default=0, help="spawn this many local processes (no torchrun)")
    ap.add_argument("--json", type=Path)
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------ dispatch tests
def dispatch_tests(impl, g: C.Groups, dev) -> dict:
    """TODO 3 and TODO 4 against known traffic. Rows carry labels (source, destination, k) so misplaced rows show."""
    out = {}
    e_loc = 4 // g.ep
    cases = {"ragged": C.forced_experts("ragged", torch.arange(512), 4, 512),
             "all_to_one": C.forced_experts("all_to_one", torch.arange(512), 4, 512)}
    for name, idx in cases.items():
        per_src = 512 // C.N_SOURCES
        mine = idx[g.rank * per_src:(g.rank + 1) * per_src]
        send = torch.bincount(mine, minlength=4).view(g.ep, e_loc).to(dev)
        recv = impl.exchange_counts(send, g.ep_group)
        # expected: what every EP peer sends to my local experts
        exp_recv = torch.stack([torch.bincount(idx[p * per_src:(p + 1) * per_src], minlength=4)
                                .view(g.ep, e_loc)[g.ep_rank] for p in g.ep_ranks]).to(dev)
        out[f"exchange_counts.{name}"] = {"ok": bool(torch.equal(recv, exp_recv)), "rel": 0.0}
        in_splits, out_splits = send.sum(1).tolist(), exp_recv.sum(1).tolist()
        # payload: row = [source rank, destination EP rank, k-th row to that destination]
        rows = [torch.tensor([[g.rank, j, k] for k in range(n)], dtype=torch.float32).view(-1, 3)
                for j, n in enumerate(in_splits)]
        x = torch.cat(rows).to(dev).requires_grad_(True)
        y = impl.AllToAllV.apply(x, out_splits, in_splits, g.ep_group)
        want = torch.cat([torch.tensor([[p, g.ep_rank, k] for k in range(n)], dtype=torch.float32).view(-1, 3)
                          for p, n in zip(g.ep_ranks, out_splits)]).to(dev)
        out[f"alltoallv.forward.{name}"] = {"ok": bool(torch.equal(y.detach(), want)), "rel": 0.0}
        # backward: the receiver multiplies each row by a code of (receiver rank); the sender must get exactly the
        # codes of the ranks it sent to, on the rows it sent.
        code = torch.full_like(y, float(g.rank + 1))
        (y * code).sum().backward()
        dst_ranks = torch.cat([torch.full((n, 3), float(g.ep_ranks[j] + 1)) for j, n in enumerate(in_splits)]).to(dev)
        out[f"alltoallv.backward.{name}"] = {"ok": bool(x.grad is not None and torch.equal(x.grad, dst_ranks)),
                                             "rel": 0.0}
        out[f"zero_splits.{name}"] = {"ok": True, "rel": 0.0, "note": f"in={in_splits} out={out_splits}"}
    return out


# ------------------------------------------------------------------------------------------------ parity
def expected_collectives(cfg: C.MoEConfig, g: C.Groups, aux_scope: str) -> dict:
    """Per rank per step: each MoE layer does 1 count all_to_all + 2 payload all_to_alls forward (dispatch,
    combine) and their 2 reverses backward; with the global aux loss, one all_reduce of E counts forward."""
    L = cfg.n_layers
    exp = {"forward.all_to_all_single": 3 * L, "backward.all_to_all_single": 2 * L}
    if aux_scope == "global":
        exp["forward.all_reduce"] = L
    return exp


def run_case(impl_name: str, case: str, args, cfg, g: C.Groups, dev, dtype, base_all) -> dict:
    impl, pattern = moe_controls.load_impl(impl_name)
    pattern = pattern or case
    base = base_all[pattern]
    tol = C.REL_TOL[dtype]
    opts = RouteOpts(pattern=pattern, n_tokens_global=cfg.tokens, aux_scope=args.aux_scope,
                     validate=not args.no_validate)
    model = build_model(cfg, g, impl, opts, dev, dtype)
    tokens, labels = C.make_batch(cfg)
    b_r = cfg.batch // g.world
    sl = slice(g.rank * b_r, (g.rank + 1) * b_r)
    tok_sl = slice(g.rank * b_r * cfg.seq_len, (g.rank + 1) * b_r * cfg.seq_len)
    x, y = tokens[sl].to(dev), labels[sl].to(dev)
    res: dict = {}
    with C.CollectiveCounter() as counter:
        try:
            counter.phase = "forward"
            logits = model(x, token_offset=g.rank * b_r * cfg.seq_len)
            ce = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))     # mean over MY tokens
            counter.phase = "backward"
            ce.backward()
            counter.phase = "finish"
            impl.finish_grads(model, g)
            counter.phase = None
        except moe_ref.SplitMetadataError as e:
            return {"raised:SplitMetadataError": {"ok": False, "msg": str(e)[:300]}}
        counts = counter.take()

    def add(name, got, ref):
        res[name] = C.compare(name, got, ref, tol)

    def mean_over_ranks(v: torch.Tensor) -> torch.Tensor:
        v = v.detach().clone().to(torch.float64)
        dist.all_reduce(v)
        return v / g.world

    for k, layer in enumerate(model.moe_layers()):
        st = layer.stats
        ref_idx = base["idx"][k][tok_sl]
        same = torch.equal(st["idx"].cpu(), ref_idx)
        res[f"routing.L{k}"] = {"ok": same, "rel": 0.0 if same else 1.0,
                                "mismatched_tokens": int((st["idx"].cpu() != ref_idx).sum())}
        m = C.send_matrix(base["idx"][k], cfg.n_experts, C.N_SOURCES, g.ep)
        want_in = m[g.rank].tolist()
        want_out = [int(m[p, g.ep_rank]) for p in g.ep_ranks]
        ok = st["send"] == want_in and st["recv"] == want_out
        res[f"counts.L{k}"] = {"ok": ok, "rel": 0.0 if ok else 1.0, "send": st["send"], "recv": st["recv"]}
        add(f"loss.aux.L{k}", mean_over_ranks(st["aux"]), base["aux"][k].double())
        add(f"loss.z.L{k}", mean_over_ranks(st["z"]), base["z"][k].double())
        r = layer.router
        add(f"grad.router.L{k}", torch.cat([r.weight.grad.reshape(-1), r.bias.grad]),
            torch.cat([base["grads"][f"blocks.{k}.moe.router.weight"].reshape(-1),
                       base["grads"][f"blocks.{k}.moe.router.bias"]]))
        for e in layer.owned:
            ex = layer.experts[str(e)]
            names = [f"blocks.{k}.moe.experts.{e}.{leaf}" for leaf in ("w1", "b1", "w2", "b2")]
            add(f"grad.expert.L{k}.e{e}",
                torch.cat([getattr(ex, leaf).grad.reshape(-1) for leaf in ("w1", "b1", "w2", "b2")]),
                torch.cat([base["grads"][n].reshape(-1) for n in names]))
    add("logits", logits, base["logits"][sl])
    add("loss.ce", mean_over_ranks(ce), base["ce"].double())
    # rank r back-propagated ITS mean; the global mean is 1/DP of it on rank r's tokens (they affect no other loss)
    add("dx", model.h0.grad / g.world, base["dx"][sl])
    dense = [(n, p) for n, p in model.named_parameters() if ".moe." not in n]
    add("grad.dense", torch.cat([p.grad.reshape(-1) for _, p in dense]),
        torch.cat([base["grads"][n].reshape(-1) for n, _ in dense]))
    worst = max(((n, C.compare(n, p.grad, base["grads"][n], tol)) for n, p in dense), key=lambda t: t[1]["rel"])
    res["grad.dense"]["worst_param"] = worst[0]

    # replicated params must be bitwise identical on every rank; expert replicas bitwise identical across EDP
    rep = [p.grad for n, p in model.named_parameters() if not getattr(p, "expert_parallel", False)]
    exp_ = [p.grad for n, p in model.named_parameters() if getattr(p, "expert_parallel", False)]
    digests = [None] * g.world
    dist.all_gather_object(digests, (C.tensor_digest(rep), C.tensor_digest(exp_), g.ep_rank))
    rep_ok = len({d[0] for d in digests}) == 1
    edp_ok = all(len({d[1] for d in digests if d[2] == j}) == 1 for j in range(g.ep))
    res["replicated"] = {"ok": rep_ok and edp_ok, "rel": 0.0, "dense_same": rep_ok, "expert_replicas_same": edp_ok}

    want = expected_collectives(cfg, g, args.aux_scope)
    got = {k: v for k, v in counts.items() if k.startswith(("forward.", "backward."))}
    res["collectives"] = {"ok": got == want, "rel": 0.0, "got": got, "want": want, "finish": {
        k: v for k, v in counts.items() if k.startswith("finish.")}}
    return res


def merge(per_rank: list[dict]) -> dict:
    merged: dict = {}
    for r, res in enumerate(per_rank):
        for name, v in res.items():
            m = merged.setdefault(name, {"ok": True, "rel": 0.0, "ratio": float("nan"), "fail_ranks": [],
                                         "bitwise": True, "detail": {}})
            m["ok"] = m["ok"] and bool(v.get("ok"))
            if not v.get("ok"):
                m["fail_ranks"].append(r)
            rel = v.get("rel", 0.0)
            if rel is not None and rel >= m["rel"]:
                m["rel"] = rel
                m["ratio"] = v.get("ratio", float("nan"))
            m["bitwise"] = m["bitwise"] and bool(v.get("bitwise", True))
            extra = {k: v[k] for k in ("send", "recv", "worst_param", "got", "want", "finish", "msg",
                                       "mismatched_tokens", "note", "dense_same", "expert_replicas_same") if k in v}
            if extra:
                m["detail"][r] = extra
    return merged


def print_case(title: str, merged: dict) -> None:
    print(f"\n=== {title}")
    print(f"{'check':28s} {'verdict':7s} {'max rel err':>12s} {'ratio':>8s} {'bitwise':>8s}  ranks failing / detail")
    for name in sorted(merged, key=_order):
        m = merged[name]
        ratio = "" if m["ratio"] != m["ratio"] else f"{m['ratio']:.4f}"
        det = ""
        if name.startswith("counts") and m["detail"]:
            det = " ".join(f"r{r}:send{d['send']}/recv{d['recv']}" for r, d in sorted(m["detail"].items()))
        elif name == "collectives":
            d = m["detail"].get(0, {})
            det = f"rank0 got {d.get('got')} finish {d.get('finish')}"
        elif name == "grad.dense":
            det = f"worst param {m['detail'].get(0, {}).get('worst_param')}"
        elif name.startswith("raised") or name.startswith("routing"):
            d = m["detail"].get(0, {})
            det = d.get("msg", "") or (f"mismatched tokens r0: {d.get('mismatched_tokens')}" if d else "")
        fr = f"fail on {m['fail_ranks']} " if m["fail_ranks"] else ""
        print(f"{name:28s} {'PASS' if m['ok'] else 'FAIL':7s} {m['rel']:12.3e} {ratio:>8s} "
              f"{'yes' if m['bitwise'] else 'no':>8s}  {fr}{det}")


def _order(name: str):
    pref = ["routing", "counts", "logits", "loss.ce", "loss.aux", "loss.z", "dx", "grad.dense", "grad.router",
            "grad.expert", "replicated", "collectives", "raised", "exchange", "alltoallv", "zero"]
    for i, p in enumerate(pref):
        if name.startswith(p):
            return (i, name)
    return (len(pref), name)


def worker(argv) -> None:
    args = parse(argv)
    dev, backend = C.init_distributed(args.cpu)
    dist.barrier()
    C.set_exact_math()
    g = C.build_groups(args.ep)
    rank0 = g.rank == 0
    cfg = C.get_preset(args.preset)
    dtype = C.DTYPES[args.dtype]
    hw = "CPU (gloo)" if backend == "gloo" else f"GPU ({torch.cuda.get_device_name(dev)}, NCCL)"
    if rank0:
        lay = C.layout_ranks(g.world, args.ep)
        print(f"[layout] world {g.world}  EP={g.ep}  EDP={g.edp}  dense DP={g.dp}  TP=PP=CP=ETP=1  on {hw}")
        print(f"[layout] dense DP groups {lay['dp']}  EP groups {lay['ep']}  EDP groups {lay['edp']}")
        print(f"[layout] experts per rank {4 // g.ep}; rank r owns experts "
              f"{[list(range(r % g.ep * (4 // g.ep), (r % g.ep + 1) * (4 // g.ep))) for r in range(g.world)]}")
    exit_code = 0
    summary = {"impl": args.impl, "tests": args.tests, "ep": g.ep, "edp": g.edp, "world": g.world,
               "backend": backend, "hardware": hw, "preset": C.config_dict(cfg), "dtype": args.dtype,
               "aux_scope": args.aux_scope, "env": C.environment(), "cases": {}}
    if args.tests == "dispatch":
        impl, _ = moe_controls.load_impl("ref" if args.impl == "controls" else args.impl)
        res = dispatch_tests(impl, g, dev)
        per_rank = [None] * g.world
        dist.all_gather_object(per_rank, res)
        if rank0:
            merged = merge(per_rank)
            print_case(f"dispatch tests, impl {args.impl}, EP={g.ep}", merged)
            ok = all(m["ok"] for m in merged.values())
            summary["cases"]["dispatch"] = {"merged": merged, "passed": ok}
            print(f"\nDISPATCH ({args.impl}, EP={g.ep}, {hw}): {'PASSED' if ok else 'FAILED'}")
            exit_code = 0 if ok else 1
    else:
        assert args.baseline is not None and args.baseline.is_file(), "--baseline FILE (from baseline_1gpu.py)"
        base_all = torch.load(args.baseline, map_location="cpu", weights_only=False)
        if rank0:
            print(f"[baseline] {args.baseline} ({base_all['_meta']['hardware']}, sha {base_all['_meta']['sha'][:12]})")
            assert base_all["_meta"]["preset"] == C.config_dict(cfg), "baseline made with a different preset"
        runs = ([(c, None) for c in moe_controls.CONTROLS] if args.impl == "controls"
                else [(args.impl, c) for c in args.cases.split(",")])
        all_ok = True
        for impl_name, case in runs:
            t0 = time.perf_counter()
            res = run_case(impl_name, case, args, cfg, g, dev, dtype, base_all)
            per_rank = [None] * g.world
            dist.all_gather_object(per_rank, res)
            dist.barrier()
            if not rank0:
                continue
            merged = merge(per_rank)
            pattern = case or moe_controls.CONTROLS[impl_name][1]
            title = f"{impl_name} | routing {pattern} | EP={g.ep} EDP={g.edp} | {hw} | {time.perf_counter() - t0:.1f} s"
            print_case(title, merged)
            failed = sorted(n for n, m in merged.items() if not m["ok"])
            if impl_name in moe_controls.CONTROLS:
                expected = sorted(moe_controls.EXPECTED[impl_name])
                ok = failed == expected
                verdict = ("DETECTED as predicted" if ok and expected else
                           "blind spot as predicted (nothing fails)" if ok else "NOT as predicted")
                if not ok:
                    print(f"  expected to fail: {expected}\n  actually failed:  {failed}")
            else:
                ok = not failed
                verdict = "PASSED" if ok else f"FAILED ({', '.join(failed)})"
            print(f"CASE {impl_name}/{pattern}: {verdict}")
            summary["cases"][f"{impl_name}/{pattern}"] = {"merged": merged, "failed": failed, "passed": ok}
            all_ok = all_ok and ok
        if rank0:
            label = "CONTROLS" if args.impl == "controls" else "PARITY"
            print(f"\n{label} ({args.impl}, EP={g.ep}, EDP={g.edp}, {args.dtype}, {hw}): "
                  f"{'PASSED' if all_ok else 'FAILED'}")
            exit_code = 0 if all_ok else 1
    if rank0 and args.json:
        C.write_json(args.json, summary)
    t = torch.tensor([exit_code], device=dev)
    dist.broadcast(t, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if int(t.item()):
        sys.exit(int(t.item()))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = parse(argv)
    if args.impl == "mine":
        import my_moe
        left = moe_controls.todo_left(my_moe)
        needed = ({"exchange_counts", "AllToAllV.forward", "AllToAllV.backward"} if args.tests == "dispatch"
                  else set(left))
        missing = [n for n in left if n in needed]
        if missing:
            C.incomplete(NotImplementedError(", ".join(missing)))
    if args.nproc:
        C.spawn_local(worker, argv, args.nproc)
    else:
        worker(argv)


if __name__ == "__main__":
    main()
