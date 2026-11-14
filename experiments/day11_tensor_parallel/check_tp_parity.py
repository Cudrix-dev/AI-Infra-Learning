"""Day 11 - tensor-parallel MLP vs the single-GPU MLP: unit tests of f/g, then forward and gradient parity.

Launch with torchrun, one process per GPU (TP group = all ranks):

  # unit tests of the shard functions and the two autograd Functions (no baseline needed; CPU works too)
  torchrun --standalone --nproc_per_node=2 experiments/day11_tensor_parallel/check_tp_parity.py --impl mine --test unit
  # parity against a baseline written by baseline_1gpu.py, plus the four negative controls
  torchrun --standalone --nproc_per_node=2 experiments/day11_tensor_parallel/check_tp_parity.py --impl ref \
      --test parity --baseline outputs/day11_tp/tensors/baseline_float32_b1_RID.pt --controls --run-id RID

--impl ref | mine | megatron    (megatron = Megatron Core's ColumnParallelLinear/RowParallelLinear; needs
                                 PYTHONPATH pointing at a Megatron-LM checkout, optional)
--device cuda (NCCL, default) | cpu (gloo; for checking logic without GPUs)

What the parity test compares, all against the SAME tensors the baseline used (loaded from its .pt file):
  y           forward output (replicated: rank 0's copy)
  dx          gradient w.r.t. the input (replicated: rank 0's copy)
  blkK.W1/b1  column-parallel weight/bias gradients: shards all-gathered and concatenated along dim 0
  blkK.W2     row-parallel weight gradient: shards concatenated along dim 1
  blkK.b2     row-parallel bias gradient: replicated (rank 0's copy)
  replicated  y, dx and every blkK.b2 gradient bitwise identical on all ranks
Pass: max|tp - dense| / max|dense| <= 1e-5 (fp32) or 1e-12 (fp64), for every tensor, AND the all-reduce count per
phase is (1 forward, 1 backward) per block. Bitwise equality with the baseline is reported, not required.

Exit codes: 0 all passed, 1 a check failed (or a control was not detected as predicted), 3 my_tp.py not filled in.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tp_ref  # noqa: E402
from tp_common import (OUT, REL_TOL, CollectiveCounter, compare, environment, loss_fn,  # noqa: E402
                       set_exact_math, write_json)
from tp_controls import CONTROLS, EXPECTED, EXPECTED_COUNTS_PER_BLOCK  # noqa: E402

RANK = 0


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def gather(t: torch.Tensor) -> list[torch.Tensor]:
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return out


# ---------------------------------------------------------------------------------------------------------------
# Unit tests: shard functions and the two Functions, with values small enough to check by eye
# ---------------------------------------------------------------------------------------------------------------

def run_unit(impl, group, device, counter: CollectiveCounter) -> tuple[int, int]:
    rank, P = dist.get_rank(), dist.get_world_size()
    results = []  # (name, status, detail) ; status in ok / FAIL / todo

    def record(name, fn):
        try:
            ok, detail = fn()
            results.append((name, "ok" if ok else "FAIL", detail))
        except NotImplementedError as e:
            results.append((name, "todo", f"NotImplementedError({e})"))
        finally:
            counter.phase = None
            counter.take()

    def t_col():
        W = torch.arange(8 * 6, dtype=torch.float32).view(8, 6)
        b = torch.arange(8, dtype=torch.float32)
        Wr, br = impl.shard_column_parallel(W, b, rank, P)
        eW, eb = W.chunk(P, dim=0)[rank], b.chunk(P, dim=0)[rank]
        own = Wr.untyped_storage().data_ptr() != W.untyped_storage().data_ptr()
        ok = Wr.shape == eW.shape and torch.equal(Wr, eW) and torch.equal(br, eb) and own
        return ok, f"W[8,6] -> {list(Wr.shape)} (expect {list(eW.shape)}, rows {rank * 8 // P}..{(rank + 1) * 8 // P - 1}), " \
                   f"b[8] -> {list(br.shape)}, owns storage: {own}"

    def t_row():
        W = torch.arange(6 * 8, dtype=torch.float32).view(6, 8)
        Wr = impl.shard_row_parallel(W, rank, P)
        eW = W.chunk(P, dim=1)[rank]
        ok = Wr.shape == eW.shape and torch.equal(Wr, eW) and Wr.is_contiguous()
        return ok, f"W[6,8] -> {list(Wr.shape)} (expect {list(eW.shape)}, cols {rank * 8 // P}..{(rank + 1) * 8 // P - 1}), " \
                   f"contiguous: {Wr.is_contiguous()}"

    tri = P * (P + 1) / 2  # 1 + 2 + ... + P

    def t_f():
        x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
        counter.phase = "forward"
        y = impl.CopyToTP.apply(x, group)
        fwd = [e["op"] for e in counter.take()]
        counter.phase = "backward"
        y.backward(torch.full((2, 3), 10.0 * (rank + 1), device=device))
        bwd = [e["op"] for e in counter.take()]
        ok = torch.equal(y.detach(), x.detach()) and torch.all(x.grad == 10 * tri).item() and fwd == [] \
            and bwd == ["all_reduce"]
        return ok, (f"fwd y == x: {torch.equal(y.detach(), x.detach())}, collectives fwd {fwd} bwd {bwd}; "
                    f"x.grad = {x.grad.flatten()[0].item():g} (expect {10 * tri:g} = sum over ranks of 10*(r+1))")

    def t_g():
        x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
        counter.phase = "forward"
        y = impl.ReduceFromTP.apply(x, group)
        fwd = [e["op"] for e in counter.take()]
        intact = torch.all(x.detach() == rank + 1).item()
        counter.phase = "backward"
        y.backward(torch.full((2, 3), 10.0 * (rank + 1), device=device))
        bwd = [e["op"] for e in counter.take()]
        ok = torch.all(y.detach() == tri).item() and intact and fwd == ["all_reduce"] and bwd == [] \
            and torch.all(x.grad == 10.0 * (rank + 1)).item()
        return ok, (f"y = {y.flatten()[0].item():g} (expect {tri:g}), input intact: {intact}, collectives fwd {fwd} "
                    f"bwd {bwd}; x.grad = {x.grad.flatten()[0].item():g} (expect own 10*(r+1) = {10.0 * (rank + 1):g})")

    record("shard_column_parallel", t_col)
    record("shard_row_parallel", t_row)
    record("CopyToTP (f)", t_f)
    record("ReduceFromTP (g)", t_g)

    allres = [None] * P
    dist.all_gather_object(allres, results)
    n_fail = n_todo = 0
    say("\nUNIT TESTS (value on rank r is r+1; upstream gradient on rank r is 10*(r+1))")
    for i in range(len(results)):
        for r in range(P):
            name, status, detail = allres[r][i]
            n_fail += status == "FAIL"
            n_todo += status == "todo"
            say(f"  [{status:>4}] rank {r} {name:<22} {detail}")
    return n_fail, n_todo


# ---------------------------------------------------------------------------------------------------------------
# Parity: one forward + backward of the TP model, gathered and compared with the baseline
# ---------------------------------------------------------------------------------------------------------------

def run_variant(name, build, problem, ref, device, rel_tol, counter) -> dict:
    rank, P = dist.get_rank(), dist.get_world_size()
    model = build()
    act_shapes = {}
    blk0 = model.blocks[0]
    hooks = [blk0.fc1.register_forward_hook(
        lambda m, i, o: act_shapes.__setitem__("h_r", list((o[0] if isinstance(o, tuple) else o).shape))),
        blk0.fc2.register_forward_pre_hook(lambda m, i: act_shapes.__setitem__("a_r", list(i[0].shape)))]

    x = problem["x"].to(device).requires_grad_(True)
    counter.take()
    counter.phase = "forward"
    y = model(x)
    loss = loss_fn(y, problem["target"].to(device))
    counter.phase = "backward"
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    counter.phase = None
    events = counter.take()
    for h in hooks:
        h.remove()

    grads = model.grads()
    local_shapes = {"x": list(x.shape), **act_shapes, "y": list(y.shape), "dx": list(x.grad.shape)}
    for k, g in enumerate(grads):
        for n, t in g.items():
            local_shapes[f"blk{k}.d{n}"] = list(t.shape)

    checks, rep_ok, rep_detail = [], True, []
    ys, dxs = gather(y.detach()), gather(x.grad)
    for nm, lst in (("y", ys), ("dx", dxs)):
        same = all(torch.equal(lst[0], t) for t in lst[1:])
        rep_ok &= same
        rep_detail.append(f"{nm}:{'same' if same else 'DIFFER'}")
    checks.append(compare("y", ys[0], ref["y"], rel_tol))
    checks.append(compare("dx", dxs[0], ref["dx"], rel_tol))
    for k, (g, gref) in enumerate(zip(grads, ref["grads"])):
        checks.append(compare(f"blk{k}.W1", torch.cat(gather(g["W1"]), 0), gref["W1"], rel_tol))
        checks.append(compare(f"blk{k}.b1", torch.cat(gather(g["b1"]), 0), gref["b1"], rel_tol))
        checks.append(compare(f"blk{k}.W2", torch.cat(gather(g["W2"]), 1), gref["W2"], rel_tol))
        b2s = gather(g["b2"])
        same = all(torch.equal(b2s[0], t) for t in b2s[1:])
        rep_ok &= same
        rep_detail.append(f"blk{k}.b2:{'same' if same else 'DIFFER'}")
        checks.append(compare(f"blk{k}.b2", b2s[0], gref["b2"], rel_tol))
    checks.append({"name": "replicated", "ok": rep_ok, "detail": ", ".join(rep_detail)})

    n_blocks = len(grads)
    fwd = [e for e in events if e["phase"] == "forward"]
    bwd = [e for e in events if e["phase"] == "backward"]
    exp_f, exp_b = EXPECTED_COUNTS_PER_BLOCK[name]
    counts_ok = (sum(e["op"] == "all_reduce" for e in fwd) == exp_f * n_blocks
                 and sum(e["op"] == "all_reduce" for e in bwd) == exp_b * n_blocks
                 and all(e["op"] == "all_reduce" for e in events))
    all_shapes = [None] * P
    dist.all_gather_object(all_shapes, local_shapes)
    return {"variant": name, "checks": checks, "events": events, "counts_ok": counts_ok,
            "expected_counts": [exp_f * n_blocks, exp_b * n_blocks], "local_shapes_per_rank": all_shapes,
            "loss": loss.item()}


def print_variant(res: dict) -> None:
    say(f"\n=== {res['variant']} ===   loss {res['loss']:.10g}")
    say(f"  {'check':<11} {'gathered shape':<16} {'max|tp-1gpu|':>12} {'rel':>10} {'bitwise':>8} {'ratio':>8}  verdict")
    for c in res["checks"]:
        if c["name"] == "replicated":
            say(f"  {'replicated':<11} {c['detail']:<60} {'PASS' if c['ok'] else 'FAIL'}")
            continue
        say(f"  {c['name']:<11} {str(c['shape']):<16} {c['max_abs']:>12.3e} {c['rel']:>10.2e} "
            f"{'yes' if c['bitwise'] else 'no':>8} {c['ratio']:>8.4f}  {'PASS' if c['ok'] else 'FAIL'}")
    f = [e for e in res["events"] if e["phase"] == "forward"]
    b = [e for e in res["events"] if e["phase"] == "backward"]

    def desc(evs):
        return ", ".join(f"{e['op']}{e['shape']}{' async' if e['async'] else ''}" for e in evs) or "none"

    say(f"  collectives forward ({len(f)}): {desc(f)}")
    say(f"  collectives backward ({len(b)}): {desc(b)}")
    say(f"  all-reduce count fwd/bwd expected {res['expected_counts'][0]}/{res['expected_counts'][1]}: "
        f"{'PASS' if res['counts_ok'] else 'FAIL'}")


def print_shapes(res: dict) -> None:
    say("\nLOCAL SHAPES PER RANK (one block shown; 'd' = gradient)")
    per = res["local_shapes_per_rank"]
    keys = [k for k in per[0] if not k.startswith("blk") or k.startswith("blk0")]
    say(f"  {'tensor':<10} " + " ".join(f"{'rank ' + str(r):<16}" for r in range(len(per))))
    for k in keys:
        say(f"  {k:<10} " + " ".join(f"{str(p[k]):<16}" for p in per))


def main() -> None:
    global RANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ref", "mine", "megatron"], default="ref")
    ap.add_argument("--test", choices=["unit", "parity", "all"], default="parity")
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--controls", action="store_true", help="also run the four negative controls (built from ref)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    if args.test != "unit" and args.baseline is None:
        ap.error("--baseline is required for --test parity/all")
    if args.impl == "megatron" and args.test != "parity":
        ap.error("--impl megatron supports --test parity only (its f/g are tested by the parity run)")

    set_exact_math()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    RANK = dist.get_rank()
    P = dist.get_world_size()
    say(f"TP group size {P}, backend {dist.get_backend()}, impl {args.impl}, test {args.test}")
    if device.type == "cuda":
        names = [None] * P
        dist.all_gather_object(names, f"rank {RANK} -> cuda:{local} {torch.cuda.get_device_name(local)}")
        for n in names:
            say("  " + n)

    group = dist.group.WORLD
    impl = None
    if args.impl == "ref":
        impl = tp_ref
    elif args.impl == "mine":
        impl = importlib.import_module("my_tp")
    record = {"run_id": args.run_id, "impl": args.impl, "test": args.test, "world_size": P,
              "environment": environment()}
    code = 0
    with CollectiveCounter() as counter:
        if args.test in ("unit", "all"):
            n_fail, n_todo = run_unit(impl, group, device, counter)
            record["unit"] = {"failed": n_fail, "not_implemented": n_todo}
            verdict = "FAILED" if n_fail else ("INCOMPLETE (TODOs left)" if n_todo else "PASSED")
            say(f"UNIT TESTS: {verdict}")
            code = 1 if n_fail else (3 if n_todo else 0)
            if code:
                finish(record, args, code, f"unit_{args.impl}")

        if args.test in ("parity", "all"):
            data = torch.load(args.baseline, map_location="cpu", weights_only=True)
            problem, ref = data["problem"], data["result"]
            dtype = problem["x"].dtype
            rel_tol = REL_TOL[dtype]
            n_blocks = len(problem["blocks"])
            tag = f"{data['dtype']}_b{n_blocks}"
            say(f"\nbaseline {args.baseline} ({data['dtype']}, {n_blocks} block(s), shapes {data['shapes']}), "
                f"pass if rel <= {rel_tol:g}")
            record.update({"baseline": str(args.baseline), "dtype": data["dtype"], "blocks": n_blocks,
                           "rel_tol": rel_tol})

            if RANK == 0 and device.type == "cuda":   # is the single-GPU baseline itself reproducible here?
                from baseline_1gpu import run_dense
                from tp_common import Shapes
                rep = run_dense(problem, Shapes(**data["shapes"]), device, dtype)
                same = torch.equal(rep["y"], ref["y"]) and torch.equal(rep["dx"], ref["dx"])
                record["dense_replay_bitwise"] = same
                say(f"dense replay on rank 0 bitwise equal to the saved baseline: {same}")

            builders = {}
            if args.impl == "megatron":
                import megatron_impl
                group = megatron_impl.setup(P)
                builders["megatron"] = lambda: megatron_impl.build(problem, RANK, P, group, device)
            else:
                builders[args.impl] = lambda: tp_ref.build_tp_model(impl, problem, RANK, P, group, device)
            if args.controls:
                for cname, cimpl in CONTROLS.items():
                    builders[cname] = (lambda ci=cimpl: tp_ref.build_tp_model(ci, problem, RANK, P, group, device))

            variants = []
            for vname, build in builders.items():
                try:
                    res = run_variant(vname, build, problem, ref, device, rel_tol, counter)
                except NotImplementedError as e:
                    say(f"\n=== {vname} ===\n  NotImplementedError({e}): fill in my_tp.py first")
                    finish(record, args, 3, f"parity_{args.impl}_{tag}")
                variants.append(res)
                if vname == args.impl:
                    print_shapes(res)
                print_variant(res)

            main_res = variants[0]
            main_ok = all(c["ok"] for c in main_res["checks"]) and main_res["counts_ok"]
            say(f"\nPARITY ({args.impl}, {data['dtype']}, {n_blocks} block(s)): {'PASSED' if main_ok else 'FAILED'}")
            code = 0 if main_ok else 1
            ctrl_summary = {}
            if args.controls:
                say("\nNEGATIVE CONTROLS (must fail exactly the predicted checks)")
                for res in variants[1:]:
                    failed = sorted(c["name"] for c in res["checks"] if not c["ok"])
                    expected = sorted(EXPECTED[res["variant"]][n_blocks])
                    as_pred = failed == expected and res["counts_ok"]
                    ratios = {c["name"]: round(c["ratio"], 4) for c in res["checks"]
                              if c["name"] != "replicated" and not c["ok"]}
                    ctrl_summary[res["variant"]] = {"failed": failed, "expected": expected, "as_predicted": as_pred,
                                                    "ratios_of_failed": ratios}
                    say(f"  {res['variant']:<20} failed {failed}  ->  "
                        f"{'DETECTED as predicted' if as_pred else 'NOT AS PREDICTED (expected ' + str(expected) + ')'}")
                    say(f"  {'':<20} ratios of failed checks: {ratios}")
                all_pred = all(v["as_predicted"] for v in ctrl_summary.values())
                say(f"CONTROLS: {'ALL DETECTED AS PREDICTED' if all_pred else 'SOME NOT AS PREDICTED'}")
                if not all_pred:
                    code = 1
            record.update({"variants": variants, "controls": ctrl_summary, "passed": main_ok})
            finish(record, args, code, f"parity_{args.impl}_{tag}")
    finish(record, args, code, f"unit_{args.impl}")


def finish(record: dict, args, code: int, stem: str) -> None:
    codes = [None] * dist.get_world_size()
    dist.all_gather_object(codes, code)
    code = max(codes)
    record["exit_code"] = code
    if RANK == 0:
        write_json(OUT / f"{stem}_{args.run_id}.json", record)
    dist.destroy_process_group()
    sys.exit(code)


if __name__ == "__main__":
    main()
