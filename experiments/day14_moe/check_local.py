"""Day 14 - single-process checks of the GUIDE's hand-worked examples (no process group; CPU is enough).

    python experiments/day14_moe/check_local.py                 # reference
    python experiments/day14_moe/check_local.py --impl mine     # your TODO 1-2 (unfinished parts: INCOMPLETE, exit 3)

  gating    P1: top-1 weight from the full softmax, its gradient d w / d logits, and the renormalized variant
            whose router gradient from the task loss is EXACTLY zero; the tie case (argmax picks the first).
  permute   P3: stable order, counts with an idle expert, unpermute(x[order]) == x bitwise, a local MoE (permute ->
            experts -> unpermute) == the mask-based oracle in forward and backward; the wrong inverse is caught on
            non-uniform routing and NOT caught on presorted routing.
  capacity  P2: T = 16, E = 4, counts [7, 5, 3, 1]: C, kept / dropped, a dropped token's MoE output is exactly 0
            and its block output is the residual stream; drop-and-pad rows.
  aux       P8: L_aux / alpha for balanced, collapsed and the P8 example, its gradient (only through P), and the
            autoscaler gives the same router gradient as adding the aux loss to the task loss.
  splits    [simulated] split-metadata controls on 4 logical ranks: every wrong list is flagged, the right one not.
  ep_sim    [simulated] the whole dispatch / combine for EP = 4 (E = 4) and EP = 8 (E = 8) by slicing tensors in
            one process: equal to EP = 1; per-rank receive rows and the largest receive buffer.
  top2      reading only: renormalized top-2 (two weights that still carry gradient) with colliding index_add.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

import moe_common as C
import moe_controls
import moe_ref
from moe_model import Expert, MoELayer, RouteOpts

RESULTS: list[tuple[str, str, str]] = []     # (test, verdict, detail)


def record(test: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((test, "PASS" if ok else "FAIL", detail))
    print(f"[{test}] {'PASS' if ok else 'FAIL'}  {detail}")


def guarded(test_fn):
    def run(*a, **k):
        try:
            test_fn(*a, **k)
        except NotImplementedError as e:
            RESULTS.append((test_fn.__name__, "INCOMPLETE", str(e)))
            print(f"[{test_fn.__name__}] INCOMPLETE  {e}")
    run.__name__ = test_fn.__name__
    return run


def make_experts(n, d, ffn, seed=7):
    torch.manual_seed(seed)
    ex = nn.ModuleList([Expert(d, ffn) for _ in range(n)])
    with torch.no_grad():
        for e, m in enumerate(ex):
            for name, p in m.named_parameters():
                p.copy_(C.init_value(f"x.experts.{e}.{name}", tuple(p.shape), seed, torch.float64))
    return ex.double()


# ------------------------------------------------------------------------------------------------ gating (P1)
@guarded
def gating(impl):
    logits = torch.tensor([[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                          dtype=torch.float64, requires_grad=True)
    idx, w, probs = impl.top1_route(logits)
    print("  probs:\n" + "\n".join("   " + " ".join(f"{v:.4f}" for v in row) for row in probs.tolist()))
    print(f"  idx {idx.tolist()}  weight {[round(v, 4) for v in w.tolist()]}")
    record("gating.idx_weight", idx.tolist() == [0, 2, 0] and torch.allclose(w, probs[[0, 1, 2], [0, 2, 0]]),
           "token 2 is a 4-way tie: argmax returns the FIRST maximum (expert 0)")
    w[0].backward()                     # d w_0 / d logits_0 = p_e (delta_ej - p_j)
    p = probs[0].detach()
    analytic = p[0] * (torch.eye(4, dtype=p.dtype)[0] - p)
    record("gating.grad", torch.allclose(logits.grad[0], analytic, atol=1e-12) and logits.grad[1:].abs().max() == 0,
           f"d w0/d logits0 = {[round(v, 4) for v in logits.grad[0].tolist()]} (sums to 0)")
    # renormalized top-1 vs full-softmax weight: router gradient of the TASK loss only (no aux / z here)
    torch.manual_seed(0)
    d, E = 8, 4
    x = torch.randn(32, d, dtype=torch.float64)
    ex = make_experts(E, d, 16)
    target = torch.randn(32, d, dtype=torch.float64)
    norms = {}
    for name, route in (("full_softmax", impl.top1_route), ("renormalized", moe_ref.renormalized_top1)):
        wr = (torch.randn(E, d, dtype=torch.float64, generator=torch.Generator().manual_seed(3)) * 0.5).requires_grad_()
        logits = x @ wr.T
        i, wt, _ = route(logits)
        y = moe_ref.naive_moe(x, i, wt, ex, E)
        ((y - target) ** 2).mean().backward()
        norms[name] = wr.grad.norm().item()
    record("gating.renorm_kills_router_grad", norms["renormalized"] == 0.0 and norms["full_softmax"] > 0,
           f"|d task / d W_router|: full softmax {norms['full_softmax']:.3e}, renormalized {norms['renormalized']}")


# ------------------------------------------------------------------------------------------------ permute (P3)
@guarded
def permute(impl):
    idx = torch.tensor([2, 0, 3, 0, 2, 0, 3, 3])          # P3: expert 1 receives nothing
    order, counts = impl.permute(idx, 4)
    print(f"  idx {idx.tolist()} -> order {order.tolist()}  counts {counts.tolist()}")
    stable = all(order[k] < order[k + 1] for k in range(len(order) - 1) if idx[order[k]] == idx[order[k + 1]])
    record("permute.order_counts", order.tolist() == [1, 3, 5, 0, 4, 2, 6, 7] and counts.tolist() == [3, 0, 2, 3]
           and stable, "stable: equal experts keep token order; idle expert 1 appears as 0")
    x = torch.randn(8, 5, dtype=torch.float64)
    back = impl.unpermute(x[order], order, 8)
    record("permute.inverse", torch.equal(back, x), "unpermute(x[order], order) == x bitwise")
    wrong = moe_controls.unpermute_wrong_inverse(x[order], order, 8)
    record("permute.wrong_inverse_caught", not torch.equal(wrong, x),
           f"x[order][order] != x for order {order.tolist()} (not an involution)")
    pre = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    o2, _ = impl.permute(pre, 4)
    record("permute.wrong_inverse_blind_spot", torch.equal(moe_controls.unpermute_wrong_inverse(x[o2], o2, 8), x),
           "presorted routing: order is the identity, so the wrong inverse passes (why controls need ragged data)")
    # a local MoE built from the pieces vs the mask oracle, forward AND backward
    torch.manual_seed(1)
    T, d, E = 64, 8, 4
    ex = make_experts(E, d, 16)
    x = torch.randn(T, d, dtype=torch.float64, requires_grad=True)
    wr = torch.randn(E, d, dtype=torch.float64, requires_grad=True)
    i, wt, _ = impl.top1_route(x @ wr.T, C.forced_experts("hot", torch.arange(T), E, T))
    o, cnt = impl.permute(i, E)
    y = impl.unpermute(moe_ref.run_experts(x[o], cnt.tolist(), list(ex)), o, T) * wt.unsqueeze(1)
    g = torch.randn_like(y)
    grads = torch.autograd.grad((y * g).sum(), [x, wr] + list(ex.parameters()))
    i2, w2, _ = moe_ref.top1_route(x @ wr.T, C.forced_experts("hot", torch.arange(T), E, T))
    y2 = moe_ref.naive_moe(x, i2, w2, ex, E)
    grads2 = torch.autograd.grad((y2 * g).sum(), [x, wr] + list(ex.parameters()))
    gerr = max((a - b).abs().max().item() for a, b in zip(grads, grads2))
    record("permute.moe_vs_oracle", torch.allclose(y, y2, atol=1e-12) and gerr < 1e-12,
           f"forward max err {(y - y2).abs().max().item():.1e}, backward max err {gerr:.1e} (float64)")


# ------------------------------------------------------------------------------------------------ capacity (P2)
P2_IDX = torch.tensor([0, 0, 1, 0, 2, 0, 1, 0, 3, 1, 0, 2, 1, 0, 1, 2])   # counts [7, 5, 3, 1]


@guarded
def capacity(impl):
    T, E = 16, 4
    cfg = C.MoEConfig(d_model=8, ffn=16, n_experts=E, aux_coeff=0.0, z_coeff=0.0)
    for cf, exp_drop in ((1.0, [7, 10, 13, 14]), (1.25, [10, 13]), (2.0, [])):
        cap = C.capacity(T, E, cf)
        layer = MoELayer(cfg, None, impl, RouteOpts(forced_idx=P2_IDX, n_tokens_global=T, aux_scope="off",
                                                    capacity=cap)).double()
        with torch.no_grad():
            for n, p in layer.named_parameters():
                p.copy_(C.init_value(n, tuple(p.shape), 5, torch.float64))
        h = torch.randn(T, 8, dtype=torch.float64)
        y = layer(h, torch.arange(T))
        keep = layer.stats["keep"]
        dropped = (~keep).nonzero().flatten().tolist()
        kept = torch.bincount(P2_IDX[keep], minlength=E).tolist()
        zero_rows = bool((y[~keep] == 0).all()) and bool((y[keep].abs().sum(1) > 0).all())
        block_out = h + y                                  # Block.forward: h + moe(ln2(h))
        passthrough = torch.equal(block_out[~keep], h[~keep])
        pad_rows = E * cap
        record(f"capacity.cf{cf}", cap == math.ceil(cf * T / E) and dropped == exp_drop and zero_rows and passthrough,
               f"C={cap} kept/expert {kept} dropped tokens {dropped} ({len(dropped) / T:.1%}); dropped rows: MoE "
               f"output 0, block output = residual; drop-and-pad computes {pad_rows} rows for {T - len(dropped)} kept")


# ------------------------------------------------------------------------------------------------ aux (P8)
def aux():
    E, T = 4, 16
    uniform = torch.full((T, E), 0.25, dtype=torch.float64)
    bal = moe_ref.aux_loss(uniform, torch.tensor([4, 4, 4, 4]), T, E).item()
    one_hot = torch.zeros(T, E, dtype=torch.float64)
    one_hot[:, 0] = 1
    col = moe_ref.aux_loss(one_hot, torch.tensor([16, 0, 0, 0]), T, E).item()
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64).repeat(T, 1).requires_grad_()
    ex = moe_ref.aux_loss(probs, torch.tensor([7, 5, 3, 1]), T, E)
    ex.backward()
    dP = (probs.grad.sum(0)).tolist()                     # d L / d P_i = E f_i (sum over tokens of d/d probs * T)
    record("aux.values", abs(bal - 1) < 1e-12 and abs(col - 4) < 1e-12 and abs(ex.item() - 1.25) < 1e-12,
           f"balanced {bal:.3f}, collapsed {col:.3f} (= E), P8 example {ex.item():.4f}")
    record("aux.grad_only_through_P", torch.allclose(torch.tensor(dP, dtype=torch.float64),
                                                     torch.tensor([1.75, 1.25, 0.75, 0.25], dtype=torch.float64)),
           f"d L_aux / d P_i = E f_i = {[round(v, 3) for v in dP]}; counts are integers: no gradient")
    # autoscaler: router grad of task loss with aux attached == grad of (task + coeff * aux)
    torch.manual_seed(2)
    d = 8
    x = torch.randn(T, d, dtype=torch.float64)
    ex4 = make_experts(E, d, 16)
    grads = []
    for mode in ("explicit", "autoscaler"):
        wr = torch.randn(E, d, dtype=torch.float64, generator=torch.Generator().manual_seed(4)).requires_grad_()
        logits = x @ wr.T
        i, w, probs = moe_ref.top1_route(logits)
        a = moe_ref.aux_loss(probs, torch.bincount(i, minlength=E), T, E) * 0.01 + 1e-3 * moe_ref.z_loss(logits)
        if mode == "autoscaler":
            w = moe_ref.AuxLossAutoScaler.apply(w, a)
        y = moe_ref.naive_moe(x, i, w, ex4, E)
        loss = (y ** 2).mean() + (a if mode == "explicit" else 0)
        loss.backward()
        grads.append(wr.grad.clone())
    record("aux.autoscaler", torch.allclose(grads[0], grads[1], atol=1e-14),
           f"max |explicit - autoscaler| = {(grads[0] - grads[1]).abs().max().item():.1e}; forward output unchanged")


# ------------------------------------------------------------------------------------------------ splits [simulated]
def splits():
    T = 512
    for ep in (4, 2):
        send = C.send_matrix(C.forced_experts("ragged", torch.arange(T), 4, T), 4, C.N_SOURCES, ep)
        if ep == 2:   # EP=2 layout with EDP=2: one EP group = ranks [0, 1]; keep only that group's traffic
            send = torch.stack([send[0], send[1]])
        res = moe_controls.local_split_controls(send)
        print(f"  [simulated] EP={ep} send matrix (rows = source, cols = destination): {send.tolist()}")
        for name, problems in res.items():
            ok = (not problems) if name == "reference" else bool(problems)
            msg = "no problem" if not problems else problems[0]
            record(f"splits.ep{ep}.{name}", ok, f"[simulated] {'accepted' if name == 'reference' else 'flagged'}: {msg}")


# ------------------------------------------------------------------------------------------------ ep_sim [simulated]
def simulate_ep(x, idx, weight, experts, E, ep, n_sources):
    """Dispatch + experts + combine for `ep` logical ranks, by slicing (what all_to_all_single would move)."""
    per_src = x.shape[0] // n_sources
    e_loc = E // ep
    sent, orders = [], []
    for s in range(n_sources):
        xi, ii = x[s * per_src:(s + 1) * per_src], idx[s * per_src:(s + 1) * per_src]
        o, cnt = moe_ref.permute(ii, E)
        orders.append((o, cnt))
        sent.append(torch.split(xi[o], cnt.view(ep, e_loc).sum(1).tolist()))
    recv_rows, outputs = [], [[None] * ep for _ in range(n_sources)]
    for r in range(ep):                                   # receiver r gets chunk r from every source
        chunks = [sent[s][r] for s in range(n_sources)]
        recv_rows.append(sum(c.shape[0] for c in chunks))
        for s in range(n_sources):                        # per source, experts in order (E_loc may be > 1)
            cnt = orders[s][1].view(ep, e_loc)[r].tolist()
            parts = torch.split(chunks[s], cnt)
            outputs[s][r] = torch.cat([experts[r * e_loc + le](p) for le, p in enumerate(parts)])
    y = torch.cat([moe_ref.unpermute(torch.cat(outputs[s]), orders[s][0], per_src) for s in range(n_sources)])
    return y * weight.unsqueeze(1), recv_rows


def ep_sim():
    torch.manual_seed(3)
    d = 16
    for ep, E, n_src in ((4, 4, 4), (8, 8, 8)):
        T = 64 * n_src
        ex = make_experts(E, d, 32)
        x = torch.randn(T, d, dtype=torch.float64)
        for pattern in ("balanced", "hot", "all_to_one"):
            idx = C.forced_experts(pattern, torch.arange(T), E, T) if n_src == C.N_SOURCES else \
                _pattern_any(pattern, T, E)
            w = torch.rand(T, dtype=torch.float64)
            y_ep, rows = simulate_ep(x, idx, w, ex, E, ep, n_src)
            y_1 = moe_ref.naive_moe(x, idx, w, ex, E)
            record(f"ep_sim.EP{ep}.{pattern}", torch.allclose(y_ep, y_1, atol=1e-12),
                   f"[simulated] {n_src} sources x {T // n_src} tokens -> receive rows per rank {rows}, largest "
                   f"buffer {max(rows)} rows = {max(rows) / (T / ep):.1f} x the balanced share")


def _pattern_any(pattern, T, E):
    t = torch.arange(T)
    if pattern == "balanced":
        return t % E
    if pattern == "all_to_one":
        return torch.full_like(t, E - 1)
    h = ((t * 2654435761) % (2 ** 32)).double() / 2 ** 32
    return torch.where(h < C.HOT_FRACTION, torch.zeros_like(t), 1 + t % (E - 1))


# ------------------------------------------------------------------------------------------------ top2 (reading)
def top2():
    torch.manual_seed(4)
    T, d, E = 32, 8, 4
    ex = make_experts(E, d, 16)
    x = torch.randn(T, d, dtype=torch.float64)
    wr = torch.randn(E, d, dtype=torch.float64, requires_grad=True)
    logits = x @ wr.T
    top_v, top_i = logits.topk(2, dim=-1)
    w = torch.softmax(top_v, dim=-1)                       # renormalized over the TWO picks: still differentiable
    flat_i, flat_w = top_i.reshape(-1), w.reshape(-1)
    src = torch.arange(T).repeat_interleave(2)             # each token appears twice
    o, cnt = moe_ref.permute(flat_i, E)
    y_sorted = moe_ref.run_experts(x[src[o]], cnt.tolist(), list(ex)) * flat_w[o].unsqueeze(1)
    y = x.new_zeros(T, d).index_add(0, src[o], y_sorted)  # two rows add into one token: collisions (atomics on GPU)
    oracle = sum(moe_ref.naive_moe(x, top_i[:, k], w[:, k], ex, E) for k in range(2))
    y.sum().backward()
    record("top2.reading", torch.allclose(y, oracle, atol=1e-12) and wr.grad.norm() > 0,
           f"renormalized top-2 router grad norm {wr.grad.norm().item():.3e} (> 0, unlike renormalized top-1)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="ref", choices=["ref", "mine"])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    impl, _ = moe_controls.load_impl(args.impl)
    torch.set_float32_matmul_precision("highest")
    print(f"check_local: impl {args.impl}, torch {torch.__version__}, CPU, float64 unless noted")
    gating(impl)
    permute(impl)
    capacity(impl)
    aux()
    splits()
    ep_sim()
    top2()
    n = {v: sum(1 for _, x, _ in RESULTS if x == v) for v in ("PASS", "FAIL", "INCOMPLETE")}
    verdict = "PASSED" if not n["FAIL"] and not n["INCOMPLETE"] else ("FAILED" if n["FAIL"] else "INCOMPLETE")
    print(f"\nCHECK_LOCAL ({args.impl}): {n['PASS']} passed, {n['FAIL']} failed, {n['INCOMPLETE']} incomplete: {verdict}")
    if args.json:
        C.write_json(args.json, {"impl": args.impl, "results": RESULTS, "verdict": verdict})
    return 0 if verdict == "PASSED" else (3 if verdict == "INCOMPLETE" else 1)


if __name__ == "__main__":
    sys.exit(main())
