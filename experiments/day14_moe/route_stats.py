"""Day 14 Experiment 1 - routing and capacity on ONE GPU (the baseline role), same global token count throughout.

    CUDA_VISIBLE_DEVICES=0 python experiments/day14_moe/route_stats.py --json <file>
    python experiments/day14_moe/route_stats.py --cpu                                # also fine: it is counting
    CUDA_VISIBLE_DEVICES=0 python experiments/day14_moe/route_stats.py --aux-train 200   # optional, a hypothesis

Day 01's medium shapes: 16 x 256 = 4096 tokens, d 512, 6 MoE layers of E = 4 experts. For each routing:
  * learned       the router at initialization, on real hidden states (one forward of the medium model);
  * learned+bias  the same with a router bias of +BIAS on expert 0 in every layer (a controlled-logit skew);
  * balanced / hot / ragged / all_to_one: forced assignments (moe_common.forced_experts), same 4096 tokens.
it prints tokens per expert, max/mean, CV, load entropy, idle experts, and (learned only, where it means
something) the mean router entropy. Then capacity: for capacity factor 1.0 / 1.25 / 2.0,
  C = ceil(cf * T * k / E) over the GLOBAL batch (one GPU) vs per SOURCE rank (4 x 1024 tokens, as Megatron applies
  it before dispatch), the tokens dropped in each case, and the rows drop-and-pad would compute (E x C).
Finally the per-rank receive rows of an EP=4 dispatch and the largest receive buffer, plus EP=8 [simulated].

--aux-train N (optional): train the small model for N steps from a skewed router (bias +2 on expert 0) with
alpha = 0, 1e-2, 1e-1 and report layer 0's max/mean every 20 steps. Whether the load recovers or collapses
within N steps is a HYPOTHESIS to test, not a required result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import moe_common as C
import moe_ref
from moe_model import RouteOpts, build_model

BIAS = 1.5
CFS = (1.0, 1.25, 2.0)


def drops(idx: torch.Tensor, E: int, cap: int, policy="position", weight=None) -> torch.Tensor:
    return ~moe_ref.capacity_keep_mask(idx, E, cap, policy, weight)


def learned_routing(cfg, dev, bias: float):
    model = build_model(cfg, None, moe_ref, RouteOpts(pattern="learned", n_tokens_global=cfg.tokens,
                                                      aux_scope="off"), dev)
    if bias:
        with torch.no_grad():
            for layer in model.moe_layers():
                layer.router.bias[0] += bias
    tokens, _ = C.make_batch(cfg)
    with torch.no_grad():
        model(tokens.to(dev), token_offset=0)
    return [(l.stats["idx"].cpu(), l.stats["probs"].cpu(), l.stats["counts"].cpu()) for l in model.moe_layers()]


def report_pattern(name, idx, probs, E, T, d, rows: list):
    m = C.load_metrics(torch.bincount(idx, minlength=E))
    ent = f"{C.router_entropy(probs):.3f}" if probs is not None else "  -  "
    print(f"  {name:13s} tokens/expert {m['counts']}  max/mean {m['max_over_mean']:.3f}  CV {m['cv']:.3f}  "
          f"load entropy {m['load_entropy']:.3f}  idle {m['idle_experts']}  router entropy {ent}")
    cap_rows = []
    per_src = T // C.N_SOURCES
    for cf in CFS:
        cg = C.capacity(T, E, cf)
        dg = drops(idx, E, cg)
        cs = C.capacity(per_src, E, cf)
        ds = torch.cat([drops(idx[s * per_src:(s + 1) * per_src], E, cs) for s in range(C.N_SOURCES)])
        same = bool(torch.equal(dg, ds))
        cap_rows.append({"cf": cf, "C_global": cg, "dropped_global": int(dg.sum()), "C_per_source": cs,
                         "dropped_per_source": int(ds.sum()), "same_tokens_dropped": same,
                         "pad_rows_global": E * cg, "kept_global": T - int(dg.sum())})
        print(f"      cf {cf:4.2f}: global C={cg:5d} drops {int(dg.sum()):5d} ({dg.float().mean():6.1%}) | per source "
              f"C={cs:4d} drops {int(ds.sum()):5d} ({ds.float().mean():6.1%}) | same tokens: {'yes' if same else 'no'}"
              f" | drop-and-pad rows {E * cg} for {T - int(dg.sum())} kept")
    send = C.send_matrix(idx, E, C.N_SOURCES, 4)
    recv = send.sum(0).tolist()
    mib = max(recv) * d * 4 / 2 ** 20
    print(f"      EP=4: send matrix {send.tolist()} -> receive rows per rank {recv}; largest receive buffer "
          f"{max(recv)} rows = {mib:.1f} MiB FP32 (balanced share {T // 4} rows = {T // 4 * d * 4 / 2 ** 20:.1f} MiB)")
    rows.append({"pattern": name, **m, "router_entropy": None if probs is None else C.router_entropy(probs),
                 "capacity": cap_rows, "ep4_send": send.tolist(), "ep4_recv_rows": recv, "ep4_max_recv_mib": mib})


def aux_train(cfg, dev, steps: int) -> dict:
    out = {}
    for alpha in (0.0, 1e-2, 1e-1):
        c = C.get_preset("small", aux_coeff=alpha)
        model = build_model(c, None, moe_ref, RouteOpts(pattern="learned", n_tokens_global=c.tokens,
                                                        aux_scope="global"), dev)
        with torch.no_grad():
            for layer in model.moe_layers():
                layer.router.bias[0] += 2.0
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist = []
        for step in range(steps + 1):
            tokens, labels = C.make_batch(c, step)
            logits = model(tokens.to(dev), token_offset=0)
            loss = F.cross_entropy(logits.reshape(-1, c.vocab), labels.to(dev).reshape(-1))
            st = model.moe_layers()[0].stats
            if step % 20 == 0:
                m = C.load_metrics(st["counts"])
                hist.append({"step": step, "ce": loss.item(), "counts": m["counts"],
                             "max_over_mean": m["max_over_mean"], "aux": st["aux"].item()})
                print(f"    alpha {alpha:5.2f} step {step:4d}  CE {loss.item():.4f}  layer-0 tokens/expert "
                      f"{m['counts']}  max/mean {m['max_over_mean']:.3f}  L_aux/alpha {st['aux'].item():.3f}")
            if step == steps:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        out[str(alpha)] = hist
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--aux-train", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    C.set_exact_math()
    dev = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    cfg = C.get_preset(args.preset)
    E, T, d = cfg.n_experts, cfg.tokens, cfg.d_model
    hw = "CPU" if dev.type == "cpu" else torch.cuda.get_device_name(dev)
    print(f"[route_stats] one {hw}; {cfg.batch} x {cfg.seq_len} = {T} tokens, E = {E}, top-1, d {d}, "
          f"{cfg.n_layers} MoE layers; capacity factors {CFS}")
    rows: list = []
    summary = {"hardware": hw, "preset": C.config_dict(cfg), "env": C.environment(), "patterns": rows}
    for bias, label in ((0.0, "learned"), (BIAS, "learned+bias")):
        layers = learned_routing(cfg, dev, bias)
        print(f"\n[{label}] router at init{'' if not bias else f', bias +{BIAS} on expert 0'}; per layer max/mean "
              f"{[round(C.load_metrics(c)['max_over_mean'], 3) for _, _, c in layers]}; layer 0 in detail:")
        idx, probs, _ = layers[0]
        report_pattern(label, idx, probs, E, T, d, rows)
        if bias == 0.0:   # probs vs position drop policy on the same learned routing (Megatron default: probs)
            w = probs.gather(1, idx.unsqueeze(1)).squeeze(1)
            cap = C.capacity(T, E, 1.0)
            a, b = drops(idx, E, cap), drops(idx, E, cap, "probs", w)
            print(f"      cf 1.00 drop policy: position drops {int(a.sum())}, probs drops {int(b.sum())}, "
                  f"same tokens: {int((a & b).sum())} of them")
            summary["policy_overlap_cf1"] = {"position": int(a.sum()), "probs": int(b.sum()), "both": int((a & b).sum())}
    print("\n[forced] same 4096 tokens, assignment fixed by pattern (router entropy not meaningful):")
    for p in ("balanced", "hot", "ragged", "all_to_one"):
        report_pattern(p, C.forced_experts(p, torch.arange(T), E, T), None, E, T, d, rows)
    print("\n[simulated] EP=8 with E=8 experts, 8 sources x 512 tokens (no GPU run; arithmetic only):")
    t = torch.arange(T)
    sim = {}
    for p in ("balanced", "hot", "all_to_one"):
        if p == "balanced":
            idx8 = t % 8
        elif p == "all_to_one":
            idx8 = torch.full_like(t, 7)
        else:
            h = ((t * 2654435761) % (2 ** 32)).double() / 2 ** 32
            idx8 = torch.where(h < C.HOT_FRACTION, torch.zeros_like(t), 1 + t % 7)
        recv = C.send_matrix(idx8, 8, 8, 8).sum(0).tolist()
        sim[p] = recv
        print(f"  {p:11s} receive rows per rank {recv}; largest {max(recv)} = {max(recv) / (T / 8):.1f} x balanced")
    summary["ep8_simulated_recv_rows"] = sim
    if args.aux_train:
        print(f"\n[aux-train, optional] small preset, router bias +2 on expert 0, Adam 1e-3, {args.aux_train} steps "
              f"(a hypothesis: recovery or collapse is not guaranteed in a short run)")
        summary["aux_train"] = aux_train(cfg, dev, args.aux_train)
    if args.json:
        C.write_json(args.json, summary)
    print("\nROUTE_STATS: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
