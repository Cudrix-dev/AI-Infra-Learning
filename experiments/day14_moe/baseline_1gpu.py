"""Day 14 - the one-GPU parity oracle: EP=1, dense DP=1, EDP=1, all four experts local, the FULL global batch.

    CUDA_VISIBLE_DEVICES=0 python experiments/day14_moe/baseline_1gpu.py --run-id <rid>
    python experiments/day14_moe/baseline_1gpu.py --cpu --run-id try1          # for the local CPU checks

For every routing pattern it runs one forward + backward of the global mean loss and saves what check_ep.py
compares: logits, task CE, aux and z per layer, the expert index of every token per layer, the input gradient
(d loss / d h0) and every parameter gradient. The file is small (~25 MB for the small preset) and is not archived.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import moe_common as C
import moe_controls
from moe_model import RouteOpts, build_model


def run_pattern(cfg, impl, pattern, dev, dtype) -> dict:
    opts = RouteOpts(pattern=pattern, n_tokens_global=cfg.tokens, aux_scope="global", validate=False)
    model = build_model(cfg, None, impl, opts, dev, dtype)
    tokens, labels = C.make_batch(cfg)
    logits = model(tokens.to(dev), token_offset=0)
    ce = F.cross_entropy(logits.reshape(-1, cfg.vocab), labels.to(dev).reshape(-1))   # mean over ALL tokens
    ce.backward()
    layers = model.moe_layers()
    margins = []
    for layer in layers:                         # how close was the closest routing decision? (determinism, P12)
        top2 = layer.stats["probs"].topk(2, dim=-1).values
        margins.append(float((top2[:, 0] - top2[:, 1]).min()))
    return {
        "logits": logits.detach().cpu(), "ce": ce.detach().cpu(), "dx": model.h0.grad.detach().cpu(),
        "aux": [l.stats["aux"].cpu() for l in layers], "z": [l.stats["z"].cpu() for l in layers],
        "idx": [l.stats["idx"].cpu() for l in layers], "counts": [l.stats["counts"].cpu() for l in layers],
        "grads": {n: p.grad.detach().cpu() for n, p in model.named_parameters()},
        "min_top1_margin": margins,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--dtype", default="float32", choices=list(C.DTYPES))
    ap.add_argument("--impl", default="ref", help="ref | mine (TODO 1-2 are used by the EP=1 path too)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.impl == "mine":
        import my_moe
        left = [n for n in moe_controls.todo_left(my_moe) if n in ("top1_route", "permute", "unpermute")]
        if left:
            C.incomplete(NotImplementedError(", ".join(left)))
    impl, _ = moe_controls.load_impl(args.impl)
    C.set_exact_math()
    dev = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    hw = "CPU" if dev.type == "cpu" else torch.cuda.get_device_name(dev)
    cfg = C.get_preset(args.preset)
    dtype = C.DTYPES[args.dtype]
    print(f"[baseline] EP=1 dense DP=1 EDP=1, all {cfg.n_experts} experts on one {hw}; preset {args.preset} "
          f"({cfg.n_layers} layers, d {cfg.d_model}, ffn {cfg.ffn}, {cfg.batch} x {cfg.seq_len} = {cfg.tokens} tokens), "
          f"{args.dtype}, TF32 off, impl {args.impl}")
    out = {}
    for pattern in C.PATTERNS:
        t0 = time.perf_counter()
        r = run_pattern(cfg, impl, pattern, dev, dtype)
        out[pattern] = r
        loads = " | ".join(" ".join(f"{int(c):4d}" for c in cnt) for cnt in r["counts"])
        print(f"  {pattern:11s} CE {r['ce'].item():.7f}  aux/L {[round(a.item(), 5) for a in r['aux']]}  "
              f"z/L {[round(z.item(), 4) for z in r['z']]}  tokens/expert per layer [{loads}]  "
              f"min top-1 margin {min(r['min_top1_margin']):.2e}  ({time.perf_counter() - t0:.1f} s)")
    # identical baselines from two runs <=> identical sha (repeatability of the oracle itself)
    sha = C.sha256_bytes({p: {k: v for k, v in r.items() if k != "min_top1_margin"} for p, r in out.items()})
    out["_meta"] = {"preset": C.config_dict(cfg), "dtype": args.dtype, "hardware": hw, "impl": args.impl,
                    "env": C.environment(), "sha": sha}
    path = args.out or C.TENSORS / f"baseline_{args.preset}_{args.run_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    C.write_json(path.with_suffix(".json"), {"meta": out["_meta"], "patterns": {
        p: {"ce": r["ce"].item(), "aux": [a.item() for a in r["aux"]], "z": [z.item() for z in r["z"]],
            "counts": [c.tolist() for c in r["counts"]], "min_top1_margin": r["min_top1_margin"]}
        for p, r in out.items() if p != "_meta"}})
    print(f"BASELINE written: {path} (sha256 {sha[:16]})")


if __name__ == "__main__":
    sys.exit(main())
