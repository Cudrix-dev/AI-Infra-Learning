"""Day 09 probe, one GPU, one optimizer step per configuration, same init and same batch everywhere.

--part dtypes  fp32 vs bf16 autocast vs fp16 autocast (+ bf16_pure: the whole model in bf16, no fp32 master):
               * dtype of the parameters, selected activations, the loss, p.grad, AdamW's exp_avg / exp_avg_sq,
                 and the GradScaler scale
               * which aten ops ran in which dtype in forward (a TorchDispatchMode sees every op AFTER autocast)
               * autocast's weight casts: the low-precision copies of fp32 parameters made in forward, and how many
                 bytes of them autograd keeps for backward
               * what autograd saved for backward, by kind (activation / weight cast) and dtype
               * gradient error vs fp32, and the fraction of parameter elements the first AdamW step did NOT change
--part ckpt    checkpointing off vs on at the same precision (fp32 and bf16; m = 32 like the matrix):
               * what is saved (census by module), what is recomputed (block forward calls and the forward ops that
                 now run inside backward), memory held after forward and the backward peak
               * correctness: loss and every gradient compared with the non-checkpointed path (bitwise if possible)

  python experiments/day09_mixed_precision/probe_amp.py --part all
  python experiments/day09_mixed_precision/probe_amp.py --part ckpt --impl mine   # checks YOUR model_forward

The census uses saved_tensors_hooks (autograd calls pack() for every tensor it saves; unique storages are summed, as
in Day 01's memory probe). Only the innermost hook pair is active: inside a checkpointed block, checkpoint's own hook
takes the saves (and drops them), so the census sees what checkpoint itself keeps (the block inputs), not the inside.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "day01_single_gpu"))
sys.path.insert(0, str(HERE))
from train import PRESETS, Config, build, env_info, get_batch, setup_reproducibility  # noqa: E402

from amp_setup import AmpNet, load_impl, make_scaler, sdpa_context  # noqa: E402

OUT = HERE.parents[1] / "outputs" / "day09_amp"
MiB = 2**20
SHORT = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16", torch.int64: "i64", torch.bool: "bool",
         torch.uint8: "u8", torch.int32: "i32", torch.float64: "fp64"}
# modules whose OUTPUT dtype goes into the table (module name, how to describe it)
SELECTED = [("model.token_emb", "token embedding out"), ("model.blocks.0.norm1", "block 0 LayerNorm out"),
            ("model.blocks.0.self_attn", "block 0 attention out (after out_proj)"),
            ("model.blocks.0.linear1", "block 0 FFN linear1 out"), ("model.blocks.0.linear2", "block 0 FFN linear2 out"),
            ("model.blocks.0", "block 0 out = residual stream"), ("model.norm", "final LayerNorm out"),
            ("model.head", "logits (head out)")]
# forward ops whose output dtype is worth a row (aten overload packet names)
OPS = ["embedding", "native_layer_norm", "_to_copy", "addmm", "mm", "bmm", "_safe_softmax", "relu", "add",
       "_log_softmax", "nll_loss_forward"]
VIEW_OPS = {"detach", "view", "transpose", "_unsafe_view", "t", "expand", "unsqueeze", "squeeze", "select", "permute",
            "alias", "slice", "as_strided", "reshape", "unflatten", "split", "chunk", "unbind"}


def short(dt: torch.dtype) -> str:
    return SHORT.get(dt, str(dt).replace("torch.", ""))


class OpTally(TorchDispatchMode):
    """Counts (phase, op, output dtype) for every aten op, and remembers autocast's casts of parameters.

    A TorchDispatchMode runs BELOW autograd and autocast in the dispatcher, so it sees the ops that actually execute:
    the _to_copy casts autocast inserted, and in backward the backward kernels (and any recomputed forward ops)."""

    def __init__(self, param_ptrs: dict[int, str]):
        super().__init__()
        self.param_ptrs, self.phase = param_ptrs, "fwd"
        self.ops: Counter = Counter()
        self.casts: dict[int, tuple[str, str, int]] = {}  # output storage ptr -> (param, dtype, bytes)
        self.cast_calls: Counter = Counter()  # phase -> number of parameter casts

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        t = out if isinstance(out, torch.Tensor) else next(
            (o for o in out if isinstance(o, torch.Tensor)), None) if isinstance(out, (tuple, list)) else None
        if t is not None:
            self.ops[(self.phase, func.overloadpacket.__name__, short(t.dtype))] += 1
        if func is torch.ops.aten._to_copy.default and args and isinstance(args[0], torch.Tensor):
            name = self.param_ptrs.get(args[0].untyped_storage().data_ptr())
            if name is not None and self.phase in ("fwd", "bwd"):  # "opt" = the probe's own CPU copies: ignore
                self.cast_calls[self.phase] += 1
                if self.phase == "fwd":
                    self.casts[out.untyped_storage().data_ptr()] = (name, short(out.dtype),
                                                                    out.untyped_storage().nbytes())
        return out


class Probe:
    """Everything recorded around ONE window_step call: the `on` callback, module hooks and the saved-tensor census."""

    def __init__(self, net, device, n_micro: int):
        self.net, self.device, self.n_micro = net, device, n_micro
        self.params = dict(net.named_parameters())
        self.param_ptrs = {p.untyped_storage().data_ptr(): k for k, p in self.params.items()}
        self.tally = OpTally(self.param_ptrs)
        self.saved: dict[int, dict] = {}
        self.stack: list[str] = []
        self.act_dtype: dict[str, str] = {}
        self.calls: Counter = Counter()  # (block, phase) -> forward calls
        self.mem: dict[str, float] = {}
        self.n_bwd = 0
        self.handles = []
        for name, mod in net.named_modules():
            if not name:
                continue
            self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self.handles.append(mod.register_forward_hook(self._post(name)))

    # -- module hooks: attribution stack, block call counts, selected output dtypes
    def _pre(self, name):
        def hook(_m, _a):
            self.stack.append(name)
            parts = name.split(".")
            if len(parts) == 3 and parts[1] == "blocks":
                self.calls[(name, self.tally.phase)] += 1
        return hook

    def _post(self, name):
        def hook(_m, _a, out):
            if self.stack and self.stack[-1] == name:
                self.stack.pop()
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and name not in self.act_dtype and self.tally.phase == "fwd":
                self.act_dtype[name] = short(t.dtype)
        return hook

    # -- saved-tensor census (unique storages, first saver wins)
    def pack(self, t: torch.Tensor) -> torch.Tensor:
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr not in self.saved:
            kind = "param" if ptr in self.param_ptrs else "weight cast" if ptr in self.tally.casts else "activation"
            where = self.stack[-1] if self.stack else "(outside modules: loss, or checkpoint's saved inputs)"
            self.saved[ptr] = {"kind": kind, "dtype": short(t.dtype), "bytes": st.nbytes(), "shape": list(t.shape),
                               "where": where}
        return t

    # -- the window_step callback
    def _sync_mem(self, key: str) -> None:
        torch.cuda.synchronize(self.device)
        self.mem[key] = torch.cuda.memory_allocated(self.device) / MiB
        self.mem["peak_" + key] = torch.cuda.max_memory_allocated(self.device) / MiB
        torch.cuda.reset_peak_memory_stats(self.device)

    def on(self, event: str) -> None:
        if event == "fwd":
            self._sync_mem("after_fwd")
            self.tally.phase = "bwd"
        elif event == "bwd":
            self.n_bwd += 1
            self._sync_mem("after_bwd")
            self.tally.phase = "fwd" if self.n_bwd < self.n_micro else "opt"
        elif event == "pre_clip":
            self.tally.phase = "opt"
            # copies go to the CPU so they do not show up in the GPU memory numbers
            self.grads = {k: p.grad.detach().float().cpu() for k, p in self.params.items()}
            self.grad_dtypes = sorted({short(p.grad.dtype) for p in self.params.values()})
        elif event == "pre_step":
            self.before_step = {k: p.detach().cpu().clone() for k, p in self.params.items()}
        elif event == "opt":
            self._sync_mem("after_opt")

    def detach(self) -> None:
        for h in self.handles:
            h.remove()


def run_one(cfg, device, impl, precision: str, ckpt: bool, batch, sdpa: str) -> tuple[Probe, dict]:
    """Fresh model from the same seed, one window of K = 1 microbatch, everything recorded."""
    gc.collect()  # the previous configuration's model (hooks make reference cycles) must be gone before "start"
    torch.cuda.empty_cache()
    setup_reproducibility(cfg)
    model, optimizer, _sched = build(cfg, device)  # LambdaLR has already set lr = 3e-4 * lr_lambda(0) = 1.5e-5
    lr0 = optimizer.param_groups[0]["lr"]
    run_prec = precision
    if precision == "bf16_pure":
        model = model.to(torch.bfloat16)  # params AND (below) AdamW state in bf16: there is no fp32 copy anywhere
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=cfg.weight_decay, foreach=False,
                                      fused=False)
        run_prec = "fp32"  # no autocast: the model itself is bf16
    net = AmpNet(model, ckpt, impl.model_forward)
    scaler = make_scaler(precision)
    x, y = batch
    probe = Probe(net, device, n_micro=1)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    probe.mem["start"] = torch.cuda.memory_allocated(device) / MiB
    with sdpa_context(sdpa), probe.tally, torch.autograd.graph.saved_tensors_hooks(probe.pack, lambda t: t):
        out = impl.window_step(net, optimizer, scaler, [(x, y, 1.0)], precision=run_prec, max_norm=1.0,
                               on=probe.on)
    probe.detach()
    params = probe.params
    unchanged = {}
    groups = {"all": list(params), "matrix weights (2-D, not embeddings)":
              [k for k, p in params.items() if p.dim() == 2 and "emb" not in k],
              "embeddings": [k for k in params if "emb" in k],
              "1-D (biases, LayerNorm)": [k for k, p in params.items() if p.dim() == 1]}
    for g, keys in groups.items():
        same = sum(int((params[k].detach().cpu() == probe.before_step[k]).sum()) for k in keys)
        unchanged[g] = same / sum(params[k].numel() for k in keys)
    state = sorted({f"{k}={short(v.dtype)}" for st in optimizer.state.values() for k, v in st.items()
                    if torch.is_tensor(v)})
    info = {"precision": precision, "ckpt": ckpt, "loss": out["loss"].item(), "norm_seen_by_clip": out["norm"].item(),
            "scale": out["scale"], "skipped": out["skipped"],
            "param_dtypes": sorted({short(p.dtype) for p in params.values()}), "grad_dtypes": probe.grad_dtypes,
            "adamw_state": state, "scaler_scale_dtype": short(scaler._scale.dtype) if scaler is not None else None,
            "unchanged_after_step": unchanged, "memory_mib": probe.mem}
    probe.optimizer = optimizer
    return probe, info


def census(probe: Probe) -> dict:
    by_kind, by_where = Counter(), Counter()
    for s in probe.saved.values():
        by_kind[f"{s['kind']} {s['dtype']}"] += s["bytes"]
        w = s["where"]
        top = ".".join(w.split(".")[:3]) if w.startswith("model.blocks.") else w
        by_where[top] += s["bytes"]
    return {"by_kind_mib": {k: v / MiB for k, v in sorted(by_kind.items())},
            "by_module_mib": {k: v / MiB for k, v in by_where.items()},
            "total_mib": sum(s["bytes"] for s in probe.saved.values()) / MiB}


def block0_detail(probe: Probe, top: int = 14) -> list[dict]:
    rows = [s for s in probe.saved.values() if s["where"].startswith("model.blocks.0")]
    return sorted(rows, key=lambda s: -s["bytes"])[:top]


def rel_err(g: dict, ref: dict) -> float:
    num = sum(float((g[k] - ref[k]).double().pow(2).sum()) for k in ref)
    den = sum(float(ref[k].double().pow(2).sum()) for k in ref)
    return (num / den) ** 0.5


# ============================================================================ --part dtypes
def part_dtypes(cfg, device, impl, args) -> dict:
    print(f"\n=== Part dtypes: one optimizer step, same init and batch (medium, m = {cfg.batch_size}, K = 1, "
          f"SDPA {args.sdpa}) ===")
    batch = get_batch(cfg, 0, device)
    precs = ["fp32", "bf16", "fp16", "bf16_pure"]
    res, grads = {}, {}
    for prec in precs:
        probe, info = run_one(cfg, device, impl, prec, False, batch, args.sdpa)
        info["census"] = census(probe)
        info["weight_casts"] = {"count": len(probe.tally.casts),
                                "mib": sum(c[2] for c in probe.tally.casts.values()) / MiB,
                                "saved_for_backward_mib": info["census"]["by_kind_mib"].get(
                                    f"weight cast {LOWS.get(prec, '-')}", 0.0)}
        info["activation_dtypes"] = {d: probe.act_dtype.get(n, "-") for n, d in SELECTED}
        info["fwd_ops"] = {f"{op} {dt}": c for (ph, op, dt), c in sorted(probe.tally.ops.items()) if ph == "fwd"}
        info["block0_saved"] = block0_detail(probe)
        res[prec], grads[prec] = info, probe.grads  # CPU copies only; the model is released
        del probe
    for prec in precs:
        res[prec]["grad_rel_err_vs_fp32"] = rel_err(grads[prec], grads["fp32"]) if prec != "fp32" else 0.0

    w = 16
    print("\n-- dtypes of what you own and what autograd made (one row per tensor kind) --")
    print(f"{'':48s}" + "".join(f"{p:>{w}s}" for p in precs))
    rows = [("parameters (p.dtype)", lambda r: "/".join(r["param_dtypes"]))]
    rows += [(desc, (lambda n: lambda r: r["activation_dtypes"][n])(desc)) for _n, desc in SELECTED]

    def op_dtypes(op):
        return lambda r: "/".join(sorted({k.split()[1] for k in r["fwd_ops"] if k.split()[0] == op})) or "-"
    rows += [("attention scores bmm (SDPA math)", op_dtypes("bmm")),
             ("attention softmax (_safe_softmax)", op_dtypes("_safe_softmax")),
             ("log_softmax (inside cross_entropy)", op_dtypes("_log_softmax")),
             ("loss (nll_loss_forward)", op_dtypes("nll_loss_forward")),
             ("p.grad (after unscale_)", lambda r: "/".join(r["grad_dtypes"])),
             ("AdamW exp_avg / exp_avg_sq", lambda r: "/".join(sorted({s.split("=")[1] for s in r["adamw_state"]
                                                                       if s.startswith("exp_avg")}))),
             ("AdamW step counter", lambda r: "/".join(s.split("=")[1] for s in r["adamw_state"]
                                                       if s.startswith("step"))),
             ("GradScaler scale (value, dtype)", lambda r: f"{r['scale']:.0f} {r['scaler_scale_dtype']}"
              if r["scaler_scale_dtype"] else "-")]
    for label, f in rows:
        print(f"{label:48s}" + "".join(f"{f(res[p]):>{w}s}" for p in precs))

    print("\n-- forward ops by output dtype (op: count per dtype; a TorchDispatchMode sees ops AFTER autocast) --")
    print(f"{'':20s}" + "".join(f"{p:>{w + 6}s}" for p in precs))
    for op in OPS:
        cells = []
        for p in precs:
            c = {k.split()[1]: v for k, v in res[p]["fwd_ops"].items() if k.split()[0] == op}
            cells.append(" ".join(f"{d}x{n}" for d, n in sorted(c.items())) or "-")
        print(f"{op:20s}" + "".join(f"{c:>{w + 6}s}" for c in cells))

    print("\n-- what autograd keeps for backward (unique storages, MiB) and autocast's weight copies --")
    kinds = sorted({k for p in precs for k in res[p]["census"]["by_kind_mib"]})
    print(f"{'':48s}" + "".join(f"{p:>{w}s}" for p in precs))
    for k in kinds:
        print(f"{'saved: ' + k:48s}" + "".join(f"{res[p]['census']['by_kind_mib'].get(k, 0.0):{w}.1f}" for p in precs))
    print(f"{'saved: total':48s}" + "".join(f"{res[p]['census']['total_mib']:{w}.1f}" for p in precs))
    print(f"{'weight casts made in forward (count)':48s}" + "".join(f"{res[p]['weight_casts']['count']:{w}d}"
                                                                     for p in precs))
    print(f"{'weight casts made in forward (MiB)':48s}" + "".join(f"{res[p]['weight_casts']['mib']:{w}.1f}"
                                                                   for p in precs))
    for key, label in [("start", "allocated at start (params only)"),
                       ("after_fwd", "allocated after forward + loss"), ("peak_after_bwd", "peak during backward"),
                       ("after_opt", "allocated after step (+ grads + AdamW state)")]:
        print(f"{label:48s}" + "".join(f"{res[p]['memory_mib'][key]:{w}.1f}" for p in precs))

    print("\n-- numerics of the step --")
    print(f"{'':48s}" + "".join(f"{p:>{w}s}" for p in precs))
    print(f"{'loss':48s}" + "".join(f"{res[p]['loss']:{w}.5f}" for p in precs))
    print(f"{'gradient rel. error vs fp32 ||g-g32||/||g32||':48s}" +
          "".join(f"{res[p]['grad_rel_err_vs_fp32']:{w}.2e}" for p in precs))
    for g in res["fp32"]["unchanged_after_step"]:
        print(f"{'unchanged by step 1: ' + g:48s}" + "".join(f"{res[p]['unchanged_after_step'][g]:{w}.1%}"
                                                              for p in precs))

    print("\n-- block 0: the largest tensors autograd saved (bf16 autocast vs fp32) --")
    for p in ("fp32", "bf16"):
        print(f"  [{p}]")
        for s in res[p]["block0_saved"]:
            print(f"    {s['where']:28s} {s['kind']:12s} {s['dtype']:5s} {str(s['shape']):22s} {s['bytes'] / MiB:7.2f}")

    inv = all(res[p]["param_dtypes"] == ["fp32"] and res[p]["grad_dtypes"] == ["fp32"]
              and {s.split("=")[1] for s in res[p]["adamw_state"] if s.startswith("exp_avg")} == {"fp32"}
              for p in ("fp32", "bf16", "fp16"))
    inv_pure = res["bf16_pure"]["param_dtypes"] == ["bf16"] and res["bf16_pure"]["grad_dtypes"] == ["bf16"]
    print(f"\nDTYPE PROBE: params, grads and AdamW state fp32 under fp32/bf16/fp16 autocast: {inv}; "
          f"all bf16 in bf16_pure: {inv_pure}")
    return res


LOWS = {"bf16": "bf16", "fp16": "fp16"}


# ============================================================================ --part ckpt
def part_ckpt(cfg, device, impl, args) -> dict:
    print(f"\n=== Part ckpt: checkpointing off vs on, same precision, same init and batch (medium, m = "
          f"{cfg.batch_size}, K = 1, SDPA {args.sdpa}) ===")
    batch = get_batch(cfg, 0, device)
    out, ok_all = {}, True
    block_input_mib = cfg.n_layers * cfg.batch_size * cfg.seq_len * cfg.d_model * 4 / MiB
    for prec in args.ckpt_precisions.split(","):
        r = {}
        for ck in (False, True):
            probe, info = run_one(cfg, device, impl, prec, ck, batch, args.sdpa)
            info["census"] = census(probe)
            info["block_calls"] = {f"{b} {ph}": c for (b, ph), c in sorted(probe.calls.items())}
            info["bwd_ops"] = {f"{op} {dt}": c for (ph, op, dt), c in probe.tally.ops.items() if ph == "bwd"}
            info["param_casts_by_phase"] = dict(probe.tally.cast_calls)
            info["_grads"] = probe.grads
            del probe
            r["on" if ck else "off"] = info
        off, on = r["off"], r["on"]
        g0, g1 = off.pop("_grads"), on.pop("_grads")
        bitwise = all(torch.equal(g0[k], g1[k]) for k in g0)
        maxdiff = max(float((g0[k] - g1[k]).abs().max()) for k in g0)
        loss_eq = off["loss"] == on["loss"]
        calls_ok = (all(v == 1 for k, v in off["block_calls"].items() if k.endswith(" fwd"))
                    and not any(k.endswith(" bwd") for k in off["block_calls"])
                    and all(on["block_calls"].get(f"model.blocks.{i} {ph}") == 1 for i in range(cfg.n_layers)
                            for ph in ("fwd", "bwd")))
        mem_ok = on["memory_mib"]["after_fwd"] < off["memory_mib"]["after_fwd"]
        rel = rel_err(g1, g0)
        ok = loss_eq and calls_ok and mem_ok and (bitwise or rel < 1e-6)
        ok_all &= ok
        extra = Counter(on["bwd_ops"])
        extra.subtract(off["bwd_ops"])
        recomputed = {k: v for k, v in sorted(extra.items(), key=lambda kv: -kv[1])
                      if v > 0 and k.split()[0] not in VIEW_OPS}
        r["check"] = {"loss_equal": loss_eq, "grads_bitwise": bitwise, "grad_max_abs_diff": maxdiff,
                      "grad_rel_err": rel, "block_calls_ok": calls_ok, "held_memory_lower": mem_ok, "pass": ok,
                      "recomputed_ops_in_backward": recomputed}
        out[prec] = r

        print(f"\n[{prec}]  {'':44s}{'ckpt off':>12s}{'ckpt on':>12s}")
        for key, label in [("start", "allocated at start (params)"),
                           ("after_fwd", "allocated after forward + loss"),
                           ("peak_after_fwd", "peak during forward"), ("peak_after_bwd", "peak during backward")]:
            print(f"  {label:48s}{off['memory_mib'][key]:12.1f}{on['memory_mib'][key]:12.1f}")
        print(f"  {'act held = after forward - start':48s}"
              f"{off['memory_mib']['after_fwd'] - off['memory_mib']['start']:12.1f}"
              f"{on['memory_mib']['after_fwd'] - on['memory_mib']['start']:12.1f}")
        print("  census of what autograd saved, by module (MiB):")
        mods = list(dict.fromkeys([*off["census"]["by_module_mib"], *on["census"]["by_module_mib"]]))
        for mod in mods:
            print(f"    {mod:46s}{off['census']['by_module_mib'].get(mod, 0.0):12.1f}"
                  f"{on['census']['by_module_mib'].get(mod, 0.0):12.1f}")
        print(f"    {'total':46s}{off['census']['total_mib']:12.1f}{on['census']['total_mib']:12.1f}")
        print(f"  expected checkpoint inputs: {cfg.n_layers} blocks x [B, T, D] fp32 = {block_input_mib:.1f} MiB "
              f"(the residual stream entering each block)")
        fwd_calls = sum(v for k, v in on["block_calls"].items() if k.endswith(" fwd"))
        bwd_calls = sum(v for k, v in on["block_calls"].items() if k.endswith(" bwd"))
        print(f"  block forward calls with ckpt on: {fwd_calls} in forward, {bwd_calls} inside backward (recompute)")
        print(f"  parameter casts by phase: off {off['param_casts_by_phase']}, on {on['param_casts_by_phase']}")
        print("  forward compute ops that ran inside backward only with ckpt on (op dtype: extra count; views and "
              "detach omitted):")
        print("    " + ", ".join(f"{k}: {v}" for k, v in list(recomputed.items())[:14]))
        print(f"  correctness: loss equal {loss_eq} | grads bitwise {bitwise} (max |diff| {maxdiff:.3e}, rel "
              f"{rel:.2e}) | block calls {calls_ok} | held memory lower {mem_ok}  =>  {'PASS' if ok else 'FAIL'}")
    print(f"\nCKPT CHECK: {'PASSED' if ok_all else 'FAILED'} ("
          + ", ".join(f"{p}: {'bitwise' if r['check']['grads_bitwise'] else 'close'}" for p, r in out.items()) + ")")
    out["passed"] = ok_all
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["dtypes", "ckpt", "all"], default="all")
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument("--preset", choices=PRESETS, default="medium")
    ap.add_argument("--dtype-batch", type=int, default=8, help="rows for --part dtypes")
    ap.add_argument("--ckpt-batch", type=int, default=32, help="rows for --part ckpt (the matrix's m)")
    ap.add_argument("--ckpt-precisions", default="fp32,bf16")
    ap.add_argument("--sdpa", choices=["math", "auto"], default="math")
    ap.add_argument("--run-id", default="local")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    impl = load_impl(args.impl)
    print(f"torch {torch.__version__} | {env_info(device).get('gpu')} | impl={args.impl} | TF32 matmul "
          f"{torch.backends.cuda.matmul.allow_tf32}")
    OUT.mkdir(parents=True, exist_ok=True)
    # cuBLAS / cuBLASLt allocate their workspaces (~64 MiB here) lazily, during the first forward AND backward matmuls.
    # One discarded fp32 + bf16 step makes them exist before any measurement, so every "allocated at start" row
    # includes them equally.
    wcfg = Config(**PRESETS[args.preset])
    wcfg.batch_size = 2
    for prec in ("fp32", "bf16"):
        run_one(wcfg, device, impl, prec, False, get_batch(wcfg, 0, device), args.sdpa)
    if args.part in ("dtypes", "all"):
        cfg = Config(**PRESETS[args.preset])
        cfg.batch_size = args.dtype_batch
        res = part_dtypes(cfg, device, impl, args)
        (OUT / f"probe_dtypes_{args.impl}_{args.run_id}.json").write_text(json.dumps(res, indent=2, default=str))
    ok = True
    if args.part in ("ckpt", "all"):
        cfg = Config(**PRESETS[args.preset])
        cfg.batch_size = args.ckpt_batch
        res = part_ckpt(cfg, device, impl, args)
        ok = res["passed"]
        (OUT / f"probe_ckpt_{args.impl}_{args.run_id}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {OUT}/probe_*_{args.impl}_{args.run_id}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
