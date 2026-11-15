"""Day 12 - the single-GPU reference: the whole 4-block model on one GPU, for every microbatch count M.

    CUDA_VISIBLE_DEVICES=0 python experiments/day12_pipeline_parallel/baseline_1gpu.py --run-id RID \
        [--microbatches 1,2,4,8] [--dtype float32|float64]

For each M it runs plain gradient accumulation: forward + backward of microbatch 0, 1, ..., M-1 in that order, each
with mb_loss() (sum / N, so the M losses add up to the full-batch mean), and records the concatenated output y, the
summed loss and every parameter gradient BEFORE any optimizer step. M = 1 is the ordinary full-batch step.

Why accumulation and not only the full batch: a pipeline with M microbatches does the same arithmetic as
accumulation with M microbatches, in the same order on every parameter (both schedules accumulate microbatch 0
first). So the pipeline can be compared with two references: accumulation with the same M (same arithmetic) and the
full batch (same mathematics, different rounding order).

Writes outputs/day12_pp/tensors/baseline_<dtype>_<RID>.pt (not archived; reproducible from the seed) and
outputs/day12_pp/baseline_<dtype>_<RID>.json (losses, differences between M values, determinism, CUDA memory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pp_common import (DTYPES, OUT, REL_TOL, TENSORS, Shapes, build_blocks, compare, environment,  # noqa: E402
                       make_problem, mb_loss, microbatches, set_exact_math, sha256_tensors, shapes_to_dict,
                       write_json, zero_grads)


def run_accumulation(model, problem, shapes: Shapes, M: int, device) -> dict:
    zero_grads(model)
    x = problem["x"].to(device)
    t = problem["target"].to(device)
    xs, ts = microbatches(x, M), microbatches(t, M)
    n_total = t.numel()
    ys, loss = [], torch.zeros((), dtype=x.dtype, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    for m in range(M):
        y = model(xs[m])
        lm = mb_loss(y, ts[m], n_total)
        lm.backward()
        ys.append(y.detach())
        loss += lm.detach()
    peak = None
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    return {"y": torch.cat(ys, dim=1).cpu(), "loss": loss.cpu(),
            "grads": {k: v.detach().cpu().clone() for k, v in model.grads().items()}, "peak_mib": peak}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--microbatches", default="1,2,4,8")
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    set_exact_math()
    dtype = DTYPES[args.dtype]
    shapes = Shapes()
    Ms = [int(v) for v in args.microbatches.split(",")]
    if 1 not in Ms:
        Ms = [1] + Ms                       # the full-batch reference is always needed
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)

    problem = make_problem(shapes, dtype)
    model = build_blocks(problem, range(shapes.blocks), shapes, device, dtype)
    results, summary = {}, {}
    full = None
    print(f"Single-device 4-block model, {args.dtype}, device {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}); effective batch {shapes.batch} "
          f"sequences x {shapes.seq} tokens, d {shapes.d_model}, ffn {shapes.ffn}")
    print(f"{'M':>3} {'microbatch':>16} {'loss':>16} {'repeat bitwise':>15} {'y vs M=1 rel':>13} "
          f"{'worst grad vs M=1 rel':>22} {'peak MiB':>9}")
    for M in Ms:
        r = run_accumulation(model, problem, shapes, M, device)
        r2 = run_accumulation(model, problem, shapes, M, device)
        rep = torch.equal(r["y"], r2["y"]) and torch.equal(r["loss"], r2["loss"]) and all(
            torch.equal(r["grads"][k], r2["grads"][k]) for k in r["grads"])
        if full is None:
            full = r
        cy = compare("y", r["y"], full["y"], REL_TOL[dtype])
        cg = [compare(k, r["grads"][k], full["grads"][k], REL_TOL[dtype]) for k in r["grads"]]
        worst = max(cg, key=lambda c: c["rel"])
        mb_shape = [shapes.seq, shapes.batch // M, shapes.d_model]
        print(f"{M:>3} {str(mb_shape):>16} {r['loss'].item():>16.10g} {str(rep):>15} {cy['rel']:>13.3e} "
              f"{worst['rel']:>13.3e} ({worst['name']}) {r['peak_mib'] if r['peak_mib'] is None else round(r['peak_mib'], 1)!s:>9}")
        results[M] = {"y": r["y"], "loss": r["loss"], "grads": r["grads"]}
        summary[M] = {"microbatch_shape": mb_shape, "loss": r["loss"].item(), "repeat_bitwise": rep,
                      "y_vs_full": cy, "grads_vs_full": cg, "peak_mib_accumulation": r["peak_mib"]}
    print("peak MiB = torch.cuda.max_memory_allocated() - memory_allocated() at the start of the M microbatches,\n"
          "  with every .grad pre-allocated and the inputs already on the GPU: one microbatch's saved activations\n"
          "  for the WHOLE model plus backward temporaries (plain accumulation never holds two microbatches).")

    TENSORS.mkdir(parents=True, exist_ok=True)
    path = TENSORS / f"baseline_{args.dtype}_{args.run_id}.pt"
    if path.exists():
        raise SystemExit(f"{path} exists; use a new --run-id")
    torch.save({"problem": problem, "results": results, "shapes": shapes_to_dict(shapes), "dtype": args.dtype}, path)
    print(f"saved {path}")
    write_json(OUT / f"baseline_{args.dtype}_{args.run_id}.json", {
        "run_id": args.run_id, "dtype": args.dtype, "shapes": shapes_to_dict(shapes), "microbatches": Ms,
        "tensor_file": str(path.relative_to(OUT.parents[1])),
        "problem_sha256": sha256_tensors({"x": problem["x"], "target": problem["target"],
                                          **{f"blk{k}.{n}": t for k, b in enumerate(problem["blocks"])
                                             for n, t in b.items()}}),
        "per_M": summary, "repeat_bitwise": all(v["repeat_bitwise"] for v in summary.values()),
        "environment": environment(),
    })
    print("BASELINE: SAVED")


if __name__ == "__main__":
    main()
