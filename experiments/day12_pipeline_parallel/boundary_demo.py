"""Day 12 - autograd across a stage boundary, with numbers you can check by hand (one CPU process, no communication).

    python experiments/day12_pipeline_parallel/boundary_demo.py

Two "stages" with one scalar weight each, two microbatches:
    stage 0:  h = w0 * x                 w0 = 3
    stage 1:  y = w1 * h,  L_m = y^2/2   w1 = 2      total loss L = L_0 + L_1   (weight 1 per microbatch here)
    microbatches x_0 = 1, x_1 = 2

The "send" is a detach-and-copy: only numbers cross, no graph. The script runs:
  A. the uncut graph (what one GPU computes)                         -> reference gradients
  B. the cut graph, the pipeline way (requires_grad on the receiver, backward with grad_tensors on the sender)
  C. three mistakes: the receiver forgets requires_grad; the sender calls backward() without the received gradient;
     the sender pairs microbatch 0's gradient with microbatch 1's saved graph
and prints what is SAVED on each side for each microbatch and what CROSSES the boundary.
"""

from __future__ import annotations

import torch


def params():
    return torch.tensor(3.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)


XS = [torch.tensor(1.0), torch.tensor(2.0)]


def uncut():
    w0, w1 = params()
    for x in XS:
        y = w1 * (w0 * x)
        (y * y / 2).backward()
    return w0.grad.item(), w1.grad.item()


def cut(mistake: str | None = None, verbose: bool = False):
    w0, w1 = params()
    saved0, saved1, wire = {}, {}, []
    # ---- forward of both microbatches (GPipe order) ----
    for m, x in enumerate(XS):
        h = w0 * x                                    # stage 0 output, still attached to stage 0's graph
        saved0[m] = h                                 # stage 0 keeps the OUTPUT tensor (its grad_fn is the graph)
        sent = h.detach().clone()                     # what crosses: a number, no history
        wire.append(f"F: stage 0 -> stage 1  activation a{m} = {sent.item():g}")
        r = sent if mistake == "no_requires_grad" else sent.requires_grad_()   # stage 1's received leaf
        y = w1 * r
        loss = y * y / 2
        saved1[m] = (r, loss)                         # stage 1 keeps its INPUT leaf and its OUTPUT (the loss)
    if verbose:
        for m in saved0:
            print(f"  saved on stage 0 for mb{m}: output h = {saved0[m].item():g} (grad_fn {type(saved0[m].grad_fn).__name__})")
            r, loss = saved1[m]
            print(f"  saved on stage 1 for mb{m}: input r = {r.item():g} (leaf, requires_grad={r.requires_grad}), "
                  f"loss = {loss.item():g}")
    # ---- backward of both microbatches ----
    order0 = [1, 0] if mistake == "swap" else [0, 1]   # which saved graph stage 0 uses for the m-th gradient
    for m in (0, 1):
        r, loss = saved1[m]
        loss.backward()                               # stage 1: scalar loss, starts from dL/dL = 1
        g = r.grad
        if g is None:
            wire.append(f"B: stage 1 -> stage 0  gradient g{m} = None  (r was not a leaf requiring grad)")
            continue
        g = g.detach().clone()
        wire.append(f"B: stage 1 -> stage 0  gradient g{m} = dL/da{m} = {g.item():g}")
        h = saved0[order0[m]]
        if mistake == "no_grad_tensors":
            try:
                h.backward()                          # h is a scalar here, so this RUNS: it uses 1.0, not g
            except RuntimeError as e:
                wire.append(f"   stage 0 backward() error: {e}")
        else:
            torch.autograd.backward(h, grad_tensors=g)
    if verbose:
        for line in wire:
            print("  " + line)
    return (None if w0.grad is None else w0.grad.item()), w1.grad.item()


def main() -> None:
    ref = uncut()
    print("A. uncut graph (one device):                         w0.grad = %g, w1.grad = %g" % ref)
    print("   by hand: mb0 h=3 y=6 dL/dy=6 -> dw1 += 6*3 = 18, dL/dh = 6*w1 = 12 -> dw0 += 12*x0 = 12")
    print("            mb1 h=6 y=12 dL/dy=12 -> dw1 += 12*6 = 72, dL/dh = 24 -> dw0 += 24*x1 = 48")
    print("            totals w0.grad = 60, w1.grad = 90")
    print("\nB. cut graph, pipeline style:")
    got = cut(verbose=True)
    print(f"   -> w0.grad = {got[0]:g}, w1.grad = {got[1]:g}   equal to A: {got == ref}")
    checks = [got == ref == (60.0, 90.0)]
    print("\nC. mistakes:")
    a = cut("no_requires_grad")
    print(f"   receiver forgets requires_grad: w0.grad = {a[0]}, w1.grad = {a[1]:g} "
          f"(stage 1 is right, stage 0 gets nothing to backpropagate)")
    b = cut("no_grad_tensors")
    print(f"   sender calls h.backward() without g: w0.grad = {b[0]:g} (= 1*x0 + 1*x1 = 3; with a non-scalar h, as "
          f"in the real model, backward() raises 'grad can be implicitly created only for scalar outputs')")
    c = cut("swap")
    print(f"   sender pairs g0 with mb1's graph and g1 with mb0's: w0.grad = {c[0]:g} (= 12*x1 + 24*x0 = 48, "
          f"not 60). Same shapes, no error, wrong answer.")
    checks += [a[0] is None and a[1] == 90.0, b[0] == 3.0, c[0] == 48.0]
    print(f"\nBOUNDARY DEMO: {'PASSED' if all(checks) else 'FAILED'}")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
