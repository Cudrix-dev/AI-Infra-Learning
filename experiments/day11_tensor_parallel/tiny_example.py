"""Day 11 - the tiny example from GUIDE 1.1, computed by hand-sized integers on the CPU, both "ranks" in one process.

    python experiments/day11_tensor_parallel/tiny_example.py

No GPUs, no torch.distributed: the "all-reduce" is a Python sum over the two rank tensors. Use it to check your
predictions about what each rank owns, then compare with the real two-process run.

Model (one block, TP = 2): y = W2 . relu(W1 . x + b1) + b2, with 2 tokens, d = 4, ffn = 4 (2 per rank).
ReLU instead of GELU only to keep every number an integer; any elementwise function behaves the same.
Loss: L = sum(y * G) for a fixed integer matrix G, so the upstream gradient dL/dy is exactly G.
"""

from __future__ import annotations

import torch

T = lambda rows: torch.tensor(rows, dtype=torch.float64)  # noqa: E731

x = T([[1, 2, 0, -1],
       [0, 1, 1, 2]])                    # [tokens=2, d=4]  replicated on both ranks
W1 = T([[1, 0, -1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1]])                  # [ffn=4, d=4]  nn.Linear(4, 4).weight layout [out, in]
b1 = T([0, -1, 0, 1])                    # [ffn=4]
W2 = T([[1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 1]])                   # [d=4, ffn=4]  [out, in]
b2 = T([1, 0, 0, -1])                    # [d=4]
G = T([[1, 0, 0, 1],
       [0, 1, -1, 0]])                   # dL/dy, [tokens, d]
P = 2


def show(name, t):
    t = t + 0.0  # turn -0 into 0 for display
    rows = t.tolist() if t.dim() > 1 else [t.tolist()]
    body = " ; ".join(" ".join(f"{v:3g}" for v in r) for r in rows)
    print(f"    {name:<10} {str(list(t.shape)):<8} [{body}]")


def main() -> None:
    torch.set_printoptions(precision=0)
    # ---- dense (single GPU) with autograd -------------------------------------------------------------------
    xs = x.clone().requires_grad_(True)
    p = [t.clone().requires_grad_(True) for t in (W1, b1, W2, b2)]
    h = xs @ p[0].T + p[1]
    a = torch.relu(h)
    y = a @ p[2].T + p[3]
    (y * G).sum().backward()
    print("DENSE (one device)")
    for n, t in [("h", h), ("a", a), ("y", y), ("dx", xs.grad), ("dW1", p[0].grad), ("db1", p[1].grad),
                 ("dW2", p[2].grad), ("db2", p[3].grad)]:
        show(n, t.detach())

    # ---- tensor parallel, written out per rank ---------------------------------------------------------------
    n = W1.shape[0] // P
    parts = {}
    for r in range(P):
        s = slice(r * n, (r + 1) * n)
        W1_r, b1_r = W1[s], b1[s]                 # column-parallel: ROWS of the [out, in] weight
        W2_r = W2[:, s]                           # row-parallel: COLUMNS of the [out, in] weight
        h_r = x @ W1_r.T + b1_r                   # [2, 2] sharded: columns s of the dense h
        a_r = torch.relu(h_r)                     # elementwise: still exactly columns s of the dense a
        z_r = a_r @ W2_r.T                        # [2, 4] PARTIAL: one rank's share of the sum over ffn
        parts[r] = dict(W1_r=W1_r, b1_r=b1_r, W2_r=W2_r, h_r=h_r, a_r=a_r, z_r=z_r)
    z = sum(parts[r]["z_r"] for r in range(P))    # forward all-reduce (g)
    y_tp = z + b2                                 # bias once, after the sum
    dy = G                                        # replicated: every rank computes the same loss from the same y
    db2_tp = dy.sum(0)                            # replicated, computed locally on each rank, no communication
    for r in range(P):
        q = parts[r]
        q["dz_r"] = dy                            # g backward = identity
        q["dW2_r"] = dy.T @ q["a_r"]              # [4, 2] local; columns s of the dense dW2
        q["da_r"] = dy @ q["W2_r"]                # [2, 2] sharded
        q["dh_r"] = q["da_r"] * (q["h_r"] > 0)    # relu'
        q["dW1_r"] = q["dh_r"].T @ x              # [2, 4] local; rows s of the dense dW1
        q["db1_r"] = q["dh_r"].sum(0)
        q["dx_r"] = q["dh_r"] @ q["W1_r"]         # [2, 4] PARTIAL: only the paths through this rank's ffn units
    dx_tp = sum(parts[r]["dx_r"] for r in range(P))   # backward all-reduce (f)

    for r in range(P):
        print(f"\nRANK {r} owns ffn units {r * n}..{(r + 1) * n - 1}")
        q = parts[r]
        for k in ["W1_r", "b1_r", "W2_r", "h_r", "a_r", "z_r", "dW2_r", "da_r", "dh_r", "dW1_r", "db1_r", "dx_r"]:
            show(k, q[k])

    print("\nAFTER THE TWO ALL-REDUCES")
    show("y = z+b2", y_tp)
    show("dx = sum", dx_tp)
    show("db2", db2_tp)

    checks = {
        "y": torch.equal(y_tp, y.detach()),
        "dx (summed)": torch.equal(dx_tp, xs.grad),
        "dx without reduce (rank 0 only)": torch.equal(parts[0]["dx_r"], xs.grad),
        "dW1 = cat rows": torch.equal(torch.cat([parts[r]["dW1_r"] for r in range(P)], 0), p[0].grad),
        "db1 = cat": torch.equal(torch.cat([parts[r]["db1_r"] for r in range(P)], 0), p[1].grad),
        "dW2 = cat cols": torch.equal(torch.cat([parts[r]["dW2_r"] for r in range(P)], 1), p[2].grad),
        "db2": torch.equal(db2_tp, p[3].grad),
        "h = cat cols of h_r": torch.equal(torch.cat([parts[r]["h_r"] for r in range(P)], 1), h.detach()),
    }
    print("\nCHECKS (exact, integers)")
    for k, v in checks.items():
        print(f"    {k:<34} {v}")
    expected_false = {"dx without reduce (rank 0 only)"}
    ok = all(v == (k not in expected_false) for k, v in checks.items())
    print("TINY EXAMPLE:", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
