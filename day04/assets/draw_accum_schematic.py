"""Draw day04/assets/accum_schematic.svg: one rank, one optimizer step with K = 4 microbatches, three ways to sync.

Schematic, not to scale. Collective counts per optimizer step agree with the pre-test's profiler count
(count_collectives.py, Guide Appendix A.2): DDP every microbatch = 16 bucket all-reduces + 4 buffer broadcasts,
DDP no_sync = 4 + 1, flat accumulate-then-sync = 1 all-reduce on the compute stream.

  python day04/assets/draw_accum_schematic.py
"""
import sys
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "accum_schematic.svg"
FONT = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
INK, MUTED = "#1f2d3d", "#5b6b7d"
FWD, BWD, OPT = "#c9d8f5", "#2f6fdb", "#e3e8ef"
COMM, BCAST, LOCAL = "#6b46c1", "#b794f4", "#3aa17e"

WIDTH, X0, ROW_H = 1200, 210, 30
F_W, B_W, OPT_W, K = 46, 110, 70, 4
svg = []


def t(x, y, s, size=14, color=INK, anchor="start", weight=400):
    svg.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" '
               f'text-anchor="{anchor}">{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", rx=5):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.5"/>')


def block(x0, x1, y, fill, label="", color=INK, size=13):
    rect(x0, y, x1 - x0, ROW_H, fill)
    if label:
        t((x0 + x1) / 2, y + ROW_H / 2 + 5, label, size, color, "middle", 600)


def panel(y, title, sub, mode):
    """mode: 'every' | 'nosync' | 'flat'. Returns the panel height."""
    h = 170
    rect(20, y, WIDTH - 40, h, "#f6f8fb", "#d5dde7", rx=14)
    t(40, y + 30, title, 17, INK, weight=700)
    t(40, y + 52, sub, 13, MUTED)
    yc, yn = y + 72, y + 72 + ROW_H + 18  # compute lane, NCCL lane
    t(X0 - 12, yc + ROW_H / 2 + 5, "compute stream", 13, INK, "end", 600)
    t(X0 - 12, yn + ROW_H / 2 + 5, "NCCL stream", 13, INK, "end", 600)
    x = X0
    n_ar = n_bc = 0
    for j in range(K):
        sync = mode == "every" or (mode == "nosync" and j == K - 1)
        if mode != "flat" and (mode == "every" or j == 0):
            block(x, x + 16, yn, BCAST)  # DDP buffer broadcast before a forward whose predecessor synced
            n_bc += 1
        block(x, x + F_W, yc, FWD, f"F{j + 1}")
        x += F_W
        block(x, x + B_W, yc, BWD, f"B{j + 1}", "#ffffff")
        if sync:  # 4 buckets, launched as backward produces them (head first, embedding last)
            for b in range(4):
                bx = x + 18 + b * 25
                block(bx, bx + 19, yn, COMM)
                n_ar += 1
        else:
            t(x + B_W / 2, yn + ROW_H / 2 + 5, "hooks return early" if mode != "flat" else "", 11, LOCAL, "middle", 600)
        x += B_W + 4
    if mode == "flat":  # Day 02: one synchronous all_reduce of the accumulated gradient, on the compute stream
        block(x, x + 34, yc, COMM, "AR", "#ffffff", 12)
        x += 38
        n_ar = 1
    elif mode != "flat":
        x += 14  # the last bucket's tail: exposed
    block(x, x + OPT_W, yc, OPT, "optimizer")
    t(x + OPT_W + 14, yc + ROW_H / 2 + 5, f"{n_ar} all-reduce" + ("s" if n_ar > 1 else "")
      + (f" + {n_bc} broadcast" + ("s" if n_bc > 1 else "") if n_bc else ""), 13, COMM, weight=700)
    t(x + OPT_W + 14, yc + ROW_H / 2 + 23, "per optimizer step", 12, MUTED)
    return h


H = 92 + 3 * 186 + 70
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{H}" viewBox="0 0 {WIDTH} {H}" '
           f'font-family="{FONT}">')
svg.append(f'<rect x="0" y="0" width="{WIDTH}" height="{H}" rx="16" fill="#ffffff"/>')
t(WIDTH / 2, 42, "One optimizer step = K = 4 microbatches on one rank: when do gradients travel?", 21, INK, "middle", 700)
t(WIDTH / 2, 66, "medium model, DDP buckets 1/25 MiB (4 buckets) · time flows right · schematic, not to scale", 13,
  MUTED, "middle")
y = 92
y += panel(y, "DDP, sync every microbatch", "every forward arms the Reducer: K x (4 bucket all-reduces) and K buffer "
           "broadcasts; correct, but K times the traffic", "every") + 16
y += panel(y, "DDP, no_sync() around microbatches 1..K-1", "forward under no_sync does not call prepare_for_backward: "
           "hooks fire and return; p.grad accumulates locally; one sync", "nosync") + 16
y += panel(y, "Day 02 flat, accumulate then sync once", "no hooks: just do not call sync_grads until the last "
           "microbatch; the one all-reduce is not overlapped", "flat") + 16
items = [(FWD, "forward"), (BWD, "backward"), (COMM, "all-reduce (bucket)"), (BCAST, "buffer broadcast"),
         (OPT, "optimizer.step + scheduler.step (once)")]
lx = 40
for color, label in items:
    rect(lx, y + 6, 18, 18, color)
    t(lx + 26, y + 20, label, 13, INK)
    lx += 34 + 8.2 * len(label) + 20
svg.append("</svg>")
OUT.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {OUT}")
