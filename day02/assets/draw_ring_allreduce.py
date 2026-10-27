"""Draw day02/assets/ring_allreduce_w3.svg: flowchart of simulate_ring.py for W = 3, with the state grid after each step.

  python day02/assets/draw_ring_allreduce.py
"""
import sys
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "ring_allreduce_w3.svg"
W = 3
FONT = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"
INK, MUTED = "#1f2d3d", "#5b6b7d"
ADD, OVR, FULL = "#6b46c1", "#1f9d6b", "#2f6fdb"
FILL = {1: "#f1f4f8", 2: "#c9d8f5", 3: FULL}

# ---- run the same simulation as simulate_ring.py, recording every step ----
state = [[{r} for _ in range(W)] for r in range(W)]
snap = lambda: [[set(c) for c in row] for row in state]
steps = [("start", None, snap(), [])]
for s in range(W - 1):
    sends = {r: (r - s) % W for r in range(W)}
    pay = {r: set(state[r][sends[r]]) for r in range(W)}
    recv = []
    for r in range(W):
        c = (r - s - 1) % W
        state[r][c] |= pay[(r - 1) % W]
        recv.append((r, c))
    steps.append(("rs", s, snap(), recv, sends))
owners = {c: r for r in range(W) for c in range(W) if len(state[r][c]) == W}
for s in range(W - 1):
    sends = {r: (r + 1 - s) % W for r in range(W)}
    pay = {r: set(state[r][sends[r]]) for r in range(W)}
    recv = []
    for r in range(W):
        c = (r - s) % W
        state[r][c] = set(pay[(r - 1) % W])
        recv.append((r, c))
    steps.append(("ag", s, snap(), recv, sends))
assert all(len(c) == W for row in state for c in row)

svg = []
def t(x, y, s, size=15, color=INK, anchor="start", weight=400, mono=False):
    fam = MONO if mono else FONT
    svg.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')
def rect(x, y, w, h, fill, stroke="none", sw=1.5, rx=8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def arrow(x1, y1, x2, y2, color=MUTED):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')

WIDTH = 1200
FX, FW = 40, 500          # flowchart column
GX = 600                  # grid column
MX = 950                  # moves column
CW, CH, LW = 84, 34, 64   # cell width/height, row-label width

def grid(y, st, recv=(), color=None, sends=None):
    for c in range(W):
        t(GX + LW + c * CW + CW / 2, y + 14, f"chunk {c}", 13, MUTED, "middle")
    for r in range(W):
        cy = y + 22 + r * (CH + 4)
        t(GX, cy + 22, f"rank {r}", 14, MUTED)
        for c in range(W):
            cx = GX + LW + c * CW
            n = len(st[r][c])
            hit = (r, c) in recv
            sent = sends is not None and sends[r] == c
            rect(cx + 2, cy, CW - 4, CH, FILL[n], color if hit else ("#7a8899" if sent else "#d5dde7"),
                 3 if hit else 1.5, 6, "4 3" if sent and not hit else None)
            label = "{" + ",".join(map(str, sorted(st[r][c]))) + "}"
            t(cx + CW / 2, cy + 22, label, 14, "#ffffff" if n == W else INK, "middle", 600 if n == W else 400, True)

def flowbox(y, h, title, lines, fill="#ffffff", stroke="#9aa8b8"):
    rect(FX, y, FW, h, fill, stroke, 1.5, 10)
    t(FX + 18, y + 28, title, 17, INK, weight=700)
    for i, (s, mono) in enumerate(lines):
        t(FX + 18, y + 54 + i * 22, s, 14, MUTED, mono=mono)

H = 1320
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{H}" viewBox="0 0 {WIDTH} {H}">')
svg.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5b6b7d"/></marker></defs>')
rect(0, 0, WIDTH, H, "#ffffff", rx=16)
t(WIDTH / 2, 46, "Ring all-reduce as simulated by simulate_ring.py (W = 3)", 24, INK, "middle", 700)
t(WIDTH / 2, 76, "Each cell lists whose data is already summed into that rank's copy of that chunk. Rows are ranks, columns are chunks.", 15, MUTED, "middle")

# ring topology inset
cx0, cy0, R = 290, 150, 46
pos = [(cx0 + R * dx, cy0 + R * dy) for dx, dy in ((0, -1), (0.866, 0.5), (-0.866, 0.5))]
for r in range(W):
    (x1, y1), (x2, y2) = pos[r], pos[(r + 1) % W]
    k = 17 / ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    arrow(x1 + (x2 - x1) * k, y1 + (y2 - y1) * k, x2 - (x2 - x1) * k, y2 - (y2 - y1) * k)
for r, (x, y) in enumerate(pos):
    svg.append(f'<circle cx="{x}" cy="{y}" r="16" fill="{FULL}"/>')
    t(x, y + 5, f"r{r}", 13, "#ffffff", "middle", 700)
t(cx0 + 90, cy0 - 8, "every rank sends to rank r+1", 14, MUTED)
t(cx0 + 90, cy0 + 14, "and receives from rank r−1", 14, MUTED)
t(cx0 + 90, cy0 + 36, "(rank 2 sends to rank 0)", 14, MUTED)

# legend
ly = 118
for i, (fill, stroke, dash, label) in enumerate([
    (FILL[1], "#d5dde7", None, "1 of 3 ranks summed"),
    (FILL[2], "#d5dde7", None, "2 of 3"),
    (FILL[3], "#d5dde7", None, "all 3: fully reduced"),
    ("#ffffff", "#7a8899", "4 3", "chunk this rank sends"),
    ("#ffffff", ADD, None, "received and ADDED"),
    ("#ffffff", OVR, None, "received and OVERWRITTEN"),
]):
    x = 700 + (i // 3) * 240
    y = ly + (i % 3) * 28
    rect(x, y, 30, 20, fill, stroke, 3 if stroke in (ADD, OVR) else 1.5, 4, dash)
    t(x + 40, y + 15, label, 14, MUTED)

# phase bands
row_y = {}
y = 220
row_y["start"] = y
y += 186
p1 = y - 46
for s in range(W - 1):
    row_y[("rs", s)] = y
    y += 160
p1_end = y - 12
row_y["check"] = y
y += 156
p2 = y - 46
for s in range(W - 1):
    row_y[("ag", s)] = y
    y += 160
p2_end = y - 12
row_y["done"] = y

rect(20, p1, WIDTH - 40, p1_end - p1, "#f6f2fe", "#d9cdf6", 1.5, 14)
t(FX + FW / 2 + 18, p1 + 24, "Phase 1: reduce-scatter, W−1 = 2 steps (repeat for s = 0 … W−2)", 15, ADD, weight=700)
rect(20, p2, WIDTH - 40, p2_end - p2, "#effaf5", "#bfe6d4", 1.5, 14)
t(FX + FW / 2 + 18, p2 + 24, "Phase 2: all-gather, W−1 = 2 steps (repeat for s = 0 … W−2)", 15, OVR, weight=700)

def moves(y, sends):
    t(MX, y + 40, "this step:", 14, MUTED, weight=600)
    for r in range(W):
        t(MX, y + 64 + r * 22, f"r{r} → r{(r + 1) % W}: chunk {sends[r]}", 14, INK, mono=True)

# start
y0 = row_y["start"]
flowbox(y0, 100, "Start", [("state[r][c] = {r} for every chunk c", True), ("each rank holds only its own data", False)], "#f6f8fb")
grid(y0, steps[0][2])
arrow(FX + FW / 2, y0 + 100, FX + FW / 2, row_y[("rs", 0)] - 2)

for kind, s, st, recv, sends in steps[1:]:
    y = row_y[(kind, s)]
    if kind == "rs":
        flowbox(y, 120, f"Reduce-scatter step s = {s}", [
            ("send   chunk (r − s) mod W      → rank r+1", True),
            ("recv   chunk (r − s − 1) mod W  ← rank r−1", True),
            ("receiver ADDS it to its own copy", False)], "#ffffff", ADD)
        grid(y, st, recv, ADD, sends)
    else:
        flowbox(y, 120, f"All-gather step s = {s}", [
            ("send   chunk (r + 1 − s) mod W  → rank r+1", True),
            ("recv   chunk (r − s) mod W      ← rank r−1", True),
            ("receiver OVERWRITES its copy", False)], "#ffffff", OVR)
        grid(y, st, recv, OVR, sends)
    moves(y, sends)

# arrows between flow boxes
def nxt(a, b, h=120):
    arrow(FX + FW / 2, row_y[a] + h, FX + FW / 2, row_y[b] - 2)
nxt(("rs", 0), ("rs", 1))
nxt(("rs", 1), "check")
nxt("check", ("ag", 0), 90)
nxt(("ag", 0), ("ag", 1))
nxt(("ag", 1), "done")

yc = row_y["check"]
flowbox(yc, 90, "Check after phase 1", [("rank r owns the fully reduced chunk (r + 1) mod W", False)], "#f6f8fb")
own = ", ".join(f"rank {r} → chunk {c}" for c, r in sorted(owners.items(), key=lambda kv: kv[1]))
t(GX, yc + 34, "owners of the dark cells above:", 14, MUTED, weight=600)
t(GX, yc + 58, own, 14, INK, mono=True)

yd = row_y["done"]
flowbox(yd, 100, "Done", [("assert every cell = {0,1,2}: each rank counted once", False), (f"2(W−1) = {2 * (W - 1)} steps; each rank sent {2 * (W - 1)}/{W} · S", False)], "#f6f8fb")
t(GX, yd + 34, "result: every rank holds the full sum", 14, MUTED, weight=600)
t(GX, yd + 58, "in every chunk, i.e. all_reduce(SUM)", 14, MUTED, weight=600)

svg.append("</svg>")
OUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
print("wrote", OUT, "height used", yd + 100)
