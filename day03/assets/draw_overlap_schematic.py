"""Draw day03/assets/overlap_schematic.svg: one rank's step, Day 02 flat_sync vs DDP-style bucketed overlap.

Schematic, not to scale. Bucket sizes are the medium model's rebuilt layout (hook_order.py, Guide 1.4); the stream
placement agrees with the learner's a100_20261028T040220Z profiler traces (Guide Appendix A.5): flat_sync's all_reduce runs on
the compute stream, the hook-launched bucket all_reduces on a separate NCCL stream.

  python day03/assets/draw_overlap_schematic.py
"""
import sys
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "overlap_schematic.svg"
FONT = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
INK, MUTED = "#1f2d3d", "#5b6b7d"
FWD, BWD, OPT, COPY = "#c9d8f5", "#2f6fdb", "#e3e8ef", "#9aa8b8"
COMM, HOOK, TAIL = "#6b46c1", "#f2a33a", "#d64545"

WIDTH = 1200
X0 = 200        # left edge of the time axis (row labels sit left of it)
ROW_H = 34

svg = []


def t(x, y, s, size=15, color=INK, anchor="start", weight=400, italic=False):
    st = ' font-style="italic"' if italic else ""
    svg.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" '
               f'text-anchor="{anchor}"{st}>{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=1.5, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, color=MUTED, sw=2, dash=None, arrow=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#a)"' if arrow else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')


def block(x0, x1, y, fill, label="", text_color=INK, size=14, weight=600):
    rect(x0, y, x1 - x0, ROW_H, fill)
    if label:
        t((x0 + x1) / 2, y + ROW_H / 2 + 5, label, size, text_color, "middle", weight)


def row_label(y, name, sub=""):
    t(X0 - 14, y + ROW_H / 2 + (0 if sub else 5), name, 15, INK, "end", 600)
    if sub:
        t(X0 - 14, y + ROW_H / 2 + 15, sub, 12, MUTED, "end")


def bracket(x0, x1, y, label, color):
    svg.append(f'<path d="M{x0},{y + 8} L{x0},{y} L{x1},{y} L{x1},{y + 8}" stroke="{color}" stroke-width="2" fill="none"/>')
    t((x0 + x1) / 2, y - 8, label, 13, color, "middle", 600)


H = 728
svg.append(f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
           f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')
rect(0, 0, WIDTH, H, "#ffffff", rx=16)  # explicit background: readable on dark Markdown themes
t(WIDTH / 2, 42, "One training step on one rank: synchronous sync vs bucketed overlap", 22, INK, "middle", 700)
t(WIDTH / 2, 66, "medium model, 105 MiB of fp32 gradients · time flows right · schematic, not to scale", 14, MUTED,
  "middle")

# time stamps shared by both panels
F0, F1 = X0, X0 + 150          # forward
B0, B1 = F1, F1 + 480          # backward
READY = [B0 + 40, B0 + 190, B0 + 340, B1]  # moments when buckets 0..3 hold all their gradients

# ---------------------------------------------------------------- panel 1: Day 02 flat_sync
p1 = 92
rect(20, p1, WIDTH - 40, 196, "#f6f8fb", "#d5dde7", 1.5, 14)
t(40, p1 + 30, "Day 02  flat_sync", 18, INK, "start", 700)
t(40, p1 + 52, "backward ends, then ONE all_reduce; nothing overlaps", 14, MUTED)
y = p1 + 80
row_label(y, "compute stream", "(also runs the NCCL kernel)")
block(F0, F1, y, FWD, "forward")
block(B0, B1, y, BWD, "backward:  head → block5 … block0 → emb", "#ffffff")
c0 = B1
block(c0, c0 + 30, y, COPY, "cat", size=12)
block(c0 + 30, c0 + 110, y, COMM, "all_reduce", "#ffffff", 13)
block(c0 + 110, c0 + 160, y, COPY, "copy", size=12)
block(c0 + 160, c0 + 310, y, OPT, "optimizer")
line(B1, y - 14, B1, y + ROW_H + 14, TAIL, 2, "5 4")
t(B1, y + ROW_H + 32, "every gradient is ready only here: all communication is exposed", 13, TAIL, "middle", 600)
bracket(c0, c0 + 160, y - 6, "exposed sync", TAIL)

# ---------------------------------------------------------------- panel 2: DDP-style overlap
p2 = 306
rect(20, p2, WIDTH - 40, 328, "#f6f2fe", "#d9cdf6", 1.5, 14)
t(40, p2 + 30, "Day 03  MiniDDP / DDP", 18, INK, "start", 700)
t(40, p2 + 52, "a hook fires when each gradient is final; a full bucket is all-reduced on the NCCL stream "
               "while backward continues", 14, MUTED)
yc, yn = p2 + 124, p2 + 224
row_label(yc, "compute stream")
row_label(yn, "NCCL stream")
block(F0, F1, yc, FWD, "forward")
segs = [("head", B0, READY[0]), ("norm, block5–4", READY[0], READY[1]), ("block3–2", READY[1], READY[2]),
        ("block1–0, emb", READY[2], READY[3])]
for k, (name, a, b) in enumerate(segs):
    rect(a, yc, b - a, ROW_H, BWD)
    if b - a > 60:
        t((a + b) / 2, yc + ROW_H / 2 + 5, name, 13, "#ffffff", "middle", 600)
    if k:  # thin separators between bucket groups
        line(a, yc + 3, a, yc + ROW_H - 3, "#ffffff", 1.5)
for k in range(6, 470, 12):  # per-parameter hook work inside backward: the price of overlap
    rect(B0 + k, yc + ROW_H - 10, 3, 10, HOOK, rx=0)
t((B0 + B1) / 2, yc - 10, "backward  (orange ticks = per-parameter hooks)", 13, MUTED, "middle")

sizes = ["b0  16 MiB", "b1  28 MiB", "b2  28 MiB", "b3  32.5 MiB"]
widths = [52, 70, 70, 80]
ends = []
for k, (r, s, w) in enumerate(zip(READY, sizes, widths)):
    line(r, yc + ROW_H + 4, r + 2, yn - 4, MUTED, 1.5, "4 3", arrow=True)
    rect(r + 2, yn, w, ROW_H, COMM)
    t(r + 2 + w / 2, yn + ROW_H / 2 + 5, s.split()[0], 13, "#ffffff", "middle", 700)
    t(r + 2 + w / 2, yn + ROW_H + 18, s.split(maxsplit=1)[1], 12, MUTED, "middle")
    ends.append(r + 2 + w)
t(READY[0] + 2, yn + ROW_H + 36, "launched in bucket order, each as soon as it is full", 12, MUTED)

w0 = ends[-1]  # compute stream waits for the last bucket, then copies back
rect(B1, yc, w0 - B1, ROW_H, "#ffffff", MUTED, 1.5, dash="5 4")
t((B1 + w0) / 2, yc + ROW_H / 2 + 5, "wait", 13, MUTED, "middle", 600)
block(w0, w0 + 50, yc, COPY, "copy", size=12)
block(w0 + 50, w0 + 200, yc, OPT, "optimizer")
bracket(B1, w0 + 50, yc - 30, "exposed tail: last bucket + copy-back + rank skew", TAIL)
line(B1, yc - 14, B1, yn + ROW_H + 8, TAIL, 2, "5 4")

# ---------------------------------------------------------------- legend
ly = 660
items = [(FWD, "forward"), (BWD, "backward"), (HOOK, "hook work"), (COMM, "all_reduce (NCCL kernel)"),
         (COPY, "cat / copy back"), (OPT, "optimizer")]
x = 40
for color, label in items:
    rect(x, ly, 18, 18, color, rx=4)
    t(x + 26, ly + 14, label, 14, INK)
    x += 44 + len(label) * 7.6
t(40, ly + 48, "Overlap hides communication behind backward but adds per-parameter hook work inside it. "
               "On NVLink at local batch 8 that work cost more than it hid (Appendix A.4).", 13, MUTED)

svg.insert(0, f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{H}" viewBox="0 0 {WIDTH} {H}" '
              f'font-family="{FONT}">')
svg.append("</svg>")
OUT.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {OUT}")
