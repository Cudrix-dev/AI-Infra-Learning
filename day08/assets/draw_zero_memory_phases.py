"""Draw day08/assets/zero_memory_phases.svg: what ONE rank holds under DeepSpeed ZeRO-1, -2 and -3 at seven moments of
one optimizer step (FP32 model, client AdamW, K >= 2 microbatches, W ranks, sized buckets).

Rows: stage 1, 2, 3. Columns: start of step -> forward mb 0 -> backward mb 0 -> between microbatches ->
end of the boundary backward (mb K-1) -> optimizer step -> after the step.
Block labels are in units of P (model parameters in FP32 = 4 Psi bytes) and W. Heights are drawn for W = 2 (today's
run), and activations A and gathered modules G are not to scale. The ledger follows guide sections 1.2-1.5 and the
pre-test census (Appendix A.3, notes B6).

  python day08/assets/draw_zero_memory_phases.py
"""
import sys
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "zero_memory_phases.svg"
FONT = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
MONO = "'Cascadia Code', Consolas, 'SFMono-Regular', Menlo, monospace"
INK, MUTED, GRID, PANEL = "#1f2d3d", "#5b6b7d", "#d8dee6", "#f6f8fb"
PARAM, PART = "#2f6fdb", "#8fb1ef"      # full parameters / this rank's parameter partition
MASTER, STATE = "#6b46c1", "#9aa8b8"    # fp32 master partition / AdamW exp_avg + exp_avg_sq partition
GRAD, GPART = "#e07a2f", "#2e9e78"      # full local p.grad / this rank's reduced gradient partition
FLIGHT = "#f3b47e"                      # gradients in flight (copied into the bucket, then freed)
BUCKET, ACT = "#c0392b", "#f2d16b"      # reduction bucket / activations
AG, RS, AR = "#2f6fdb", "#e07a2f", "#c0392b"

W, H = 1240, 1297
U = 13            # px per unit; P = 4 units, so P/W = 2 units at W = 2
P, PW, B, A, G = 4, 2, 1.2, 6, 2
svg = []


def t(x, y, s, size=13, color=INK, anchor="start", weight=400, italic=False, mono=False):
    st = ' font-style="italic"' if italic else ""
    svg.append(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
               f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{st}>{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=1.2, rx=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{sw}"{d}/>')


# A block is (label, units, fill, text colour, dashed outline?)
def S():  # optimizer state every stage keeps: fp32 master partition + AdamW moments, both sharded
    return [("AdamW 2P/W", 2 * PW, STATE, INK, False), ("master P/W", PW, MASTER, "#ffffff", False)]


def full_p(note="params P"):
    return [(note, P, PARAM, "#ffffff", False)]


ACT_FULL = ("A", A, ACT, INK, True)
ACT_HALF = ("A (freeing)", A / 2, ACT, INK, True)
BKT = ("B", B, BUCKET, "#ffffff", False)

COLS = [
    ("1  start of step", "grads freed"),
    ("2  forward, mb 0", "activation peak"),
    ("3  backward, mb 0", "mid-backward"),
    ("4  between", "microbatches"),
    ("5  end of boundary", "backward (mb K−1)"),
    ("6  optimizer step", "engine.step()"),
    ("7  after the step", "cleanup"),
]

# Each stage: (name, subtitle, [blocks per column], [total per column], [caption per column: (text, colour)])
STAGES = [
    ("ZeRO-1", "shards optimizer state", [
        S() + full_p(),
        S() + full_p() + [ACT_FULL],
        S() + full_p() + [("p.grad → P", P, GRAD, "#ffffff", False), BKT, ACT_HALF],
        S() + full_p() + [("p.grad P", P, GRAD, "#ffffff", False), BKT],
        S() + full_p() + [("avg grad P/W", PW, GPART, "#ffffff", False)],
        S() + full_p("params P ←") + [("avg grad P/W", PW, GPART, "#ffffff", False)],
        S() + full_p(),
    ], [
        "P + 3P/W", "P + 3P/W + A", "≤ 2P + 3P/W + B + A", "2P + 3P/W + B", "P + 4P/W", "P + 4P/W", "P + 3P/W",
    ], [
        ("", MUTED), ("no communication", MUTED), ("p.grad accumulates locally;|1st hook allocates B", MUTED),
        ("full gradient + B|carried to next mb", MUTED), ("5 bucket all-reduces,|keep own 1/W; free p.grad, B", AR),
        ("AdamW on 1/W, copy in,|all-gather params P", AG), ("free avg grad", MUTED),
    ]),
    ("ZeRO-2", "+ shards gradients", [
        S() + full_p(),
        S() + full_p() + [ACT_FULL],
        S() + full_p() + [("avg grad P/W", PW, GPART, "#ffffff", False), ("g in flight", B, FLIGHT, INK, False),
                          BKT, ACT_HALF],
        S() + full_p() + [("avg grad P/W", PW, GPART, "#ffffff", False)],
        S() + full_p() + [("avg grad P/W", PW, GPART, "#ffffff", False)],
        S() + full_p("params P ←") + [("avg grad P/W", PW, GPART, "#ffffff", False)],
        S() + full_p(),
    ], [
        "P + 3P/W", "P + 3P/W + A", "≤ P + 4P/W + B + g + A", "P + 4P/W", "P + 4P/W", "P + 4P/W", "P + 3P/W",
    ], [
        ("", MUTED), ("no communication", MUTED), ("bucket all-reduce EVERY mb,|keep own 1/W, free p.grad", AR),
        ("only my partition|survives; B released", MUTED), ("same as every mb:|nothing extra here", MUTED),
        ("AdamW on 1/W, copy in,|all-gather params P", AG), ("free avg grad", MUTED),
    ]),
    ("ZeRO-3", "+ shards parameters", [
        S() + [("part. P/W", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT],
        S() + [("part. P/W", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT,
               ("G gathered", G, "url(#hatch-ag)", INK, True), ACT_FULL],
        S() + [("part. P/W", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT,
               ("G again", G, "url(#hatch-ag)", INK, True), ("g in flight", B, FLIGHT, INK, False),
               ACT_HALF],
        S() + [("part. P/W", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT],
        S() + [("part. P/W", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT],
        S() + [("part. P/W ←", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT],
        S() + [("part. P/W", PW, PART, INK, False), ("grad buf P/W", PW, GPART, "#ffffff", False), BKT],
    ], [
        "5P/W + B", "5P/W + B + G + A", "≤ 5P/W + B + G + g + A", "5P/W + B", "5P/W + B", "5P/W + B", "5P/W + B",
    ], [
        ("buffer and bucket|are persistent", MUTED), ("all-gather per module|(+prefetch), release after", AG),
        ("re-gather; reduce-scatter|bucket into grad buf", RS), ("nothing new", MUTED),
        ("same as every mb", MUTED), ("AdamW on 1/W; all-gather|each persistent param", AG),
        ("zero grad buf, keep it", MUTED),
    ]),
]

# ------------------------------------------------------------------------------------------------ canvas, defs
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
svg.append('<defs><pattern id="hatch-ag" width="7" height="7" patternUnits="userSpaceOnUse" '
           f'patternTransform="rotate(45)"><rect width="7" height="7" fill="#e6eefc"/><line x1="0" y1="0" x2="0" '
           f'y2="7" stroke="{AG}" stroke-width="2.2"/></pattern>'
           f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>')

t(30, 38, "DeepSpeed ZeRO-1 / 2 / 3: what ONE rank holds during one optimizer step of K microbatches", size=20,
  weight=700)
t(30, 62, "FP32 model, client AdamW, W ranks, sized buckets · P = model parameters in FP32 (4Ψ bytes) · "
          "B = reduction bucket · A = activations of one microbatch · G = gathered module parameters", size=13,
  color=MUTED)
t(30, 81, "Heights drawn for W = 2 (today's run); A and G not to scale. Microbatches 1 … K−2 repeat columns 2–4 "
          "on top of what column 4 carries.", size=13, color=MUTED, italic=True)

LEFT, CW, BW = 160, 152, 78
X0 = [LEFT + i * CW for i in range(len(COLS))]

# column headers and the step timeline
for x, (a, b) in zip(X0, COLS):
    t(x + CW / 2, 112, a, size=13, weight=700, anchor="middle")
    t(x + CW / 2, 128, b, size=12, color=MUTED, anchor="middle")
svg.append(f'<line x1="{LEFT + 10}" y1="140" x2="{LEFT + len(COLS) * CW - 10}" y2="140" stroke="{MUTED}" '
           f'stroke-width="1.5" marker-end="url(#arr)"/>')

ROW_TOP, ROW_H = 150, 345
for r, (name, sub, cols, totals, caps) in enumerate(STAGES):
    top = ROW_TOP + r * ROW_H
    rect(16, top, W - 32, ROW_H - 12, PANEL, rx=10)
    base = top + ROW_H - 52  # bar baseline
    t(30, top + 30, name, size=18, weight=700)
    t(30, top + 50, sub, size=12.5, color=MUTED)
    # dotted reference line: this stage's start-of-step height, to see what each phase adds
    start_h = sum(b[1] for b in cols[0]) * U
    svg.append(f'<line x1="{LEFT}" y1="{base - start_h}" x2="{LEFT + len(COLS) * CW}" y2="{base - start_h}" '
               f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 4"/>')
    t(30, base - start_h + 4, "start-of-step level", size=11, color=MUTED, italic=True)
    for x, blocks, total, (cap, capc) in zip(X0, cols, totals, caps):
        bx = x + (CW - BW) / 2
        y = base
        for label, units, fill, tc, dashed in blocks:
            h = units * U
            y -= h
            rect(bx, y, BW, h - 1, fill, stroke=MUTED if dashed else "none", dash="4 3" if dashed else None)
            if h >= 12:
                if fill.startswith("url("):  # white halo so the label stays readable on the hatch
                    svg.append(f'<text x="{bx + BW / 2}" y="{y + h / 2 + 4}" font-family="{MONO}" font-size="10.5" '
                               f'text-anchor="middle" fill="{tc}" stroke="#ffffff" stroke-width="3.5" '
                               f'paint-order="stroke">{escape(label)}</text>')
                else:
                    t(bx + BW / 2, y + h / 2 + 4, label, size=10.5 if len(label) > 10 else 11, color=tc,
                      anchor="middle", mono=True)
        t(x + CW / 2, y - 7, total, size=11.5 if len(total) <= 18 else 10, weight=700, anchor="middle", mono=True)
        for i, line in enumerate(cap.split("|") if cap else []):
            t(x + CW / 2, base + 17 + 14 * i, line, size=10.5, color=capc, anchor="middle")

# ------------------------------------------------------------------------------------------------ legend + key
ly = ROW_TOP + 3 * ROW_H + 8
legend = [("full params P", PARAM), ("param partition P/W", PART), ("gathered modules G", "url(#hatch-ag)"),
          ("fp32 master P/W", MASTER), ("AdamW state 2P/W", STATE), ("full local p.grad P", GRAD),
          ("reduced grad partition P/W", GPART), ("g: grads in flight", FLIGHT), ("bucket B", BUCKET),
          ("activations A", ACT)]
for i, (label, fill) in enumerate(legend):
    lx, yy = 30 + (i % 5) * 238, ly + (i // 5) * 22
    rect(lx, yy, 14, 14, fill, stroke=GRID)
    t(lx + 20, yy + 12, label, size=11.5)
ly += 22
t(30, ly + 40, "Start of step:  z3 − z1 = (5P/W + B) − (P + 3P/W) = B − P(1 − 2/W).   At W = 2 stage 3 starts B higher "
               "(pre-test: +25.2 MiB over z2); for W > 2 it starts lower once P(1 − 2/W) > B.", size=13,
  weight=600)
t(30, ly + 60, "Between microbatches z1 carries P + B, z2 carries P/W, z3 carries nothing extra; arrows ← mark the "
               "tensor the step rewrites (flat params, or the partition in z3).", size=12.5, color=MUTED)

svg.insert(0, f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
svg.append("</svg>")
OUT.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {OUT}")
