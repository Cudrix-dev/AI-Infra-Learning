"""Draw day07/assets/fsdp2_group_lifecycle.svg: what FSDP2 does to ONE parameter group (one fully_shard call) during
one microbatch, on W = 2 ranks.

Panel A: where the group's bytes live at each stage (rank 0 / rank 1).
Panel B: one rank's timeline: the hooks FSDP2 fires, what runs on which CUDA stream, and what this group holds in
         memory meanwhile.
Schematic, not to scale. Names and streams follow torch 2.14.1's _fully_shard/ sources (Guide Block 3) and the
`--sequence` output of analyze_fsdp_trace.py (all-gathers on one stream, reduce-scatters on another).

  python day07/assets/draw_fsdp2_group_lifecycle.py
"""
import sys
from html import escape
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fsdp2_group_lifecycle.svg"
FONT = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"
MONO = "'Cascadia Code', Consolas, 'SFMono-Regular', Menlo, monospace"
INK, MUTED, GRID, PANEL = "#1f2d3d", "#5b6b7d", "#d8dee6", "#f6f8fb"
P0, P1 = "#2f6fdb", "#8fb1ef"          # parameter shard 0 / shard 1
G0, G1 = "#e07a2f", "#f3b47e"          # local gradients of rank 0 / rank 1
GR = "#2e9e78"                          # reduced (averaged) gradient shard
FWD, BWD, COPY = "#c9d8f5", "#5b8fe6", "#9aa8b8"
AG, RS, HOOK = "#2f6fdb", "#e07a2f", "#6b46c1"

W, H = 1200, 770
svg = []


def t(x, y, s, size=14, color=INK, anchor="start", weight=400, italic=False, mono=False):
    st = ' font-style="italic"' if italic else ""
    svg.append(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
               f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{st}>{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke="none", sw=1.5, rx=5, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{o} stroke="{stroke}" '
               f'stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, color=MUTED, sw=2, dash=None, arrow=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#arr-{color[1:]})"' if arrow else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')


def bar_label(x, y, w, h, s, color="#ffffff", size=12, mono=True):
    t(x + w / 2, y + h / 2 + size * 0.36, s, size=size, color=color, anchor="middle", mono=mono)


# ------------------------------------------------------------------------------------------------ header
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
defs = ["<defs>"]
for c in (MUTED, AG, RS, HOOK, INK):
    defs.append(f'<marker id="arr-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
defs.append('<pattern id="hatch-ag" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="7" height="7" fill="#e6eefc"/><line x1="0" y1="0" x2="0" y2="7" stroke="{AG}" '
            'stroke-width="2.2"/></pattern>')
defs.append('<pattern id="hatch-rs" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="7" height="7" fill="#fdeedf"/><line x1="0" y1="0" x2="0" y2="7" stroke="{RS}" '
            'stroke-width="2.2"/></pattern>')
defs.append("</defs>")
svg += defs

t(30, 38, "FSDP2 = ZeRO-3 organised by module: the life of ONE group (one fully_shard call) in one microbatch",
  size=20, weight=700)
t(30, 62, "W = 2 ranks · group = the parameters one fully_shard(m) call claimed (e.g. blocks.3: 12 tensors, "
          "12.03 MiB) · P_g = the group's size", size=13, color=MUTED)

# ------------------------------------------------------------------------------------------------ panel A
rect(20, 78, W - 40, 222, PANEL, rx=10)
t(36, 104, "A   Where the group's bytes live", size=15, weight=700)
t(330, 104, "every element of a parameter, its gradient and its AdamW moments lives on exactly ONE rank "
            "between uses: no replicas", size=13, color=MUTED, italic=True)

COLS = [110, 400, 690, 980]
BW, BH = 180, 34
ROWS = [150, 212]
for i, y in enumerate(ROWS):
    t(36, y + 22, f"rank {i}", size=14, weight=600)

titles = [("1  stored between uses", "DTensor Shard(0): P_g / W per rank"),
          ("2  gathered for compute", "plain unsharded params on the module"),
          ("3  backward: local gradients", "full size, different on each rank"),
          ("4  after reduce-scatter", "sharded .grad = my slice of the mean")]
for x, (a, b) in zip(COLS, titles):
    t(x, 132, a, size=13, weight=700)
    t(x, 272, b, size=12, color=MUTED)

# stage 1: each rank holds its own half
for r, y in enumerate(ROWS):
    x = COLS[0]
    rect(x, y, BW, BH, "#ffffff", stroke=GRID, dash="5 4")
    hx = x if r == 0 else x + BW / 2
    rect(hx, y, BW / 2, BH, P0 if r == 0 else P1)
    bar_label(hx, y, BW / 2, BH, f"shard {r}", color="#ffffff" if r == 0 else INK)
# stage 2: both ranks hold the full group, made of both shards
for y in ROWS:
    x = COLS[1]
    rect(x, y, BW / 2, BH, P0)
    rect(x + BW / 2, y, BW / 2, BH, P1)
    bar_label(x, y, BW / 2, BH, "shard 0")
    bar_label(x + BW / 2, y, BW / 2, BH, "shard 1", color=INK)
# stage 3: local gradients differ
for r, y in enumerate(ROWS):
    rect(COLS[2], y, BW, BH, G0 if r == 0 else G1)
    bar_label(COLS[2], y, BW, BH, f"g{'⁰' if r == 0 else '¹'}  (rank {r}'s batch)", color="#ffffff" if r == 0 else INK)
# stage 4: reduced halves
for r, y in enumerate(ROWS):
    x = COLS[3]
    rect(x, y, BW, BH, "#ffffff", stroke=GRID, dash="5 4")
    hx = x if r == 0 else x + BW / 2
    rect(hx, y, BW / 2, BH, GR)
    bar_label(hx, y, BW / 2, BH, f"½(g⁰+g¹)[{r}]", size=11)

arrows = [("all-gather", "before fwd and bwd", AG), ("forward +", "backward", MUTED), ("reduce-scatter", "op = AVG", RS)]
for (a, b, c), x0, x1 in zip(arrows, [COLS[0] + BW, COLS[1] + BW, COLS[2] + BW], COLS[1:]):
    mid = (ROWS[0] + ROWS[1] + BH) / 2
    line(x0 + 10, mid, x1 - 10, mid, color=c, sw=2.5, arrow=True)
    t((x0 + x1) / 2, mid - 10, a, size=12, color=c, anchor="middle", weight=700)
    t((x0 + x1) / 2, mid + 20, b, size=11, color=c, anchor="middle")
t(COLS[3] + BW, 290, "AdamW then updates only this shard, with only this shard's exp_avg / exp_avg_sq",
  size=12, color=GR, anchor="end", weight=600)

# ------------------------------------------------------------------------------------------------ panel B
rect(20, 314, W - 40, 368, PANEL, rx=10)
t(36, 340, "B   One rank, one microbatch: hooks, CUDA streams, and what this group holds in memory", size=15,
  weight=700)

X0, X1 = 200, 1160
LANES = {"hooks": 372, "compute": 400, "ag": 444, "rs": 488}
LH = 30
for name, label, sub in [("compute", "compute stream", "default"), ("ag", "all-gather stream", "+ copy-in stream"),
                         ("rs", "reduce-scatter", "stream")]:
    y = LANES[name]
    t(36, y + 14, label, size=13, weight=600)
    t(36, y + 28, sub, size=11, color=MUTED)
    line(X0, y + LH / 2, X1, y + LH / 2, color=GRID, sw=1)

# time points (schematic)
x_pre, x_ag_end, x_co_end, x_post = 230, 290, 306, 450
x_gap_end = 590
x_bpre, x_bag_end, x_bco_end, x_bpost = 590, 650, 666, 880
x_ci_end, x_rs_end, x_final = 898, 980, 1110

# hooks (CPU side), as labelled ticks above the compute lane
hooks = [(x_pre, "fwd pre-hook", "unshard()"), (x_post, "fwd post-hook", "reshard()"),
         (x_bpre, "pre-backward", "tensor hook on output"), (x_bpost, "post-backward", "RegisterPostBackwardFn"),
         (x_final, "final callback", "wait for RS")]
for x, a, b in hooks:
    line(x, LANES["hooks"] - 4, x, LANES["rs"] + LH + 4, color=HOOK, sw=1.3, dash="3 3")
    svg.append(f'<circle cx="{x}" cy="{LANES["hooks"] - 6}" r="5" fill="{HOOK}"/>')
    t(x + 9, LANES["hooks"] - 12, a, size=12, color=HOOK, weight=700)
    t(x + 9, LANES["hooks"] + 2, b, size=10.5, color=HOOK, mono=True)

yc, ya, yr = LANES["compute"], LANES["ag"], LANES["rs"]
# forward
rect(x_pre, ya, x_ag_end - x_pre, LH, AG)
bar_label(x_pre, ya, x_ag_end - x_pre, LH, "AG", size=12)
rect(x_ag_end, yc, x_co_end - x_ag_end, LH, COPY, rx=3)
rect(x_co_end, yc, x_post - x_co_end, LH, FWD)
bar_label(x_co_end, yc, x_post - x_co_end, LH, "forward of this group", color=INK, size=12, mono=False)
# gap
rect(x_post + 8, yc, x_gap_end - x_post - 16, LH, "#ffffff", stroke=GRID, dash="5 4")
bar_label(x_post + 8, yc, x_gap_end - x_post - 16, LH, "other groups …", color=MUTED, size=12, mono=False)
# backward
rect(x_bpre, ya, x_bag_end - x_bpre, LH, AG)
bar_label(x_bpre, ya, x_bag_end - x_bpre, LH, "AG", size=12)
rect(x_bag_end, yc, x_bco_end - x_bag_end, LH, COPY, rx=3)
rect(x_bco_end, yc, x_bpost - x_bco_end, LH, BWD)
bar_label(x_bco_end, yc, x_bpost - x_bco_end, LH, "backward of this group", size=12, mono=False)
rect(x_bpost, yc, x_ci_end - x_bpost, LH, COPY, rx=3)
rect(x_ci_end, yr, x_rs_end - x_ci_end, LH, RS)
bar_label(x_ci_end, yr, x_rs_end - x_ci_end, LH, "RS", size=12)
rect(x_ci_end + 6, yc, x_final - x_ci_end - 12, LH, "#ffffff", stroke=GRID, dash="5 4")
bar_label(x_ci_end + 6, yc, x_final - x_ci_end - 12, LH, "previous group's backward", color=MUTED, size=11.5,
          mono=False)
t(x_ag_end + 8, yc - 6, "copy-out", size=10.5, color=MUTED)
t(x_bpost + 2, yc - 6, "chunk_cat", size=10.5, color=MUTED)

# memory lane
MY = 536
t(36, MY + 14, "memory held", size=13, weight=600)
t(36, MY + 28, "for this group", size=11, color=MUTED)
BAND = 22
y_shard, y_full, y_buf, y_grad = MY + 98, MY + 72, MY + 46, MY + 20
# always: the parameter shard
rect(X0, y_shard, X1 - X0, BAND, P1, rx=3)
t(X0 + 8, y_shard + 15, "parameter shard  P_g / W  (always)", size=11.5, color=INK, mono=True)
# unsharded params (fwd) and the all-gather buffer
rect(x_pre, y_full, x_post - x_pre, BAND, P0, rx=3)
bar_label(x_pre, y_full, x_post - x_pre, BAND, "unsharded params  P_g", size=11)
rect(x_pre, y_buf, x_co_end - x_pre + 40, BAND, "url(#hatch-ag)", stroke=AG, sw=1, rx=3)
# kept-if-nrf outline across the gap
rect(x_post, y_full, x_bpre - x_post, BAND, "none", stroke=P0, sw=1.5, rx=3, dash="5 4")
gx = (x_post + x_bpre) / 2
t(gx, y_grad + 16, "kept gathered if", size=11, color=P0, anchor="middle", italic=True)
t(gx, y_buf + 9, "reshard_after_forward", size=10.5, color=P0, anchor="middle", mono=True)
t(gx, y_buf + 22, "=False (nrf, root)", size=10.5, color=P0, anchor="middle", mono=True)
# unsharded params (bwd)
rect(x_bpre, y_full, x_bpost - x_bpre, BAND, P0, rx=3)
bar_label(x_bpre, y_full, x_bpost - x_bpre, BAND, "unsharded params again  P_g", size=11)
rect(x_bpre, y_buf, x_bco_end - x_bpre + 20, BAND, "url(#hatch-ag)", stroke=AG, sw=1, rx=3)
# unsharded gradients, RS input, sharded gradient
svg.append(f'<polygon points="{x_bco_end},{y_grad + BAND} {x_bpost},{y_grad} {x_bpost},{y_grad + BAND}" '
           f'fill="{G0}" fill-opacity="0.85"/>')
rect(x_bpost, y_grad, x_ci_end - x_bpost, BAND, G0, rx=2)
rect(x_bpost, y_buf, x_rs_end - x_bpost, BAND, "url(#hatch-rs)", stroke=RS, sw=1, rx=3)
rect(x_rs_end, y_full, X1 - x_rs_end, BAND, GR, rx=3)
bar_label(x_rs_end, y_full, X1 - x_rs_end, BAND, "sharded grad  P_g / W", size=11)

# memory legend
lx, ly = X0, MY + 136
items = [("url(#hatch-ag)", AG, "all-gather buffer (freed after the NEXT copy-out in fwd)"),
         (G0, None, "unsharded gradients (grow during backward)"),
         ("url(#hatch-rs)", RS, "reduce-scatter input")]
for fill, stroke, label in items:
    rect(lx, ly - 11, 22, 14, fill, stroke=stroke or "none", sw=1, rx=2)
    t(lx + 30, ly, label, size=11.5, color=MUTED)
    lx += 34 + 7.0 * len(label)

# callouts
t(x_bpost - 4, y_grad - 6, "nosync: these stay; no RS until the last microbatch", size=10.5, color=RS,
  anchor="end", italic=True)

# ------------------------------------------------------------------------------------------------ footer cards
cards = [("No replicas", "Nothing to compare across ranks: a wrong model", "cannot diverge. Check gradients and state."),
         ("No initial broadcast", "fully_shard keeps chunk r of RANK r's tensor.",
          "Broadcast (or seed identically) first."),
         ("Optimizer after fully_shard", "AdamW over model.parameters() = DTensor shards;",
          "its state is sharded with no extra code.")]
cw = (W - 40 - 2 * 16) / 3
for i, (a, b, c) in enumerate(cards):
    x = 20 + i * (cw + 16)
    rect(x, 696, cw, 62, "#ffffff", stroke=GRID, rx=10)
    rect(x, 696, 6, 62, HOOK, rx=3)
    t(x + 18, 718, a, size=14, weight=700)
    t(x + 18, 736, b, size=11.5, color=MUTED)
    t(x + 18, 751, c, size=11.5, color=MUTED)

svg.insert(0, f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
svg.append("</svg>")
OUT.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {OUT}")
