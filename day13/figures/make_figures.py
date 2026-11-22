"""Generate day13/figures/d13_sp_*.svg (sequence-parallelism diagrams for GUIDE §4).
Usage: python day13/figures/make_figures.py day13/figures"""
import html
import re
import sys
from pathlib import Path

FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"
INK, GREY, MUTED = "#1f2937", "#6b7280", "#94a3b8"
C = {
    "rep": ("#e8f0fe", "#3b6fd4"),   # replicated: full tensor, same on both ranks
    "hid": ("#e3f5e9", "#2f9e5b"),   # TP region: hidden dim split (Day 11)
    "seq": ("#f1eafd", "#7048c8"),   # SP region: sequence split
    "part": ("#fff1dc", "#e3891b"),  # partial sum, must be reduced
    "AR": ("#fde8f3", "#c0267a"),
    "AG": ("#ddf4f5", "#0b7f86"),
    "RS": ("#fde2e2", "#b91c1c"),
    "ID": ("#f8fafc", "#94a3b8"),
    "note": ("#f8fafc", "#cbd5e1"),
    "empty": ("#ffffff", "#cbd5e1"),
}


def rich(s):
    """Escape, then turn _{x} into a subscript and ^{x} into a superscript."""
    s = html.escape(s, quote=False)
    s = re.sub(r"_\{([^}]*)\}", r'<tspan baseline-shift="sub" font-size="75%">\1</tspan>', s)
    s = re.sub(r"\^\{([^}]*)\}", r'<tspan baseline-shift="super" font-size="75%">\1</tspan>', s)
    return s


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.p = w, h, []
        self.p.append(f'<rect width="{w}" height="{h}" rx="16" fill="#ffffff"/>')

    def text(self, x, y, s, size=13, fill=INK, anchor="start", weight="normal", mono=False, italic=False):
        fam = MONO if mono else FONT
        st = ' font-style="italic"' if italic else ""
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}" '
                      f'text-anchor="{anchor}" font-weight="{weight}" xml:space="preserve"{st}>{rich(s)}</text>')

    def rect(self, x, y, w, h, kind="note", rx=8, dash=False, sw=1.5, fill=None, stroke=None):
        f, s = C[kind]
        f, s = fill or f, stroke or s
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{f}" '
                      f'stroke="{s}" stroke-width="{sw}"{d}/>')

    def lines(self, cx, y, lines, gap=16):
        """lines: list of (text, dict(size, weight, mono, fill, italic))."""
        for i, (t, kw) in enumerate(lines):
            self.text(cx, y + i * gap, t, anchor="middle", **kw)

    def box(self, x, y, w, h, kind, lines, gap=16, dash=False, top=None):
        self.rect(x, y, w, h, kind, dash=dash)
        n = len(lines)
        y0 = top if top is not None else y + h / 2 - (n - 1) * gap / 2 + 4.5
        self.lines(x + w / 2, y0, lines, gap)

    def arrow(self, x1, y1, x2, y2, color=GREY, sw=1.6, dash=False, head=True):
        mid = "ah" + color.strip("#")
        d = ' stroke-dasharray="5 4"' if dash else ""
        m = f' marker-end="url(#{mid})"' if head else ""
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                      f'stroke-width="{sw}"{d}{m}/>')

    def save(self, path):
        colors = {GREY, INK, MUTED} | {v[1] for v in C.values()}
        defs = "".join(
            f'<marker id="ah{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in sorted(colors))
        body = "\n".join(self.p)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}"><defs>{defs}</defs>\n{body}\n</svg>\n', encoding="utf-8")


def T(t, size=13, weight="normal", mono=False, fill=INK, italic=False):
    return (t, dict(size=size, weight=weight, mono=mono, fill=fill, italic=italic))


def B(t, size=13, fill=INK):
    return T(t, size=size, weight="700", fill=fill)


def M(t, size=12, fill=INK):
    return T(t, size=size, mono=True, fill=fill)


def G(t, size=11):
    return T(t, size=size, fill=GREY)


def legend(s, x, y, items):
    for kind, label in items:
        s.rect(x, y, 18, 18, kind, rx=4)
        s.text(x + 25, y + 14, label, size=13)
        x += 25 + 7.4 * len(label) + 26


def title(s, t, sub):
    s.text(24, 40, t, size=22, weight="700")
    s.text(24, 64, sub, size=13, fill=GREY)


def coll_bar(s, x, y, w, h, kind, lines, dash=False):
    """A collective drawn as a tall bar across both rank lanes."""
    s.rect(x, y, w, h, kind, rx=10, sw=2, dash=dash)
    s.lines(x + w / 2, y + h / 2 - (len(lines) - 1) * 8 + 4.5, lines)


LEG_STATE = [("rep", "replicated (full, same on both ranks)"), ("seq", "sequence-sharded (own tokens)"),
             ("hid", "hidden-sharded (TP region)"), ("part", "partial sum (needs a reduction)")]


# ---------------------------------------------------------------- figure 1: the three collectives
def tensor(s, x, y, w, h, top, bottom, label_top, label_bot):
    """A [256, 2, 512] tensor drawn as two sequence halves (tokens 0-127 on top, 128-255 below)."""
    for i, (kind, lab) in enumerate(((top, label_top), (bottom, label_bot))):
        yy = y + i * h / 2
        if kind is None:
            s.rect(x, yy, w, h / 2, "empty", rx=4, dash=True)
            s.text(x + w / 2, yy + h / 4 + 4, "(not held)", size=11, fill=MUTED, anchor="middle")
        else:
            s.rect(x, yy, w, h / 2, kind, rx=4)
            s.text(x + w / 2, yy + h / 4 + 4.5, lab, size=12, anchor="middle")


def fig_collectives(out):
    W, H = 1484, 600
    s = Svg(W, H)
    title(s, "The three TP collectives on two ranks, drawn on one activation [s, b, h] = [256, 2, 512]",
          "Each tensor is drawn as two halves of the sequence: tokens 0–127 (top) and 128–255 (bottom). "
          "P_{0}, P_{1} = partial sums from a row-parallel matmul; X = an activation.")
    legend(s, 24, 82, LEG_STATE)
    panels = [
        ("AR", "ALL-REDUCE", "everyone gets the full sum",
         [("part", "part", "P_{0}", "P_{0}"), ("part", "part", "P_{1}", "P_{1}")],
         [("rep", "rep", "P_{0}+P_{1}", "P_{0}+P_{1}"), ("rep", "rep", "P_{0}+P_{1}", "P_{0}+P_{1}")],
         "in [256,2,512]  →  out [256,2,512]",
         ["fwd: after a row-parallel linear (g), SP off", "bwd: input grad of a column-parallel", "      linear (f), SP off",
          "also: embedding, cross-entropy, and", "      LN grads once per step (SP on)"]),
        ("RS", "REDUCE-SCATTER", "sum, then keep only my tokens",
         [("part", "part", "P_{0}", "P_{0}"), ("part", "part", "P_{1}", "P_{1}")],
         [("seq", None, "P_{0}+P_{1}", ""), (None, "seq", "", "P_{0}+P_{1}")],
         "in [256,2,512]  →  out [128,2,512]",
         ["fwd: after a row-parallel linear, SP on", "      (layers.py:1476)", "bwd: input grad of a column-parallel",
          "      linear, SP on (layers.py:650)"]),
        ("AG", "ALL-GATHER", "concatenate everyone's tokens",
         [("seq", None, "X[0:128]", ""), (None, "seq", "", "X[128:256]")],
         [("rep", "rep", "X[0:128]", "X[128:256]"), ("rep", "rep", "X[0:128]", "X[128:256]")],
         "in [128,2,512]  →  out [256,2,512]",
         ["fwd: input of a column-parallel linear,", "      SP on (layers.py:571)", "bwd: gradient of a forward reduce-scatter",
          "      (mappings.py:377), and the re-gather", "      of x for the weight grad (layers.py:617)"]),
    ]
    pw = 352
    for i, (kind, name, what, before, after, shape, uses) in enumerate(panels):
        x0 = 24 + i * (pw + 16)
        s.rect(x0, 112, pw, 470, "note", rx=12, fill="#ffffff", stroke="#e2e8f0")
        s.rect(x0, 112, pw, 46, kind, rx=12, sw=2)
        s.text(x0 + pw / 2, 133, name, size=16, weight="700", fill=C[kind][1], anchor="middle")
        s.text(x0 + pw / 2, 151, what, size=12, fill=INK, anchor="middle")
        tw, th = 118, 96
        xs = [x0 + 40, x0 + pw - 40 - tw]
        s.text(x0 + 14, 182, "before", size=12, fill=GREY, weight="700")
        for r in range(2):
            s.text(xs[r] + tw / 2, 182, f"rank {r}", size=12, fill=GREY, anchor="middle")
            tensor(s, xs[r], 190, tw, th, before[r][0], before[r][1], before[r][2], before[r][3])
        # collective band
        s.rect(x0 + 20, 300, pw - 40, 30, kind, rx=8, sw=2)
        s.text(x0 + pw / 2, 320, shape, size=12, mono=True, anchor="middle")
        for r in range(2):
            s.arrow(xs[r] + tw / 2, 288, xs[r] + tw / 2, 299, color=C[kind][1])
            s.arrow(xs[r] + tw / 2, 331, xs[r] + tw / 2, 346, color=C[kind][1])
        s.text(x0 + 14, 344, "after", size=12, fill=GREY, weight="700")
        for r in range(2):
            tensor(s, xs[r], 350, tw, th, after[r][0], after[r][1], after[r][2], after[r][3])
        s.text(x0 + 14, 470, "Used here for", size=12, weight="700")
        for j, u in enumerate(uses):
            s.text(x0 + 14, 489 + j * 17, u, size=12)
    # rule card
    x0 = 24 + 3 * (pw + 16)
    rw = W - 24 - x0
    s.rect(x0, 112, rw, 470, "note", rx=12, fill="#f8fafc", stroke="#cbd5e1")
    s.text(x0 + 16, 138, "Two facts to remember", size=15, weight="700")
    s.text(x0 + 16, 166, "1.  ALL-REDUCE = REDUCE-SCATTER, then", size=13, weight="700")
    s.text(x0 + 16, 184, "     ALL-GATHER", size=13, weight="700")
    s.text(x0 + 16, 204, "A ring all-reduce runs exactly these two", size=12)
    s.text(x0 + 16, 220, "phases. SP just stops halfway, does the", size=12)
    s.text(x0 + 16, 236, "cheap per-token work (LayerNorm, residual)", size=12)
    s.text(x0 + 16, 252, "on half the tokens, and finishes later.", size=12)
    s.text(x0 + 16, 268, "Same bytes on the wire.", size=12)
    s.text(x0 + 16, 300, "2.  Every forward op has a mirror", size=13, weight="700")
    s.text(x0 + 16, 318, "     in backward", size=13, weight="700")
    rows = [("identity (f)", "ID", "ALL-REDUCE", "AR"), ("ALL-REDUCE (g)", "AR", "identity", "ID"),
            ("ALL-GATHER", "AG", "REDUCE-SCATTER", "RS"), ("REDUCE-SCATTER", "RS", "ALL-GATHER", "AG"),
            ("split (no comm)", "ID", "ALL-GATHER", "AG")]
    s.text(x0 + 16, 344, "forward", size=11, fill=GREY, weight="700")
    s.text(x0 + rw - 16, 344, "backward", size=11, fill=GREY, weight="700", anchor="end")
    for j, (a, ka, b, kb) in enumerate(rows):
        yy = 352 + j * 30
        bw = 128
        s.rect(x0 + 12, yy, bw, 24, ka, rx=6, dash=(ka == "ID"))
        s.text(x0 + 12 + bw / 2, yy + 16.5, a, size=11, anchor="middle", weight="700", fill=C[ka][1] if ka != "ID" else GREY)
        s.arrow(x0 + 12 + bw + 4, yy + 12, x0 + rw - 12 - bw - 6, yy + 12, color=GREY)
        s.rect(x0 + rw - 12 - bw, yy, bw, 24, kb, rx=6, dash=(kb == "ID"))
        s.text(x0 + rw - 12 - bw / 2, yy + 16.5, b, size=11, anchor="middle", weight="700", fill=C[kb][1] if kb != "ID" else GREY)
    s.text(x0 + 16, 520, "Why: the gradient of \"copy to everyone\" is", size=12)
    s.text(x0 + 16, 536, "\"sum everyone's gradients\"; the gradient of", size=12)
    s.text(x0 + 16, 552, "\"concatenate\" is \"take my slice of the sum\".", size=12)
    s.save(out / "d13_sp_collectives.svg")


# ---------------------------------------------------------------- figure 2: forward of one block
def fig_forward(out):
    W, H = 1484, 790
    s = Svg(W, H)
    title(s, "Forward of one MLP block at TP = 2: SP off vs SP on",
          "s = 256 tokens, b = 2 sequences, h = 512, ffn = 2048. Matmuls are written on the flattened (s·b) × h = "
          "512 × 512 activation. Both rank lanes are shown; a tall bar is a collective between them.")
    legend(s, 24, 82, LEG_STATE)
    # columns
    cx = dict(inp=(104, 150), ln=(276, 128), enter=(428, 128), fc1=(580, 200), act=(804, 126),
              fc2=(954, 196), leave=(1174, 132), out=(1330, 130))
    # region brackets
    def bracket(x1, x2, y, label, color):
        s.p.append(f'<path d="M{x1},{y + 8} L{x1},{y} L{x2},{y} L{x2},{y + 8}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        s.text((x1 + x2) / 2, y - 6, label, size=12, weight="700", fill=color, anchor="middle")
    bracket(cx["inp"][0], cx["ln"][0] + cx["ln"][1], 134, "outside the TP region (residual stream)", "#475569")
    bracket(cx["fc1"][0], cx["fc2"][0] + cx["fc2"][1], 134, "TP region: hidden dim split, exactly Day 11  (identical with SP on or off)", C["hid"][1])
    bracket(cx["out"][0], cx["out"][0] + cx["out"][1], 134, "outside again", "#475569")
    s.text(cx["enter"][0] + cx["enter"][1] / 2, 128, "ENTER", size=12, weight="700", fill="#475569", anchor="middle")
    s.text(cx["leave"][0] + cx["leave"][1] / 2, 128, "LEAVE", size=12, weight="700", fill="#475569", anchor="middle")

    def panel(y0, sp):
        name = "SP on  (--sp)" if sp else "SP off  (main run)"
        s.text(24, y0 + 6, name, size=16, weight="700", fill=C["seq"][1] if sp else C["rep"][1])
        lane_y = [y0 + 22, y0 + 116]
        lh = 84
        for r in range(2):
            s.text(24, lane_y[r] + lh / 2 + 5, f"rank {r}", size=13, weight="700", fill=GREY)
        toks = ["tokens 0–127", "tokens 128–255"]
        for r in range(2):
            y = lane_y[r]
            if sp:
                s.box(*cx["inp"][:1], y, cx["inp"][1], lh, "seq", [B("x (shard)"), M("[128, 2, 512]"), G(toks[r])])
                s.box(cx["ln"][0], y, cx["ln"][1], lh, "seq", [B("LayerNorm"), M("[128, 2, 512]"), G("own tokens only")])
            else:
                s.box(cx["inp"][0], y, cx["inp"][1], lh, "rep", [B("x"), M("[256, 2, 512]"), G("all 256 tokens")])
                s.box(cx["ln"][0], y, cx["ln"][1], lh, "rep", [B("LayerNorm"), M("[256, 2, 512]"), G("same on both")])
            s.box(cx["fc1"][0], y, cx["fc1"][1], lh, "hid",
                  [B("fc1 (column-parallel)"), T("h_{r} = x·A1_{r} + b1_{r}", 13),
                   M("(512×512)·(512×1024)", 11), M("→ [256, 2, 1024]")], gap=17)
            s.box(cx["act"][0], y, cx["act"][1], lh, "hid", [B("GELU"), T("a_{r} = gelu(h_{r})"), M("[256, 2, 1024]")])
            s.box(cx["fc2"][0], y, cx["fc2"][1], lh, "part",
                  [B("fc2 (row-parallel)"), T("z_{r} = a_{r}·A2_{r}", 13), M("(512×1024)·(1024×512)", 11),
                   M("→ [256, 2, 512] partial")], gap=17)
            if sp:
                s.box(cx["out"][0], y, cx["out"][1], lh, "seq", [B("y = x + z + b2"), M("[128, 2, 512]"), G(toks[r])])
            else:
                s.box(cx["out"][0], y, cx["out"][1], lh, "rep", [B("y = x + z + b2"), M("[256, 2, 512]"), G("same on both")])
            for a, b in (("inp", "ln"), ("ln", "enter"), ("enter", "fc1"), ("fc1", "act"), ("act", "fc2"),
                         ("fc2", "leave"), ("leave", "out")):
                xa = cx[a][0] + cx[a][1]
                s.arrow(xa + 1, y + lh / 2, cx[b][0] - 2, y + lh / 2)
        top, hgt = lane_y[0] - 6, lane_y[1] + lh + 6 - (lane_y[0] - 6)
        if sp:
            coll_bar(s, cx["enter"][0], top, cx["enter"][1], hgt, "AG",
                     [B("ALL-GATHER", 14, C["AG"][1]), T("concat along s"), M("[128,2,512]", 11), M("→ [256,2,512]", 11),
                      G("layers.py:571"), G("(sync, compute stream)")])
            coll_bar(s, cx["leave"][0], top, cx["leave"][1], hgt, "RS",
                     [B("REDUCE-SCATTER", 13, C["RS"][1]), T("sum, keep my tokens"), M("[256,2,512]", 11),
                      M("→ [128,2,512]", 11), G("layers.py:1476"), G("(sync, compute stream)")])
        else:
            coll_bar(s, cx["enter"][0], top, cx["enter"][1], hgt, "ID",
                     [B("f = identity", 13, GREY), T("x is already full"), T("on both ranks:"), T("no communication")],
                     dash=True)
            coll_bar(s, cx["leave"][0], top, cx["leave"][1], hgt, "AR",
                     [B("ALL-REDUCE", 14, C["AR"][1]), T("g: z = z_{0} + z_{1}"), M("[256,2,512]", 11),
                      M("→ [256,2,512]", 11), G("layers.py:1480"), G("→ mappings.py:35")])
        return lane_y[1] + lh + 6

    yb = panel(170, sp=False)
    s.text(104, yb + 22, "SP off: 1 ALL-REDUCE per block in forward. Outside the TP region both ranks hold the same full "
           "[256, 2, 512] activation: a duplicate copy.", size=13)
    yb = panel(yb + 62, sp=True)
    s.text(104, yb + 22, "SP on: 1 ALL-GATHER + 1 REDUCE-SCATTER per block (together the same bytes as one all-reduce). "
           "Outside the TP region each rank holds only its half: [128, 2, 512].", size=13)
    # rule box
    ry = yb + 40
    s.rect(24, ry, W - 48, 76, "note", rx=10, fill="#f8fafc", stroke="#cbd5e1")
    s.text(40, ry + 24, "Rule:", size=14, weight="700")
    s.text(90, ry + 24, "ENTER the TP region (column-parallel input needs every token):  SP off → nothing (identity f)   ·   "
           "SP on → ALL-GATHER", size=13)
    s.text(90, ry + 46, "LEAVE the TP region (row-parallel output is a partial sum):  SP off → ALL-REDUCE (all tokens to all)   ·   "
           "SP on → REDUCE-SCATTER (my tokens to me)", size=13)
    s.text(90, ry + 66, "Attention is the same pattern: column = linear_qkv, A_{qkv,r} is 512×768 (4 of 8 heads); core attention per rank "
           "→ [256, 2, 256]; row = linear_proj, A_{proj,r} is 256×512.", size=12, fill=GREY)
    s.save(out / "d13_sp_forward.svg")


# ---------------------------------------------------------------- figure 3: backward of one block
def fig_backward(out):
    W, H = 1484, 700
    s = Svg(W, H)
    title(s, "Backward of the same MLP block, SP on (the SP-off alternative is in grey in each bar)",
          "Read left to right in execution order: the gradient enters from the block above and leaves to the block "
          "below. Each forward collective is mirrored (figure 1, fact 2).")
    legend(s, 24, 82, LEG_STATE)
    cx = dict(dy=(104, 150), ag1=(276, 150), fc2=(448, 206), act=(676, 168), fc1=(866, 190), rs=(1078, 160),
              ln=(1260, 200))
    lane_y = [140, 236]
    lh = 84
    toks = ["tokens 0–127", "tokens 128–255"]
    for r in range(2):
        y = lane_y[r]
        s.text(24, y + lh / 2 + 5, f"rank {r}", size=13, weight="700", fill=GREY)
        s.box(cx["dy"][0], y, cx["dy"][1], lh, "seq", [B("dy (shard)"), M("[128, 2, 512]"), G(toks[r])])
        s.box(cx["fc2"][0], y, cx["fc2"][1], lh, "hid",
              [B("fc2 backward (local)"), T("da_{r} = dz·A2_{r}^{T}"), M("(512×512)·(512×1024)", 11),
               M("→ [256, 2, 1024]")], gap=17)
        s.box(cx["act"][0], y, cx["act"][1], lh, "hid",
              [B("GELU backward"), T("dh_{r} = da_{r} ⊙ gelu′(h_{r})"), M("[256, 2, 1024]")])
        s.box(cx["fc1"][0], y, cx["fc1"][1], lh, "part",
              [B("fc1 input grad"), T("dx_{r} = dh_{r}·A1_{r}^{T}"), M("(512×1024)·(1024×512)", 11),
               M("→ [256, 2, 512] partial")], gap=17)
        s.box(cx["ln"][0], y, cx["ln"][1], lh, "seq",
              [B("LayerNorm backward"), M("dx [128, 2, 512]"), G("+ residual grad, own tokens"), G("→ to the block below")])
        for a, b in (("dy", "ag1"), ("ag1", "fc2"), ("fc2", "act"), ("act", "fc1"), ("fc1", "rs"), ("rs", "ln")):
            s.arrow(cx[a][0] + cx[a][1] + 1, y + lh / 2, cx[b][0] - 2, y + lh / 2)
    top, hgt = lane_y[0] - 6, lane_y[1] + lh + 12 - lane_y[0]
    coll_bar(s, cx["ag1"][0], top, cx["ag1"][1], hgt, "AG",
             [B("ALL-GATHER #1", 13, C["AG"][1]), T("mirror of the fwd RS"), M("[128,2,512]", 11), M("→ dz [256,2,512]", 11),
              G("mappings.py:377 → :118"), G("sync, compute stream"), T("SP off: identity (mirror of g)", 11, fill=GREY, italic=True)])
    coll_bar(s, cx["rs"][0], top, cx["rs"][1], hgt, "RS",
             [B("REDUCE-SCATTER", 13, C["RS"][1]), T("mirror of the fwd AG"), M("[256,2,512]", 11), M("→ [128,2,512]", 11),
              G("layers.py:650, async"), G("NCCL stream"), T("SP off: ALL-REDUCE (f),", 11, fill=GREY, italic=True),
              T("layers.py:639", 11, fill=GREY, italic=True)])
    # weight-gradient path
    wy = 360
    s.text(24, wy - 10, "Weight gradients of fc1 (inside the same backward call, rank 0 shown; rank 1 is the same with its own shard)",
           size=13, weight="700")
    s.box(104, wy, 190, 78, "seq", [B("saved forward input x"), M("[128, 2, 512]"), G("only the shard was kept (memory)")])
    coll_bar(s, 330, wy - 4, 170, 86, "AG",
             [B("ALL-GATHER #2", 13, C["AG"][1]), M("→ x_full [256,2,512]", 11), G("layers.py:617, async,"),
              G("issued before the dgrad GEMM")])
    s.box(536, wy, 320, 78, "hid",
          [B("dA1_{r} = x_{full}^{T}·dh_{r}"), M("(512×512)ᵀ·(512×1024) → [512, 1024]", 11),
           G("needs every token of x, hence the re-gather")])
    s.arrow(295, wy + 39, 328, wy + 39)
    s.arrow(501, wy + 39, 534, wy + 39)
    s.arrow(cx["act"][0] + 84, lane_y[1] + lh + 2, cx["act"][0] + 84, wy - 2, color=C["hid"][1], dash=True)
    s.text(772, wy - 14, "dh_{r} from the GELU backward", size=11, fill=C["hid"][1])
    s.box(896, wy, 548, 78, "note",
          [T("SP off: x was saved full ([256, 2, 512]), so there is no ALL-GATHER #2.", 12),
           T("That is SP's trade: half the stored activation, one extra all-gather per column-parallel linear.", 12),
           T("fc2's weight grad dA2_{r} = a_{r}^{T}·dz needs nothing extra: a_{r} is in the TP region.", 12, fill=GREY)], gap=19)
    # LN grads note
    ly = 476
    s.rect(24, ly, W - 48, 70, "AR", rx=10, sw=1.5, fill="#fff7fb")
    s.text(40, ly + 26, "Once per step, after the schedule: ALL-REDUCE of the LayerNorm γ, β and row-parallel bias (b2) gradients",
           size=13, weight="700", fill=C["AR"][1])
    s.text(40, ly + 48, "With SP each rank saw only its 128 tokens, so its dγ, dβ, db2 are partial sums. 9216 values on stage 0 "
           "(3 layers × (2 LN × 1024 + 2 biases × 512)), 10240 on stage 1 (+ final LN).", size=12)
    s.text(40, ly + 64, "Megatron: finalize_model_grads.py:452 (_allreduce_non_tensor_model_parallel_grads); here: "
           "allreduce_sequence_parallel_grads in train_gpt.py.", size=11, fill=GREY)
    # summary
    sy = 570
    s.rect(24, sy, W - 48, 106, "note", rx=10, fill="#f8fafc", stroke="#cbd5e1")
    s.text(40, sy + 26, "Per block, per microbatch", size=14, weight="700")
    rows = [("", "forward", "backward"),
            ("SP off", "1 ALL-REDUCE", "1 ALL-REDUCE  (async, overlaps the weight-grad GEMM)"),
            ("SP on", "1 ALL-GATHER + 1 REDUCE-SCATTER", "2 ALL-GATHER (#1 mirror, #2 re-gather) + 1 REDUCE-SCATTER")]
    for j, (a, b, c) in enumerate(rows):
        yy = sy + 50 + j * 20
        w = "700" if j == 0 else "normal"
        s.text(40, yy, a, size=13, weight="700")
        s.text(140, yy, b, size=13, weight=w)
        s.text(470, yy, c, size=13, weight=w)
    s.save(out / "d13_sp_backward.svg")


# ---------------------------------------------------------------- figure 4: one microbatch through both stages
def fig_pipeline(out):
    W, H = 1484, 560
    s = Svg(W, H)
    title(s, "One microbatch through both pipeline stages with SP on: every collective, and where it comes from",
          "Lanes are TP ranks. Stage 0 = ranks 0, 1; stage 1 = ranks 2, 3. Each \"3 layers\" box contains "
          "3 × (attention block + MLP block), each block as in figures 2 and 3.")
    legend(s, 24, 82, LEG_STATE)
    lane_y = [160, 256]
    lh = 80
    s.text(24, lane_y[0] + lh / 2 + 5, "TP rank 0", size=13, weight="700", fill=GREY)
    s.text(24, lane_y[1] + lh / 2 + 5, "TP rank 1", size=13, weight="700", fill=GREY)
    s.text(104, 140, "Stage 0  (ranks 0, 1)", size=15, weight="700")
    s.text(870, 140, "Stage 1  (ranks 2, 3)", size=15, weight="700")
    s.p.append(f'<line x1="836" y1="124" x2="836" y2="{lane_y[1] + lh + 12}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 5"/>')
    cols = dict(emb=(104, 158), ar=(282, 104), split=(406, 150), l0=(578, 170), l1=(870, 150), lnf=(1040, 92),
                ag=(1150, 100), out=(1268, 192))
    vocab = ["vocab rows 0–4095", "vocab rows 4096–8191"]
    toks = ["tokens 0–127", "tokens 128–255"]
    for r in range(2):
        y = lane_y[r]
        s.box(cols["emb"][0], y, cols["emb"][1], lh, "part",
              [B("word embedding"), G(vocab[r]), M("[256, 2, 512] partial")])
        s.box(cols["split"][0], y, cols["split"][1], lh, "seq",
              [B("+ position, split"), M("→ [128, 2, 512]"), G(toks[r] + ", no comm")])
        s.box(cols["l0"][0], y, cols["l0"][1], lh, "seq",
              [B("3 layers"), T("fwd: 2 AG + 2 RS per layer", 12), T("bwd: 4 AG + 2 RS per layer", 12), M("[128, 2, 512]")])
        s.box(cols["l1"][0], y, cols["l1"][1], lh, "seq",
              [B("3 layers"), T("fwd: 2 AG + 2 RS per layer", 12), T("bwd: 4 AG + 2 RS per layer", 12), M("[128, 2, 512]")])
        s.box(cols["lnf"][0], y, cols["lnf"][1], lh, "seq", [B("final LN"), M("[128,2,512]", 11)])
        s.box(cols["out"][0], y, cols["out"][1], lh, "hid",
              [B("output layer (column)"), T("A_{out,r}: 512×4096", 12), M("→ [256, 2, 4096]")])
        for a, b in (("emb", "ar"), ("ar", "split"), ("split", "l0"), ("l1", "lnf"), ("lnf", "ag"), ("ag", "out")):
            s.arrow(cols[a][0] + cols[a][1] + 1, y + lh / 2, cols[b][0] - 2, y + lh / 2)
        # P2P
        s.arrow(cols["l0"][0] + cols["l0"][1] + 1, y + lh / 2, cols["l1"][0] - 2, y + lh / 2, color=C["seq"][1], sw=2.4)
        s.text(810, y + lh / 2 - 8, "P2P", size=12, weight="700", fill=C["seq"][1], anchor="middle")
        s.text(810, y + lh / 2 + 18, f"{r} → {r + 2}", size=12, weight="700", fill=C["seq"][1], anchor="middle")
    top, hgt = lane_y[0] - 6, lane_y[1] + lh + 12 - lane_y[0]
    coll_bar(s, cols["ar"][0], top, cols["ar"][1], hgt, "AR",
             [B("ALL-REDUCE", 12, C["AR"][1]), M("[256,2,512]", 11), G("layers.py:377"), G("bwd: identity")])
    coll_bar(s, cols["ag"][0], top, cols["ag"][1], hgt, "AG",
             [B("ALL-GATHER", 12, C["AG"][1]), M("→ [256,2,512]", 11), G("layers.py:571"), G("bwd: RS")])
    s.text(cols["split"][0] + 75, lane_y[1] + lh + 26, "bwd: ALL-GATHER (mappings.py:297)", size=11, fill=C["AG"][1], anchor="middle")
    s.text(810, lane_y[1] + lh + 26, "[128, 2, 512] each way", size=11, fill=C["seq"][1], anchor="middle")
    s.text(W - 24, lane_y[1] + lh + 26, "→ cross-entropy: 3 × ALL-REDUCE of [256, 2] = 512 values", size=11,
           fill=C["AR"][1], anchor="end")
    # notes
    ny = 392
    s.rect(24, ny, 700, 148, "note", rx=10, fill="#f8fafc", stroke="#cbd5e1")
    s.text(40, ny + 24, "Why the embedding still all-reduces", size=14, weight="700")
    for j, t in enumerate([
            "Each rank looks up only its vocabulary rows; the other rows give 0, so the",
            "result is a partial sum. With learned position embeddings Megatron sums it with",
            "an ALL-REDUCE, adds positions, then just slices out its own tokens",
            "(language_model_embedding.py:48–52, :141). Without learned positions it",
            "would use a REDUCE-SCATTER directly (layers.py:372)."]):
        s.text(40, ny + 46 + j * 19, t, size=12)
    s.rect(744, ny, W - 768, 148, "note", rx=10, fill="#f8fafc", stroke="#cbd5e1")
    s.text(760, ny + 24, "Why the P2P pairs rank 0 → 2 and 1 → 3", size=14, weight="700")
    for j, t in enumerate([
            "With SP on, stage 0's output is sequence-sharded: rank 0 has tokens 0–127,",
            "rank 1 has 128–255. Rank 2 is TP rank 0 of stage 1, so it must receive",
            "exactly tokens 0–127. The PP groups [0, 2] and [1, 3] pair equal TP ranks, and",
            "get_tensor_shapes divides s by TP (schedules.py:2114): [128, 2, 512], half",
            "of the SP-off transfer. Crossed pairs would give the right shape, wrong tokens."]):
        s.text(760, ny + 46 + j * 19, t, size=12)
    s.save(out / "d13_sp_pipeline.svg")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig_collectives(out)
    fig_forward(out)
    fig_backward(out)
    fig_pipeline(out)
    print("wrote", sorted(p.name for p in out.glob("d13_sp_*.svg")))
