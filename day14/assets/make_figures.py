"""Generate day14/assets/d14_ep_flow.svg (one top-1 MoE layer at EP=4, GUIDE Block 1).
Usage: python day14/assets/make_figures.py day14/assets
Same drawing helpers and palette as day13/figures/make_figures.py."""
import html
import re
import sys
from pathlib import Path

FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"
INK, GREY, MUTED = "#1f2937", "#6b7280", "#94a3b8"
C = {
    "rep": ("#e8f0fe", "#3b6fd4"),    # tensors on this rank
    "loc": ("#e3f5e9", "#2f9e5b"),    # local compute, no communication
    "w": ("#f1eafd", "#7048c8"),      # the combine weight / router gradient path
    "sync": ("#fff1dc", "#e3891b"),   # device -> host synchronization
    "a2a": ("#ddf4f5", "#0b7f86"),    # collective over the EP group
    "hi": ("#fde2e2", "#b91c1c"),     # highlight (row = what I send)
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
        for i, (t, kw) in enumerate(lines):
            self.text(cx, y + i * gap, t, anchor="middle", **kw)

    def box(self, x, y, w, h, kind, lines, gap=17, dash=False, top=None):
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

    def path(self, d, color=GREY, sw=1.6, dash=False, head=True):
        mid = "ah" + color.strip("#")
        da = ' stroke-dasharray="5 4"' if dash else ""
        m = f' marker-end="url(#{mid})"' if head else ""
        self.p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{m}/>')

    def save(self, path):
        colors = {GREY, INK, MUTED} | {v[1] for v in C.values()}
        defs = "".join(
            f'<marker id="ah{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in sorted(colors))
        body = "\n".join(self.p)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}"><defs>{defs}</defs>\n{body}\n</svg>\n', encoding="utf-8", newline="\n")


def T(t, size=12.5, weight="normal", mono=False, fill=INK, italic=False):
    return (t, dict(size=size, weight=weight, mono=mono, fill=fill, italic=italic))


def B(t, size=13.5, fill=INK):
    return T(t, size=size, weight="700", fill=fill)


def M(t, size=12, fill=INK):
    return T(t, size=size, mono=True, fill=fill)


def ep_flow(out: Path) -> None:
    W, H = 1240, 860
    s = Svg(W, H)
    s.text(30, 38, "One top-1 MoE layer at EP=4, seen from rank r (it owns expert e = r)", size=19, weight="700")
    s.text(30, 60, "Numbered in execution order. Steps 1–3 and 7, 9–10 are local; 4, 6, 8 are collectives over the "
           "EP group [0,1,2,3]; 5 is the one device→host sync.", size=12.5, fill=GREY)

    bw, bh, gap, x0 = 204, 132, 40, 30
    col = [x0 + i * (bw + gap) for i in range(5)]
    y1, y2 = 92, 312
    tag = dict(size=11.5, fill=GREY)

    row1 = [
        ("rep", [B("1 · Input"), M("x_{r}  [T_{r}, d]"), T("T_{r} = T / 4 tokens"), T("small preset: 128 × 128", **tag)]),
        ("loc", [B("2 · Router, top-1"), M("logits = x_{r}W^{T} + b"), M("probs = softmax(logits)"),
                 M("idx = argmax(probs)"), M("w = probs[t, idx_{t}]")]),
        ("loc", [B("3 · Permute"), M("order = argsort(idx,"), M("        stable=True)"), M("counts = bincount(idx)"),
                 T("x_{r}[order]: grouped by expert", **tag)]),
        ("a2a", [B("4 · Count exchange"), T("all_to_all, EQUAL splits"), M("send counts.view(EP, E_{loc})"),
                 T("I send row r, I get column r"), T("int64, a few bytes", **tag)]),
        ("sync", [B("5 · Host sync"), M("one .tolist()  (D→H)"), M("input_splits  = row r"),
                  M("output_splits = column r"), T("host waits for steps 2–4", **tag)]),
    ]
    row2 = [  # drawn right to left: index 0 sits under step 5
        ("a2a", [B("6 · Dispatch"), T("all_to_all_single, UNEVEN"), M("send x_{r}[order]"), M("in  = input_splits"),
                 M("out = output_splits"), T("zero-size splits allowed", **tag)]),
        ("loc", [B("7 · Expert e = r"), M("fc1 → GELU → fc2"), T("on Σ output_splits rows"),
                 T("0 rows? it still runs", **tag), T("E_{loc} > 1: sort_chunks first", **tag)]),
        ("a2a", [B("8 · Combine"), T("all_to_all_single"), T("splits SWAPPED:"), M("in  = output_splits"),
                 M("out = input_splits")]),
        ("loc", [B("9 · Unpermute, × w"), M("y[order[k]] = y_sorted[k]"), M("zeros.index_add(0,"),
                 M("          order, y_sorted)"), M("y_{r} = y · w  (on rank r)")]),
        ("rep", [B("10 · Output"), M("y_{r}  [T_{r}, d]"), M("block: h + y_{r}"), T("dropped token: + 0", **tag)]),
    ]
    for i, (kind, ls) in enumerate(row1):
        s.box(col[i], y1, bw, bh, kind, ls)
        if i < 4:
            s.arrow(col[i] + bw + 3, y1 + bh / 2, col[i + 1] - 4, y1 + bh / 2, color=INK)
    for i, (kind, ls) in enumerate(row2):
        c = col[4 - i]
        s.box(c, y2, bw, bh, kind, ls)
        if i < 4:
            s.arrow(c - 3, y2 + bh / 2, col[3 - i] + bw + 4, y2 + bh / 2, color=INK)

    # step 5 -> step 6: the split lists go down as Python ints
    cx = col[4] + bw / 2
    s.arrow(cx, y1 + bh + 3, cx, y2 - 4, color=C["sync"][1], sw=2)
    s.text(cx + 8, y1 + bh + 46, "split lists", size=12, fill=C["sync"][1], weight="600")
    s.text(cx + 8, y1 + bh + 61, "(Python ints)", size=11.5, fill=C["sync"][1])
    # step 2 -> step 9: the combine weight never leaves rank r
    wx = col[1] + bw / 2
    s.arrow(wx, y1 + bh + 3, wx, y2 - 4, color=C["w"][1], sw=2, dash=True)
    s.text(wx + 10, y1 + bh + 40, "w [T_{r}] stays on rank r: the router's", size=12, fill=C["w"][1], weight="600")
    s.text(wx + 10, y1 + bh + 56, "only task-loss gradient path (§1.1)", size=12, fill=C["w"][1], weight="600")

    # ---- inset: the count matrix (ragged example, small preset)
    iy = 500
    s.rect(30, iy, 590, 300, "note", rx=12)
    s.text(48, iy + 28, "Steps 4–6 on the ragged example (small preset, 128 tokens per source)", size=13.5, weight="700")
    s.text(48, iy + 48, "send matrix: row = source rank, column = destination rank (= expert)", size=12, fill=GREY)
    mat = [[128, 0, 0, 0], [86, 42, 0, 0], [0, 32, 0, 96], [64, 0, 0, 64]]
    cw, ch, mx, my = 60, 34, 100, iy + 92
    for j in range(4):
        s.text(mx + j * cw + cw / 2, my - 10, f"dst {j}", size=11.5, fill=GREY, anchor="middle")
    for i in range(4):
        s.text(mx - 12, my + i * ch + ch / 2 + 4, f"src {i}", size=11.5, fill=GREY, anchor="end")
        for j in range(4):
            kind = "hi" if i == 1 else ("a2a" if j == 1 else "empty")
            s.rect(mx + j * cw, my + i * ch, cw, ch, kind, rx=3, sw=1.2)
            v = mat[i][j]
            s.text(mx + j * cw + cw / 2, my + i * ch + ch / 2 + 4.5, str(v), size=13, mono=True,
                   fill=MUTED if v == 0 else INK, anchor="middle", weight="700" if (i == 1 or j == 1) and v else "normal")
    lx = mx + 4 * cw + 22
    s.rect(lx, my + 2, 14, 14, "hi", rx=3)
    s.text(lx + 22, my + 14, "row 1 = rank 1's input_splits", size=12)
    s.text(lx + 22, my + 30, "[86, 42, 0, 0]: known locally", size=11.5, fill=GREY, mono=True)
    s.rect(lx, my + 48, 14, 14, "a2a", rx=3)
    s.text(lx + 22, my + 60, "column 1 = rank 1's output_splits", size=12)
    s.text(lx + 22, my + 76, "[0, 42, 32, 0]: from step 4", size=11.5, fill=GREY, mono=True)
    s.text(lx, my + 106, "rank 1's expert computes 74 rows;", size=12)
    s.text(lx, my + 122, "rank 2's column is all zeros: 0 rows,", size=12)
    s.text(lx, my + 138, "but it still enters every collective.", size=12)

    # ---- inset: backward
    bx = 640
    s.rect(bx, iy, 570, 300, "note", rx=12)
    s.text(bx + 18, iy + 28, "Backward: the same split lists, reversed (no new count exchange)", size=13.5, weight="700")
    rows = [
        ("a2a", "8ᵀ  combine backward", "all_to_all(in = input_splits, out = output_splits)"),
        ("loc", "7ᵀ  expert backward", "dW_{e} = Σ over every source's rows, on the owner"),
        ("a2a", "6ᵀ  dispatch backward", "all_to_all(in = output_splits, out = input_splits)"),
        ("w", "2ᵀ  router", "gradient through w (step 9) and the aux loss"),
        ("note", "finish_grads", "dense: all-reduce, ÷DP · expert: ÷DP, no reduce (EDP = 1)"),
    ]
    for k, (kind, name, what) in enumerate(rows):
        yy = iy + 50 + k * 44
        s.rect(bx + 18, yy, 178, 32, kind, rx=6)
        s.text(bx + 107, yy + 21, name, size=12.5, weight="700", anchor="middle")
        s.text(bx + 208, yy + 21, what, size=12, mono=k != 3 and k != 4, fill=INK)
    s.text(bx + 18, iy + 284, "Per layer per step: 3 all_to_all forward (4, 6, 8) + 2 backward (8ᵀ, 6ᵀ).", size=12,
           fill=GREY)

    # ---- legend
    ly = 830
    items = [("loc", "local compute"), ("a2a", "collective over the EP group"), ("sync", "device→host sync"),
             ("w", "combine weight / router gradient"), ("rep", "tensor on rank r")]
    x = 30
    for kind, label in items:
        s.rect(x, ly - 12, 16, 16, kind, rx=3)
        s.text(x + 24, ly + 1, label, size=12, fill=GREY)
        x += 24 + len(label) * 6.6 + 34
    s.save(out / "d14_ep_flow.svg")


# ------------------------------------------------------------------ all four ranks at once (notes §4)
RAGGED = [[128, 0, 0, 0], [86, 42, 0, 0], [0, 32, 0, 96], [64, 0, 0, 64]]   # [source][expert], small preset
EXPERT_COLOR = ["#3b6fd4", "#2f9e5b", "#7048c8", "#e3891b"]
EXPERT_FILL = ["#dbe6fb", "#d5efdf", "#e6dcfa", "#fde6c8"]


def _band(s, x1, y1a, y1b, x2, y2a, y2b, color):
    xm = (x1 + x2) / 2
    d = (f"M{x1:.1f},{y1a:.1f} C{xm:.1f},{y1a:.1f} {xm:.1f},{y2a:.1f} {x2:.1f},{y2a:.1f} "
         f"L{x2:.1f},{y2b:.1f} C{xm:.1f},{y2b:.1f} {xm:.1f},{y1b:.1f} {x1:.1f},{y1b:.1f} Z")
    s.p.append(f'<path d="{d}" fill="{color}" fill-opacity="0.38" stroke="{color}" stroke-opacity="0.7" '
               f'stroke-width="0.8"/>')


def _layout(k=0.85, y0=138.0, exp_gap=30.0, exp_min=66.0):
    """y ranges of every (source, expert) stripe on the source bars and on the expert bars."""
    E = 4
    tot = [sum(RAGGED[r][j] for r in range(4)) for j in range(E)]
    slot = [max(t * k, exp_min) for t in tot]
    exp_top, y = [], y0
    for j in range(E):
        exp_top.append(y)
        y += slot[j] + exp_gap
    total_h = y - exp_gap - y0
    src_h = 128 * k
    src_gap = (total_h - 4 * src_h) / 3
    src_top = [y0 + r * (src_h + src_gap) for r in range(4)]
    src_stripe, exp_stripe, exp_bar = {}, {}, []
    for r in range(4):
        yy = src_top[r]
        for j in range(E):
            h = RAGGED[r][j] * k
            src_stripe[r, j] = (yy, yy + h)
            yy += h
    for j in range(E):
        bh = tot[j] * k
        yy = exp_top[j] + (slot[j] - bh) / 2
        exp_bar.append((yy, yy + bh))
        for r in range(4):
            h = RAGGED[r][j] * k
            exp_stripe[j, r] = (yy, yy + h)
            yy += h
    return dict(tot=tot, slot=slot, exp_top=exp_top, src_top=src_top, src_h=src_h, src_stripe=src_stripe,
                exp_stripe=exp_stripe, exp_bar=exp_bar, bottom=y0 + total_h)


def _bars_and_bands(s, L, xa, xb, xc, xd, bw=34):
    """Source bars at xa (left) and xd (right), expert boxes from xb to xc, bands between them."""
    for r in range(4):
        for j in range(4):
            a, b = L["src_stripe"][r, j]
            if b > a:
                for x in (xa, xd):
                    s.rect(x, a, bw, b - a, "empty", rx=0, sw=0.8, fill=EXPERT_FILL[j], stroke=EXPERT_COLOR[j])
        for x in (xa, xd):
            s.rect(x, L["src_top"][r], bw, L["src_h"], "empty", rx=3, sw=1.4, fill="none", stroke=INK)
    for j in range(4):
        top, h = L["exp_top"][j], L["slot"][j]
        empty = L["tot"][j] == 0
        s.rect(xb, top, xc - xb, h, "empty", rx=8, sw=1.5, fill=EXPERT_FILL[j] if not empty else "#ffffff",
               stroke=EXPERT_COLOR[j], dash=empty)
        for r in range(4):
            a, b = L["exp_stripe"][j, r]
            if b > a:
                for x in (xb, xc - bw):
                    s.rect(x, a, bw, b - a, "empty", rx=0, sw=0.8, fill="#ffffff", stroke=EXPERT_COLOR[j])
                    s.text(x + bw / 2, (a + b) / 2 + 4, str(r), size=10.5, fill=GREY, anchor="middle")
    for r in range(4):
        for j in range(4):
            n = RAGGED[r][j]
            if not n:
                continue
            a1, b1 = L["src_stripe"][r, j]
            a2, b2 = L["exp_stripe"][j, r]
            _band(s, xa + bw, a1, b1, xb, a2, b2, EXPERT_COLOR[j])
            _band(s, xc, a2, b2, xd, a1, b1, EXPERT_COLOR[j])
            for x1, x2, ya, yb in ((xa + bw, xb, (a1 + b1) / 2, (a2 + b2) / 2), (xc, xd, (a2 + b2) / 2, (a1 + b1) / 2)):
                if n >= 30:
                    s.text((x1 + x2) / 2, (ya + yb) / 2 + 4, str(n), size=11.5, weight="700", fill=EXPERT_COLOR[j],
                           anchor="middle")


def _header(s, cols, y, direction_right=True, x_from=160, x_to=1080):
    for x, title, sub in cols:
        s.text(x, y, title, size=13, weight="700", anchor="middle")
        if sub:
            s.text(x, y + 17, sub, size=11.5, fill=GREY, anchor="middle", mono=True)
    ay = y - 26
    if direction_right:
        s.arrow(x_from, ay, x_to, ay, color=MUTED, sw=2)
    else:
        s.arrow(x_to, ay, x_from, ay, color=MUTED, sw=2)


def ep_butterfly(out: Path, backward: bool) -> None:
    W = 1240
    L = _layout()
    H = int(L["bottom"]) + (230 if backward else 92)
    s = Svg(W, H)
    xa, xb, xc, xd, bw = 168, 452, 788, 1038, 34
    if not backward:
        s.text(30, 36, "Forward at EP=4: every rank routes its own tokens, each expert gets its rows from all four",
               size=18, weight="700")
        s.text(30, 57, "ragged routing, small preset (4 × 128 tokens), one MoE layer; band width ∝ rows; a stripe's "
               "number on an expert bar = the source rank", size=12.5, fill=GREY)
        _header(s, [(110, "Source ranks", "route, permute"), (318, "Dispatch all_to_all", "in=row, out=column"),
                    (620, "Experts (rank j runs expert j)", "fc1 → GELU → fc2"),
                    (913, "Combine all_to_all", "splits swapped"), (1140, "Source ranks", "unpermute, × w, + h")],
                110, True)
    else:
        s.text(30, 36, "Backward at EP=4: gradients take the same bands in reverse, then finish_grads", size=18,
               weight="700")
        s.text(30, 57, "same ragged example; autograd replays the saved split lists, so there is no count exchange "
               "and no host sync in backward", size=12.5, fill=GREY)
        _header(s, [(110, "dx_r on the sources", "scatter to token order"),
                    (318, "Dispatch backward", "in=column, out=row"),
                    (620, "Expert backward", "dX rows + dW_j"),
                    (913, "Combine backward", "in=row, out=column"), (1140, "dy_r on the sources", "× w, regroup")],
                110, False)
    _bars_and_bands(s, L, xa, xb, xc, xd, bw)

    send = ["[128,0,0,0]", "[86,42,0,0]", "[0,32,0,96]", "[64,0,0,64]"]
    for r in range(4):
        cy = L["src_top"][r] + L["src_h"] / 2
        if not backward:
            left = [B(f"rank {r}"), T(f"x_{{{r}}}: 128 tokens", size=12), M(f"send {send[r]}", size=11)]
            right = [B(f"rank {r}"), T("its 128 rows back", size=12), T("unpermute → × w → + h", size=11.5)]
        else:
            left = [B(f"rank {r}"), T(f"dx_{{{r}}}: 128 rows", size=12), T("+ router grad (w, aux)", size=11.5)]
            right = [B(f"rank {r}"), T(f"dy_{{{r}}}: 128 rows", size=12), T("dz = permute(dy · w)", size=11.5)]
        for i, (t, kw) in enumerate(left):
            s.text(xa - 10, cy - 14 + i * 16, t, anchor="end", **kw)
        for i, (t, kw) in enumerate(right):
            s.text(xd + bw + 10, cy - 14 + i * 16, t, anchor="start", **kw)

    srcs = {0: "ranks 0, 1, 3", 1: "ranks 1, 2", 2: "nobody", 3: "ranks 2, 3"}
    cols = ["[128,86,0,64]", "[0,42,32,0]", "[0,0,0,0]", "[0,0,96,64]"]
    for j in range(4):
        cy = L["exp_top"][j] + L["slot"][j] / 2
        n = L["tot"][j]
        if not backward:
            ls = [B(f"expert {j} · rank {j}", fill=EXPERT_COLOR[j]), M(f"recv {cols[j]}", size=11),
                  T(f"{n} rows" if n else "0 rows: runs on [0, d]", size=12)]
        else:
            ls = [B(f"expert {j} · rank {j}", fill=EXPERT_COLOR[j]),
                  T(f"dW_{{{j}}} = Σ over {n} rows ({srcs[j]})" if n else "dW_{2} = 0 exactly (not None)", size=12),
                  T("dX rows go back to their sources" if n else "still enters both all_to_alls", size=11.5,
                    fill=GREY)]
        if L["slot"][j] < 90:
            ls = ls[:3]
        s.lines((xb + xc) / 2, cy - (len(ls) - 1) * 8 + 4, ls, 16)

    y = L["bottom"] + 34
    if not backward:
        s.text(30, y, "Before the dispatch: one all_to_all of the [EP, E_loc] counts plus one .tolist() gives every "
               "rank its ROW of this matrix (what it sends) and its COLUMN (what it receives).", size=12.5)
        x = 30
        for j in range(4):
            s.rect(x, y + 20, 16, 16, "empty", rx=3, fill=EXPERT_FILL[j], stroke=EXPERT_COLOR[j])
            s.text(x + 24, y + 33, f"tokens routed to expert {j}", size=12, fill=GREY)
            x += 210
        s.save(out / "d14_ep_forward.svg")
        return

    # finish_grads panel
    py = y - 10
    s.rect(30, py, 575, 168, "note", rx=12)
    s.text(48, py + 26, "finish_grads, dense parameters", size=13.5, weight="700")
    s.text(48, py + 44, "attention, LayerNorms, router, embeddings, head: a copy on every rank", size=12, fill=GREY)
    for r in range(4):
        s.box(48 + r * 62, py + 62, 52, 40, "rep", [M(f"g_{{{r}}}", size=12)])
    s.arrow(300, py + 82, 352, py + 82, color=C["a2a"][1], sw=2)
    s.box(358, py + 60, 228, 44, "a2a", [T("all-reduce SUM over DP [0,1,2,3]", size=12), T("then ÷ DP (= 4)", size=12)],
          gap=16)
    s.text(48, py + 128, "result identical on all ranks (replicated check, bitwise)", size=12)
    s.text(48, py + 148, "[derived, not a control here] forget ÷ DP → every dense grad 4×", size=12, fill=GREY)

    s.rect(625, py, 585, 168, "note", rx=12)
    s.text(643, py + 26, "finish_grads, expert parameters", size=13.5, weight="700")
    s.text(643, py + 44, "expert j lives only on rank j (EP = 4, EDP = 1)", size=12, fill=GREY)
    s.text(643, py + 68, "after backward, rank j already holds the sum over all sources:", size=12)
    s.text(643, py + 85, "the bands above brought every source's rows to it.", size=12)
    s.text(643, py + 106, "→ ÷ DP (= 4), and NO all-reduce", size=12.5, weight="700", fill=C["a2a"][1])
    s.text(643, py + 128, "EP=2 × EDP=2: ÷ 4, then all-reduce over EDP [j, j+2]", size=12)
    s.text(643, py + 148, "controls: expert_sum_over_ep (ratio ≈ 0.25), expert_no_dp_scale (4.0000)", size=12,
           fill=GREY)
    s.save(out / "d14_ep_backward.svg")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    out.mkdir(parents=True, exist_ok=True)
    ep_flow(out)
    ep_butterfly(out, backward=False)
    ep_butterfly(out, backward=True)
    print(f"wrote {out / 'd14_ep_flow.svg'}, d14_ep_forward.svg, d14_ep_backward.svg")
