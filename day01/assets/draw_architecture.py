from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
W, H = 2400, 1800
im = Image.new("RGB", (W, H), "#f5f7fb")
d = ImageDraw.Draw(im)
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="#f5f7fb"/>',
]
FONT = "C:/Windows/Fonts/msyh.ttc"
INK = "#192d48"


def text(x, y, s, size=28, color=INK, center=False):
    f = ImageFont.truetype(FONT, size)
    d.text((x, y), s, font=f, fill=color, anchor="mt" if center else "lt")
    svg.append(
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei,sans-serif" font-size="{size}" fill="{color}" dominant-baseline="text-before-edge" text-anchor="{"middle" if center else "start"}">{escape(s)}</text>'
    )


def rect(x, y, w, h, fill="#ffffff", stroke="#dbe3ee", radius=18):
    d.rounded_rectangle((x, y, x + w, y + h), radius, fill, stroke, 2)
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )


def line(points, color="#536b87", width=4, arrow=False):
    d.line(points, fill=color, width=width, joint="curve")
    svg.append(
        f'<polyline points="{" ".join(f"{x},{y}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="{width}"/>'
    )
    if arrow:
        x, y = points[-1]
        px, py = points[-2]
        if x == px:
            sign = 1 if y > py else -1
            tri = [(x, y), (x - 9, y - sign * 15), (x + 9, y - sign * 15)]
        else:
            sign = 1 if x > px else -1
            tri = [(x, y), (x - sign * 15, y - 9), (x - sign * 15, y + 9)]
        d.polygon(tri, fill=color)
        svg.append(
            f'<polygon points="{" ".join(f"{a},{b}" for a, b in tri)}" fill="{color}"/>'
        )


def box(x, y, w, h, title, sub="", fill="#eaf1fc"):
    rect(x, y, w, h, fill)
    text(x + w / 2, y + 16, title, 29, center=True)
    if sub:
        text(x + w / 2, y + 57, sub, 25, "#536b87", True)


def plus(x, y):
    d.ellipse(
        (x - 25, y - 25, x + 25, y + 25), fill="#ffffff", outline="#536b87", width=3
    )
    svg.append(
        f'<circle cx="{x}" cy="{y}" r="25" fill="white" stroke="#536b87" stroke-width="3"/>'
    )
    text(x, y - 22, "+", 34, center=True)


text(80, 48, "Day 01 · TinyTransformer Model Architecture", 54)
text(
    82,
    125,
    "Matches train.py: learned positional embeddings / causal self-attention / Pre-Norm / FP32",
    29,
    "#536b87",
)
for x, w in [(60, 700), (790, 780), (1600, 740)]:
    rect(x, 205, w, 1270)
text(90, 235, "① Feed Forward Process", 35)
text(820, 235, "② Single Transformer Block", 35)
text(1630, 235, "③ Attention and Causal Constraints", 35)

# Overall model; token and learned position embeddings meet at addition.
box(95, 310, 300, 95, "Input token IDs", "[B,T] integer")
box(425, 310, 300, 95, "position index", "0 … T−1")
line([(245, 405), (245, 445)], arrow=True)
line([(575, 405), (575, 445)], arrow=True)
box(95, 445, 300, 110, "Token Embedding", "V → D  [B,T,D]", "#def3ed")
box(425, 445, 300, 110, "Position Embedding", "T → D  [T,D]", "#def3ed")
line([(245, 555), (245, 610), (385, 610)], arrow=True)
line([(575, 555), (575, 610), (435, 610)], arrow=True)
plus(410, 610)
text(490, 635, "broadcast across batch", 24, "#536b87")
line([(410, 635), (410, 700)], arrow=True)
box(150, 700, 520, 110, "Transformer Block × L", "output [B,T,D] each layer", "#e6edfd")
line([(410, 810), (410, 860)], arrow=True)
box(150, 860, 520, 95, "Final LayerNorm", "[B,T,D]", "#ece6fa")
line([(410, 955), (410, 1005)], arrow=True)
box(150, 1005, 520, 95, "Linear Head", "D → V (with bias)", "#ece6fa")
line([(410, 1100), (410, 1150)], arrow=True)
box(150, 1150, 520, 100, "Output Logits", "[B,T,V], Softmax", "#ece6fa")
text(95, 1305, "• Token Embedding and Head do not share weights", 25)
text(95, 1355, "• No cross-attention; no RoPE", 25)
text(95, 1405, "• forward() returns logits only", 25)

# Exact pre-norm block, with two separate residual paths.
cx = 1210
text(cx, 310, "Input h  [B,T,D]", 29, center=True)
line([(cx, 355), (cx, 395)], arrow=True)
box(1030, 395, 360, 80, "LayerNorm 1", fill="#edf1f8")
line([(cx, 475), (cx, 515)], arrow=True)
box(1030, 515, 360, 110, "Causal Multi-Head", "Self-Attention", "#e2edff")
line([(cx, 625), (cx, 685)], arrow=True)
plus(cx, 710)
line([(cx, 365), (910, 365), (910, 710), (1185, 710)], color="#218b79", arrow=True)
text(855, 490, "Residual 1", 25, "#218b79")
text(1270, 690, "h′", 28)
line([(cx, 735), (cx, 785)], arrow=True)
box(1030, 785, 360, 80, "LayerNorm 2", fill="#edf1f8")
line([(cx, 865), (cx, 905)], arrow=True)
box(1030, 905, 360, 80, "Linear  D → 4D", fill="#fff0d8")
line([(cx, 985), (cx, 1015)], arrow=True)
box(1030, 1015, 360, 70, "ReLU", fill="#fff0d8")
line([(cx, 1085), (cx, 1115)], arrow=True)
box(1030, 1115, 360, 80, "Linear  4D → D", fill="#fff0d8")
line([(cx, 1195), (cx, 1240)], arrow=True)
plus(cx, 1265)
line([(cx, 755), (1480, 755), (1480, 1265), (1235, 1265)], color="#218b79", arrow=True)
text(1430, 965, "Residual 2", 25, "#218b79")
line([(cx, 1290), (cx, 1320)], arrow=True)
text(cx, 1325, "Output h_out  [B,T,D]", 28, center=True)
text(830, 1390, "h′ = h + Attention(LN1(h))", 25)
text(830, 1430, "h_out = h′ + FFN(LN2(h′))   ·   dropout=0", 25)

# Attention internals.
box(1660, 315, 620, 80, "Hidden states from LayerNorm 1", fill="#edf1f8")
for x, label in [(1660, "Q"), (1880, "K"), (2100, "V")]:
    line([(1970, 395), (1970, 420), (x + 90, 420), (x + 90, 450)], arrow=True)
    box(x, 450, 180, 80, f"{label} Projection", fill="#e2edff")
text(1970, 555, "Split into H heads: [B,H,T,d_h]", 28, center=True)
text(1970, 600, "d_h = D / H", 27, "#536b87", True)
line([(1970, 640), (1970, 675)], arrow=True)
box(
    1660,
    675,
    620,
    110,
    "softmax(QK^T / sqrt(d_h) + M) V",
    "M: future positions = −∞; all others = 0",
    "#e2edff",
)
line([(1970, 785), (1970, 830)], arrow=True)
box(
    1660,
    830,
    620,
    95,
    "Concatenate heads → output projection",
    "[B,T,D] → [B,T,D]",
    "#e2edff",
)
text(1640, 970, "Causal mask (example T=4)", 29)
gx, gy, cell = 1740, 1035, 65
for r in range(4):
    for c in range(4):
        rect(
            gx + c * cell,
            gy + r * cell,
            cell - 5,
            cell - 5,
            "#ccece4" if c <= r else "#e7ebf2",
            radius=5,
        )
        text(
            gx + c * cell + 30,
            gy + r * cell + 12,
            "O" if c <= r else "×",
            28,
            center=True,
        )
text(2050, 1060, "Rows: query", 24)
text(2050, 1110, "Columns: key", 24)
text(1660, 1320, "Allowed: current and past positions", 27, "#218b79")
text(1660, 1370, "Blocked: future positions in the upper triangle", 27, "#536b87")
text(1660, 1420, "Mask before softmax to prevent label leakage", 24)

rect(60, 1505, 2280, 245, "#ffffff")
text(90, 1530, "Default tiny configuration: ", 30)
text(470, 1532, "B=16   T=64   V=128   D=128   L=2   H=4   d_h=32", 29)
text(
    90,
    1590,
    "Shapes: [16,64] → [16,64,128] → logits [16,64,128]; D and V have the same value but different meanings.",
    28,
)
text(
    90,
    1650,
    "Training: reshape logits to [B·T,V], labels to [B·T] → Cross Entropy → scalar loss",
    28,
)
text(
    90,
    1700,
    "Cross Entropy handles log_softmax internally; do not add Softmax after the model Head.",
    25,
    "#536b87",
)
svg.append("</svg>")
im.save(OUT / "model_architecture.png")
(OUT / "model_architecture.svg").write_text("\n".join(svg), encoding="utf-8")
print(OUT / "model_architecture.png")
