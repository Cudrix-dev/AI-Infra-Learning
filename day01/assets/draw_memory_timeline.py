"""Day 01 - "One training step, as a memory timeline" (GUIDE.md, block 1.1).

Numbers are the medium preset's steady-state step (iteration 1 of the memory probe):
B=16, T=256, V=8192, D=512, 6 layers, FP32, deterministic (math attention).
Writes memory_timeline.png and memory_timeline.svg next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent

# ---------------------------------------------------------------- fonts / colors
for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
PHASE_BG = ["#f3f2ef", "#fcfcfb"]

# Categorical slots in the validated fixed order; stack order = slot order (adjacent pairs pass CVD).
CATS = [  # key, label, color, note
    ("params", "Parameters", "#2a78d6", "allocated at model.to(cuda)\nlives forever"),
    ("ws", "cuBLAS workspaces", "#eb6834", "2 × 32 MiB (forward + autograd thread)\nlives forever"),
    ("adam", "Adam m, v", "#1baf7a", "created lazily at the first step()\nlives forever"),
    ("grads", "Gradients", "#eda100", "born in backward (AccumulateGrad)\nfreed at the next zero_grad"),
    ("acts", "Saved activations", "#e87ba4", "saved during forward\nfreed node by node in backward"),
    ("logits", "Logits [B·T, V]", "#008300", "freed when the last reference\ngoes (del logits)"),
    ("lgrads", "Logit-sized grads", "#6250d6", "dL/dlogp + dL/dlogits\n2 × 128 MiB, briefly"),
    ("tmp", "Adam temporaries", "#e34948", "sqrt(v) and divide temps\nfor the largest tensor"),
]

# ---------------------------------------------------------------- phases (abstract time)
PH = [("zero_grad", 0.0, 0.8), ("forward", 0.8, 5.6), ("loss +\ndel logits", 5.6, 6.8),
      ("backward", 6.8, 11.8), ("optimizer\n.step()", 11.8, 13.4)]
x = np.linspace(0, 13.4, 4000)

# ---------------------------------------------------------------- memory (MiB), medium preset, iteration 1
P, WS, ADAM, G, ACT, LOG, LG, TMP = 105.81, 64.0, 210.38, 105.69, 1008.47, 128.0, 256.0, 32.0
BWD0, BWD_T, BWD1 = 6.8, 7.1, 11.8  # backward start, end of log_softmax backward, backward end


def ramp(x, a, b):  # 0 before a, 1 after b, linear in between
    return np.clip((x - a) / (b - a), 0, 1)


series = {
    "params": np.full_like(x, P),
    "ws": np.full_like(x, WS),
    "adam": np.full_like(x, ADAM),
    "grads": G * (x < 0.4) + G * ramp(x, BWD_T, BWD1),  # last step's grads, freed by zero_grad; rebuilt in backward
    "acts": ACT * ramp(x, 0.8, 5.4) * (1 - ramp(x, BWD_T, BWD1)),
    "logits": LOG * ((x >= 5.4) & (x < 6.2)),
    "lgrads": LG * ((x >= BWD0) & (x < BWD_T)),
    "tmp": TMP * ((x >= 12.3) & (x < 12.9)),
}
stack = np.vstack([series[k] for k, *_ in CATS])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 11.2), dpi=200, facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.12, left=0.155, right=0.765, top=0.86, bottom=0.07)
ax_g, ax_m = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)

fig.text(0.04, 0.955, "One training step, as a memory timeline", fontsize=24, weight="bold", color=INK)
fig.text(0.04, 0.925, "Medium preset (27.4M params, B=16, T=256, V=8192, FP32, math attention) · steady-state step "
         "(iteration 1 of --memory-probe) · MiB above baseline", fontsize=12.5, color=INK2)

for ax in (ax_g, ax_m):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 13.4)
    for i, (name, a, b) in enumerate(PH):
        ax.axvspan(a, b, color=PHASE_BG[i % 2], zorder=0, lw=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xticks([])

# phase labels above the top panel
for name, a, b in PH:
    ax_g.text((a + b) / 2, len(CATS) - 0.1, name, ha="center", va="bottom", fontsize=12, color=INK, weight="semibold",
              linespacing=1.0)

# ---------------------------------------------------------------- panel A: lifetimes (thickness = share of own max)
n = len(CATS)
for i, (k, label, color, note) in enumerate(CATS):
    yc = n - 1 - i
    v = series[k]
    h = 0.62 * v / v.max()
    ax_g.fill_between(x, yc - h / 2, yc + h / 2, color=color, lw=0, zorder=3)
    size = {"params": "105.8", "ws": "64", "adam": "210.4", "grads": "105.7", "acts": "1008.5",
            "logits": "128", "lgrads": "256", "tmp": "32"}[k]
    ax_g.text(-0.15, yc + 0.08, label, ha="right", va="center", fontsize=12.5, color=INK, weight="semibold")
    ax_g.text(-0.15, yc - 0.24, f"{size} MiB", ha="right", va="center", fontsize=10.5, color=INK2)
    ax_g.text(13.55, yc, note, ha="left", va="center", fontsize=10.2, color=INK2, wrap=True)
    ax_g.add_patch(Rectangle((-0.02, yc - 0.31), 0.0, 0.62, color=color))
ax_g.set_ylim(-0.6, n + 0.35)
ax_g.set_yticks([])
ax_g.text(0.0, -0.55, "Bar thickness = that category's size relative to its own maximum.", fontsize=10, color=MUTED,
          va="bottom")

# ---------------------------------------------------------------- panel B: stacked allocated memory
ax_m.stackplot(x, stack, colors=[c for _, _, c, _ in CATS], edgecolor=SURFACE, linewidth=0.8, zorder=3)
ax_m.set_ylim(0, 1800)
ax_m.set_yticks(range(0, 1801, 300))
ax_m.yaxis.grid(True, color=GRID, lw=0.8, zorder=1)
ax_m.set_axisbelow(True)
ax_m.tick_params(axis="y", labelsize=11, labelcolor=INK2)
ax_m.set_ylabel("memory_allocated (MiB)", fontsize=12, color=INK2, labelpad=10)

# measured points from the probe (iteration 1)
meas = [(0.8, 380.19, "380.2"), (5.8, 1516.60, "1516.6"), (6.6, 1388.60, "1388.6"),
        (BWD0 + 0.15, 1644.60, "1644.6"), (11.8, 488.88, "488.9"), (12.6, 520.88, "520.9")]
mx, my = zip(*[(a, b) for a, b, _ in meas])
ax_m.scatter(mx, my, s=70, color=INK, edgecolor=SURFACE, linewidth=2, zorder=6, label="measured by --memory-probe")
for (a, b, t), dx, dy, ha in zip(meas, [0.0, -0.15, -0.12, 0.2, -0.15, 0.0], [45, 25, -95, 25, -95, 45],
                                 ["center", "right", "right", "left", "right", "center"]):
    ax_m.text(a + dx, b + dy, t, fontsize=10.5, color=INK, ha=ha, weight="semibold", zorder=7)

# callouts
ax_m.annotate("PEAK 1644.6 MiB\nstart of backward: all activations\n+ two 128 MiB logit-sized grads",
              xy=(BWD0 + 0.15, 1644.6), xytext=(8.3, 1500), fontsize=11, color=INK, va="center",
              arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax_m.annotate("between steps: 380 MiB\n= params + Adam + workspaces\n(3P + 64 MiB)", xy=(0.8, 380), xytext=(1.0, 1500),
              fontsize=11, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax_m.annotate("del logits: −128 MiB", xy=(6.2, 1450), xytext=(3.6, 1720), fontsize=11, color=INK, va="center",
              arrowprops=dict(arrowstyle="-", color=INK2, lw=1))

# legend (always present; panel A rows also name every series directly)
handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in CATS]
leg = ax_m.legend(handles[::-1] + [ax_m.collections[-1]], [l for _, l, _, _ in CATS][::-1] + ["measured by --memory-probe"],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=11, labelcolor=INK,
                  handlelength=1.2, handleheight=1.2, borderaxespad=0.2, title="stacked, top to bottom")
leg.get_title().set_color(INK2)
leg.get_title().set_fontsize(10.5)

fig.text(0.04, 0.025, "Source: experiments/day01_single_gpu/train.py --memory-probe --preset medium "
         "(identical on RTX 5060 Laptop, RTX 4090 and RTX 5090). Time axis is schematic; memory values are measured.",
         fontsize=10, color=MUTED)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"memory_timeline.{ext}", facecolor=SURFACE)
print("wrote", OUT / "memory_timeline.png")
