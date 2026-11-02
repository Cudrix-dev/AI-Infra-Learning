"""Plot real unprofiled rank boundaries, with explicitly target-selected examples.

This is NOT a GPU-kernel trace: core_start/core_end are same-host perf_counter
readings bracketing the synchronized core step. Selection minimizes distance
of BOTH absolute offsets from 0.2 ms over all 150 measured steps per variant.
"""
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/day03_cuda_graph_view/a100_nvlink_view_20261102"
ASSETS = Path(__file__).resolve().parent
COLORS = ["#3268a8", "#9656a4"]


def main():
    evidence = {}
    for impl in ("mini_ddp", "torch_ddp"):
        rows, hashes = [], {}
        for repeat in range(3):
            folder = OUT / f"timing_r{repeat}_{impl}_graph_view"
            meta = json.loads((folder / "metrics.json").read_text())["meta"]
            assert not meta["args"]["profile"]
            warm = meta["warmup_steps"]
            files = [folder / f"boundaries_rank{r}.json" for r in (0, 1)]
            ranks = [json.loads(p.read_text()) for p in files]
            hashes.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
            for a, b in zip(ranks[0][warm:], ranks[1][warm:]):
                assert a["step"] == b["step"]
                start = (b["core_start"] - a["core_start"]) * 1000
                end = (b["core_end"] - a["core_end"]) * 1000
                rows.append(dict(repeat=repeat, step=a["step"], rank0=a, rank1=b,
                                 start_offset_ms=start, end_offset_ms=end,
                                 target_distance=abs(abs(start) - .2) + abs(abs(end) - .2)))
        assert len(rows) == 150
        chosen = min(rows, key=lambda r: r["target_distance"])
        evidence[impl] = dict(rows=rows, selected=chosen, selected_index=rows.index(chosen), input_sha256=hashes,
                              pooled_median_abs_start_ms=statistics.median(abs(r["start_offset_ms"]) for r in rows),
                              pooled_median_abs_end_ms=statistics.median(abs(r["end_offset_ms"]) for r in rows))

    fig = plt.figure(figsize=(15, 9.5), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.45])
    fig.suptitle("Optimizer CUDA Graph + bucket views: real unprofiled step boundaries", fontsize=17, weight="bold")
    names = {"mini_ddp": "MiniDDP rebuild", "torch_ddp": "PyTorch DDP"}
    for row_index, (impl, data) in enumerate(evidence.items()):
        chosen = data["selected"]
        ranks = [chosen["rank0"], chosen["rank1"]]
        origin = min(r["core_start"] for r in ranks)
        starts = [(r["core_start"] - origin) * 1000 for r in ranks]
        ends = [(r["core_end"] - origin) * 1000 for r in ranks]
        for col, title in enumerate(("Full core interval", "Start zoom", "End zoom")):
            ax = fig.add_subplot(grid[row_index, col])
            for rank in (0, 1):
                ax.broken_barh([(starts[rank], ends[rank] - starts[rank])], (1-rank-.16, .32),
                               facecolors=COLORS[rank], alpha=.85)
                ax.scatter([starts[rank], ends[rank]], [1-rank, 1-rank], color=COLORS[rank], marker="|", s=230)
            ax.set(yticks=[1, 0], yticklabels=["rank 0", "rank 1"], ylim=(-.6, 1.6))
            ax.grid(axis="x", alpha=.25)
            ax.set_axisbelow(True)
            ax.set_xlabel("ms from earliest rank core start", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_xlim(-.4, max(ends)+.4)
                ax.set_title(f"{names[impl]} | repeat {chosen['repeat']}, step {chosen['step']}\n{title}", fontsize=10)
                ax.text(.5, .05, "Whole core interval; not continuous GPU busy time", transform=ax.transAxes,
                        ha="center", fontsize=8)
            elif col == 1:
                ax.set_xlim(-.05, max(starts)+.35)
                ax.set_title(f"{title}: |offset| = {abs(chosen['start_offset_ms']):.3f} ms", fontsize=11)
            else:
                ax.set_xlim(min(ends)-.35, max(ends)+.05)
                ax.set_title(f"{title}: |offset| = {abs(chosen['end_offset_ms']):.3f} ms", fontsize=11)

    bottom = grid[2, :].subgridspec(1, 2)
    for column, (impl, data) in enumerate(evidence.items()):
        ax = fig.add_subplot(bottom[0, column])
        ax.axhspan(.1, .3, color="#eeeeee")
        ax.axhline(.2, color="#888888", linestyle="--", linewidth=1, label="0.2 ms target")
        for field, color, marker, label in (("start_offset_ms", "#207e73", "o", "start"),
                                             ("end_offset_ms", "#c17b26", ".", "end")):
            vals = [abs(r[field]) for r in data["rows"]]
            med = statistics.median(vals)
            ax.scatter(range(150), vals, s=13, marker=marker, color=color, alpha=.65,
                       label=f"{label}: pooled median {med:.3f} ms")
            k = data["selected_index"]
            ax.scatter(k, vals[k], s=150, marker="*", color="#bd3030", edgecolors="white", zorder=4)
        for x in (49.5, 99.5):
            ax.axvline(x, color="#bbb", linewidth=.7)
        ax.set(title=f"{names[impl]}: ALL 150 measured steps", xlabel="Measured step index (3 repeats x 50)",
               ylabel="Absolute rank offset (ms)", ylim=(-.08, 3.35))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.supxlabel("Top examples were selected for offsets near 0.2 ms, not as median or random steps. Red stars locate them below.\n"
                  "CPU wall-clock boundaries around synchronized GPU work; no profiler and no internal kernel timing. Step/repeat indices are zero-based.",
                  fontsize=10)
    fig.savefig(ASSETS / "unprofiled_view_boundaries.png", dpi=160)
    fig.savefig(ASSETS / "unprofiled_view_boundaries.svg")
    plt.close(fig)
    output = dict(selection_rule="minimize |abs(start_offset)-0.2| + |abs(end_offset)-0.2| over 150 measured steps per implementation",
                  clock="same-host perf_counter; CPU core boundaries, not GPU kernel timestamps", evidence=evidence)
    (ASSETS / "unprofiled_view_boundaries.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for impl, data in evidence.items():
        s = data["selected"]
        print(impl, "repeat", s["repeat"], "step", s["step"], "start/end offsets", s["start_offset_ms"], s["end_offset_ms"])


if __name__ == "__main__":
    main()
