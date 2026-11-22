"""Day 13 - read `nvidia-smi topo -m`, rank the GPU pairs by link, and choose where TP and PP go.

    python experiments/day13_megatron/topo_check.py                 # on the instance
    python experiments/day13_megatron/topo_check.py --file topo.txt  # a saved matrix (e.g. from a downloaded log)

Rules (GUIDE Block 1, Section 5):
  - TP all-reduces activations twice per layer per microbatch in forward and twice in backward, on the critical
    path: the TP pairs get the fastest links.
  - PP sends one activation per microbatch forward and one gradient back: it tolerates a slower link.
  - With DP=1 TP=2 PP=2 and Megatron's order, ranks (0,1) and (2,3) are TP pairs and (0,2), (1,3) PP pairs; global
    rank r runs on the r-th entry of CUDA_VISIBLE_DEVICES. So choose CUDA_VISIBLE_DEVICES=a,b,c,d with (a,b) and
    (c,d) on the best links.

Prints the link of every pair, the fastest pair (for the two-GPU smoke tests) and a 4-GPU order; writes
`pair=...` and `order=...` lines with --out. Link ranking: NV# (more links = faster) > PIX > PXB > PHB > NODE > SYS.
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
from pathlib import Path

RANK = {"PIX": 5, "PXB": 4, "PHB": 3, "NODE": 2, "SYS": 1}


def score(link: str) -> int:
    m = re.fullmatch(r"NV(\d+)", link)
    return 100 + int(m[1]) if m else RANK.get(link, 0)


def parse(text: str) -> dict[tuple[int, int], str]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)   # nvidia-smi colours the header
    rows = [l.split() for l in text.splitlines() if re.match(r"^GPU\d+\s", l)]
    n = len(rows)
    links = {}
    for i, row in enumerate(rows):
        for j in range(n):
            if i != j:
                links[(i, j)] = row[1 + j]
    return links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = Path(args.file).read_text() if args.file else subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True).stdout
    links = parse(text)
    gpus = sorted({i for i, _ in links})
    print(f"{len(gpus)} GPUs. Links (higher score = faster):")
    for i, j in itertools.combinations(gpus, 2):
        print(f"  GPU{i} - GPU{j}: {links[(i, j)]:<5} score {score(links[(i, j)])}")
    best = max(itertools.combinations(gpus, 2), key=lambda p: (score(links[p]), -p[0], -p[1]))
    print(f"fastest pair (two-GPU smoke tests): {best[0]},{best[1]} ({links[best]})")
    order = None
    if len(gpus) >= 4:
        # Choose 4 GPUs and split them into TP pairs (a,b), (c,d) maximising the WORST TP link, then the PP links.
        cands = []
        for quad in itertools.combinations(gpus, 4):
            for a, b, c, d in itertools.permutations(quad):
                if a < b and c < d and a < c:
                    tp = min(score(links[(a, b)]), score(links[(c, d)]))
                    pp = min(score(links[(a, c)]), score(links[(b, d)]))
                    cands.append(((tp, pp), (a, b, c, d)))
        (tp, pp), order = max(cands, key=lambda x: (x[0], [-v for v in x[1]]))
        a, b, c, d = order
        print(f"4-GPU order for DP=1 TP=2 PP=2: CUDA_VISIBLE_DEVICES={a},{b},{c},{d}")
        print(f"  TP pairs: GPU{a}-GPU{b} {links[(a, b)]}, GPU{c}-GPU{d} {links[(c, d)]}")
        print(f"  PP pairs: GPU{a}-GPU{c} {links[(a, c)]}, GPU{b}-GPU{d} {links[(b, d)]}")
        if len({links[p] for p in itertools.combinations(gpus, 2)}) == 1:
            print("  every pair has the same link (e.g. an NVSwitch board): any order is equally good here")
    if args.out:
        lines = [f"pair={best[0]},{best[1]}"] + ([f"order={','.join(map(str, order))}"] if order else [])
        Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
