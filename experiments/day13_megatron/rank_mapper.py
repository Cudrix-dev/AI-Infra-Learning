"""Day 13 - rank -> (TP, DP, PP) mapper and the process groups it implies (no torch, no GPU needed).

    python experiments/day13_megatron/rank_mapper.py                         # reference mapper, DP=1 TP=2 PP=2
    python experiments/day13_megatron/rank_mapper.py --impl mine             # YOUR mapper (my_coords / my_groups)
    python experiments/day13_megatron/rank_mapper.py --megatron /workspace/Megatron-LM   # also check RankGenerator
    python experiments/day13_megatron/rank_mapper.py --layouts all           # the larger layouts too (simulated)

What it does
------------
Megatron Core lays ranks out in the order "tp-cp-ep-dp-pp": the leftmost name changes FASTEST. With CP = EP = 1 a
global rank decomposes like a mixed-radix number whose lowest digit is the TP rank:

    rank = tp_rank + TP * (dp_rank + DP * pp_rank)

A process group of kind X is the set of ranks that agree on every coordinate EXCEPT X. For DP=1, TP=2, PP=2:

    rank  (tp, dp, pp)      TP group = same (dp, pp)   PP group = same (tp, dp)   DP group = same (tp, pp)
     0    (0, 0, 0)          [0, 1]                     [0, 2]                     [0]
     1    (1, 0, 0)          [0, 1]                     [1, 3]                     [1]
     2    (0, 0, 1)          [2, 3]                     [0, 2]                     [2]
     3    (1, 0, 1)          [2, 3]                     [1, 3]                     [3]

The model-parallel group ("tp-pp") is every rank with the same dp_rank. The embedding group is the first and the
last rank of each PP group (word embedding on stage 0, output layer on the last stage); the position-embedding group
is the first rank of each PP group only (learned positions live on stage 0).

Checks
------
1. The chosen mapper (ref or mine) against the reference closed form, for every layout in --layouts.
2. With --megatron PATH (needs torch, because parallel_state imports it): against Megatron's own
   `RankGenerator(tp, ep=1, dp, pp, cp=1, order="tp-cp-ep-dp-pp").get_ranks(...)` - the same call
   `initialize_model_parallel` makes - and `default_embedding_ranks` / `default_position_embedding_ranks`.
   This is pure Python: no process group is created, so it is a [simulated] check for layouts with more ranks than
   GPUs. train_gpt.py repeats the comparison inside the real job against the groups NCCL actually built.

Exit codes: 0 all checks passed, 1 a mismatch, 3 a YOU WRITE function still raises NotImplementedError.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (tp, dp, pp). The first is Day 13's job; the next three are the smoke tests and the 1-GPU reference; the rest are
# larger layouts you cannot run here (labelled [simulated] in the output).
DAY13_LAYOUTS = [(2, 1, 2), (2, 1, 1), (1, 1, 2), (1, 1, 1)]
EXTRA_LAYOUTS = [(2, 2, 2), (4, 2, 1), (1, 2, 4), (2, 4, 1), (8, 1, 2)]
GROUP_KINDS = ["tp", "pp", "dp", "mp", "embedding", "position_embedding"]


# ----------------------------------------------------------------------------------------------- reference mapper
def ref_coords(rank: int, tp: int, dp: int, pp: int) -> tuple[int, int, int]:
    """(tp_rank, dp_rank, pp_rank) of a global rank under Megatron's default order tp-cp-ep-dp-pp (CP = EP = 1)."""
    assert 0 <= rank < tp * dp * pp
    return rank % tp, (rank // tp) % dp, rank // (tp * dp)


def ref_groups(tp: int, dp: int, pp: int) -> dict[str, list[list[int]]]:
    """Every group of each kind, each group's ranks ascending, groups ordered by their smallest rank."""
    world = tp * dp * pp
    coords = {r: ref_coords(r, tp, dp, pp) for r in range(world)}

    def same(
        *keep,
    ):  # ranks that agree on the coordinates listed in `keep` (0 = tp, 1 = dp, 2 = pp)
        buckets: dict[tuple, list[int]] = {}
        for r in range(world):
            buckets.setdefault(tuple(coords[r][i] for i in keep), []).append(r)
        return sorted(buckets.values())

    pp_groups = same(0, 1)
    return {
        "tp": same(1, 2),  # vary tp only
        "pp": pp_groups,  # vary pp only
        "dp": same(0, 2),  # vary dp only
        "mp": same(1),  # vary tp and pp ("tp-pp"): one full model replica
        "embedding": [[g[0], g[-1]] if len(g) > 1 else [g[0]] for g in pp_groups],
        "position_embedding": [[g[0]] for g in pp_groups],
    }


# ------------------------------------------------------------------------------------------------- YOUR mapper
def my_coords(rank: int, tp: int, dp: int, pp: int) -> tuple[int, int, int]:
    """YOU WRITE (TODO 1, ~3 lines): return (tp_rank, dp_rank, pp_rank) for `rank` under order tp-cp-ep-dp-pp.

    Hint: write rank as a mixed-radix number. Which coordinate is the lowest digit, which the highest?
    Check by hand: TP=2, DP=1, PP=2 must give rank 2 -> (0, 0, 1) and rank 3 -> (1, 0, 1).
    """
    tp_rank = rank % tp
    dp_rank = (rank // tp) % dp
    pp_rank = rank // (tp * dp)
    return tp_rank, dp_rank, pp_rank


def my_groups(tp: int, dp: int, pp: int) -> dict[str, list[list[int]]]:
    """YOU WRITE (TODO 2, ~15 lines): the groups of each kind in GROUP_KINDS, built from my_coords.

    Return {"tp": [[...], ...], "pp": ..., "dp": ..., "mp": ..., "embedding": ..., "position_embedding": ...}.
    Each group ascending; the order of groups does not matter (the check sorts them).
    - a group of kind X = the ranks that share every coordinate except X ("mp" = share dp only);
    - embedding = [first, last] rank of each PP group ([first] if the PP group has one rank);
    - position_embedding = [first] rank of each PP group.
    """
    pp_buckets = {}
    tp_buckets = {}
    dp_buckets = {}
    mp_buckets = {}
    for r in range(tp * dp * pp):
        tp_rank, dp_rank, pp_rank = my_coords(r, tp, dp, pp)
        pp_buckets.setdefault((tp_rank, dp_rank), []).append(r)
        tp_buckets.setdefault((dp_rank, pp_rank), []).append(r)
        dp_buckets.setdefault((tp_rank, pp_rank), []).append(r)
        mp_buckets.setdefault(dp_rank, []).append(r)
    embedding_buckets = [
        [g[0], g[-1]] if len(g) > 1 else [g[0]] for g in pp_buckets.values()
    ]
    position_embedding_buckets = [[g[0]] for g in pp_buckets.values()]
    return {
        "tp": list(tp_buckets.values()),
        "pp": list(pp_buckets.values()),
        "dp": list(dp_buckets.values()),
        "mp": list(mp_buckets.values()),
        "embedding": embedding_buckets,
        "position_embedding": position_embedding_buckets,
    }


# ------------------------------------------------------------------------------------------------------ Megatron
def megatron_groups(tp: int, dp: int, pp: int) -> dict[str, list[list[int]]]:
    """The groups Megatron would build, from the same pure-Python helpers initialize_model_parallel calls."""
    from megatron.core.parallel_state import (
        RankGenerator,
        default_embedding_ranks,
        default_position_embedding_ranks,
    )

    gen = RankGenerator(tp=tp, ep=1, dp=dp, pp=pp, cp=1, order="tp-cp-ep-dp-pp")
    pp_groups = gen.get_ranks("pp")
    return {
        "tp": gen.get_ranks("tp"),
        "pp": pp_groups,
        "dp": gen.get_ranks("dp"),
        "mp": gen.get_ranks("tp-pp"),
        "embedding": [list(default_embedding_ranks(g)) for g in pp_groups],
        "position_embedding": [
            list(default_position_embedding_ranks(g)) for g in pp_groups
        ],
    }


def canon(groups: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    return {k: sorted(sorted(g) for g in groups[k]) for k in GROUP_KINDS}


def table(tp: int, dp: int, pp: int, coords_fn) -> str:
    g = ref_groups(tp, dp, pp)
    find = lambda kind, r: next(x for x in g[kind] if r in x)  # noqa: E731
    lines = [
        "rank  (tp, dp, pp)   TP group      PP group      DP group      MP group          embedding"
    ]
    for r in range(tp * dp * pp):
        c = coords_fn(r, tp, dp, pp)
        lines.append(
            f"{r:>4}  {str(c):<12}  {str(find('tp', r)):<12}  {str(find('pp', r)):<12}  "
            f"{str(find('dp', r)):<12}  {str(find('mp', r)):<16}  {find('embedding', r)}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--impl", choices=["ref", "mine"], default="ref")
    ap.add_argument(
        "--megatron",
        default=None,
        help="Megatron-LM checkout to import RankGenerator from",
    )
    ap.add_argument("--layouts", choices=["day13", "all"], default="day13")
    ap.add_argument("--json", default=None, help="write the groups and results here")
    args = ap.parse_args()

    coords_fn, groups_fn = (
        (ref_coords, ref_groups) if args.impl == "ref" else (my_coords, my_groups)
    )
    layouts = DAY13_LAYOUTS + (EXTRA_LAYOUTS if args.layouts == "all" else [])
    if args.megatron:
        sys.path.insert(0, str(Path(args.megatron).resolve()))

    try:
        print(
            f"Mapper: {args.impl}. Order tp-cp-ep-dp-pp (CP = EP = 1). Day 13 layout DP=1, TP=2, PP=2:\n"
        )
        print(table(2, 1, 2, coords_fn))
        results, ok = [], True
        for tp, dp, pp in layouts:
            world = tp * dp * pp
            mine = canon(groups_fn(tp, dp, pp))
            coords_ok = all(
                coords_fn(r, tp, dp, pp) == ref_coords(r, tp, dp, pp)
                for r in range(world)
            )
            ref_ok = mine == canon(ref_groups(tp, dp, pp))
            mg_ok = None
            if args.megatron:
                mg = megatron_groups(tp, dp, pp)
                mg_ok = mine == canon(mg)
            tag = (
                ""
                if (tp, dp, pp) in DAY13_LAYOUTS
                else " [simulated: more ranks than this host has]"
            )
            status = coords_ok and ref_ok and mg_ok is not False
            ok &= status
            print(
                f"\nTP={tp} DP={dp} PP={pp} (world {world}){tag}: coords {'ok' if coords_ok else 'FAIL'}, "
                f"groups vs closed form {'ok' if ref_ok else 'FAIL'}, "
                f"vs Megatron RankGenerator {'not run' if mg_ok is None else ('ok' if mg_ok else 'FAIL')}"
            )
            for k in GROUP_KINDS:
                print(f"  {k:<19} {mine[k]}")
            results.append(
                {
                    "tp": tp,
                    "dp": dp,
                    "pp": pp,
                    "groups": mine,
                    "coords_ok": coords_ok,
                    "ref_ok": ref_ok,
                    "megatron_ok": mg_ok,
                }
            )
    except NotImplementedError as e:
        print(f"\nINCOMPLETE (TODOs left): {e}")
        return 3
    if args.json:
        Path(args.json).write_text(
            json.dumps({"impl": args.impl, "results": results, "passed": ok}, indent=1)
        )
    where = " and Megatron's RankGenerator" if args.megatron else ""
    print(
        f"\nRANK MAPPER ({args.impl}, {len(layouts)} layouts, vs closed form{where}): "
        f"{'PASSED' if ok else 'FAILED'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
