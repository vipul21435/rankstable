"""Measure how close the Kemeny-Young local search comes to the best order.

    python examples/kemeny_gap.py
    python examples/kemeny_gap.py --profiles 60 --seed 0 --table scores.csv

Random tables of 10 to 12 models are small enough to solve exactly, so the order found by
the local search can be compared with the best one. The objective is the number of pairwise
wins an order keeps (see ``kemeny_score``). With ``--table`` the objective of the
local-search order on that table is also set beside the orders of ``copeland`` and
``mean-rank``, which is a sanity check and not a bound on the distance to the optimum.
"""

from __future__ import annotations

import argparse

import numpy as np

from rankstable import (
    Ranking,
    ScoreTable,
    aggregate,
    kemeny_score,
    kemeny_young,
    pairwise_wins,
    read_table,
)


def objective(table: ScoreTable, wins: np.ndarray, ranking: Ranking) -> int:
    """Pairwise wins kept by the order of a ranking (ties are listed alphabetically)."""
    index = {model: i for i, model in enumerate(table.models)}
    return kemeny_score(wins, [index[model] for model in ranking.order()])


def random_table(rng: np.random.Generator) -> ScoreTable:
    """A table whose benchmarks share a common skill signal of random strength plus noise."""
    n_models = int(rng.integers(10, 13))
    n_benchmarks = int(rng.integers(3, 13))
    signal = rng.uniform(0.0, 1.5)
    scores = signal * rng.normal(size=(n_models, 1)) + rng.normal(size=(n_models, n_benchmarks))
    return ScoreTable(
        tuple(f"m{i:02d}" for i in range(n_models)),
        tuple(f"b{j:02d}" for j in range(n_benchmarks)),
        scores,
    )


def study(n_profiles: int, seed: int) -> tuple[int, float, float]:
    """Number of profiles solved to the optimum, mean and largest relative shortfall."""
    rng = np.random.default_rng(seed)
    shortfalls = []
    for _ in range(n_profiles):
        table = random_table(rng)
        wins = pairwise_wins(table)
        best = objective(table, wins, kemeny_young(table, exact_limit=table.n_models))
        found = objective(table, wins, kemeny_young(table, exact_limit=0))
        shortfalls.append((best - found) / best)
    solved = sum(shortfall == 0 for shortfall in shortfalls)
    return solved, float(np.mean(shortfalls)), float(np.max(shortfalls))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", type=int, default=60, help="random tables (default 60)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--table", default=None, help="also report objectives on this table")
    args = parser.parse_args()

    solved, mean, worst = study(args.profiles, args.seed)
    print(
        f"{args.profiles} random tables of 10 to 12 models and 3 to 12 benchmarks, seed {args.seed}"
    )
    print(
        f"local search reached the best order in {solved} of {args.profiles} "
        f"({100 * solved / args.profiles:.1f} %)"
    )
    print(f"shortfall of the objective: mean {100 * mean:.3f} %, largest {100 * worst:.2f} %")

    if args.table:
        table = read_table(args.table)
        wins = pairwise_wins(table)
        total = int(np.triu(wins + wins.T, k=1).sum())
        print(f"{args.table}: {table.n_models} models, {total} decided pairwise comparisons")
        for rule in ("kemeny", "copeland", "mean-rank"):
            kept = objective(table, wins, aggregate(table, rule))
            print(f"  kept by the {rule} order: {kept} ({100 * kept / total:.3f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
