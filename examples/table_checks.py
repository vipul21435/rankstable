"""Report properties of a score table that shape its rankings.

    python examples/table_checks.py scores.csv

Zero scores matter for the geometric mean: a model with a zero on any benchmark gets a
geometric mean of zero and ties with every other such model. The script counts them and
gives Kendall's tau-b between ``geomean`` and other rules with and without those models.
"""

from __future__ import annotations

import argparse

from rankstable import ScoreTable, aggregate, kendall_tau, read_table

OTHER_RULES = ("mean", "minmax", "zscore", "mean-rank", "copeland")


def report_zero_scores(table: ScoreTable) -> None:
    zero = table.scores == 0
    with_zero = zero.any(axis=1)
    print(f"models with a zero score on at least one benchmark: {int(with_zero.sum())}")
    counts = zip(table.benchmarks, zero.sum(axis=0), strict=True)
    print("  per benchmark: " + ", ".join(f"{name} {int(count)}" for name, count in counts))
    if not with_zero.any() or with_zero.all():
        return
    try:
        geomean = aggregate(table, "geomean")
    except ValueError as error:
        print(f"  geomean cannot rank this table: {error}")
        return
    at_zero = geomean.values == 0
    print(
        f"  models with a geometric mean of zero: {int(at_zero.sum())}, "
        f"sharing rank {int(geomean.ranks[at_zero].min())} of {table.n_models}"
    )
    kept = table.select(
        models=[model for model, flag in zip(table.models, with_zero, strict=True) if not flag]
    )
    kept_geomean = aggregate(kept, "geomean")
    print(
        "Kendall tau-b between geomean and other rules: "
        f"all {table.n_models} models / the {kept.n_models} models without a zero score"
    )
    for rule in OTHER_RULES:
        full = kendall_tau(aggregate(table, rule).ranks, geomean.ranks)
        part = kendall_tau(aggregate(kept, rule).ranks, kept_geomean.ranks)
        print(f"  {rule:10s} {full:.3f} / {part:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", help="wide CSV or long JSONL")
    args = parser.parse_args()
    table = read_table(args.table)
    print(f"{table.n_models} models, {table.n_benchmarks} benchmarks")
    report_zero_scores(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
