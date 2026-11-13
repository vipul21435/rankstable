"""Draw a critical-difference diagram for the models with the best mean rank in a table.

    python examples/cd_figure.py scores.csv figure.png --top 10

The models are labelled by their position on the full table (``model 01`` has the best mean
rank) instead of their names. Needs matplotlib: ``pip install "rankstable[plot]"``.
"""

from __future__ import annotations

import argparse

from rankstable import ScoreTable, aggregate, friedman_test, read_table
from rankstable.plot import plot_critical_difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", help="wide CSV or long JSONL without missing cells")
    parser.add_argument("output", help="path of the image to write")
    parser.add_argument("--top", type=int, default=10, help="number of models (default 10)")
    parser.add_argument("--title", default=None, help="title of the diagram")
    args = parser.parse_args()

    table = read_table(args.table)
    best = table.select(models=aggregate(table, "mean-rank").order()[: args.top])
    labels = tuple(f"model {place:02d}" for place in range(1, best.n_models + 1))
    result = friedman_test(ScoreTable(labels, best.benchmarks, best.scores, best.higher_is_better))
    plot_critical_difference(result, args.output, title=args.title)
    print(
        f"{best.n_models} models, {best.n_benchmarks} benchmarks: "
        f"chi-square {result.chi2:.3f} (p = {result.chi2_p:.4g}), "
        f"critical difference {result.critical_difference:.3f}"
    )
    for group in result.groups():
        print("not separated:", ", ".join(group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
