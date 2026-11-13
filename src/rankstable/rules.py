"""Aggregation rules: each turns a score table into one ranking of the models.

Every rule works on the cells that are present. Score-based rules average over the benchmarks
a model has a score on, rank-based rules rank the models that have a score on a benchmark, and
pairwise rules compare two models on the benchmarks both have. A model without any score gets
no value and is ranked last.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .ranking import Ranking, average_ranks
from .table import ScoreTable


def _row_mean(cells: np.ndarray) -> np.ndarray:
    """Mean of each row over its non-missing cells; ``nan`` for a row that has none."""
    present = ~np.isnan(cells)
    count = present.sum(axis=1)
    total = np.where(present, cells, 0.0).sum(axis=1)
    out = np.full(cells.shape[0], np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _magnitude(cells: np.ndarray) -> float:
    """Largest absolute present value, 0 when there is none."""
    largest = float(np.fmax.reduce(np.abs(cells), axis=None))
    return 0.0 if np.isnan(largest) else largest


def _column_range(oriented: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smallest and largest present value of each column (``nan`` for an empty column)."""
    return np.fmin.reduce(oriented, axis=0), np.fmax.reduce(oriented, axis=0)


def arithmetic_mean(table: ScoreTable) -> Ranking:
    """Mean of the raw scores, with lower-is-better benchmarks negated."""
    oriented = table.oriented()
    return Ranking.from_values(
        "mean", table.models, _row_mean(oriented), scale=_magnitude(oriented)
    )


def minmax_mean(table: ScoreTable) -> Ranking:
    """Mean of min-max normalised scores.

    On each benchmark the worst present score maps to 0 and the best to 1. A benchmark on
    which all present scores are equal says nothing about the order and is skipped.
    """
    oriented = table.oriented()
    low, high = _column_range(oriented)
    spread = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (oriented - low) / spread
    scaled[:, ~(spread > 0)] = np.nan
    return Ranking.from_values("minmax", table.models, _row_mean(scaled), scale=1.0)


def zscore_mean(table: ScoreTable) -> Ranking:
    """Mean of z-normalised scores.

    Each benchmark is centred on the mean of its present scores and divided by their
    population standard deviation. A benchmark on which all present scores are equal is
    skipped.
    """
    oriented = table.oriented()
    low, high = _column_range(oriented)
    present = ~np.isnan(oriented)
    count = np.maximum(present.sum(axis=0), 1)
    centre = np.where(present, oriented, 0.0).sum(axis=0) / count
    deviation = oriented - centre
    variance = np.where(present, deviation**2, 0.0).sum(axis=0) / count
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = deviation / np.sqrt(variance)
    scaled[:, ~(high > low)] = np.nan
    return Ranking.from_values("zscore", table.models, _row_mean(scaled), scale=_magnitude(scaled))


def geometric_mean(table: ScoreTable) -> Ranking:
    """Geometric mean of the scores, taking ``1 / score`` on lower-is-better benchmarks.

    Scores must not be negative. A zero on a higher-is-better benchmark makes the geometric
    mean zero; a zero on a lower-is-better benchmark has no reciprocal and is an error.
    """
    scores = table.scores
    lower = ~np.asarray(table.higher_is_better, dtype=bool)
    for j, name in enumerate(table.benchmarks):
        column = scores[:, j]
        if np.any(column < 0):
            raise ValueError(
                f"the geometric mean needs non-negative scores; benchmark {name!r} has a "
                "negative one"
            )
        if lower[j] and np.any(column == 0):
            raise ValueError(
                f"the geometric mean takes 1/score on lower-is-better benchmark {name!r}, "
                "which has a zero score"
            )
    with np.errstate(divide="ignore"):
        logs = np.log(scores)
    logs[:, lower] = -logs[:, lower]
    return Ranking.from_values("geomean", table.models, np.exp(_row_mean(logs)))


def benchmark_ranks(table: ScoreTable) -> np.ndarray:
    """Rank of every model on every benchmark, 1 for the best, ties sharing the average rank.

    The ranking on a benchmark covers the models that have a score there; the other cells
    are ``nan``.
    """
    oriented = table.oriented()
    return np.column_stack([average_ranks(oriented[:, j]) for j in range(table.n_benchmarks)])


def _benchmark_wins(table: ScoreTable) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, the number of models beaten on that benchmark (a tie counts one half).

    Also returns the number of models with a score on each benchmark.
    """
    ranks = benchmark_ranks(table)
    present = (~np.isnan(ranks)).sum(axis=0)
    return present - ranks, present


def mean_rank(table: ScoreTable) -> Ranking:
    """Mean of the per-benchmark ranks over the benchmarks a model has a score on.

    Lower is better. Ranks are not rescaled, so with missing cells a rank on a benchmark that
    few models have a score on counts the same as that rank on a full benchmark; the mean win
    rate is the variant that corrects for this.
    """
    ranks = benchmark_ranks(table)
    return Ranking.from_values("mean-rank", table.models, _row_mean(ranks), higher_is_better=False)


def mean_win_rate(table: ScoreTable) -> Ranking:
    """Mean over benchmarks of the fraction of the other models that a model beats.

    A tie counts as half a win. A benchmark on which the model is the only one with a score
    offers no comparison and is skipped.
    """
    wins, present = _benchmark_wins(table)
    opponents = present - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = wins / opponents
    rate[:, opponents <= 0] = np.nan
    return Ranking.from_values("win-rate", table.models, _row_mean(rate))


def borda_count(table: ScoreTable) -> Ranking:
    """Borda count: on each benchmark one point per model beaten, half a point per tie.

    Points are summed, not averaged, so a benchmark on which a model has no score gives it
    no points.
    """
    wins, _ = _benchmark_wins(table)
    has_score = ~np.isnan(wins)
    points = np.where(has_score, wins, 0.0).sum(axis=1)
    points[~has_score.any(axis=1)] = np.nan
    return Ranking.from_values("borda", table.models, points)


def pairwise_wins(table: ScoreTable) -> np.ndarray:
    """``wins[i, k]``: the number of benchmarks on which model ``i`` scores better than ``k``.

    Only benchmarks on which both models have a score count, and ties count for neither.
    """
    oriented = table.oriented()
    wins = np.zeros((table.n_models, table.n_models), dtype=np.int64)
    for j in range(table.n_benchmarks):
        column = oriented[:, j]
        wins += column[:, None] > column[None, :]
    return wins


def copeland(table: ScoreTable) -> Ranking:
    """Copeland score: pairwise contests won minus pairwise contests lost.

    Model ``i`` wins its contest against ``k`` when it is better on more benchmarks than
    ``k`` is. A contest that is level, or that has no common benchmark, counts for neither.
    """
    wins = pairwise_wins(table)
    beats = wins > wins.T
    score = (beats.sum(axis=1) - beats.sum(axis=0)).astype(float)
    score[np.isnan(table.scores).all(axis=1)] = np.nan
    return Ranking.from_values("copeland", table.models, score)


#: Rule name to function, in the order used for reports.
RULES: dict[str, Callable[[ScoreTable], Ranking]] = {
    "mean": arithmetic_mean,
    "minmax": minmax_mean,
    "zscore": zscore_mean,
    "geomean": geometric_mean,
    "mean-rank": mean_rank,
    "win-rate": mean_win_rate,
    "borda": borda_count,
    "copeland": copeland,
}


def aggregate(table: ScoreTable, rule: str = "mean") -> Ranking:
    """Rank the models of ``table`` with the rule called ``rule`` (a key of :data:`RULES`)."""
    try:
        function = RULES[rule]
    except KeyError:
        raise ValueError(f"unknown rule {rule!r}; choose from {', '.join(RULES)}") from None
    return function(table)
