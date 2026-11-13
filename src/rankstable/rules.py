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
    wins = np.zeros((table.n_models, table.n_models), dtype=np.int32)
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


#: Largest number of models for which Kemeny-Young is solved exactly.
KEMENY_EXACT_LIMIT = 12


def kemeny_score(wins: np.ndarray, order: list[int] | np.ndarray) -> int:
    """Pairwise agreement of an order: ``wins[i, k]`` summed over each ``i`` placed before ``k``."""
    placed = np.asarray(order, dtype=np.int64)
    return int(np.triu(np.asarray(wins)[np.ix_(placed, placed)], k=1).sum())


def _kemeny_exact(wins: np.ndarray) -> list[int]:
    """The best order by dynamic programming over subsets, in O(2^n n) steps.

    ``best[S]`` is the largest agreement any order of the subset ``S`` can reach; the model
    placed first in ``S`` collects its wins over the rest of ``S``. Among equally good orders
    the one that puts lower indices first is returned.
    """
    n = wins.shape[0]
    size = 1 << n
    member = (np.arange(size)[:, None] >> np.arange(n)) & 1
    ahead = (member @ wins.T.astype(np.int64)).tolist()
    best = [0] * size
    for subset in range(1, size):
        best[subset] = max(
            best[subset ^ (1 << i)] + ahead[subset ^ (1 << i)][i]
            for i in range(n)
            if subset >> i & 1
        )
    order = []
    subset = size - 1
    while subset:
        for i in range(n):
            rest = subset ^ (1 << i)
            if subset >> i & 1 and best[rest] + ahead[rest][i] == best[subset]:
                order.append(i)
                subset = rest
                break
    return order


def _kemeny_local_search(wins: np.ndarray, max_sweeps: int = 100) -> list[int]:
    """A heuristic order: start from the total number of wins, then move single models.

    Each sweep takes every model in turn and moves it to the position that raises the pairwise
    agreement most, until a sweep moves nobody or ``max_sweeps`` is reached. The result cannot
    be improved by moving one model, but it is not guaranteed to be the best order.
    """
    n = wins.shape[0]
    margin = wins.astype(np.int64) - wins.T
    order = sorted(range(n), key=lambda i: (-int(wins[i].sum()), i))
    for _ in range(max_sweeps):
        moved = False
        for model in list(order):
            position = order.index(model)
            ahead_of = margin[model, order]
            # gain[q]: change in agreement when the model is taken out and put back at slot q
            before = np.concatenate((np.cumsum(ahead_of[:position][::-1])[::-1], [0]))
            after = -np.cumsum(ahead_of[position + 1 :])
            gain = np.concatenate((before, after))
            target = int(np.argmax(gain))
            if gain[target] > 0:
                order.pop(position)
                order.insert(target, model)
                moved = True
        if not moved:
            break
    return order


def kemeny_young(table: ScoreTable, *, exact_limit: int = KEMENY_EXACT_LIMIT) -> Ranking:
    """Kemeny-Young: the order that agrees with the benchmarks on the most pairs.

    An order scores ``wins[i, k]`` (see :func:`pairwise_wins`) for every pair it puts in the
    order ``i`` before ``k``. With at most ``exact_limit`` models the best order is found
    exactly; above that a local search is used and the result is only a heuristic. The value
    of a model is its position in that order, so lower is better and there are no ties. When
    several orders are equally good, the one that is first alphabetically is reported.
    """
    scored = [i for i in np.argsort(np.array(table.models, dtype=object), kind="stable")]
    scored = [int(i) for i in scored if not np.isnan(table.scores[i]).all()]
    wins = pairwise_wins(table)[np.ix_(scored, scored)]
    order = _kemeny_exact(wins) if len(scored) <= exact_limit else _kemeny_local_search(wins)
    position = np.full(table.n_models, np.nan)
    for place, index in enumerate(order, start=1):
        position[scored[index]] = place
    return Ranking.from_values("kemeny", table.models, position, higher_is_better=False)


def bradley_terry(
    table: ScoreTable, *, prior: float = 0.5, max_iter: int = 2000, tol: float = 1e-10
) -> Ranking:
    """Bradley-Terry strengths fitted to the pairwise wins; the value is the log-strength.

    Every benchmark is a round of contests between the models scored on it: a better score is
    a win and a tie is half a win for each side. The model says ``i`` beats ``k`` with
    probability ``p_i / (p_i + p_k)``, and the strengths are fitted by the
    minorisation-maximisation update ``p_i <- W_i / sum_k N_ik / (p_i + p_k)``, with ``W_i``
    the wins of ``i`` and ``N_ik`` the contests between ``i`` and ``k``.

    ``prior`` adds that many wins and as many losses against a virtual opponent of strength 1,
    which keeps the strengths finite when a model wins or loses everything. With ``prior=0``
    the plain maximum-likelihood fit is attempted and the log-strengths are centred on zero.
    """
    if prior < 0:
        raise ValueError("prior cannot be negative")
    wins, _ = _benchmark_wins(table)
    present = ~np.isnan(wins)
    scored = np.flatnonzero(present.any(axis=1))
    total = np.where(present, wins, 0.0).sum(axis=1)[scored] + prior
    membership = present[scored].astype(np.float64)
    contests = membership @ membership.T
    np.fill_diagonal(contests, 0.0)
    strength = np.ones(scored.size)
    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore"):
            load = (contests / (strength[:, None] + strength[None, :])).sum(axis=1)
            updated = total / (load + 2 * prior / (strength + 1))
        updated = np.where(np.isfinite(updated), updated, strength)
        if prior == 0:
            positive = updated > 0
            if positive.any():
                updated = updated / np.exp(np.log(updated[positive]).mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.abs(np.log(updated) - np.log(strength))
        strength = updated
        if not np.nanmax(shift, initial=0.0) > tol:
            break
    value = np.full(table.n_models, np.nan)
    with np.errstate(divide="ignore"):
        value[scored] = np.log(strength)
    return Ranking.from_values("bradley-terry", table.models, value)


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
    "kemeny": kemeny_young,
    "bradley-terry": bradley_terry,
}


def aggregate(table: ScoreTable, rule: str = "mean") -> Ranking:
    """Rank the models of ``table`` with the rule called ``rule`` (a key of :data:`RULES`)."""
    try:
        function = RULES[rule]
    except KeyError:
        raise ValueError(f"unknown rule {rule!r}; choose from {', '.join(RULES)}") from None
    return function(table)
