"""Ranks, tie handling and rank correlation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Two aggregate values closer than this fraction of the largest absolute value are tied.
TIE_REL_TOL = 1e-12


def average_ranks(
    values: Sequence[float] | np.ndarray, *, higher_is_better: bool = True
) -> np.ndarray:
    """Fractional ranks with 1 for the best value; equal values share their average rank.

    ``nan`` entries are left out of the ranking and stay ``nan``. Ties are exact equality,
    which is the right notion for raw scores on one benchmark.
    """
    column = np.asarray(values, dtype=float)
    ranks = np.full(column.shape, np.nan)
    present = ~np.isnan(column)
    if not present.any():
        return ranks
    key = column[present] if higher_is_better else -column[present]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    strictly_better = key.size - np.cumsum(counts)
    ranks[present] = (strictly_better + (counts + 1) / 2.0)[inverse]
    return ranks


def competition_ranks(
    values: Sequence[float] | np.ndarray,
    *,
    higher_is_better: bool = True,
    rel_tol: float = TIE_REL_TOL,
    scale: float = 0.0,
) -> np.ndarray:
    """Integer ranks with 1 for the best value; tied values share the smallest rank (1, 2, 2, 4).

    Values are tied when they differ by at most ``rel_tol`` times the largest absolute value,
    so that two aggregates that are equal on paper do not get ordered by rounding error. The
    grouping walks the sorted values and starts a new group at every larger gap. ``scale`` is
    the size of the terms the values were computed from; pass it when terms can cancel, since
    the rounding error of a sum is relative to its terms and not to the sum. ``nan`` entries
    share the last place, one below the number of ranked values.
    """
    column = np.asarray(values, dtype=float)
    present = np.flatnonzero(~np.isnan(column))
    ranks = np.full(column.shape, present.size + 1, dtype=np.int64)
    if present.size == 0:
        return ranks
    key = column[present] if higher_is_better else -column[present]
    order = np.argsort(-key, kind="stable")
    descending = key[order]
    tolerance = rel_tol * max(float(np.max(np.abs(descending))), scale)
    starts_group = np.ones(order.size, dtype=bool)
    starts_group[1:] = (descending[:-1] - descending[1:]) > tolerance
    group_start = np.maximum.accumulate(np.where(starts_group, np.arange(order.size), 0))
    ranks[present[order]] = group_start + 1
    return ranks


def kendall_tau(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Kendall's tau-b between two equally long sequences of numbers.

    ``tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))`` with ``C`` concordant and ``D``
    discordant pairs, ``n0 = n(n-1)/2`` pairs in all, and ``n1``, ``n2`` the pairs tied in the
    first and in the second sequence. It is 1 for the same order, -1 for the reverse order and
    ``nan`` when either sequence is constant or shorter than two. Runs in O(n^2) time.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("kendall_tau needs two one-dimensional sequences of the same length")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("kendall_tau does not accept nan")
    n = x.size
    balance = 0
    tied_x = 0
    tied_y = 0
    for i in range(n - 1):
        dx = np.sign(x[i + 1 :] - x[i])
        dy = np.sign(y[i + 1 :] - y[i])
        balance += int(np.dot(dx, dy))
        tied_x += int(np.count_nonzero(dx == 0))
        tied_y += int(np.count_nonzero(dy == 0))
    pairs = n * (n - 1) // 2
    scale = math.sqrt((pairs - tied_x) * (pairs - tied_y))
    return balance / scale if scale > 0 else math.nan


@dataclass(frozen=True, eq=False)
class Ranking:
    """The outcome of one aggregation rule.

    ``values[i]`` is the rule's own number for ``models[i]`` (a mean score, a mean rank, a
    count of wins) and ``higher_is_better`` says which way it points. ``ranks`` are
    competition ranks with 1 for the best model; a model for which the rule has no value
    (``nan``) is placed last.
    """

    rule: str
    models: tuple[str, ...]
    values: np.ndarray
    higher_is_better: bool
    ranks: np.ndarray

    @classmethod
    def from_values(
        cls,
        rule: str,
        models: Sequence[str],
        values: Sequence[float] | np.ndarray,
        *,
        higher_is_better: bool = True,
        scale: float = 0.0,
    ) -> Ranking:
        """Build a ranking from one value per model.

        ``scale`` is passed on to :func:`competition_ranks` for the tie tolerance.
        """
        column = np.array(values, dtype=float)
        if column.shape != (len(models),):
            raise ValueError(f"expected {len(models)} values, got shape {column.shape}")
        ranks = competition_ranks(column, higher_is_better=higher_is_better, scale=scale)
        column.setflags(write=False)
        ranks.setflags(write=False)
        return cls(rule, tuple(models), column, higher_is_better, ranks)

    def order(self) -> list[str]:
        """Models from best to worst; tied models are listed alphabetically."""
        return [self.models[i] for i in self._positions()]

    def rank_of(self, model: str) -> int:
        """The competition rank of one model."""
        try:
            return int(self.ranks[self.models.index(model)])
        except ValueError:
            raise KeyError(model) from None

    def rows(self) -> list[dict[str, object]]:
        """One ``{"rank", "model", "value"}`` record per model, best first.

        A missing value is ``None`` so that the records can go straight into JSON.
        """
        return [
            {
                "rank": int(self.ranks[i]),
                "model": self.models[i],
                "value": None if math.isnan(self.values[i]) else float(self.values[i]),
            }
            for i in self._positions()
        ]

    def _positions(self) -> list[int]:
        return sorted(range(len(self.models)), key=lambda i: (int(self.ranks[i]), self.models[i]))
