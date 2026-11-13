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
    ``nan`` when either sequence is constant or shorter than two.

    The pairs are counted by sorting on the first sequence and counting inversions of the
    second with a merge sort, so the cost is O(n log^2 n) rather than O(n^2).
    """
    concordant, discordant, tied_x, tied_y, _, pairs = pair_counts(a, b)
    scale = math.sqrt((pairs - tied_x) * (pairs - tied_y))
    return (concordant - discordant) / scale if scale > 0 else math.nan


def pair_counts(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray
) -> tuple[int, int, int, int, int, int]:
    """Count the pairs of positions by how the two sequences order them.

    Returns ``(concordant, discordant, tied_a, tied_b, tied_both, pairs)``: pairs ordered the
    same way, pairs ordered oppositely, pairs tied in ``a``, pairs tied in ``b``, pairs tied
    in both (these are also counted in the two previous numbers) and all ``n(n-1)/2`` pairs.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("kendall_tau needs two one-dimensional sequences of the same length")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("kendall_tau does not accept nan")
    n = x.size
    pairs = n * (n - 1) // 2
    if n < 2:
        return 0, 0, 0, 0, 0, pairs
    order = np.lexsort((y, x))
    x = x[order]
    y = y[order]
    tied_x = _tied_pairs(np.diff(x) != 0)
    tied_both = _tied_pairs((np.diff(x) != 0) | (np.diff(y) != 0))
    tied_y = _tied_pairs(np.diff(np.sort(y)) != 0)
    # Within equal x the y are ascending, so an inversion of y is a discordant pair.
    discordant = _inversions(y)
    concordant = pairs - tied_x - tied_y + tied_both - discordant
    return concordant, discordant, tied_x, tied_y, tied_both, pairs


def _tied_pairs(changes: np.ndarray) -> int:
    """Pairs inside runs of equal neighbours; ``changes[i]`` says element i+1 starts a run."""
    edges = np.flatnonzero(np.concatenate(([True], changes, [True])))
    runs = np.diff(edges)
    return int((runs * (runs - 1) // 2).sum())


def _inversions(values: np.ndarray) -> int:
    """Number of pairs ``i < j`` with ``values[i] > values[j]``, by bottom-up merge sort.

    Each pass merges neighbouring sorted runs of length ``width`` with one stable sort. An
    element of a right-hand run that lands at position ``p`` of its merged block, having been
    at position ``width + q`` before, has ``p - q`` left-hand elements at or below it; the
    other ``width - (p - q)`` left-hand elements are larger and each forms one inversion.
    """
    n = values.size
    index = np.arange(n)
    sequence = values
    total = 0
    width = 1
    while width < n:
        block = index // (2 * width)
        order = np.lexsort((sequence, block))
        offset = order % (2 * width)
        from_right = offset >= width
        landed = index[from_right] % (2 * width)
        total += int((offset[from_right] - landed).sum())
        sequence = sequence[order]
        width *= 2
    return total


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
