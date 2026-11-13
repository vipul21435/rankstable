"""How much of a ranking survives another rule, another set of benchmarks or another field."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .ranking import Ranking, kendall_tau, pair_counts
from .rules import aggregate
from .table import ScoreTable


def top_set(ranking: Ranking, k: int = 1) -> frozenset[str]:
    """The models ranked ``k`` or better. A tie at the boundary makes the set larger than ``k``."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return frozenset(m for m, rank in zip(ranking.models, ranking.ranks, strict=True) if rank <= k)


def top_k_overlap(a: Ranking, b: Ranking, k: int) -> float:
    """Share of the top ``k`` that two rankings have in common: ``|A & B| / max(|A|, |B|)``.

    ``A`` and ``B`` are the :func:`top_set` of each ranking. Without ties both have ``k``
    members and this is the usual ``|A & B| / k``.
    """
    first, second = top_set(a, k), top_set(b, k)
    return len(first & second) / max(len(first), len(second))


def _mean(values: list[float]) -> float:
    kept = [value for value in values if not math.isnan(value)]
    return sum(kept) / len(kept) if kept else math.nan


@dataclass(frozen=True)
class SubsetOutcome:
    """The ranking after leaving one set of benchmarks out, compared with the full ranking."""

    dropped: tuple[str, ...]
    top1: tuple[str, ...]
    top1_changed: bool
    topk_changed: bool
    tau: float


@dataclass(frozen=True)
class LeaveOutResult:
    """Leave-``k``-benchmarks-out for one rule; ``exhaustive`` is false for sampled subsets."""

    rule: str
    k: int
    top: int
    exhaustive: bool
    full_top1: tuple[str, ...]
    outcomes: tuple[SubsetOutcome, ...]

    @property
    def n_subsets(self) -> int:
        return len(self.outcomes)

    @property
    def top1_changed(self) -> float:
        """Share of subsets after which the set of first-ranked models differs."""
        return sum(o.top1_changed for o in self.outcomes) / len(self.outcomes)

    @property
    def topk_changed(self) -> float:
        """Share of subsets after which the top-``top`` set differs."""
        return sum(o.topk_changed for o in self.outcomes) / len(self.outcomes)

    @property
    def mean_tau(self) -> float:
        return _mean([o.tau for o in self.outcomes])

    @property
    def min_tau(self) -> float:
        kept = [o.tau for o in self.outcomes if not math.isnan(o.tau)]
        return min(kept) if kept else math.nan


def leave_out(
    table: ScoreTable,
    rule: str = "mean",
    *,
    k: int = 1,
    top: int = 5,
    max_subsets: int = 1000,
    seed: int = 0,
) -> LeaveOutResult:
    """Re-rank after leaving every set of ``k`` benchmarks out.

    All ``C(m, k)`` subsets are tried when there are at most ``max_subsets`` of them;
    otherwise ``max_subsets`` distinct subsets are drawn with the given seed. Each outcome
    records whether the first-ranked set and the top-``top`` set differ from the full table's
    and Kendall's tau-b between the two rankings.
    """
    m = table.n_benchmarks
    if not 1 <= k < m:
        raise ValueError(f"cannot leave {k} of {m} benchmark(s) out")
    if max_subsets < 1:
        raise ValueError("max_subsets must be at least 1")
    full = aggregate(table, rule)
    exhaustive = math.comb(m, k) <= max_subsets
    if exhaustive:
        subsets = list(itertools.combinations(range(m), k))
    else:
        rng = np.random.default_rng(seed)
        drawn: set[tuple[int, ...]] = set()
        while len(drawn) < max_subsets:
            drawn.add(tuple(sorted(rng.choice(m, size=k, replace=False).tolist())))
        subsets = sorted(drawn)
    full_top1, full_topk = top_set(full, 1), top_set(full, top)
    outcomes = []
    for subset in subsets:
        kept = [name for j, name in enumerate(table.benchmarks) if j not in subset]
        ranking = aggregate(table.select(benchmarks=kept), rule)
        top1 = top_set(ranking, 1)
        outcomes.append(
            SubsetOutcome(
                dropped=tuple(table.benchmarks[j] for j in subset),
                top1=tuple(sorted(top1)),
                top1_changed=top1 != full_top1,
                topk_changed=top_set(ranking, top) != full_topk,
                tau=kendall_tau(full.ranks, ranking.ranks),
            )
        )
    return LeaveOutResult(rule, k, top, exhaustive, tuple(sorted(full_top1)), tuple(outcomes))


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Bootstrap over benchmarks for one rule.

    ``p_top1[i]`` and ``p_topk[i]`` are the shares of resamples in which ``models[i]`` is ranked
    first and within the top ``top``; ``rank_low`` and ``rank_high`` bound its rank at the
    chosen confidence. ``tau_low`` and ``tau_high`` bound Kendall's tau-b with the full ranking.
    """

    rule: str
    n_resamples: int
    top: int
    confidence: float
    seed: int
    models: tuple[str, ...]
    full_top1: tuple[str, ...]
    top1_changed: float
    topk_changed: float
    mean_tau: float
    tau_low: float
    tau_high: float
    p_top1: np.ndarray
    p_topk: np.ndarray
    rank_low: np.ndarray
    rank_high: np.ndarray


def bootstrap_benchmarks(
    table: ScoreTable,
    rule: str = "mean",
    *,
    n_resamples: int = 1000,
    top: int = 5,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Re-rank on benchmark sets drawn with replacement from the table's own benchmarks.

    Each resample draws ``m`` benchmarks with replacement, so some count twice and some drop
    out. Intervals are percentile intervals over the resamples.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    m = table.n_benchmarks
    full = aggregate(table, rule)
    full_top1, full_topk = top_set(full, 1), top_set(full, top)
    ranks = np.empty((n_resamples, table.n_models), dtype=np.int64)
    taus = np.empty(n_resamples)
    top1_changed = 0
    topk_changed = 0
    for b in range(n_resamples):
        ranking = aggregate(table.resample_benchmarks(rng.integers(0, m, size=m)), rule)
        ranks[b] = ranking.ranks
        taus[b] = kendall_tau(full.ranks, ranking.ranks)
        top1_changed += top_set(ranking, 1) != full_top1
        topk_changed += top_set(ranking, top) != full_topk
    tail = 100 * (1 - confidence) / 2
    rank_low, rank_high = np.percentile(ranks, [tail, 100 - tail], axis=0)
    defined = taus[~np.isnan(taus)]
    tau_low, tau_high = (
        np.percentile(defined, [tail, 100 - tail]) if defined.size else (math.nan, math.nan)
    )
    return BootstrapResult(
        rule=rule,
        n_resamples=n_resamples,
        top=top,
        confidence=confidence,
        seed=seed,
        models=table.models,
        full_top1=tuple(sorted(full_top1)),
        top1_changed=top1_changed / n_resamples,
        topk_changed=topk_changed / n_resamples,
        mean_tau=float(defined.mean()) if defined.size else math.nan,
        tau_low=float(tau_low),
        tau_high=float(tau_high),
        p_top1=(ranks == 1).mean(axis=0),
        p_topk=(ranks <= top).mean(axis=0),
        rank_low=rank_low,
        rank_high=rank_high,
    )


@dataclass(frozen=True)
class RemovalResult:
    """What removing one model at a time does to the order of the models that stay.

    ``order_changed`` is the share of removals after which at least one pair of remaining
    models is ordered differently, ``mean_pairs_changed`` the mean share of remaining pairs
    affected, and ``top1_changed`` the share of removals after which the first place among the
    remaining models goes to someone else. ``worst`` names the removal that changes most pairs.
    """

    rule: str
    n_removals: int
    exhaustive: bool
    order_changed: float
    mean_pairs_changed: float
    top1_changed: float
    worst: tuple[str, float]


def remove_models(
    table: ScoreTable, rule: str = "mean", *, max_removals: int = 200, seed: int = 0
) -> RemovalResult:
    """Remove each model in turn and check whether the others keep their relative order.

    A rule that is independent of irrelevant alternatives never changes the order of the
    remaining models. With more than ``max_removals`` models a seeded sample is removed.
    """
    n = table.n_models
    if n < 3:
        raise ValueError("removing a model needs at least three models")
    if max_removals < 1:
        raise ValueError("max_removals must be at least 1")
    full = aggregate(table, rule)
    exhaustive = n <= max_removals
    candidates = (
        list(range(n))
        if exhaustive
        else sorted(
            np.random.default_rng(seed).choice(n, size=max_removals, replace=False).tolist()
        )
    )
    shares = []
    top_changes = 0
    for removed in candidates:
        rest = [i for i in range(n) if i != removed]
        ranking = aggregate(table.select(models=[table.models[i] for i in rest]), rule)
        before = full.ranks[rest]
        concordant, _, _, _, tied_both, pairs = pair_counts(before, ranking.ranks)
        shares.append(1 - (concordant + tied_both) / pairs)
        top_changes += not np.array_equal(before == before.min(), ranking.ranks == 1)
    worst = int(np.argmax(shares))
    return RemovalResult(
        rule=rule,
        n_removals=len(candidates),
        exhaustive=exhaustive,
        order_changed=sum(share > 0 for share in shares) / len(shares),
        mean_pairs_changed=sum(shares) / len(shares),
        top1_changed=top_changes / len(shares),
        worst=(table.models[candidates[worst]], shares[worst]),
    )


def benchmark_diversity(table: ScoreTable) -> float:
    """Mean Kendall distance between the rankings that the benchmarks induce.

    The distance between two benchmarks is ``(1 - tau_b) / 2`` over the models scored on
    both: 0 when they order the models identically, 1 when one reverses the other. The
    diversity is the mean over all pairs of benchmarks for which tau-b is defined, and
    ``nan`` when there is no such pair.
    """
    oriented = table.oriented()
    distances = []
    for a, b in itertools.combinations(range(table.n_benchmarks), 2):
        both = ~np.isnan(oriented[:, a]) & ~np.isnan(oriented[:, b])
        tau = kendall_tau(oriented[both, a], oriented[both, b])
        distances.append((1 - tau) / 2)
    return _mean(distances)
