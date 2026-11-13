"""Agreement of each benchmark with the aggregate of the others."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ranking import kendall_tau
from .rules import aggregate
from .table import ScoreTable


@dataclass(frozen=True)
class BenchmarkAgreement:
    """Kendall's tau-b between one benchmark and the aggregate of all the others.

    ``n_models`` counts the models that have a score on the benchmark and a value from the
    others; ``low`` and ``high`` are a percentile bootstrap interval over those models.
    """

    benchmark: str
    n_models: int
    tau: float
    low: float
    high: float


def benchmark_agreement(
    table: ScoreTable,
    rule: str = "win-rate",
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[BenchmarkAgreement]:
    """Rank correlation of every benchmark with the ranking that the others give without it.

    For benchmark ``j`` the other benchmarks are aggregated with ``rule`` and Kendall's tau-b
    is taken between that ranking and the scores on ``j``. The interval resamples the models
    (pairs of values) with replacement ``n_resamples`` times and takes percentiles; the two
    rankings themselves are not recomputed inside a resample. Results come in table order.
    """
    if table.n_benchmarks < 2:
        raise ValueError("agreement needs at least two benchmarks")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    oriented = table.oriented()
    tail = 100 * (1 - confidence) / 2
    results = []
    for j, name in enumerate(table.benchmarks):
        others = table.select(benchmarks=[other for other in table.benchmarks if other != name])
        rest = aggregate(others, rule)
        both = ~np.isnan(oriented[:, j]) & ~np.isnan(rest.values)
        own = oriented[both, j]
        reference = -rest.ranks[both].astype(float)
        n = int(both.sum())
        tau = kendall_tau(own, reference)
        resampled = np.empty(n_resamples)
        for b in range(n_resamples):
            draw = rng.integers(0, n, size=n) if n else np.empty(0, dtype=np.int64)
            resampled[b] = kendall_tau(own[draw], reference[draw])
        defined = resampled[~np.isnan(resampled)]
        low, high = (
            np.percentile(defined, [tail, 100 - tail]) if defined.size else (math.nan, math.nan)
        )
        results.append(BenchmarkAgreement(name, n, tau, float(low), float(high)))
    return results
