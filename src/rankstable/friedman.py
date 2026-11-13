"""The Friedman test over benchmarks and the Nemenyi critical difference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rules import benchmark_ranks
from .table import ScoreTable


@dataclass(frozen=True, eq=False)
class FriedmanResult:
    """Outcome of :func:`friedman_test`.

    ``mean_ranks[i]`` is the mean rank of ``models[i]`` over the benchmarks. ``chi2`` is
    Friedman's statistic and ``f`` the Iman-Davenport statistic, each with its p-value. Two
    models whose mean ranks differ by at least ``critical_difference`` differ significantly at
    level ``alpha`` by the Nemenyi test.
    """

    models: tuple[str, ...]
    mean_ranks: np.ndarray
    n_benchmarks: int
    chi2: float
    chi2_p: float
    f: float
    f_p: float
    alpha: float
    critical_difference: float

    def groups(self) -> list[tuple[str, ...]]:
        """Maximal runs of models, by mean rank, that span less than the critical difference."""
        order = sorted(range(len(self.models)), key=lambda i: (self.mean_ranks[i], self.models[i]))
        ranks = [float(self.mean_ranks[i]) for i in order]
        found: list[tuple[int, int]] = []
        for start in range(len(order)):
            stop = start
            while (
                stop + 1 < len(order) and ranks[stop + 1] - ranks[start] < self.critical_difference
            ):
                stop += 1
            if stop > start and not (found and found[-1][1] >= stop):
                found.append((start, stop))
        return [tuple(self.models[order[i]] for i in range(a, b + 1)) for a, b in found]


def friedman_test(table: ScoreTable, *, alpha: float = 0.05) -> FriedmanResult:
    """Test whether the models' mean ranks differ, treating each benchmark as a block.

    With ``N`` benchmarks, ``k`` models and mean ranks ``R_i`` the statistic is
    ``chi2 = 12 N / (k (k + 1)) * (sum R_i^2 - k (k + 1)^2 / 4)``, divided by
    ``1 - sum (t^3 - t) / (N k (k^2 - 1))`` over the groups of ``t`` tied scores, and compared
    with a chi-square distribution on ``k - 1`` degrees of freedom. The Iman-Davenport variant
    ``F = (N - 1) chi2 / (N (k - 1) - chi2)`` follows an F distribution on ``k - 1`` and
    ``(k - 1)(N - 1)`` degrees of freedom. The Nemenyi critical difference is
    ``q_alpha * sqrt(k (k + 1) / (6 N))``, with ``q_alpha`` the upper ``alpha`` point of the
    range of ``k`` standard normal variables divided by ``sqrt(2)``. The table must be complete.
    """
    if table.n_missing:
        raise ValueError(
            "the Friedman test needs a complete table; drop models or benchmarks with a "
            "missing cell first"
        )
    k, n = table.n_models, table.n_benchmarks
    if k < 2 or n < 2:
        raise ValueError("the Friedman test needs at least two models and two benchmarks")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    ranks = benchmark_ranks(table)
    mean_ranks = ranks.mean(axis=1)
    ties = 0.0
    for j in range(n):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        ties += float((counts**3 - counts).sum())
    correction = 1 - ties / (n * k * (k * k - 1))
    raw = 12 * n / (k * (k + 1)) * (float((mean_ranks**2).sum()) - k * (k + 1) ** 2 / 4)
    if correction <= 0:
        chi2 = chi2_p = f = f_p = math.nan
    else:
        chi2 = max(raw / correction, 0.0)
        chi2_p = chi2_sf(chi2, k - 1)
        rest = n * (k - 1) - chi2
        f = (n - 1) * chi2 / rest if rest > 1e-12 * n * k else math.inf
        f_p = f_sf(f, k - 1, (k - 1) * (n - 1))
    return FriedmanResult(
        models=table.models,
        mean_ranks=mean_ranks,
        n_benchmarks=n,
        chi2=chi2,
        chi2_p=chi2_p,
        f=f,
        f_p=f_p,
        alpha=alpha,
        critical_difference=nemenyi_q(k, alpha) * math.sqrt(k * (k + 1) / (6 * n)),
    )


def chi2_sf(x: float, df: float) -> float:
    """Upper tail of the chi-square distribution: the regularised incomplete gamma ``Q``."""
    a, half = df / 2, x / 2
    if half <= 0:
        return 1.0
    scale = math.exp(-half + a * math.log(half) - math.lgamma(a))
    if half < a + 1:
        term = total = 1 / a
        step = a
        for _ in range(100000):
            step += 1
            term *= half / step
            total += term
            if term < total * 1e-17:
                break
        return min(max(1 - total * scale, 0.0), 1.0)
    return min(
        max(scale * _lentz(lambda i: -i * (i - a), lambda i: half + 2 * i + 1 - a), 0.0), 1.0
    )


def _lentz(numerator, denominator) -> float:
    """``1 / (b0 + a1 / (b1 + a2 / (b2 + ...)))`` by the modified Lentz method."""
    tiny = 1e-300
    b = denominator(0)
    c, d = 1 / tiny, 1 / (b if abs(b) > tiny else tiny)
    value = d
    for i in range(1, 100000):
        a, b = numerator(i), denominator(i)
        d = a * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + a / c
        c = c if abs(c) > tiny else tiny
        value *= d * c
        if abs(d * c - 1) < 1e-16:
            break
    return value


def f_sf(x: float, df1: float, df2: float) -> float:
    """Upper tail of the F distribution, through the regularised incomplete beta function."""
    if math.isnan(x):
        return math.nan
    if x <= 0:
        return 1.0
    if math.isinf(x):
        return 0.0
    return _betainc(df2 / 2, df1 / 2, df2 / (df2 + df1 * x))


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * _beta_fraction(a, b, x) / a
    return 1 - front * _beta_fraction(b, a, 1 - x) / b


def _beta_fraction(a: float, b: float, x: float) -> float:
    def numerator(i: int) -> float:
        m = i // 2
        if i % 2 == 0:
            return m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        return -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))

    return _lentz(numerator, lambda i: 1.0)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1 + math.erf(value / math.sqrt(2))) for value in z])


def range_cdf(r: float, k: int) -> float:
    """Probability that the range of ``k`` independent standard normals is at most ``r``."""
    z, step = np.linspace(-9.0, 9.0, 1801, retstep=True)
    density = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    inner = np.clip(_normal_cdf(z) - _normal_cdf(z - r), 0.0, 1.0)
    integrand = k * density * inner ** (k - 1)
    return float((integrand[:-1] + integrand[1:]).sum() * step / 2)


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Critical value ``q_alpha`` of the Nemenyi test for ``k`` models.

    It is the point that the range of ``k`` standard normal variables exceeds with probability
    ``alpha`` (the Studentized range with infinite degrees of freedom), divided by ``sqrt(2)``.
    """
    if k < 2:
        raise ValueError("the Nemenyi test needs at least two models")
    low, high = 0.0, 30.0
    for _ in range(40):
        middle = (low + high) / 2
        if range_cdf(middle, k) < 1 - alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2 / math.sqrt(2)
