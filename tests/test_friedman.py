import math

import numpy as np
import pytest
from scipy import stats

from rankstable.friedman import chi2_sf, f_sf, friedman_test, nemenyi_q, range_cdf
from rankstable.table import ScoreTable


def table_of(rows):
    scores = np.array(rows, dtype=float)
    models = tuple(f"m{i:02d}" for i in range(scores.shape[0]))
    benchmarks = tuple(f"b{j + 1}" for j in range(scores.shape[1]))
    return ScoreTable(models, benchmarks, scores)


def test_friedman_by_hand():
    # Three models on four benchmarks, ranked 1 2 3 on three of them and 2 1 3 on the last.
    # Mean ranks 1.25, 1.75, 3. chi2 = 12 * 4 / (3 * 4) * (1.5625 + 3.0625 + 9 - 12) = 6.5
    table = table_of([[9, 9, 9, 5], [5, 5, 5, 9], [1, 1, 1, 1]])
    result = friedman_test(table)
    np.testing.assert_allclose(result.mean_ranks, [1.25, 1.75, 3])
    assert result.n_benchmarks == 4
    assert result.chi2 == pytest.approx(6.5)
    assert result.chi2_p == pytest.approx(math.exp(-3.25))  # chi-square on 2 df
    # Iman-Davenport: 3 * 6.5 / (4 * 2 - 6.5) = 13, on 2 and 6 degrees of freedom
    assert result.f == pytest.approx(13)
    assert result.f_p == pytest.approx((1 + 2 * 13 / 6) ** -3)
    # q_0.05 for three models is 2.343: CD = 2.343 * sqrt(3 * 4 / 24)
    assert result.critical_difference == pytest.approx(2.343 * math.sqrt(0.5), abs=1e-3)
    assert result.groups() == [("m00", "m01"), ("m01", "m02")]


def test_friedman_matches_scipy_with_and_without_ties():
    rng = np.random.default_rng(0)
    for levels in (1000, 4):
        for _ in range(8):
            k, n = int(rng.integers(3, 9)), int(rng.integers(3, 12))
            scores = rng.integers(0, levels, size=(k, n)).astype(float)
            if any(np.ptp(scores[:, j]) == 0 for j in range(n)) and levels == 4:
                continue
            result = friedman_test(table_of(scores))
            expected = stats.friedmanchisquare(*(-scores))
            assert result.chi2 == pytest.approx(expected.statistic, rel=1e-10)
            assert result.chi2_p == pytest.approx(expected.pvalue, rel=1e-8)


def test_tail_probabilities_match_scipy():
    for x, df in [(0.5, 1), (3.2, 2), (10, 7), (40, 25), (900, 1000), (5300, 4575), (0, 3)]:
        assert chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=1e-9, abs=1e-300)
    for x, d1, d2 in [(0.3, 2, 6), (13, 2, 6), (2.5, 9, 40), (1.1, 4575, 22875), (7, 1, 1)]:
        assert f_sf(x, d1, d2) == pytest.approx(stats.f.sf(x, d1, d2), rel=1e-8)
    assert f_sf(math.inf, 2, 6) == 0
    assert f_sf(0, 2, 6) == 1
    assert math.isnan(f_sf(math.nan, 2, 6))


def test_nemenyi_critical_values_match_the_published_table_and_scipy():
    # Demsar (2006), Table 5(a), which gives three decimals
    published = {
        0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
        0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
    }
    for alpha, values in published.items():
        for k, value in zip(range(2, 11), values, strict=True):
            assert nemenyi_q(k, alpha) == pytest.approx(value, abs=1.5e-3)
    for k in (3, 20):
        expected = stats.studentized_range.ppf(0.95, k, 1e6) / math.sqrt(2)
        assert nemenyi_q(k, 0.05) == pytest.approx(expected, rel=1e-4)
    assert range_cdf(0, 3) == 0
    assert range_cdf(25, 3) == pytest.approx(1)
    with pytest.raises(ValueError, match="at least two models"):
        nemenyi_q(1)


def test_friedman_edge_cases():
    same = friedman_test(table_of([[3, 3, 3], [2, 2, 2], [1, 1, 1]]))
    assert same.chi2 == pytest.approx(6)  # the largest value for three models and three benchmarks
    assert (same.f, same.f_p) == (math.inf, 0)
    flat = friedman_test(table_of([[1, 1], [1, 1]]))
    assert math.isnan(flat.chi2) and math.isnan(flat.f_p)
    assert flat.groups() == [("m00", "m01")]
    with pytest.raises(ValueError, match="complete table"):
        friedman_test(table_of([[1, math.nan], [2, 3]]))
    with pytest.raises(ValueError, match="at least two models and two benchmarks"):
        friedman_test(table_of([[1], [2]]))
    with pytest.raises(ValueError, match="alpha"):
        friedman_test(table_of([[1, 2], [2, 1]]), alpha=0)


def test_groups_are_maximal_runs_below_the_critical_difference():
    rng = np.random.default_rng(1)
    result = friedman_test(table_of(rng.normal(size=(8, 30)) + np.arange(8)[:, None] * 0.4))
    ranks = dict(zip(result.models, result.mean_ranks, strict=True))
    groups = result.groups()
    for group in groups:
        assert ranks[group[-1]] - ranks[group[0]] < result.critical_difference
    assert not any(set(a) < set(b) for a in groups for b in groups)
