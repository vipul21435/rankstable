import math

import numpy as np
import pytest

from rankstable.agreement import benchmark_agreement
from rankstable.table import ScoreTable

nan = math.nan


def table_of(rows):
    scores = np.array(rows, dtype=float)
    models = tuple(f"m{i:02d}" for i in range(scores.shape[0]))
    benchmarks = tuple(f"b{j + 1}" for j in range(scores.shape[1]))
    return ScoreTable(models, benchmarks, scores)


def test_a_reversed_benchmark_disagrees_with_the_rest():
    # b1, b2 and b3 order the models the same way, b4 the other way round. Leaving b4 out
    # the rest agree; leaving one of the others out, two against one still gives their order.
    table = table_of([[4, 8, 40, 1], [3, 6, 30, 2], [2, 4, 20, 3], [1, 2, 10, 4]])
    results = benchmark_agreement(table, n_resamples=50, seed=0)
    assert [r.benchmark for r in results] == ["b1", "b2", "b3", "b4"]
    assert [r.tau for r in results] == [1, 1, 1, -1]
    assert [(r.low, r.high) for r in results] == [(1, 1), (1, 1), (1, 1), (-1, -1)]
    assert [r.n_models for r in results] == [4, 4, 4, 4]


def test_agreement_uses_the_models_scored_on_the_benchmark():
    table = table_of([[4, 8, 40, nan], [3, 6, 30, 2], [2, 4, 20, 3], [1, 2, 10, 4]])
    by_name = {r.benchmark: r for r in benchmark_agreement(table, n_resamples=20)}
    assert by_name["b4"].n_models == 3
    assert by_name["b4"].tau == -1
    assert by_name["b1"].n_models == 4


def test_agreement_interval_covers_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    skill = rng.normal(size=60)
    scores = skill[:, None] + rng.normal(scale=[0.3, 0.3, 0.3, 3.0], size=(60, 4))
    table = table_of(scores)
    results = benchmark_agreement(table, "zscore", n_resamples=150, seed=7)
    for result in results:
        assert -1 <= result.low <= result.tau <= result.high <= 1
        assert result.high - result.low > 0
    # the noisy benchmark follows the others least
    assert min(results, key=lambda r: r.tau).benchmark == "b4"
    assert results == benchmark_agreement(table, "zscore", n_resamples=150, seed=7)
    assert results != benchmark_agreement(table, "zscore", n_resamples=150, seed=8)
    narrow = benchmark_agreement(table, "zscore", n_resamples=150, confidence=0.5, seed=7)
    assert all(n.high - n.low < r.high - r.low for n, r in zip(narrow, results, strict=True))


def test_agreement_is_undefined_when_the_rest_cannot_order_the_models():
    # b2 and b3 cancel each other, so without b1 every model has the same win rate
    table = table_of([[3, 1, 3], [2, 2, 2], [1, 3, 1]])
    by_name = {r.benchmark: r for r in benchmark_agreement(table, n_resamples=10)}
    assert math.isnan(by_name["b1"].tau) and math.isnan(by_name["b1"].low)
    assert by_name["b2"].tau == -1
    flat = table_of([[1, 5, 5], [2, 5, 5], [3, 5, 5]])
    first = benchmark_agreement(flat, n_resamples=10)[0]
    assert math.isnan(first.tau) and math.isnan(first.low) and math.isnan(first.high)


def test_agreement_checks_its_arguments():
    table = table_of([[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="at least two benchmarks"):
        benchmark_agreement(table_of([[1], [2]]))
    with pytest.raises(ValueError, match="n_resamples"):
        benchmark_agreement(table, n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        benchmark_agreement(table, confidence=0)
    with pytest.raises(ValueError, match="unknown rule"):
        benchmark_agreement(table, "median")
