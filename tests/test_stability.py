import math

import numpy as np
import pytest

from rankstable.ranking import Ranking
from rankstable.stability import (
    benchmark_diversity,
    bootstrap_benchmarks,
    leave_out,
    remove_models,
    top_k_overlap,
    top_set,
)
from rankstable.table import ScoreTable

nan = math.nan


def table_of(rows, higher_is_better=()):
    scores = np.array(rows, dtype=float)
    models = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh"[: scores.shape[0]])
    benchmarks = tuple(f"b{j + 1}" for j in range(scores.shape[1]))
    return ScoreTable(models, benchmarks, scores, higher_is_better)


def ranking_of(values):
    return Ranking.from_values("demo", "ABCDEFGH"[: len(values)], values)


def test_top_set_keeps_everyone_tied_at_the_boundary():
    ranking = ranking_of([9, 7, 7, 1])
    assert top_set(ranking, 1) == {"A"}
    assert top_set(ranking, 2) == {"A", "B", "C"}
    assert top_set(ranking, 4) == {"A", "B", "C", "D"}
    with pytest.raises(ValueError, match="at least 1"):
        top_set(ranking, 0)


def test_top_k_overlap_by_hand():
    forward, backward = ranking_of([4, 3, 2, 1]), ranking_of([1, 2, 3, 4])
    assert top_k_overlap(forward, backward, 2) == 0
    assert top_k_overlap(forward, backward, 3) == pytest.approx(2 / 3)
    assert top_k_overlap(forward, backward, 4) == 1
    assert top_k_overlap(forward, forward, 1) == 1
    # a three-way tie for first against a single winner: one common model out of three
    assert top_k_overlap(forward, ranking_of([5, 5, 5, 1]), 1) == pytest.approx(1 / 3)


# Means: A 17/3, B 5, C 3. Without b1: 4, 5, 4. Without b2: 7.5, 4.5, 2. Without b3: 5.5, 5.5, 3.
SMALL = table_of([[9, 2, 6], [5, 6, 4], [1, 5, 3]])


def test_leave_one_out_by_hand():
    result = leave_out(SMALL, "mean", top=2)
    assert (result.rule, result.k, result.top, result.exhaustive) == ("mean", 1, 2, True)
    assert result.full_top1 == ("A",)
    assert [o.dropped for o in result.outcomes] == [("b1",), ("b2",), ("b3",)]
    assert [o.top1 for o in result.outcomes] == [("B",), ("A",), ("A", "B")]
    assert [o.top1_changed for o in result.outcomes] == [True, False, True]
    # top two on the full table: A, B. Without b1 A and C tie for second, so C joins.
    assert [o.topk_changed for o in result.outcomes] == [True, False, False]
    # ranks 1 2 3 against 2 1 2: one pair agrees, one disagrees, one is tied -> 0
    # ranks 1 2 3 against 1 1 3: two pairs agree, one is tied -> 2 / sqrt(3 * 2)
    np.testing.assert_allclose([o.tau for o in result.outcomes], [0, 1, 2 / math.sqrt(6)])
    assert result.n_subsets == 3
    assert result.top1_changed == pytest.approx(2 / 3)
    assert result.topk_changed == pytest.approx(1 / 3)
    assert result.mean_tau == pytest.approx((1 + 2 / math.sqrt(6)) / 3)
    assert result.min_tau == 0


def test_leave_two_out_by_hand():
    result = leave_out(SMALL, "mean", k=2, top=1)
    assert [o.dropped for o in result.outcomes] == [("b1", "b2"), ("b1", "b3"), ("b2", "b3")]
    # what is left: b3 alone (A first), b2 alone (B, C, A), b1 alone (A first)
    assert [o.top1 for o in result.outcomes] == [("A",), ("B",), ("A",)]
    assert result.top1_changed == pytest.approx(1 / 3)
    np.testing.assert_allclose([o.tau for o in result.outcomes], [1, -1 / 3, 1])


def test_leave_out_samples_distinct_subsets_when_there_are_too_many():
    rng = np.random.default_rng(0)
    table = table_of(rng.normal(size=(6, 8)))
    result = leave_out(table, "mean", k=3, max_subsets=10, seed=1)
    assert (result.n_subsets, result.exhaustive) == (10, False)
    assert len({o.dropped for o in result.outcomes}) == 10
    assert all(len(o.dropped) == 3 for o in result.outcomes)
    again = leave_out(table, "mean", k=3, max_subsets=10, seed=1)
    assert [o.dropped for o in again.outcomes] == [o.dropped for o in result.outcomes]
    other = leave_out(table, "mean", k=3, max_subsets=10, seed=2)
    assert [o.dropped for o in other.outcomes] != [o.dropped for o in result.outcomes]
    assert leave_out(table, "mean", k=3, max_subsets=56).exhaustive


def test_leave_out_rejects_impossible_requests():
    with pytest.raises(ValueError, match="cannot leave 3 of 3"):
        leave_out(SMALL, k=3)
    with pytest.raises(ValueError, match="cannot leave 0 of 3"):
        leave_out(SMALL, k=0)
    with pytest.raises(ValueError, match="max_subsets"):
        leave_out(SMALL, max_subsets=0)


def test_leave_out_reports_an_undefined_tau_as_nan():
    flat = table_of([[1, 1], [1, 1]])
    result = leave_out(flat, "mean")
    assert math.isnan(result.mean_tau) and math.isnan(result.min_tau)
    assert result.top1_changed == 0


def test_bootstrap_matches_the_exact_distribution_for_two_benchmarks():
    # Resamples (b1, b1), (b1, b2), (b2, b1), (b2, b2) are equally likely. A is first only
    # on (b1, b1); the full table puts B first.
    table = table_of([[3, 0], [2, 2]])
    result = bootstrap_benchmarks(table, "mean", n_resamples=4000, top=1, seed=3)
    assert result.full_top1 == ("B",)
    np.testing.assert_allclose(result.p_top1, [0.25, 0.75], atol=0.03)
    np.testing.assert_array_equal(result.p_topk, result.p_top1)
    assert result.top1_changed == pytest.approx(0.25, abs=0.03)
    assert result.topk_changed == result.top1_changed
    assert result.mean_tau == pytest.approx(0.5, abs=0.06)
    assert (result.tau_low, result.tau_high) == (-1, 1)
    np.testing.assert_array_equal(result.rank_low, [1, 1])
    np.testing.assert_array_equal(result.rank_high, [2, 2])


def test_bootstrap_of_a_model_that_wins_everywhere():
    table = table_of([[9, 9, 9], [1, 2, 3], [3, 1, 2]])
    result = bootstrap_benchmarks(table, "borda", n_resamples=200, top=2, seed=0)
    assert (result.n_resamples, result.top, result.confidence, result.seed) == (200, 2, 0.95, 0)
    assert result.models == ("A", "B", "C")
    assert result.p_top1.tolist() == [1, 0, 0]
    assert result.top1_changed == 0
    assert (result.rank_low[0], result.rank_high[0]) == (1, 1)
    assert 0 < result.topk_changed < 1


def test_bootstrap_is_reproducible_and_seeded():
    rng = np.random.default_rng(5)
    table = table_of(rng.normal(size=(7, 5)))
    first = bootstrap_benchmarks(table, "zscore", n_resamples=50, seed=11)
    again = bootstrap_benchmarks(table, "zscore", n_resamples=50, seed=11)
    other = bootstrap_benchmarks(table, "zscore", n_resamples=50, seed=12)
    np.testing.assert_array_equal(first.p_topk, again.p_topk)
    assert first.mean_tau == again.mean_tau
    assert first.mean_tau != other.mean_tau
    assert first.tau_low <= first.mean_tau <= first.tau_high
    assert np.all(first.rank_low <= first.rank_high)


def test_bootstrap_checks_its_arguments_and_survives_constant_rankings():
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_benchmarks(SMALL, n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_benchmarks(SMALL, confidence=1.0)
    flat = bootstrap_benchmarks(table_of([[1, 1], [1, 1]]), n_resamples=5)
    assert math.isnan(flat.mean_tau) and math.isnan(flat.tau_low) and math.isnan(flat.tau_high)


# Three benchmarks say A > B > C, two say B > C > A.
SPLIT = table_of([[3, 3, 3, 1, 1], [2, 2, 2, 3, 3], [1, 1, 1, 2, 2]])


def test_removing_the_last_model_flips_the_borda_winner():
    # Borda gives A 6, B 7, C 2. Without C it is A 3, B 2: the pair (A, B) flips.
    result = remove_models(SPLIT, "borda")
    assert (result.n_removals, result.exhaustive) == (3, True)
    assert result.order_changed == pytest.approx(1 / 3)
    assert result.mean_pairs_changed == pytest.approx(1 / 3)
    assert result.top1_changed == pytest.approx(1 / 3)
    assert result.worst == ("C", 1.0)


def test_the_mean_never_reorders_the_remaining_models():
    rng = np.random.default_rng(8)
    table = table_of(rng.integers(0, 9, size=(9, 4)))
    for rule in ("mean", "geomean"):
        result = remove_models(table, rule)
        assert (result.order_changed, result.mean_pairs_changed, result.top1_changed) == (0, 0, 0)
    assert remove_models(SPLIT, "mean").worst == ("A", 0.0)


def test_removals_are_sampled_above_the_limit():
    rng = np.random.default_rng(9)
    table = table_of(rng.normal(size=(30, 4)))
    result = remove_models(table, "minmax", max_removals=5, seed=4)
    assert (result.n_removals, result.exhaustive) == (5, False)
    again = remove_models(table, "minmax", max_removals=5, seed=4)
    assert again == result
    with pytest.raises(ValueError, match="at least three models"):
        remove_models(table_of([[1, 2], [2, 1]]))
    with pytest.raises(ValueError, match="max_removals"):
        remove_models(table, max_removals=0)


def test_diversity_by_hand():
    same = table_of([[3, 6], [2, 4], [1, 2]])
    assert benchmark_diversity(same) == 0
    assert benchmark_diversity(table_of([[3, 1], [2, 2], [1, 3]])) == 1
    # two benchmarks agree and the third reverses them: distances 0, 1, 1
    assert benchmark_diversity(table_of([[3, 3, 1], [2, 2, 2], [1, 1, 3]])) == pytest.approx(2 / 3)
    # an error rate that falls as the accuracy rises ranks the models the same way
    assert benchmark_diversity(table_of([[3, 1], [2, 2], [1, 3]], (True, False))) == 0


def test_diversity_skips_pairs_without_a_defined_correlation():
    assert math.isnan(benchmark_diversity(table_of([[3, 5], [2, 5], [1, 5]])))
    assert math.isnan(benchmark_diversity(table_of([[3], [2], [1]])))
    # b1 and b2 share only B and C, and order them oppositely; b3 is constant
    table = table_of([[3, nan, 7], [2, 1, 7], [1, 2, 7]])
    assert benchmark_diversity(table) == 1
