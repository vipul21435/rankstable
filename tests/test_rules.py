import itertools
import math

import numpy as np
import pytest
from scipy import optimize, stats

from rankstable.rules import (
    RULES,
    _kemeny_exact,
    _kemeny_local_search,
    aggregate,
    benchmark_ranks,
    borda_count,
    bradley_terry,
    copeland,
    geometric_mean,
    kemeny_score,
    kemeny_young,
    mean_rank,
    mean_win_rate,
    minmax_mean,
    pairwise_wins,
    zscore_mean,
)
from rankstable.table import ScoreTable

nan = math.nan


def table_of(rows, higher_is_better=()):
    scores = np.array(rows, dtype=float)
    models = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: scores.shape[0]])
    benchmarks = tuple(f"b{j + 1}" for j in range(scores.shape[1]))
    return ScoreTable(models, benchmarks, scores, higher_is_better)


def ranks(table, rule):
    return aggregate(table, rule).ranks.tolist()


def bradley_terry_reference(table, prior=0.5):
    """Log-strengths by direct minimisation of the penalised negative log-likelihood."""
    oriented = table.oriented()
    present = ~np.isnan(oriented)
    n = table.n_models
    points = np.zeros(n)
    contests = np.zeros((n, n))
    for j in range(table.n_benchmarks):
        for i, k in itertools.permutations(np.flatnonzero(present[:, j]), 2):
            contests[i, k] += 0.5
            contests[k, i] += 0.5
            points[i] += 1.0 if oriented[i, j] > oriented[k, j] else 0.0
            points[i] += 0.5 if oriented[i, j] == oriented[k, j] else 0.0
    contests /= 2  # every ordered pair was visited twice
    upper = np.triu_indices(n, k=1)

    def loss(theta):
        pair = np.logaddexp(theta[:, None], theta[None, :])[upper]
        return (
            -np.dot(points + prior, theta)
            + np.dot(contests[upper] * 2, pair)
            + 2 * prior * np.logaddexp(theta, 0).sum()
        )

    return optimize.minimize(loss, np.zeros(n), method="BFGS", options={"gtol": 1e-10}).x


# A complete profile worked by hand.
#        b1  b2  b3
#   A    90  10  50
#   B    80  30  40
#   C    60  20  40
COMPLETE = table_of([[90, 10, 50], [80, 30, 40], [60, 20, 40]])

COMPLETE_VALUES = {
    "mean": [50, 50, 40],
    # b1 spans 60..90, b2 spans 10..30, b3 spans 40..50
    "minmax": [(1 + 0 + 1) / 3, (2 / 3 + 1 + 0) / 3, (0 + 0.5 + 0) / 3],
    # b1: deviations 40/3, 10/3, -50/3 with sd sqrt(4200/27), i.e. 4, 1, -5 over sqrt(14)
    # b2: deviations -10, 10, 0 with sd sqrt(200/3); b3: 20/3, -10/3, -10/3 with sd sqrt(200)/3
    "zscore": [
        (4 / math.sqrt(14) - math.sqrt(1.5) + math.sqrt(2)) / 3,
        (1 / math.sqrt(14) + math.sqrt(1.5) - 1 / math.sqrt(2)) / 3,
        (-5 / math.sqrt(14) + 0 - 1 / math.sqrt(2)) / 3,
    ],
    "geomean": [45000 ** (1 / 3), 96000 ** (1 / 3), 48000 ** (1 / 3)],
    # ranks: b1 A1 B2 C3, b2 B1 C2 A3, b3 A1 and B, C tied on 2.5
    "mean-rank": [5 / 3, 5.5 / 3, 7.5 / 3],
    "win-rate": [(1 + 0 + 1) / 3, (0.5 + 1 + 0.25) / 3, (0 + 0.5 + 0.25) / 3],
    "borda": [4, 3.5, 1.5],
    # A beats B and C two benchmarks to one; B beats C two to none with one level
    "copeland": [2, 0, -2],
    # the order A, B, C keeps 2 + 2 + 2 of the pairwise wins, more than any other
    "kemeny": [1, 2, 3],
    "bradley-terry": None,  # no closed form; checked against a direct optimiser below
}

COMPLETE_RANKS = {
    "mean": [1, 1, 3],
    "minmax": [1, 2, 3],
    "zscore": [1, 2, 3],
    "geomean": [3, 1, 2],
    "mean-rank": [1, 2, 3],
    "win-rate": [1, 2, 3],
    "borda": [1, 2, 3],
    "copeland": [1, 2, 3],
    "kemeny": [1, 2, 3],
    "bradley-terry": [1, 2, 3],
}


def test_the_hand_worked_profile_covers_every_rule():
    assert set(COMPLETE_VALUES) == set(RULES) == set(COMPLETE_RANKS)


@pytest.mark.parametrize("rule", list(RULES))
def test_complete_profile_by_hand(rule):
    result = aggregate(COMPLETE, rule)
    assert result.rule == rule
    assert result.models == ("A", "B", "C")
    if rule == "bradley-terry":
        np.testing.assert_allclose(result.values, bradley_terry_reference(COMPLETE), atol=1e-6)
    else:
        np.testing.assert_allclose(result.values, COMPLETE_VALUES[rule], rtol=1e-12)
    assert result.ranks.tolist() == COMPLETE_RANKS[rule]
    assert result.higher_is_better == (rule not in ("mean-rank", "kemeny"))


def test_complete_profile_ranks_and_pairwise_wins():
    np.testing.assert_array_equal(benchmark_ranks(COMPLETE), [[1, 3, 1], [2, 1, 2.5], [3, 2, 2.5]])
    np.testing.assert_array_equal(pairwise_wins(COMPLETE), [[0, 2, 2], [1, 0, 2], [1, 0, 0]])


# The same profile with holes, and a model without any score.
#        b1  b2  b3
#   A    90   .  50
#   B    80  30   .
#   C    60  20  40
#   D     .   .   .
SPARSE = table_of([[90, nan, 50], [80, 30, nan], [60, 20, 40], [nan, nan, nan]])

SPARSE_VALUES = {
    "mean": [70, 55, 40],
    "minmax": [(1 + 1) / 2, (2 / 3 + 1) / 2, 0],
    "zscore": [
        (4 / math.sqrt(14) + 1) / 2,
        (1 / math.sqrt(14) + 1) / 2,
        (-5 / math.sqrt(14) - 1 - 1) / 3,
    ],
    "geomean": [math.sqrt(4500), math.sqrt(2400), 48000 ** (1 / 3)],
    # b1 ranks A, B, C; b2 ranks B, C only; b3 ranks A, C only
    "mean-rank": [(1 + 1) / 2, (2 + 1) / 2, (3 + 2 + 2) / 3],
    "win-rate": [(2 / 2 + 1 / 1) / 2, (1 / 2 + 1 / 1) / 2, 0],
    "borda": [2 + 1, 1 + 1, 0],
    "copeland": [2, 0, -2],
    "kemeny": [1, 2, 3],
    "bradley-terry": None,
}


@pytest.mark.parametrize("rule", list(RULES))
def test_sparse_profile_by_hand(rule):
    result = aggregate(SPARSE, rule)
    if rule == "bradley-terry":
        expected = bradley_terry_reference(SPARSE.select(models=["A", "B", "C"]))
        np.testing.assert_allclose(result.values[:3], expected, atol=1e-6)
    else:
        np.testing.assert_allclose(result.values[:3], SPARSE_VALUES[rule], rtol=1e-12, atol=1e-15)
    assert math.isnan(result.values[3])
    assert result.ranks.tolist() == [1, 2, 3, 4]


def test_condorcet_cycle_is_a_three_way_tie():
    # b1: A > B > C, b2: B > C > A, b3: C > A > B
    cycle = table_of([[3, 1, 2], [2, 3, 1], [1, 2, 3]])
    np.testing.assert_array_equal(pairwise_wins(cycle), [[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    result = copeland(cycle)
    np.testing.assert_array_equal(result.values, [0, 0, 0])
    assert result.ranks.tolist() == [1, 1, 1]
    for rule in RULES:
        if rule != "kemeny":
            assert ranks(cycle, rule) == [1, 1, 1], rule
    # Kemeny-Young has to pick an order. The three rotations of the cycle keep five pairwise
    # wins each, the other three orders four, and the alphabetically first rotation is reported.
    wins = pairwise_wins(cycle)
    scores = {order: kemeny_score(wins, order) for order in itertools.permutations(range(3))}
    assert sorted(scores.values()) == [4, 4, 4, 5, 5, 5]
    assert {order for order, score in scores.items() if score == 5} == {
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    }
    assert kemeny_young(cycle).values.tolist() == [1, 2, 3]


def test_majority_winner_can_lose_the_borda_count():
    # Three benchmarks say A > B > C and two say B > C > A. A wins both of its contests
    # three to two, yet B collects more points.
    profile = table_of([[3, 3, 3, 1, 1], [2, 2, 2, 3, 3], [1, 1, 1, 2, 2]])
    np.testing.assert_array_equal(copeland(profile).values, [2, 0, -2])
    np.testing.assert_array_equal(borda_count(profile).values, [6, 7, 2])
    assert ranks(profile, "copeland") == [1, 2, 3]
    assert ranks(profile, "borda") == [2, 1, 3]
    assert ranks(profile, "mean") == [2, 1, 3]
    # A, B, C keeps 3 + 3 + 5 pairwise wins; B, A, C keeps 2 + 5 + 3
    assert ranks(profile, "kemeny") == [1, 2, 3]
    assert kemeny_score(pairwise_wins(profile), [0, 1, 2]) == 11
    assert kemeny_score(pairwise_wins(profile), [1, 0, 2]) == 10


def test_missing_benchmarks_separate_borda_from_win_rate_and_mean_rank():
    # D has one score, the best one on b1.
    #        b1  b2  b3
    #   A     3   3   3
    #   B     2   2   2
    #   C     1   1   1
    #   D     9   .   .
    table = table_of([[3, 3, 3], [2, 2, 2], [1, 1, 1], [9, nan, nan]])
    np.testing.assert_allclose(mean_win_rate(table).values, [8 / 9, 4 / 9, 0, 1])
    np.testing.assert_allclose(mean_rank(table).values, [4 / 3, 7 / 3, 10 / 3, 1])
    np.testing.assert_array_equal(borda_count(table).values, [6, 3, 0, 3])
    np.testing.assert_array_equal(copeland(table).values, [1, -1, -3, 3])
    assert ranks(table, "win-rate") == [2, 3, 4, 1]
    assert ranks(table, "mean-rank") == [2, 3, 4, 1]
    assert ranks(table, "borda") == [1, 2, 4, 2]


def test_lower_is_better_benchmarks():
    # accuracy up, error down
    table = table_of([[0.9, 0.2], [0.8, 0.1]], higher_is_better=(True, False))
    np.testing.assert_allclose(aggregate(table, "mean").values, [0.35, 0.35])
    assert ranks(table, "mean") == [1, 1]
    np.testing.assert_allclose(geometric_mean(table).values, [math.sqrt(4.5), math.sqrt(8)])
    np.testing.assert_array_equal(benchmark_ranks(table), [[1, 2], [2, 1]])
    np.testing.assert_array_equal(minmax_mean(table).values, [0.5, 0.5])
    flipped = table_of([[0.9, -0.2], [0.8, -0.1]])
    for rule in RULES:
        if rule != "geomean":
            assert ranks(table, rule) == ranks(flipped, rule), rule


def test_a_benchmark_without_spread_is_skipped_by_the_normalised_rules():
    table = table_of([[0.1, 1], [0.1, 2], [0.1, 4]])
    np.testing.assert_allclose(minmax_mean(table).values, [0, 1 / 3, 1])
    centre, sd = 7 / 3, math.sqrt(14 / 9)
    np.testing.assert_allclose(
        zscore_mean(table).values, [(1 - centre) / sd, (2 - centre) / sd, (4 - centre) / sd]
    )
    flat = table_of([[5, 5], [5, 5]])
    for rule in ("minmax", "zscore"):
        result = aggregate(flat, rule)
        assert np.isnan(result.values).all()
        assert result.ranks.tolist() == [1, 1]


def test_zscore_matches_scipy_on_complete_tables():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(12, 5)) * [1, 10, 100, 0.1, 3] + [0, 5, -50, 1, 2]
    expected = stats.zscore(scores, axis=0).mean(axis=1)
    np.testing.assert_allclose(zscore_mean(table_of(scores)).values, expected, rtol=1e-10)


def test_geometric_mean_matches_scipy_and_handles_zero():
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.1, 100, size=(9, 4))
    np.testing.assert_allclose(
        geometric_mean(table_of(scores)).values, stats.gmean(scores, axis=1), rtol=1e-12
    )
    with_zero = table_of([[0, 4], [1, 4], [2, 8]])
    np.testing.assert_allclose(geometric_mean(with_zero).values, [0, 2, 4])


def test_geometric_mean_rejects_scores_it_cannot_use():
    with pytest.raises(ValueError, match="non-negative scores; benchmark 'b2'"):
        geometric_mean(table_of([[1, -1], [2, 3]]))
    with pytest.raises(ValueError, match="lower-is-better benchmark 'b1'"):
        geometric_mean(table_of([[0, 1], [2, 3]], higher_is_better=(False, True)))


def test_a_benchmark_with_one_model_gives_no_win_rate():
    table = table_of([[1, 7], [2, nan], [nan, nan]])
    np.testing.assert_array_equal(mean_win_rate(table).values, [0, 1, nan])
    only = table_of([[nan, 7], [nan, nan]])
    assert np.isnan(mean_win_rate(only).values).all()
    np.testing.assert_array_equal(borda_count(only).values, [0, nan])
    np.testing.assert_array_equal(mean_rank(only).values, [1, nan])


def test_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule 'median'; choose from mean, minmax"):
        aggregate(COMPLETE, "median")


def random_table(rng, n_models, n_benchmarks, missing=0.0):
    """Small integer scores, so that ties are common, with a share of cells removed."""
    scores = rng.integers(0, 6, size=(n_models, n_benchmarks)).astype(float)
    scores[rng.random(scores.shape) < missing] = nan
    directions = tuple(bool(flag) for flag in rng.random(n_benchmarks) < 0.7)
    scores[:, [not flag for flag in directions]] += 1  # keep 1/score defined
    return table_of(scores, higher_is_better=directions)


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("missing", [0.0, 0.3])
def test_rules_do_not_depend_on_the_order_of_models_or_benchmarks(rule, missing):
    rng = np.random.default_rng(4)
    for _ in range(15):
        table = random_table(rng, int(rng.integers(2, 9)), int(rng.integers(1, 7)), missing)
        base = aggregate(table, rule)
        rows = rng.permutation(table.n_models)
        shuffled = aggregate(table.select(models=[table.models[i] for i in rows]), rule)
        assert shuffled.ranks.tolist() == base.ranks[rows].tolist()
        np.testing.assert_allclose(shuffled.values, base.values[rows], rtol=1e-9, atol=1e-12)
        assert shuffled.order() == base.order()
        columns = rng.permutation(table.n_benchmarks)
        reordered = aggregate(table.select(benchmarks=[table.benchmarks[j] for j in columns]), rule)
        assert reordered.ranks.tolist() == base.ranks.tolist()


def test_mean_rank_win_rate_and_borda_agree_on_complete_tables():
    rng = np.random.default_rng(5)
    for _ in range(50):
        table = random_table(rng, int(rng.integers(2, 10)), int(rng.integers(1, 8)))
        n, m = table.n_models, table.n_benchmarks
        mean_ranks = mean_rank(table).values
        points = borda_count(table).values
        np.testing.assert_allclose(points, m * (n - mean_ranks), rtol=1e-12)
        np.testing.assert_allclose(mean_win_rate(table).values, points / (m * (n - 1)), rtol=1e-12)
        assert ranks(table, "mean-rank") == ranks(table, "borda") == ranks(table, "win-rate")


def test_per_benchmark_ranks_match_scipy():
    rng = np.random.default_rng(6)
    table = random_table(rng, 15, 6)
    oriented = table.oriented()
    expected = np.column_stack([stats.rankdata(-oriented[:, j]) for j in range(6)])
    np.testing.assert_array_equal(benchmark_ranks(table), expected)


def test_rescaling_a_benchmark_only_moves_the_rules_that_use_raw_scores():
    rng = np.random.default_rng(7)
    scores = rng.uniform(1, 10, size=(8, 5))
    stretched = scores.copy()
    stretched[:, 2] = 1000 * stretched[:, 2] + 7
    for rule in ("minmax", "zscore"):
        np.testing.assert_allclose(
            aggregate(table_of(stretched), rule).values,
            aggregate(table_of(scores), rule).values,
            rtol=1e-9,
        )
    for rule in ("mean-rank", "win-rate", "borda", "copeland"):
        np.testing.assert_array_equal(
            aggregate(table_of(stretched), rule).values, aggregate(table_of(scores), rule).values
        )
    scaled = scores.copy()
    scaled[:, 2] *= 1000
    assert ranks(table_of(scaled), "geomean") == ranks(table_of(scores), "geomean")
    np.testing.assert_allclose(
        geometric_mean(table_of(scaled)).values, geometric_mean(table_of(scores)).values * 1000**0.2
    )


def test_the_raw_mean_follows_the_benchmark_with_the_widest_scale():
    narrow = table_of([[1, 2.0], [2, 1.5]])
    wide = table_of([[1, 2000], [2, 1500]])
    assert ranks(narrow, "mean") == [2, 1]
    assert ranks(wide, "mean") == [1, 2]
    for rule in ("minmax", "zscore", "mean-rank", "win-rate", "borda", "copeland"):
        assert ranks(narrow, rule) == ranks(wide, rule) == [1, 1], rule


def test_cancelling_terms_do_not_break_a_tie():
    # 0.3 - 0.1 - 0.2 and 0.5 - 0.25 - 0.25 are both zero on paper, not in floating point.
    assert 0.3 - 0.1 - 0.2 != 0.5 - 0.25 - 0.25
    table = table_of([[0.3, 0.1, 0.2], [0.5, 0.25, 0.25]], higher_is_better=(True, False, False))
    assert ranks(table, "mean") == [1, 1]


def test_kemeny_exact_matches_brute_force_up_to_seven_models():
    rng = np.random.default_rng(10)
    for n in range(1, 8):
        for _ in range(6 if n < 7 else 3):
            table = random_table(rng, n, int(rng.integers(1, 8)), float(rng.choice([0, 0.3])))
            wins = pairwise_wins(table)
            scores = {
                order: kemeny_score(wins, order) for order in itertools.permutations(range(n))
            }
            best = max(scores.values())
            order = tuple(_kemeny_exact(wins))
            assert scores[order] == best
            assert order == min(o for o, score in scores.items() if score == best)


def test_kemeny_local_search_ends_where_no_single_move_helps():
    rng = np.random.default_rng(11)
    for _ in range(30):
        n = int(rng.integers(2, 8))
        wins = pairwise_wins(random_table(rng, n, int(rng.integers(1, 8))))
        order = _kemeny_local_search(wins)
        assert sorted(order) == list(range(n))
        reached = kemeny_score(wins, order)
        assert reached <= kemeny_score(wins, _kemeny_exact(wins))
        for position in range(n):
            rest = order[:position] + order[position + 1 :]
            for slot in range(n):
                moved = rest[:slot] + [order[position]] + rest[slot:]
                assert kemeny_score(wins, moved) <= reached


def test_kemeny_local_search_stays_close_to_the_best_order_on_random_tables():
    # Tables with a shared skill signal of random strength plus noise, small enough to solve
    # exactly. With this seed the search misses the best order once, by under one percent.
    rng = np.random.default_rng(16)
    shortfalls = []
    for _ in range(40):
        n, m = int(rng.integers(8, 11)), int(rng.integers(3, 13))
        scores = rng.uniform(0, 1.5) * rng.normal(size=(n, 1)) + rng.normal(size=(n, m))
        wins = pairwise_wins(table_of(scores))
        best = kemeny_score(wins, _kemeny_exact(wins))
        found = kemeny_score(wins, _kemeny_local_search(wins))
        assert found <= best
        shortfalls.append((best - found) / best)
    assert sum(shortfall == 0 for shortfall in shortfalls) >= 36
    assert max(shortfalls) < 0.02
    assert sum(shortfalls) / len(shortfalls) < 0.002


def test_kemeny_switches_to_the_local_search_above_the_exact_limit():
    # A transitive table: every benchmark gives the same order, so the search must find it.
    table = table_of([[5, 50], [4, 40], [3, 30], [2, 20], [1, 10]])
    assert kemeny_young(table, exact_limit=0).values.tolist() == [1, 2, 3, 4, 5]
    assert kemeny_young(table).values.tolist() == [1, 2, 3, 4, 5]
    rng = np.random.default_rng(12)
    wide = random_table(rng, 14, 5)
    result = kemeny_young(wide)
    assert sorted(result.values.tolist()) == list(range(1, 15))


def test_bradley_terry_two_models_closed_form():
    # A is better on three benchmarks of four: the fitted odds are three to one.
    table = table_of([[2, 2, 2, 1], [1, 1, 1, 2]])
    result = bradley_terry(table, prior=0)
    np.testing.assert_allclose(result.values, [math.log(3) / 2, -math.log(3) / 2], atol=1e-8)
    # a tie is half a win each: 2.5 wins against 1.5
    tied = bradley_terry(table_of([[2, 2, 1, 1], [1, 1, 2, 1]]), prior=0)
    np.testing.assert_allclose(tied.values[0] - tied.values[1], math.log(2.5 / 1.5), atol=1e-8)


def test_bradley_terry_matches_a_direct_optimiser():
    rng = np.random.default_rng(13)
    for missing in (0.0, 0.3):
        for _ in range(6):
            table = random_table(rng, int(rng.integers(3, 8)), int(rng.integers(3, 8)), missing)
            keep = [
                m
                for m, row in zip(table.models, table.scores, strict=True)
                if not np.isnan(row).all()
            ]
            table = table.select(models=keep)
            for prior in (0.5, 2.0):
                np.testing.assert_allclose(
                    bradley_terry(table, prior=prior).values,
                    bradley_terry_reference(table, prior),
                    atol=1e-5,
                )


def test_bradley_terry_orders_a_complete_table_like_borda():
    rng = np.random.default_rng(14)
    for _ in range(20):
        table = random_table(rng, int(rng.integers(2, 9)), int(rng.integers(1, 7)))
        assert ranks(table, "bradley-terry") == ranks(table, "borda")


def test_bradley_terry_prior_keeps_a_clean_sweep_finite():
    table = table_of([[3, 3, nan], [2, 2, 5], [1, 1, 4], [nan, nan, nan]])
    result = bradley_terry(table)
    assert np.isfinite(result.values[:3]).all()
    assert math.isnan(result.values[3])
    assert result.ranks.tolist() == [1, 2, 3, 4]
    assert bradley_terry(table, prior=5).values[0] < result.values[0]
    with pytest.raises(ValueError, match="prior cannot be negative"):
        bradley_terry(table, prior=-1)
