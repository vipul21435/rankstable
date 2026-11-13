import math
import warnings

import numpy as np
import pytest
from scipy import stats

from rankstable.ranking import Ranking, average_ranks, competition_ranks, kendall_tau

nan = math.nan


def test_average_ranks_give_one_to_the_best_and_share_ranks_between_ties():
    np.testing.assert_array_equal(average_ranks([3.0, 1.0, 2.0]), [1, 3, 2])
    np.testing.assert_array_equal(average_ranks([5, 5, 1]), [1.5, 1.5, 3])
    np.testing.assert_array_equal(average_ranks([2, 2, 2, 2]), [2.5] * 4)
    np.testing.assert_array_equal(average_ranks([3, 1, 2], higher_is_better=False), [3, 1, 2])


def test_average_ranks_leave_missing_values_out():
    np.testing.assert_array_equal(average_ranks([0.4, nan, 0.9, 0.4]), [2.5, nan, 1, 2.5])
    assert np.isnan(average_ranks([nan, nan])).all()
    assert average_ranks([]).shape == (0,)


def test_average_and_competition_ranks_match_scipy_rankdata():
    rng = np.random.default_rng(0)
    for _ in range(100):
        values = rng.integers(0, 6, size=rng.integers(1, 30)).astype(float)
        np.testing.assert_array_equal(
            average_ranks(values), stats.rankdata(-values, method="average")
        )
        np.testing.assert_array_equal(
            competition_ranks(values), stats.rankdata(-values, method="min")
        )
        np.testing.assert_array_equal(
            competition_ranks(values, higher_is_better=False),
            stats.rankdata(values, method="min"),
        )


def test_competition_ranks_share_the_smallest_rank():
    np.testing.assert_array_equal(competition_ranks([10, 30, 20, 20]), [4, 1, 2, 2])
    np.testing.assert_array_equal(
        competition_ranks([1.0, 2.0, 2.0, 3.0], higher_is_better=False), [1, 2, 2, 4]
    )
    np.testing.assert_array_equal(competition_ranks([0.0, 0.0]), [1, 1])


def test_competition_ranks_put_missing_values_last():
    np.testing.assert_array_equal(competition_ranks([nan, 2.0, nan, 5.0]), [3, 2, 3, 1])
    np.testing.assert_array_equal(competition_ranks([nan, nan]), [1, 1])


def test_competition_ranks_are_not_decided_by_rounding_error():
    assert 0.1 + 0.2 != 0.3
    np.testing.assert_array_equal(competition_ranks([0.1 + 0.2, 0.3, 0.2]), [1, 1, 3])
    np.testing.assert_array_equal(competition_ranks([1.000001, 1.0]), [1, 2])
    np.testing.assert_array_equal(competition_ranks([-70.1, -70.1 - 1e-13, -3.0]), [2, 2, 1])


def test_kendall_tau_hand_worked_without_ties():
    # 1 2 3 4 against 1 3 2 4: five pairs agree, the pair (2, 3) is swapped: (5 - 1) / 6.
    assert kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(2 / 3)
    assert kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_kendall_tau_b_hand_worked_with_ties():
    # x = 1 1 2 3, y = 1 2 2 3: one pair tied in x, another tied in y, the other four agree.
    # tau_b = 4 / sqrt((6 - 1) * (6 - 1)) = 0.8
    assert kendall_tau([1, 1, 2, 3], [1, 2, 2, 3]) == pytest.approx(0.8)
    # A pair tied in both sequences counts in both tie terms. Here the other two pairs
    # disagree: -2 / sqrt((3 - 1) * (3 - 1)) = -1.
    assert kendall_tau([1, 1, 2], [5, 5, 4]) == pytest.approx(-1.0)
    # Three pairs agree, two disagree, one is tied in both: (3 - 2) / sqrt(5 * 5) = 0.2.
    assert kendall_tau([1, 1, 2, 3], [5, 5, 4, 6]) == pytest.approx(0.2)


def test_kendall_tau_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        levels = int(rng.integers(2, 8))
        x = rng.integers(0, levels, size=n).astype(float)
        y = rng.integers(0, levels, size=n).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = stats.kendalltau(x, y)[0]
        np.testing.assert_allclose(kendall_tau(x, y), expected, rtol=1e-12, atol=1e-12)
    for _ in range(20):
        x = rng.normal(size=300)
        y = x + rng.normal(size=300)
        np.testing.assert_allclose(kendall_tau(x, y), stats.kendalltau(x, y)[0], rtol=1e-12)


def test_kendall_tau_undefined_cases():
    assert math.isnan(kendall_tau([1, 1, 1], [1, 2, 3]))
    assert math.isnan(kendall_tau([1], [1]))
    assert math.isnan(kendall_tau([], []))
    with pytest.raises(ValueError, match="same length"):
        kendall_tau([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="one-dimensional"):
        kendall_tau([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="nan"):
        kendall_tau([1, nan], [1, 2])


def test_ranking_lists_the_best_first_and_ties_alphabetically():
    ranking = Ranking.from_values("demo", ["zeta", "alpha", "mid", "none"], [2.0, 2.0, 5.0, nan])
    assert ranking.ranks.tolist() == [2, 2, 1, 4]
    assert ranking.order() == ["mid", "alpha", "zeta", "none"]
    assert ranking.rank_of("zeta") == 2
    with pytest.raises(KeyError):
        ranking.rank_of("nobody")
    assert ranking.rows() == [
        {"rank": 1, "model": "mid", "value": 5.0},
        {"rank": 2, "model": "alpha", "value": 2.0},
        {"rank": 2, "model": "zeta", "value": 2.0},
        {"rank": 4, "model": "none", "value": None},
    ]


def test_ranking_with_lower_is_better_values():
    ranking = Ranking.from_values("demo", ["a", "b", "c"], [2.5, 1.0, 2.5], higher_is_better=False)
    assert ranking.ranks.tolist() == [2, 1, 2]
    assert ranking.order() == ["b", "a", "c"]


def test_ranking_checks_length_and_is_read_only():
    with pytest.raises(ValueError, match="expected 2 values"):
        Ranking.from_values("demo", ["a", "b"], [1.0])
    ranking = Ranking.from_values("demo", ["a", "b"], [1.0, 2.0])
    with pytest.raises(ValueError):
        ranking.values[0] = 9.0
    with pytest.raises(ValueError):
        ranking.ranks[0] = 9
