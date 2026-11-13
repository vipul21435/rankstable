"""rankstable: how much of a leaderboard's order survives a different aggregation rule."""

from .ranking import Ranking, average_ranks, competition_ranks, kendall_tau
from .rules import (
    RULES,
    aggregate,
    arithmetic_mean,
    benchmark_ranks,
    borda_count,
    copeland,
    geometric_mean,
    mean_rank,
    mean_win_rate,
    minmax_mean,
    pairwise_wins,
    zscore_mean,
)
from .table import ScoreTable, parse_long_jsonl, parse_wide_csv, read_table

__version__ = "0.1.0"

__all__ = [
    "RULES",
    "Ranking",
    "ScoreTable",
    "aggregate",
    "arithmetic_mean",
    "average_ranks",
    "benchmark_ranks",
    "borda_count",
    "competition_ranks",
    "copeland",
    "geometric_mean",
    "kendall_tau",
    "mean_rank",
    "mean_win_rate",
    "minmax_mean",
    "pairwise_wins",
    "parse_long_jsonl",
    "parse_wide_csv",
    "read_table",
    "zscore_mean",
]
