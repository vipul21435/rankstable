"""rankstable: how much of a leaderboard's order survives a different aggregation rule."""

from .agreement import BenchmarkAgreement, benchmark_agreement
from .friedman import FriedmanResult, friedman_test, nemenyi_q
from .ranking import Ranking, average_ranks, competition_ranks, kendall_tau, pair_counts
from .rules import (
    RULES,
    aggregate,
    arithmetic_mean,
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
from .stability import (
    BootstrapResult,
    LeaveOutResult,
    RemovalResult,
    benchmark_diversity,
    bootstrap_benchmarks,
    leave_out,
    remove_models,
    top_k_overlap,
    top_set,
)
from .table import ScoreTable, parse_long_jsonl, parse_wide_csv, read_table

__version__ = "0.1.0"

__all__ = [
    "RULES",
    "BenchmarkAgreement",
    "BootstrapResult",
    "FriedmanResult",
    "LeaveOutResult",
    "Ranking",
    "RemovalResult",
    "ScoreTable",
    "benchmark_agreement",
    "benchmark_diversity",
    "bootstrap_benchmarks",
    "friedman_test",
    "leave_out",
    "nemenyi_q",
    "pair_counts",
    "remove_models",
    "top_k_overlap",
    "top_set",
    "aggregate",
    "arithmetic_mean",
    "average_ranks",
    "benchmark_ranks",
    "borda_count",
    "bradley_terry",
    "competition_ranks",
    "copeland",
    "geometric_mean",
    "kemeny_score",
    "kemeny_young",
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
