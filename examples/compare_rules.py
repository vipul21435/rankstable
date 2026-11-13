"""Rank the toy table with every rule, then look at how stable the mean ranking is.

python examples/compare_rules.py
"""

from pathlib import Path

from rankstable import RULES, aggregate, benchmark_agreement, kendall_tau, leave_out, read_table

table = read_table(Path(__file__).with_name("toy_scores.csv"))

rankings = {name: aggregate(table, name) for name in RULES}
for name, ranking in rankings.items():
    print(f"{name:14s} {' > '.join(ranking.order())}")

tau = kendall_tau(rankings["mean"].ranks, rankings["mean-rank"].ranks)
print(f"\nKendall tau-b between mean and mean-rank: {tau:.3f}")

result = leave_out(table, "mean", k=1, top=3)
print(f"\nLeave one benchmark out, rule mean: first place changes in {result.top1_changed:.0%}")
for outcome in result.outcomes:
    first = ", ".join(outcome.top1)
    print(f"  without {outcome.dropped[0]:10s} first: {first:6s} tau {outcome.tau:.3f}")

print("\nAgreement of each benchmark with the win-rate ranking of the others:")
for item in sorted(benchmark_agreement(table, n_resamples=200), key=lambda r: r.tau):
    print(f"  {item.benchmark:10s} tau {item.tau:.3f} ({item.low:.3f} to {item.high:.3f})")
