# rankstable

Aggregate model scores over several benchmarks under different published rules and measure how
stable the resulting ranking is to the rule, to dropping a benchmark and to resampling.

**Status: early.** The score table, eight aggregation rules, Kendall's tau-b between their
rankings and two commands (`rank` and `compare-rules`) work and are tested. The stability
analyses are not written yet; the list is under [Not implemented yet](#not-implemented-yet).

## Why this is hard

A leaderboard reduces a models-by-benchmarks table to one order. The reduction (mean score,
mean rank, win rate, Borda count, with or without normalisation) and the choice of benchmarks
are both decisions, and either can change the winner. Dehghani et al. (2021),
[The Benchmark Lottery](https://arxiv.org/abs/2107.07002), report that "the relative
performance of algorithms may be altered significantly simply by choosing different benchmark
tasks". Colombo et al. (2022),
[What are the best systems? New perspectives on NLP Benchmarking](https://arxiv.org/abs/2202.03799),
note that averaging "can be problematic when the metrics are on a different scale" and
aggregate the rankings induced by each task instead. rankstable applies the rules side by side
to your own table, so that you can see how much of the order depends on the rule.

## Install

```
pip install git+https://github.com/vipul21435/rankstable
```

Python 3.10 or newer. The only runtime dependency is numpy.

## Quickstart

`examples/toy_scores.csv` is a small made-up table, six models by five benchmarks, in which
one model has a single very high score:

```
model,algebra,code,dialogue,retrieval,summaries
atlas,95.0,61.0,70.5,58.0,66.0
birch,62.0,68.0,74.0,64.5,69.5
cedar,60.0,66.5,73.0,66.0,68.0
dune,58.0,67.0,71.0,63.0,70.0
ember,40.0,69.0,72.5,61.0,67.5
flint,55.0,60.0,69.0,60.0,65.0
```

The arithmetic mean puts `atlas` first:

```
$ rankstable rank examples/toy_scores.csv
Rule: mean (higher is better). 6 models, 5 benchmarks, 0 missing cells.

| rank | model | mean |
|---:|---|---:|
| 1 | atlas | 70.1 |
| 2 | birch | 67.6 |
| 3 | cedar | 66.7 |
| 4 | dune | 65.8 |
| 5 | ember | 62.0 |
| 6 | flint | 61.8 |
```

The mean rank moves it to fifth:

```
$ rankstable rank examples/toy_scores.csv --rule mean-rank
Rule: mean-rank (lower is better). 6 models, 5 benchmarks, 0 missing cells.

| rank | model | mean-rank |
|---:|---|---:|
| 1 | birch | 1.8 |
| 2 | cedar | 2.6 |
| 3 | dune | 3.0 |
| 4 | ember | 3.6 |
| 5 | atlas | 4.4 |
| 6 | flint | 5.6 |
```

`compare-rules` runs every rule and prints Kendall's tau-b between the rankings:

```
$ rankstable compare-rules examples/toy_scores.csv
6 models, 5 benchmarks, 0 missing cells.

| rule | top 3 (rank) |
|---|---|
| mean | atlas (1), birch (2), cedar (3) |
| minmax | birch (1), cedar (2), dune (3) |
| zscore | birch (1), cedar (2), dune (3) |
| geomean | atlas (1), birch (2), cedar (3) |
| mean-rank | birch (1), cedar (2), dune (3) |
| win-rate | birch (1), cedar (2), dune (3) |
| borda | birch (1), cedar (2), dune (3) |
| copeland | birch (1), cedar (2), dune (3) |

Kendall tau-b between the rankings:

| rule | mean | minmax | zscore | geomean | mean-rank | win-rate | borda | copeland |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 1.000 | 0.467 | 0.467 | 0.867 | 0.467 | 0.467 | 0.467 | 0.467 |
| minmax | 0.467 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| zscore | 0.467 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| geomean | 0.867 | 0.333 | 0.333 | 1.000 | 0.333 | 0.333 | 0.333 | 0.333 |
| mean-rank | 0.467 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| win-rate | 0.467 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| borda | 0.467 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| copeland | 0.467 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
```

Both commands take `--format json`. The same from Python:

```python
from rankstable import aggregate, kendall_tau, read_table

table = read_table("examples/toy_scores.csv")
by_mean = aggregate(table, "mean")
by_rank = aggregate(table, "mean-rank")
print(by_mean.order())
print(by_rank.order())
print(round(kendall_tau(by_mean.ranks, by_rank.ranks), 3))
```

```
['atlas', 'birch', 'cedar', 'dune', 'ember', 'flint']
['birch', 'cedar', 'dune', 'ember', 'atlas', 'flint']
0.467
```

## Input

- **Wide CSV or TSV**: one row per model, one column per benchmark; the first column holds the
  model names. An empty cell, or `NA`, `N/A`, `NaN`, `null`, `None` or `-`, is a missing score.
- **Long JSONL**: one object per line with `model`, `benchmark` and `score` (`null` for a
  missing score), and optionally `"higher_is_better": false` for a benchmark where smaller is
  better. Files ending in `.jsonl` or `.ndjson` are read this way; `--input-format` overrides
  the guess.
- `--lower-is-better NAME` marks a benchmark as smaller-is-better; repeat it for several.
- `--missing` chooses what to do with missing cells: `keep` (the default, described below),
  `drop-models`, `drop-benchmarks` or `error`.

## How it works

Write `x_ij` for the score of model `i` on benchmark `j`, and `s_ij` for the same score with
lower-is-better benchmarks negated, so that larger `s` is always better. There are `n` models
and `m` benchmarks; `n_j` models have a score on benchmark `j`. `r_ij` is the rank of model `i`
among them, 1 for the best, tied models sharing the average of their ranks.

| rule | value of model `i` | direction |
|---|---|---|
| `mean` | mean over `j` of `s_ij` | higher |
| `minmax` | mean over `j` of `(s_ij - min_j) / (max_j - min_j)` | higher |
| `zscore` | mean over `j` of `(s_ij - mean_j) / sd_j`, with the population standard deviation | higher |
| `geomean` | geometric mean over `j` of `x_ij`, or of `1 / x_ij` on a lower-is-better benchmark | higher |
| `mean-rank` | mean over `j` of `r_ij` | lower |
| `win-rate` | mean over `j` of `(n_j - r_ij) / (n_j - 1)`: the share of the other models beaten, a tie counting half | higher |
| `borda` | sum over `j` of `n_j - r_ij`: one point per model beaten, half a point per tie | higher |
| `copeland` | pairwise contests won minus contests lost; `i` wins its contest with `k` when it is better on more benchmarks than `k` is | higher |

On a table without missing cells `borda = m * (n - mean-rank)` and
`win-rate = borda / (m * (n - 1))`, so these three rules always give the same order there, as
in the tau matrix above. They differ once cells are missing.

**Missing cells.** With `--missing keep` every rule uses the cells that are present. The
means run over the benchmarks a model has a score on, and the minimum, maximum, mean and
standard deviation of a benchmark are taken over the models that have a score on it. Ranks
on a benchmark cover only the models scored there. `mean-rank` averages those ranks as they
are, so a rank on a benchmark that few models were run on counts like the same rank on a
full one; `win-rate` divides by the number of opponents and does not have this problem.
`borda` sums points, so a benchmark without a score gives a model nothing. `copeland`
compares two models on the benchmarks both have, and a pair without a common benchmark is
level. A model without any score has no value and is ranked last.

**Ties.** On one benchmark two models are tied when their scores are equal. In the final
ranking two models are tied when their values differ by no more than 1e-12 of the largest
value or term involved, so that rounding error does not order models that are equal on paper. Tied
models share the smallest rank (1, 2, 2, 4) and are listed alphabetically.

**Other conventions.** A benchmark on which all scores are equal has no spread to normalise
and is skipped by `minmax` and `zscore`. `geomean` needs non-negative scores; a zero on a
higher-is-better benchmark makes the geometric mean zero, and a zero on a lower-is-better
benchmark is an error. `compare-rules` leaves out a rule that cannot use the table and says
so.

**Kendall's tau-b** between two rankings is `(C - D) / sqrt((P - T1) * (P - T2))`, where `C`
and `D` count the concordant and discordant pairs of models, `P = n(n-1)/2` is the number of
pairs, and `T1`, `T2` count the pairs tied in the first and in the second ranking. It is 1
for identical orders, -1 for reversed ones and undefined (`n/a`) when a ranking is constant.

## Not implemented yet

- Kemeny-Young and Bradley-Terry aggregation.
- Top-k overlap between rules.
- Stability to the choice of benchmarks: leave-one-out, leave-k-out and bootstrap over
  benchmarks, and sensitivity to adding or removing a model.
- The Friedman test with the Nemenyi critical difference, and the critical-difference diagram.
- Diversity of the benchmark set, and the agreement of each benchmark with the others.
- The commands `stability`, `agreement` and `cd`.
- Results on real public data. Every number above comes from the made-up example table.

## Development

```
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

The tests check each rule against a profile worked by hand, including a Condorcet cycle, and
check ranks, Kendall's tau-b, z-scores and geometric means against scipy, which is a
test-only dependency.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
