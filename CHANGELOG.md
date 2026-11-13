# Changelog

## 0.1.0 - 2026-11-13

First release.

- Score table read from wide CSV or TSV, or from long JSONL, with missing cells and
  lower-is-better benchmarks.
- Aggregation rules: `mean`, `minmax`, `zscore`, `geomean`, `mean-rank`, `win-rate`, `borda`,
  `copeland`, `kemeny` (exact up to 12 models, local search above that) and `bradley-terry`.
- Kendall's tau-b and top-k overlap between the rankings of different rules.
- Stability: leave-k-benchmarks-out, bootstrap over benchmarks, removal of one model at a
  time, and the diversity of the benchmark set.
- Agreement of each benchmark with the aggregate of the others, with bootstrap intervals.
- Friedman test with the Iman-Davenport statistic, the Nemenyi critical difference and a
  critical-difference diagram behind the `plot` extra.
- Commands `rank`, `compare-rules`, `stability`, `agreement` and `cd`, with Markdown and JSON
  output.
