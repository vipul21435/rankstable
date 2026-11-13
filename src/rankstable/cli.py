"""Command line interface: ``rank``, ``compare-rules``, ``stability`` and ``agreement``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

from . import __version__
from .agreement import benchmark_agreement
from .ranking import Ranking, kendall_tau
from .rules import RULES, aggregate
from .stability import (
    benchmark_diversity,
    bootstrap_benchmarks,
    leave_out,
    remove_models,
    top_k_overlap,
)
from .table import ScoreTable, read_table

MISSING_POLICIES = ("keep", "drop-models", "drop-benchmarks", "error")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rankstable",
        description="Rank models over several benchmarks and compare aggregation rules.",
    )
    parser.add_argument("--version", action="version", version=f"rankstable {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "table",
        help="wide CSV or TSV (one row per model, one column per benchmark) or long JSONL "
        "(model, benchmark, score per line)",
    )
    common.add_argument(
        "--input-format",
        choices=("auto", "csv", "tsv", "jsonl"),
        default="auto",
        help="table format (default: by file suffix, else wide CSV)",
    )
    common.add_argument(
        "--lower-is-better",
        action="append",
        default=[],
        metavar="BENCHMARK",
        help="a benchmark on which a smaller score is better; repeat for several",
    )
    common.add_argument(
        "--missing",
        choices=MISSING_POLICIES,
        default="keep",
        help="missing cells: keep them and let each rule use the cells present (default), "
        "drop the models or the benchmarks that have one, or stop with an error",
    )
    common.add_argument(
        "--format", choices=("markdown", "json"), default="markdown", help="output format"
    )

    rank = commands.add_parser("rank", parents=[common], help="rank the models with one rule")
    rank.add_argument("--rule", choices=tuple(RULES), default="mean", help="aggregation rule")
    rank.set_defaults(run=_run_rank)

    compare = commands.add_parser(
        "compare-rules",
        parents=[common],
        help="rank the models with several rules and compare the rankings",
    )
    compare.add_argument(
        "--rules",
        default=None,
        metavar="NAMES",
        help=f"comma-separated rules (default: all of {', '.join(RULES)})",
    )
    compare.add_argument(
        "--top", type=int, default=3, metavar="K", help="models listed per rule (default 3)"
    )
    compare.set_defaults(run=_run_compare)

    stability = commands.add_parser(
        "stability",
        parents=[common],
        help="how often the first place and the top set change when benchmarks or models change",
    )
    stability.add_argument(
        "--rules", default=None, metavar="NAMES", help="comma-separated rules (default: all)"
    )
    stability.add_argument(
        "--top", type=int, default=5, metavar="K", help="size of the top set tracked (default 5)"
    )
    stability.add_argument(
        "--leave-out",
        default="1",
        metavar="K[,K]",
        help="how many benchmarks to leave out at a time; several values allowed (default 1)",
    )
    stability.add_argument(
        "--max-subsets",
        type=int,
        default=1000,
        metavar="N",
        help="try all subsets up to this many, sample this many beyond (default 1000)",
    )
    stability.add_argument(
        "--resamples",
        type=int,
        default=1000,
        metavar="B",
        help="bootstrap resamples of the benchmarks; 0 skips the bootstrap (default 1000)",
    )
    stability.add_argument(
        "--removals",
        type=int,
        default=200,
        metavar="N",
        help="models removed one at a time, sampled beyond this many; 0 skips (default 200)",
    )
    stability.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    stability.set_defaults(run=_run_stability)

    agreement = commands.add_parser(
        "agreement",
        parents=[common],
        help="rank correlation of each benchmark with the aggregate of the others",
    )
    agreement.add_argument(
        "--rule",
        choices=tuple(RULES),
        default="win-rate",
        help="rule that aggregates the other benchmarks (default win-rate)",
    )
    agreement.add_argument(
        "--resamples", type=int, default=1000, metavar="B", help="bootstrap resamples of the models"
    )
    agreement.add_argument(
        "--confidence", type=float, default=0.95, help="interval level (default 0.95)"
    )
    agreement.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    agreement.set_defaults(run=_run_agreement)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        output = args.run(args)
    except ValueError as error:
        print(f"rankstable: error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


def _load(args: argparse.Namespace) -> ScoreTable:
    try:
        table = read_table(args.table, fmt=args.input_format)
    except OSError as error:
        raise ValueError(f"cannot read {args.table}: {error.strerror or error}") from None
    if args.lower_is_better:
        table = table.with_lower_is_better(args.lower_is_better)
    if args.missing == "error" and table.n_missing:
        raise ValueError(f"the table has {table.n_missing} missing cell(s)")
    if args.missing == "drop-models":
        table = table.drop_incomplete("models")
    elif args.missing == "drop-benchmarks":
        table = table.drop_incomplete("benchmarks")
    return table


def _shape(table: ScoreTable) -> dict[str, int]:
    return {
        "n_models": table.n_models,
        "n_benchmarks": table.n_benchmarks,
        "n_missing": table.n_missing,
    }


def _shape_sentence(table: ScoreTable) -> str:
    return (
        f"{_count(table.n_models, 'model')}, {_count(table.n_benchmarks, 'benchmark')}, "
        f"{_count(table.n_missing, 'missing cell')}"
    )


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _run_rank(args: argparse.Namespace) -> str:
    table = _load(args)
    ranking = aggregate(table, args.rule)
    if args.format == "json":
        payload = {
            "command": "rank",
            "rule": ranking.rule,
            "higher_is_better": ranking.higher_is_better,
            **_shape(table),
            "ranking": ranking.rows(),
        }
        return json.dumps(payload, indent=2)
    direction = "higher is better" if ranking.higher_is_better else "lower is better"
    records = ranking.rows()
    values = format_column([row["value"] for row in records])
    rows = [
        [str(row["rank"]), _cell(row["model"]), value]
        for row, value in zip(records, values, strict=True)
    ]
    return "\n".join(
        [
            f"Rule: {ranking.rule} ({direction}). {_shape_sentence(table)}.",
            "",
            markdown_table(["rank", "model", ranking.rule], rows, "rlr"),
        ]
    )


def _select_rules(spec: str | None) -> list[str]:
    names = list(RULES) if spec is None else [name.strip() for name in spec.split(",")]
    names = list(dict.fromkeys(name for name in names if name))
    unknown = [name for name in names if name not in RULES]
    if unknown:
        raise ValueError(f"unknown rule(s): {', '.join(unknown)}; choose from {', '.join(RULES)}")
    if not names:
        raise ValueError("no rule given")
    return names


def _rank_all(table: ScoreTable, names: list[str]) -> tuple[dict[str, Ranking], dict[str, str]]:
    """Rank with every rule that can use the table; the others come back with the reason."""
    rankings: dict[str, Ranking] = {}
    skipped: dict[str, str] = {}
    for name in names:
        try:
            rankings[name] = aggregate(table, name)
        except ValueError as error:
            skipped[name] = str(error)
    if not rankings:
        raise ValueError("no rule could rank this table: " + "; ".join(skipped.values()))
    return rankings, skipped


def _skipped_lines(skipped: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for name, reason in skipped.items():
        lines += ["", f"Skipped {name}: {reason}."]
    return lines


def _json_number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _run_compare(args: argparse.Namespace) -> str:
    table = _load(args)
    names = _select_rules(args.rules)
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    rankings, skipped = _rank_all(table, names)
    used = list(rankings)
    tau = {a: {b: kendall_tau(rankings[a].ranks, rankings[b].ranks) for b in used} for a in used}
    overlap = {
        a: {b: top_k_overlap(rankings[a], rankings[b], args.top) for b in used} for a in used
    }

    if args.format == "json":
        payload = {
            "command": "compare-rules",
            **_shape(table),
            "rules": used,
            "skipped": skipped,
            "rankings": {name: rankings[name].rows() for name in used},
            "kendall_tau": {
                a: {b: _json_number(value) for b, value in row.items()} for a, row in tau.items()
            },
            "top": args.top,
            "top_k_overlap": overlap,
        }
        return json.dumps(payload, indent=2)

    top_rows = [
        [
            name,
            ", ".join(
                f"{_cell(row['model'])} ({row['rank']})"
                for row in rankings[name].rows()[: args.top]
            ),
        ]
        for name in used
    ]
    align = "l" + "r" * len(used)
    lines = [
        f"{_shape_sentence(table)}.",
        "",
        markdown_table(["rule", f"top {args.top} (rank)"], top_rows, "ll"),
        "",
        "Kendall tau-b between the rankings:",
        "",
        markdown_table(
            ["rule", *used], [[a] + [_fixed(tau[a][b], 3) for b in used] for a in used], align
        ),
        "",
        f"Share of the top {args.top} in common:",
        "",
        markdown_table(
            ["rule", *used], [[a] + [_fixed(overlap[a][b], 3) for b in used] for a in used], align
        ),
    ]
    return "\n".join(lines + _skipped_lines(skipped))


def _percent(share: float) -> str:
    return "n/a" if math.isnan(share) else f"{100 * share:.1f} %"


def _positive_int_list(spec: str, option: str) -> list[int]:
    try:
        values = [int(part) for part in spec.split(",") if part.strip()]
    except ValueError:
        raise ValueError(f"{option} takes whole numbers separated by commas") from None
    if not values:
        raise ValueError(f"{option} needs at least one number")
    return list(dict.fromkeys(values))


def _run_stability(args: argparse.Namespace) -> str:
    table = _load(args)
    names = _select_rules(args.rules)
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    if args.resamples < 0 or args.removals < 0:
        raise ValueError("--resamples and --removals cannot be negative")
    sizes = _positive_int_list(args.leave_out, "--leave-out")
    rankings, skipped = _rank_all(table, names)
    used = list(rankings)
    top = args.top

    left_out = {
        k: {
            name: leave_out(table, name, k=k, top=top, max_subsets=args.max_subsets, seed=args.seed)
            for name in used
        }
        for k in sizes
    }
    resampled = (
        {
            name: bootstrap_benchmarks(
                table, name, n_resamples=args.resamples, top=top, seed=args.seed
            )
            for name in used
        }
        if args.resamples
        else {}
    )
    removed = (
        {
            name: remove_models(table, name, max_removals=args.removals, seed=args.seed)
            for name in used
        }
        if args.removals and table.n_models >= 3
        else {}
    )
    diversity = benchmark_diversity(table)

    if args.format == "json":
        payload = {
            "command": "stability",
            **_shape(table),
            "top": top,
            "seed": args.seed,
            "diversity": _json_number(diversity),
            "rules": used,
            "skipped": skipped,
            "leave_out": [
                {
                    "k": k,
                    "rule": name,
                    "n_subsets": result.n_subsets,
                    "exhaustive": result.exhaustive,
                    "first": list(result.full_top1),
                    "first_changed": result.top1_changed,
                    "top_changed": result.topk_changed,
                    "mean_tau": _json_number(result.mean_tau),
                    "min_tau": _json_number(result.min_tau),
                    "subsets": [
                        {
                            "dropped": list(outcome.dropped),
                            "first": list(outcome.top1),
                            "first_changed": outcome.top1_changed,
                            "top_changed": outcome.topk_changed,
                            "tau": _json_number(outcome.tau),
                        }
                        for outcome in result.outcomes
                    ],
                }
                for k, by_rule in left_out.items()
                for name, result in by_rule.items()
            ],
            "bootstrap": [
                {
                    "rule": name,
                    "n_resamples": result.n_resamples,
                    "confidence": result.confidence,
                    "first": list(result.full_top1),
                    "first_changed": result.top1_changed,
                    "top_changed": result.topk_changed,
                    "mean_tau": _json_number(result.mean_tau),
                    "tau_low": _json_number(result.tau_low),
                    "tau_high": _json_number(result.tau_high),
                    "models": [
                        {
                            "model": model,
                            "p_first": float(result.p_top1[i]),
                            "p_top": float(result.p_topk[i]),
                            "rank_low": float(result.rank_low[i]),
                            "rank_high": float(result.rank_high[i]),
                        }
                        for i, model in enumerate(result.models)
                    ],
                }
                for name, result in resampled.items()
            ],
            "remove_models": [
                {
                    "rule": name,
                    "n_removals": result.n_removals,
                    "exhaustive": result.exhaustive,
                    "order_changed": result.order_changed,
                    "mean_pairs_changed": result.mean_pairs_changed,
                    "first_changed": result.top1_changed,
                    "largest_effect": {"model": result.worst[0], "pairs_changed": result.worst[1]},
                }
                for name, result in removed.items()
            ],
        }
        return json.dumps(payload, indent=2)

    lines = [
        f"{_shape_sentence(table)}.",
        "",
        "Diversity of the benchmark set (mean Kendall distance between benchmark rankings, "
        f"0 = same order, 1 = reversed): {_fixed(diversity, 3)}",
    ]
    for k, by_rule in left_out.items():
        any_result = next(iter(by_rule.values()))
        how = "all" if any_result.exhaustive else "a sample of"
        lines += [
            "",
            f"Leave {k} of {_count(table.n_benchmarks, 'benchmark')} out "
            f"({how} {_count(any_result.n_subsets, 'subset')}):",
            "",
            markdown_table(
                ["rule", "first on the full table", "first changes", f"top {top} changes"]
                + ["mean tau", "min tau"],
                [
                    [
                        name,
                        _cell(", ".join(result.full_top1)),
                        _percent(result.top1_changed),
                        _percent(result.topk_changed),
                        _fixed(result.mean_tau, 3),
                        _fixed(result.min_tau, 3),
                    ]
                    for name, result in by_rule.items()
                ],
                "llrrrr",
            ),
        ]
    if resampled:
        lines += [
            "",
            f"Bootstrap over benchmarks ({_count(args.resamples, 'resample')}, seed {args.seed}):",
            "",
            markdown_table(
                ["rule", "first changes", f"top {top} changes", "mean tau", "tau 95 % interval"],
                [
                    [
                        name,
                        _percent(result.top1_changed),
                        _percent(result.topk_changed),
                        _fixed(result.mean_tau, 3),
                        f"{_fixed(result.tau_low, 3)} to {_fixed(result.tau_high, 3)}",
                    ]
                    for name, result in resampled.items()
                ],
                "lrrrr",
            ),
        ]
    if removed:
        any_removal = next(iter(removed.values()))
        how = "every model" if any_removal.exhaustive else "a sample of models"
        lines += [
            "",
            f"Remove one model at a time ({how}, {_count(any_removal.n_removals, 'removal')}):",
            "",
            markdown_table(
                ["rule", "order of the others changes", "pairs changed (mean)", "first changes"]
                + ["largest effect"],
                [
                    [
                        name,
                        _percent(result.order_changed),
                        _percent(result.mean_pairs_changed),
                        _percent(result.top1_changed),
                        f"{_cell(result.worst[0])} ({_percent(result.worst[1])} of pairs)"
                        if result.worst[1] > 0
                        else "none",
                    ]
                    for name, result in removed.items()
                ],
                "lrrrl",
            ),
        ]
    return "\n".join(lines + _skipped_lines(skipped))


def _run_agreement(args: argparse.Namespace) -> str:
    table = _load(args)
    results = benchmark_agreement(
        table,
        args.rule,
        n_resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    ordered = sorted(results, key=lambda r: (math.isnan(r.tau), r.tau, r.benchmark))
    if args.format == "json":
        payload = {
            "command": "agreement",
            **_shape(table),
            "rule": args.rule,
            "n_resamples": args.resamples,
            "confidence": args.confidence,
            "seed": args.seed,
            "benchmarks": [
                {
                    "benchmark": r.benchmark,
                    "n_models": r.n_models,
                    "tau": _json_number(r.tau),
                    "low": _json_number(r.low),
                    "high": _json_number(r.high),
                }
                for r in ordered
            ],
        }
        return json.dumps(payload, indent=2)
    level = f"{100 * args.confidence:g} %"
    lines = [
        f"{_shape_sentence(table)}.",
        "",
        f"Kendall tau-b of each benchmark with the {args.rule} ranking of the other benchmarks, "
        f"lowest first ({level} bootstrap interval over models, "
        f"{_count(args.resamples, 'resample')}, seed {args.seed}):",
        "",
        markdown_table(
            ["benchmark", "tau", f"{level} interval", "models"],
            [
                [
                    _cell(r.benchmark),
                    _fixed(r.tau, 3),
                    f"{_fixed(r.low, 3)} to {_fixed(r.high, 3)}",
                    str(r.n_models),
                ]
                for r in ordered
            ],
            "lrrr",
        ),
    ]
    return "\n".join(lines)


def format_column(values: Sequence[float | None], digits: int = 4) -> list[str]:
    """Format one column of numbers with a shared number of decimals.

    The column gets as many decimals as its values need, up to ``digits``; a missing value
    (``None`` or ``nan``) becomes ``n/a``.
    """
    texts = [None if _is_missing(value) else f"{value:.{digits}f}" for value in values]
    needed = max((len(text.partition(".")[2].rstrip("0")) for text in texts if text), default=0)
    return ["n/a" if text is None else _fixed(float(text), needed) for text in texts]


def _fixed(value: float | None, digits: int) -> str:
    if _is_missing(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _is_missing(value: float | None) -> bool:
    return value is None or math.isnan(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> str:
    """A Markdown table; ``align`` has one ``l`` or ``r`` per column."""
    rule = "|" + "|".join("---:" if side == "r" else "---" for side in align) + "|"
    lines = ["| " + " | ".join(headers) + " |", rule]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|")
