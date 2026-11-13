"""Command line interface: ``rankstable rank`` and ``rankstable compare-rules``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

from . import __version__
from .ranking import Ranking, kendall_tau
from .rules import RULES, aggregate
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


def _run_compare(args: argparse.Namespace) -> str:
    table = _load(args)
    names = list(RULES) if args.rules is None else [n.strip() for n in args.rules.split(",")]
    names = list(dict.fromkeys(name for name in names if name))
    unknown = [name for name in names if name not in RULES]
    if unknown:
        raise ValueError(f"unknown rule(s): {', '.join(unknown)}; choose from {', '.join(RULES)}")
    if not names:
        raise ValueError("no rule given")
    if args.top < 1:
        raise ValueError("--top must be at least 1")

    rankings: dict[str, Ranking] = {}
    skipped: dict[str, str] = {}
    for name in names:
        try:
            rankings[name] = aggregate(table, name)
        except ValueError as error:
            skipped[name] = str(error)
    if not rankings:
        raise ValueError("no rule could rank this table: " + "; ".join(skipped.values()))
    used = list(rankings)
    tau = {a: {b: kendall_tau(rankings[a].ranks, rankings[b].ranks) for b in used} for a in used}

    if args.format == "json":
        payload = {
            "command": "compare-rules",
            **_shape(table),
            "rules": used,
            "skipped": skipped,
            "rankings": {name: rankings[name].rows() for name in used},
            "kendall_tau": {
                a: {b: None if math.isnan(value) else value for b, value in row.items()}
                for a, row in tau.items()
            },
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
    tau_rows = [[a] + [_fixed(tau[a][b], 3) for b in used] for a in used]
    lines = [
        f"{_shape_sentence(table)}.",
        "",
        markdown_table(["rule", f"top {args.top} (rank)"], top_rows, "ll"),
        "",
        "Kendall tau-b between the rankings:",
        "",
        markdown_table(["rule", *used], tau_rows, "l" + "r" * len(used)),
    ]
    for name, reason in skipped.items():
        lines += ["", f"Skipped {name}: {reason}."]
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
