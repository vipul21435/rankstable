"""The models-by-benchmarks score table and its readers."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Cell texts (compared case-insensitively, after stripping) that mean "no score".
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-"})


@dataclass(frozen=True, eq=False)
class ScoreTable:
    """Scores of ``n`` models on ``m`` benchmarks.

    ``scores[i, j]`` is the score of ``models[i]`` on ``benchmarks[j]`` and ``nan`` marks a
    missing cell. ``higher_is_better[j]`` gives the direction of benchmark ``j``; leaving it
    empty means every benchmark is higher-is-better.
    """

    models: tuple[str, ...]
    benchmarks: tuple[str, ...]
    scores: np.ndarray
    higher_is_better: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        models = tuple(str(name) for name in self.models)
        benchmarks = tuple(str(name) for name in self.benchmarks)
        scores = np.array(self.scores, dtype=float)
        if not models or not benchmarks:
            raise ValueError("a score table needs at least one model and one benchmark")
        if scores.shape != (len(models), len(benchmarks)):
            raise ValueError(
                f"scores have shape {scores.shape}, expected "
                f"({len(models)}, {len(benchmarks)}) for {len(models)} models and "
                f"{len(benchmarks)} benchmarks"
            )
        _require_unique(models, "model")
        _require_unique(benchmarks, "benchmark")
        if np.isinf(scores).any():
            raise ValueError("scores must be finite; use an empty cell for a missing score")
        directions = tuple(bool(flag) for flag in self.higher_is_better)
        if not directions:
            directions = (True,) * len(benchmarks)
        if len(directions) != len(benchmarks):
            raise ValueError(
                f"higher_is_better has {len(directions)} entries for {len(benchmarks)} benchmarks"
            )
        scores.setflags(write=False)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "benchmarks", benchmarks)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "higher_is_better", directions)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_benchmarks(self) -> int:
        return len(self.benchmarks)

    @property
    def n_missing(self) -> int:
        """Number of cells without a score."""
        return int(np.isnan(self.scores).sum())

    def oriented(self) -> np.ndarray:
        """A copy of the scores in which larger is better on every benchmark.

        Lower-is-better benchmarks are negated, which keeps their order and their spread.
        """
        out = self.scores.copy()
        lower = ~np.asarray(self.higher_is_better, dtype=bool)
        out[:, lower] = -out[:, lower]
        return out

    def select(
        self,
        models: Sequence[str] | None = None,
        benchmarks: Sequence[str] | None = None,
    ) -> ScoreTable:
        """The sub-table with the named models and benchmarks, in the order given."""
        rows = _positions(self.models, models, "model")
        cols = _positions(self.benchmarks, benchmarks, "benchmark")
        return ScoreTable(
            models=tuple(self.models[i] for i in rows),
            benchmarks=tuple(self.benchmarks[j] for j in cols),
            scores=self.scores[np.ix_(rows, cols)],
            higher_is_better=tuple(self.higher_is_better[j] for j in cols),
        )

    def resample_benchmarks(self, positions: Sequence[int] | np.ndarray) -> ScoreTable:
        """The table made of the benchmarks at ``positions``, which may repeat.

        A benchmark drawn more than once appears as separate columns, the repeats named
        ``name (2)``, ``name (3)`` and so on. This is the resampling step of a bootstrap over
        benchmarks.
        """
        seen: dict[int, int] = {}
        names = []
        for position in (int(j) for j in positions):
            seen[position] = seen.get(position, 0) + 1
            name = self.benchmarks[position]
            names.append(name if seen[position] == 1 else f"{name} ({seen[position]})")
        columns = [int(j) for j in positions]
        return ScoreTable(
            self.models,
            tuple(names),
            self.scores[:, columns],
            tuple(self.higher_is_better[j] for j in columns),
        )

    def drop_incomplete(self, axis: str = "models") -> ScoreTable:
        """Drop every model (or every benchmark) that has a missing cell."""
        missing = np.isnan(self.scores)
        if axis == "models":
            keep = [self.models[i] for i in np.flatnonzero(~missing.any(axis=1))]
            if not keep:
                raise ValueError("no model has a score on every benchmark")
            return self.select(models=keep)
        if axis == "benchmarks":
            keep = [self.benchmarks[j] for j in np.flatnonzero(~missing.any(axis=0))]
            if not keep:
                raise ValueError("no benchmark has a score for every model")
            return self.select(benchmarks=keep)
        raise ValueError(f"axis must be 'models' or 'benchmarks', got {axis!r}")

    def with_lower_is_better(self, benchmarks: Iterable[str]) -> ScoreTable:
        """The same table with the named benchmarks marked lower-is-better."""
        names = list(benchmarks)
        unknown = [name for name in names if name not in self.benchmarks]
        if unknown:
            raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
        flags = tuple(
            flag and name not in names
            for name, flag in zip(self.benchmarks, self.higher_is_better, strict=True)
        )
        return ScoreTable(self.models, self.benchmarks, self.scores, flags)


def _require_unique(names: Sequence[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if not name:
            raise ValueError(f"empty {what} name")
        if name in seen:
            raise ValueError(f"duplicate {what} name: {name!r}")
        seen.add(name)


def _positions(names: Sequence[str], wanted: Sequence[str] | None, what: str) -> list[int]:
    if wanted is None:
        return list(range(len(names)))
    index = {name: i for i, name in enumerate(names)}
    unknown = [name for name in wanted if name not in index]
    if unknown:
        raise ValueError(f"unknown {what}(s): {', '.join(map(str, unknown))}")
    return [index[name] for name in wanted]


def parse_wide_csv(text: str, *, delimiter: str = ",") -> ScoreTable:
    """Parse a wide table: one row per model, one column per benchmark.

    The first row is the header and the first column holds the model names (its own header
    cell is ignored). A cell that is empty or one of ``NA``, ``N/A``, ``NaN``, ``null``,
    ``None`` or ``-`` is a missing score. Blank lines are skipped.
    """
    rows = [
        (number, [cell.strip() for cell in row])
        for number, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter), start=1)
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        raise ValueError("the table is empty")
    _, header = rows[0]
    benchmarks = header[1:]
    if not benchmarks:
        raise ValueError("the header needs a model column followed by benchmark columns")
    models: list[str] = []
    values: list[list[float]] = []
    for number, row in rows[1:]:
        if len(row) != len(header):
            raise ValueError(f"line {number} has {len(row)} cells, the header has {len(header)}")
        models.append(row[0])
        values.append(
            [
                _parse_cell(cell, f"line {number}, column {name!r}")
                for name, cell in zip(benchmarks, row[1:], strict=True)
            ]
        )
    if not models:
        raise ValueError("the table has a header but no model rows")
    return ScoreTable(tuple(models), tuple(benchmarks), np.array(values, dtype=float))


def _parse_cell(cell: str, where: str) -> float:
    if cell.lower() in MISSING_TOKENS:
        return math.nan
    try:
        value = float(cell)
    except ValueError:
        raise ValueError(f"{where}: {cell!r} is not a number") from None
    if math.isinf(value):
        raise ValueError(f"{where}: scores must be finite")
    return value


def parse_long_jsonl(text: str) -> ScoreTable:
    """Parse a long table: one JSON object per line with ``model``, ``benchmark``, ``score``.

    ``score`` may be ``null`` for a missing cell, and a cell that never appears is missing
    too. An optional boolean ``higher_is_better`` sets the direction of the benchmark; it
    defaults to true and must not contradict itself within a benchmark. Models and benchmarks
    keep the order in which they first appear.
    """
    models: dict[str, int] = {}
    benchmarks: dict[str, int] = {}
    directions: dict[str, bool] = {}
    cells: dict[tuple[int, int], float] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {number}: invalid JSON ({error.msg})") from None
        if not isinstance(record, dict):
            raise ValueError(f"line {number}: expected a JSON object")
        model = _name_field(record, "model", number)
        benchmark = _name_field(record, "benchmark", number)
        if "score" not in record:
            raise ValueError(f"line {number}: missing 'score'")
        score = record["score"]
        if score is None:
            value = math.nan
        elif isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"line {number}: 'score' must be a number or null")
        else:
            value = float(score)
            if math.isinf(value):
                raise ValueError(f"line {number}: scores must be finite")
        flag = record.get("higher_is_better")
        if flag is not None:
            if not isinstance(flag, bool):
                raise ValueError(f"line {number}: 'higher_is_better' must be true or false")
            if directions.setdefault(benchmark, flag) != flag:
                raise ValueError(
                    f"line {number}: benchmark {benchmark!r} has conflicting "
                    "'higher_is_better' values"
                )
        key = (
            models.setdefault(model, len(models)),
            benchmarks.setdefault(benchmark, len(benchmarks)),
        )
        if key in cells:
            raise ValueError(
                f"line {number}: second score for model {model!r} on benchmark {benchmark!r}"
            )
        cells[key] = value
    if not cells:
        raise ValueError("the table is empty")
    scores = np.full((len(models), len(benchmarks)), np.nan)
    for (i, j), value in cells.items():
        scores[i, j] = value
    return ScoreTable(
        tuple(models),
        tuple(benchmarks),
        scores,
        tuple(directions.get(name, True) for name in benchmarks),
    )


def _name_field(record: dict, field: str, number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {number}: '{field}' must be a non-empty string")
    return value.strip()


def read_table(path: str | Path, *, fmt: str = "auto") -> ScoreTable:
    """Read a score table from ``path``.

    ``fmt`` is ``"csv"`` (wide, comma separated), ``"tsv"`` (wide, tab separated), ``"jsonl"``
    (long) or ``"auto"``, which decides by the file suffix and falls back to wide CSV.
    """
    path = Path(path)
    if fmt == "auto":
        suffix = path.suffix.lower()
        fmt = {".jsonl": "jsonl", ".ndjson": "jsonl", ".tsv": "tsv"}.get(suffix, "csv")
    text = path.read_text(encoding="utf-8-sig")
    if fmt == "jsonl":
        return parse_long_jsonl(text)
    if fmt == "tsv":
        return parse_wide_csv(text, delimiter="\t")
    if fmt == "csv":
        return parse_wide_csv(text)
    raise ValueError(f"unknown table format {fmt!r}; use auto, csv, tsv or jsonl")
