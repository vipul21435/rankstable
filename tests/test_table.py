import json

import numpy as np
import pytest

from rankstable.table import ScoreTable, parse_long_jsonl, parse_wide_csv, read_table

WIDE = """model,alpha,beta,gamma
a,90,10,50
b,80,,40
c,60,20,NA
"""


def test_wide_csv_reads_names_scores_and_missing_cells():
    table = parse_wide_csv(WIDE)
    assert table.models == ("a", "b", "c")
    assert table.benchmarks == ("alpha", "beta", "gamma")
    assert table.higher_is_better == (True, True, True)
    np.testing.assert_array_equal(
        table.scores,
        np.array([[90, 10, 50], [80, np.nan, 40], [60, 20, np.nan]], dtype=float),
    )
    assert (table.n_models, table.n_benchmarks, table.n_missing) == (3, 3, 2)


@pytest.mark.parametrize("token", ["", "NA", "n/a", "NaN", "null", "None", "-", "  "])
def test_wide_csv_missing_tokens(token):
    table = parse_wide_csv(f"model,x,y\na,1,{token}\nb,2,3\n")
    assert np.isnan(table.scores[0, 1])
    assert table.n_missing == 1


def test_wide_csv_skips_blank_lines_and_strips_cells():
    table = parse_wide_csv("model , x , y\n\n a , 1 , 2 \n,,\nb,3,4\n")
    assert table.models == ("a", "b")
    assert table.benchmarks == ("x", "y")
    np.testing.assert_array_equal(table.scores, [[1, 2], [3, 4]])


def test_wide_csv_tab_delimiter():
    table = parse_wide_csv("model\tx\ty\na\t1\t2\n", delimiter="\t")
    assert table.benchmarks == ("x", "y")
    np.testing.assert_array_equal(table.scores, [[1, 2]])


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "empty"),
        ("model\na\n", "benchmark columns"),
        ("model,x\n", "no model rows"),
        ("model,x,y\na,1\n", "line 2 has 2 cells"),
        ("model,x\na,1\na,2\n", "duplicate model name"),
        ("model,x,x\na,1,2\n", "duplicate benchmark name"),
        ("model,x\na,seven\n", "line 2, column 'x': 'seven' is not a number"),
        ("model,x\na,inf\n", "finite"),
        ("model,x\n,1\n", "empty model name"),
    ],
)
def test_wide_csv_rejects_malformed_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_wide_csv(text)


def _jsonl(records):
    return "\n".join(json.dumps(record) for record in records) + "\n"


def test_long_jsonl_builds_the_table_in_order_of_first_appearance():
    table = parse_long_jsonl(
        _jsonl(
            [
                {"model": "b", "benchmark": "speed", "score": 2.5, "higher_is_better": False},
                {"model": "a", "benchmark": "speed", "score": 4},
                {"model": "a", "benchmark": "accuracy", "score": 0.75},
                {"model": "c", "benchmark": "accuracy", "score": None},
            ]
        )
    )
    assert table.models == ("b", "a", "c")
    assert table.benchmarks == ("speed", "accuracy")
    assert table.higher_is_better == (False, True)
    np.testing.assert_array_equal(
        table.scores, np.array([[2.5, np.nan], [4, 0.75], [np.nan, np.nan]])
    )


@pytest.mark.parametrize(
    "lines, message",
    [
        ("", "empty"),
        ("{not json}", "line 1: invalid JSON"),
        ("[1, 2]", "line 1: expected a JSON object"),
        ('{"benchmark": "x", "score": 1}', "'model' must be a non-empty string"),
        ('{"model": "a", "benchmark": 3, "score": 1}', "'benchmark' must be a non-empty string"),
        ('{"model": "a", "benchmark": "x"}', "missing 'score'"),
        ('{"model": "a", "benchmark": "x", "score": "high"}', "must be a number or null"),
        ('{"model": "a", "benchmark": "x", "score": true}', "must be a number or null"),
        ('{"model": "a", "benchmark": "x", "score": Infinity}', "finite"),
        (
            '{"model": "a", "benchmark": "x", "score": 1}\n'
            '{"model": "a", "benchmark": "x", "score": 2}',
            "line 2: second score",
        ),
        (
            '{"model": "a", "benchmark": "x", "score": 1, "higher_is_better": true}\n'
            '{"model": "b", "benchmark": "x", "score": 2, "higher_is_better": false}',
            "conflicting",
        ),
        (
            '{"model": "a", "benchmark": "x", "score": 1, "higher_is_better": "yes"}',
            "must be true or false",
        ),
    ],
)
def test_long_jsonl_rejects_malformed_input(lines, message):
    with pytest.raises(ValueError, match=message):
        parse_long_jsonl(lines)


def test_long_jsonl_skips_blank_lines():
    table = parse_long_jsonl('\n{"model": "a", "benchmark": "x", "score": 1}\n\n')
    assert (table.models, table.benchmarks, table.scores.tolist()) == (("a",), ("x",), [[1.0]])


def test_read_table_picks_the_format_from_the_suffix(tmp_path):
    wide = tmp_path / "scores.csv"
    wide.write_text("﻿" + WIDE, encoding="utf-8")
    tabbed = tmp_path / "scores.tsv"
    tabbed.write_text(WIDE.replace(",", "\t"), encoding="utf-8")
    expected = parse_wide_csv(WIDE)
    records = [
        {"model": model, "benchmark": benchmark, "score": expected.scores[i, j]}
        for i, model in enumerate(expected.models)
        for j, benchmark in enumerate(expected.benchmarks)
        if not np.isnan(expected.scores[i, j])
    ]
    long = tmp_path / "scores.jsonl"
    long.write_text(_jsonl(records), encoding="utf-8")
    for path in (wide, tabbed, long):
        table = read_table(path)
        assert table.models == expected.models
        assert table.benchmarks == expected.benchmarks
        np.testing.assert_array_equal(table.scores, expected.scores)
    np.testing.assert_array_equal(read_table(str(wide), fmt="csv").scores, expected.scores)
    with pytest.raises(ValueError, match="unknown table format"):
        read_table(wide, fmt="xlsx")


def test_table_validates_its_parts():
    with pytest.raises(ValueError, match="at least one model"):
        ScoreTable((), ("x",), np.zeros((0, 1)))
    with pytest.raises(ValueError, match="shape"):
        ScoreTable(("a", "b"), ("x",), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="higher_is_better has 2 entries"):
        ScoreTable(("a",), ("x",), np.zeros((1, 1)), (True, False))
    with pytest.raises(ValueError, match="finite"):
        ScoreTable(("a",), ("x",), np.array([[np.inf]]))


def test_table_copies_and_freezes_its_scores():
    raw = np.array([[1.0, 2.0]])
    table = ScoreTable(("a",), ("x", "y"), raw)
    raw[0, 0] = 99.0
    assert table.scores[0, 0] == 1.0
    with pytest.raises(ValueError):
        table.scores[0, 0] = 5.0


def test_oriented_negates_lower_is_better_benchmarks():
    table = ScoreTable(("a", "b"), ("acc", "err"), [[0.9, 0.2], [0.8, np.nan]], (True, False))
    np.testing.assert_array_equal(table.oriented(), [[0.9, -0.2], [0.8, np.nan]])
    np.testing.assert_array_equal(table.scores, [[0.9, 0.2], [0.8, np.nan]])


def test_with_lower_is_better_marks_named_benchmarks():
    table = parse_wide_csv(WIDE).with_lower_is_better(["beta"])
    assert table.higher_is_better == (True, False, True)
    with pytest.raises(ValueError, match="unknown benchmark"):
        table.with_lower_is_better(["delta"])


def test_select_reorders_and_subsets():
    table = parse_wide_csv(WIDE).with_lower_is_better(["gamma"])
    part = table.select(models=["c", "a"], benchmarks=["gamma", "alpha"])
    assert part.models == ("c", "a")
    assert part.benchmarks == ("gamma", "alpha")
    assert part.higher_is_better == (False, True)
    np.testing.assert_array_equal(part.scores, [[np.nan, 60], [50, 90]])
    with pytest.raises(ValueError, match="unknown model"):
        table.select(models=["z"])


def test_drop_incomplete_models_or_benchmarks():
    table = parse_wide_csv(WIDE)
    assert table.drop_incomplete("models").models == ("a",)
    assert table.drop_incomplete("benchmarks").benchmarks == ("alpha",)
    with pytest.raises(ValueError, match="axis"):
        table.drop_incomplete("cells")
    sparse = parse_wide_csv("model,x,y\na,1,\nb,,2\n")
    with pytest.raises(ValueError, match="no model has a score on every benchmark"):
        sparse.drop_incomplete("models")
    with pytest.raises(ValueError, match="no benchmark has a score for every model"):
        sparse.drop_incomplete("benchmarks")


def test_long_jsonl_lines_end_only_at_newlines():
    # U+2028 is legal inside a JSON string and must not split the line.
    first = '{"model": "a\u2028b", "benchmark": "x", "score": 1}'
    second = '{"model": "c", "benchmark": "x", "score": 2}'
    text = first + "\r\n" + second + "\r\n"
    assert "\u2028" in first
    table = parse_long_jsonl(text)
    assert table.models == ("a\u2028b", "c")


def test_parsers_turn_unusual_input_into_value_errors():
    with pytest.raises(ValueError, match="line 1: scores must be finite"):
        parse_long_jsonl('{"model": "a", "benchmark": "x", "score": 1' + "0" * 400 + "}")
    with pytest.raises(ValueError, match="line 1: invalid JSON"):
        parse_long_jsonl("[" * 100000 + "]" * 100000)
    with pytest.raises(ValueError, match="line 1: invalid JSON"):
        parse_long_jsonl('{"model": "a", "benchmark": "x", "score": 1' + "0" * 5000 + "}")
    with pytest.raises(ValueError):
        parse_wide_csv("model,x\na,1\x00\n")
    with pytest.raises(ValueError, match="not a readable CSV table"):
        parse_wide_csv("model,x\na," + "1" * 200000 + "\n")
    quoted = parse_wide_csv('model,"x,1"\r\n"a ""b""",2\r\n')
    assert (quoted.models, quoted.benchmarks) == (('a "b"',), ("x,1",))


def test_parsers_only_ever_raise_value_errors():
    rng = np.random.default_rng(0)
    pieces = [
        ",",
        "\n",
        "\r\n",
        '"',
        "\t",
        " ",
        "a",
        "b",
        "model",
        "1",
        "-2.5",
        "1e400",
        "nan",
        "inf",
        "{",
        "}",
        "[",
        "]",
        ":",
        '"model"',
        '"benchmark"',
        '"score"',
        "null",
        "true",
        "\x00",
        "\u2028",
        "é",
        "0x1f",
        "1_0",
        "-",
        "NA",
    ]
    for _ in range(3000):
        text = "".join(rng.choice(pieces, size=int(rng.integers(0, 25))))
        for parse in (parse_wide_csv, parse_long_jsonl):
            try:
                table = parse(text)
            except ValueError:
                continue
            assert table.n_models >= 1 and table.n_benchmarks >= 1
            assert not np.isinf(table.scores).any()
