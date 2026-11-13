import json
import subprocess
import sys
from pathlib import Path

import pytest

from rankstable import __version__
from rankstable.cli import format_column, main, markdown_table

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy_scores.csv"


def run(capsys, *argv):
    code = main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_rank_prints_a_markdown_table(capsys):
    code, out, err = run(capsys, "rank", TOY)
    assert (code, err) == (0, "")
    assert out.splitlines() == [
        "Rule: mean (higher is better). 6 models, 5 benchmarks, 0 missing cells.",
        "",
        "| rank | model | mean |",
        "|---:|---|---:|",
        "| 1 | atlas | 70.1 |",
        "| 2 | birch | 67.6 |",
        "| 3 | cedar | 66.7 |",
        "| 4 | dune | 65.8 |",
        "| 5 | ember | 62.0 |",
        "| 6 | flint | 61.8 |",
    ]


def test_rank_by_mean_rank_moves_the_mean_winner_to_fifth(capsys):
    # ranks of atlas on the five benchmarks: 1, 5, 5, 6, 5; of birch: 2, 2, 1, 2, 2
    code, out, _ = run(capsys, "rank", TOY, "--rule", "mean-rank")
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("Rule: mean-rank (lower is better).")
    assert lines[4] == "| 1 | birch | 1.8 |"
    assert lines[8] == "| 5 | atlas | 4.4 |"


def test_rank_json(capsys):
    code, out, _ = run(capsys, "rank", TOY, "--rule", "borda", "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert payload["command"] == "rank"
    assert payload["rule"] == "borda"
    assert payload["higher_is_better"] is True
    assert (payload["n_models"], payload["n_benchmarks"], payload["n_missing"]) == (6, 5, 0)
    # Borda points are 5 * 6 minus the sum of the five ranks
    assert payload["ranking"][0] == {"rank": 1, "model": "birch", "value": 21.0}
    assert payload["ranking"][-1] == {"rank": 6, "model": "flint", "value": 2.0}


def test_compare_rules_markdown(capsys):
    code, out, _ = run(capsys, "compare-rules", TOY, "--rules", "mean,mean-rank,borda", "--top", 2)
    assert code == 0
    assert out.splitlines() == [
        "6 models, 5 benchmarks, 0 missing cells.",
        "",
        "| rule | top 2 (rank) |",
        "|---|---|",
        "| mean | atlas (1), birch (2) |",
        "| mean-rank | birch (1), cedar (2) |",
        "| borda | birch (1), cedar (2) |",
        "",
        "Kendall tau-b between the rankings:",
        "",
        "| rule | mean | mean-rank | borda |",
        "|---|---:|---:|---:|",
        "| mean | 1.000 | 0.467 | 0.467 |",
        "| mean-rank | 0.467 | 1.000 | 1.000 |",
        "| borda | 0.467 | 1.000 | 1.000 |",
    ]


def test_compare_rules_json_covers_every_rule(capsys):
    code, out, _ = run(capsys, "compare-rules", TOY, "--format", "json")
    assert code == 0
    payload = json.loads(out)
    rules = payload["rules"]
    assert rules == [
        "mean",
        "minmax",
        "zscore",
        "geomean",
        "mean-rank",
        "win-rate",
        "borda",
        "copeland",
    ]
    assert payload["skipped"] == {}
    tau = payload["kendall_tau"]
    for a in rules:
        assert tau[a][a] == 1.0
        for b in rules:
            assert tau[a][b] == tau[b][a]
    # Moving atlas from first to fifth swaps four of the fifteen pairs: (11 - 4) / 15.
    assert tau["mean"]["mean-rank"] == pytest.approx(7 / 15)
    assert [row["model"] for row in payload["rankings"]["mean"]][:2] == ["atlas", "birch"]


def test_compare_rules_skips_a_rule_that_cannot_use_the_scores(tmp_path, capsys):
    path = write(tmp_path, "signed.csv", "model,x,y\na,1,-2\nb,2,3\nc,3,1\n")
    code, out, _ = run(capsys, "compare-rules", path, "--rules", "mean,geomean")
    assert code == 0
    assert "| geomean |" not in out
    assert out.splitlines()[-1].startswith("Skipped geomean: the geometric mean needs")
    code, out, _ = run(capsys, "compare-rules", path, "--rules", "mean,geomean", "--format", "json")
    payload = json.loads(out)
    assert payload["rules"] == ["mean"]
    assert "non-negative" in payload["skipped"]["geomean"]
    code, _, err = run(capsys, "compare-rules", path, "--rules", "geomean")
    assert code == 2
    assert "no rule could rank this table" in err
    code, _, err = run(capsys, "rank", path, "--rule", "geomean")
    assert code == 2
    assert err.startswith("rankstable: error: the geometric mean needs non-negative scores")


def test_a_constant_ranking_has_no_tau(tmp_path, capsys):
    path = write(tmp_path, "flat.csv", "model,x\na,1\nb,1\n")
    code, out, _ = run(capsys, "compare-rules", path, "--rules", "mean,borda")
    assert code == 0
    assert "| mean | n/a | n/a |" in out
    code, out, _ = run(capsys, "compare-rules", path, "--rules", "mean,borda", "--format", "json")
    assert json.loads(out)["kendall_tau"]["mean"]["borda"] is None


HOLES = "model,x,y\na,3,1\nb,2,\nc,1,2\n"


def test_missing_cells_are_kept_by_default(tmp_path, capsys):
    path = write(tmp_path, "holes.csv", HOLES)
    code, out, _ = run(capsys, "rank", path)
    assert code == 0
    assert (
        out.splitlines()[0]
        == "Rule: mean (higher is better). 3 models, 2 benchmarks, 1 missing cell."
    )
    assert out.splitlines()[4:] == ["| 1 | a | 2.0 |", "| 1 | b | 2.0 |", "| 3 | c | 1.5 |"]


def test_missing_policies(tmp_path, capsys):
    path = write(tmp_path, "holes.csv", HOLES)
    code, out, _ = run(capsys, "rank", path, "--missing", "drop-models")
    assert "2 models, 2 benchmarks, 0 missing cells" in out
    assert "| b |" not in out
    code, out, _ = run(capsys, "rank", path, "--missing", "drop-benchmarks")
    assert "3 models, 1 benchmark, 0 missing cells" in out
    assert out.splitlines()[4:] == ["| 1 | a | 3 |", "| 2 | b | 2 |", "| 3 | c | 1 |"]
    code, _, err = run(capsys, "rank", path, "--missing", "error")
    assert code == 2
    assert err == "rankstable: error: the table has 1 missing cell(s)\n"


def test_models_without_a_value_are_shown_last(tmp_path, capsys):
    path = write(tmp_path, "empty_row.csv", "model,x\na,1\nb,\nc,2\n")
    code, out, _ = run(capsys, "rank", path)
    assert out.splitlines()[4:] == ["| 1 | c | 2 |", "| 2 | a | 1 |", "| 3 | b | n/a |"]
    code, out, _ = run(capsys, "rank", path, "--format", "json")
    assert json.loads(out)["ranking"][-1] == {"rank": 3, "model": "b", "value": None}


def test_lower_is_better_flag(tmp_path, capsys):
    path = write(tmp_path, "errors.csv", "model,error\na,0.1\nb,0.3\n")
    _, out, _ = run(capsys, "rank", path, "--rule", "mean-rank")
    assert out.splitlines()[4] == "| 1 | b | 1 |"
    _, out, _ = run(capsys, "rank", path, "--rule", "mean-rank", "--lower-is-better", "error")
    assert out.splitlines()[4] == "| 1 | a | 1 |"
    code, _, err = run(capsys, "rank", path, "--lower-is-better", "latency")
    assert code == 2
    assert "unknown benchmark(s): latency" in err


def test_long_jsonl_input(tmp_path, capsys):
    records = [
        {"model": "a", "benchmark": "error", "score": 0.1, "higher_is_better": False},
        {"model": "b", "benchmark": "error", "score": 0.3},
    ]
    path = write(tmp_path, "scores.jsonl", "\n".join(json.dumps(r) for r in records))
    _, out, _ = run(capsys, "rank", path, "--rule", "win-rate")
    assert out.splitlines()[4:] == ["| 1 | a | 1 |", "| 2 | b | 0 |"]
    renamed = write(tmp_path, "scores.txt", path.read_text(encoding="utf-8"))
    code, _, err = run(capsys, "rank", renamed)
    assert code == 2
    _, out, _ = run(capsys, "rank", renamed, "--input-format", "jsonl", "--rule", "win-rate")
    assert out.splitlines()[4] == "| 1 | a | 1 |"


def test_errors_exit_with_code_two(tmp_path, capsys):
    code, out, err = run(capsys, "rank", tmp_path / "absent.csv")
    assert (code, out) == (2, "")
    assert err.startswith("rankstable: error: cannot read ")
    assert err.rstrip().endswith("No such file or directory")
    code, _, err = run(capsys, "compare-rules", TOY, "--rules", "mean,median")
    assert code == 2
    assert "unknown rule(s): median" in err
    code, _, err = run(capsys, "compare-rules", TOY, "--rules", ",")
    assert (code, err) == (2, "rankstable: error: no rule given\n")
    code, _, err = run(capsys, "compare-rules", TOY, "--top", 0)
    assert (code, err) == (2, "rankstable: error: --top must be at least 1\n")
    with pytest.raises(SystemExit) as exit_info:
        main(["rank", str(TOY), "--rule", "median"])
    assert exit_info.value.code == 2
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"rankstable {__version__}\n"


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "rankstable", "rank", str(TOY), "--rule", "copeland"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[4] == "| 1 | birch | 5 |"


def test_format_column_shares_the_decimals_of_a_column():
    assert format_column([70.1, 62.0, None, float("nan")]) == ["70.1", "62.0", "n/a", "n/a"]
    assert format_column([3.0, 12.0]) == ["3", "12"]
    assert format_column([1 / 3, 0.5]) == ["0.3333", "0.5000"]
    assert format_column([-0.00001, 2.5]) == ["0.0", "2.5"]
    assert format_column([1.25, 2.0], digits=0) == ["1", "2"]
    assert format_column([]) == []


def test_markdown_table_escapes_nothing_itself_but_cells_are_escaped(tmp_path, capsys):
    assert markdown_table(["a", "b"], [["1", "2"]], "lr") == "| a | b |\n|---|---:|\n| 1 | 2 |"
    path = write(tmp_path, "pipes.csv", "model,x\nleft|right,1\n")
    _, out, _ = run(capsys, "rank", path)
    assert out.splitlines()[4] == "| 1 | left\\|right | 1 |"
