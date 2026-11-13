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
        "",
        "Share of the top 2 in common:",
        "",
        "| rule | mean | mean-rank | borda |",
        "|---|---:|---:|---:|",
        "| mean | 1.000 | 0.500 | 0.500 |",
        "| mean-rank | 0.500 | 1.000 | 1.000 |",
        "| borda | 0.500 | 1.000 | 1.000 |",
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
        "kemeny",
        "bradley-terry",
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
    # top three by mean: atlas, birch, cedar; by mean rank: birch, cedar, dune
    assert payload["top"] == 3
    assert payload["top_k_overlap"]["mean"]["mean-rank"] == pytest.approx(2 / 3)
    assert payload["top_k_overlap"]["borda"]["borda"] == 1


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


def test_small_shares_keep_two_significant_digits():
    from rankstable.cli import _percent

    assert _percent(0.25) == "25.0 %"
    assert _percent(0) == "0.0 %"
    assert _percent(0.000123) == "0.012 %"
    assert _percent(float("nan")) == "n/a"


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


# Means: A 17/3, B 5, C 3; the leave-one-out outcomes are worked out in test_stability.py.
SMALL = "model,b1,b2,b3\nA,9,2,6\nB,5,6,4\nC,1,5,3\n"


def test_stability_markdown_by_hand(tmp_path, capsys):
    path = write(tmp_path, "small.csv", SMALL)
    code, out, err = run(
        capsys, "stability", path, "--rules", "mean", "--top", 2, "--resamples", 0, "--removals", 0
    )
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == "3 models, 3 benchmarks, 0 missing cells."
    # b1 against b2: tau -1/3, b1 against b3: 1, b2 against b3: -1/3 -> distances 2/3, 0, 2/3
    assert lines[2].endswith("0 = same order, 1 = reversed): 0.444")
    assert lines[4] == "Leave 1 of 3 benchmarks out (all 3 subsets):"
    assert (
        lines[6]
        == "| rule | first on the full table | first changes | top 2 changes | mean tau | min tau |"
    )
    assert lines[8] == "| mean | A | 66.7 % | 33.3 % | 0.605 | 0.000 |"
    assert len(lines) == 9


def test_stability_reports_bootstrap_and_removals(tmp_path, capsys):
    # Three benchmarks say A > B > C and two say B > C > A: removing C flips the Borda order.
    path = write(tmp_path, "split.csv", "model,v,w,x,y,z\nA,3,3,3,1,1\nB,2,2,2,3,3\nC,1,1,1,2,2\n")
    code, out, _ = run(
        capsys, "stability", path, "--rules", "mean,borda", "--leave-out", "1,2", "--resamples", 20
    )
    assert code == 0
    lines = out.splitlines()
    assert "Leave 1 of 5 benchmarks out (all 5 subsets):" in lines
    assert "Leave 2 of 5 benchmarks out (all 10 subsets):" in lines
    assert "Bootstrap over benchmarks (20 resamples, seed 0):" in lines
    assert "Remove one model at a time (every model, 3 removals):" in lines
    assert lines[-2] == "| mean | 0.0 % | 0.0 % | 0.0 % | none |"
    assert lines[-1] == "| borda | 33.3 % | 33.3 % | 33.3 % | C (100.0 % of pairs) |"


def test_stability_json(tmp_path, capsys):
    path = write(tmp_path, "small.csv", SMALL)
    code, out, _ = run(
        capsys,
        "stability",
        path,
        "--rules",
        "mean,geomean",
        "--top",
        2,
        "--resamples",
        30,
        "--format",
        "json",
        "--seed",
        5,
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["command"] == "stability"
    assert (payload["top"], payload["seed"], payload["rules"]) == (2, 5, ["mean", "geomean"])
    assert payload["diversity"] == pytest.approx(4 / 9)
    first = payload["leave_out"][0]
    assert (first["k"], first["rule"], first["n_subsets"], first["exhaustive"]) == (
        1,
        "mean",
        3,
        True,
    )
    assert first["first"] == ["A"]
    assert first["first_changed"] == pytest.approx(2 / 3)
    assert first["top_changed"] == pytest.approx(1 / 3)
    assert [s["dropped"] for s in first["subsets"]] == [["b1"], ["b2"], ["b3"]]
    assert [s["first"] for s in first["subsets"]] == [["B"], ["A"], ["A", "B"]]
    boot = payload["bootstrap"][0]
    assert (boot["rule"], boot["n_resamples"], boot["confidence"]) == ("mean", 30, 0.95)
    assert [m["model"] for m in boot["models"]] == ["A", "B", "C"]
    assert all(0 <= m["p_first"] <= m["p_top"] <= 1 for m in boot["models"])
    removal = payload["remove_models"][0]
    assert (removal["rule"], removal["n_removals"], removal["order_changed"]) == ("mean", 3, 0)
    assert removal["largest_effect"] == {"model": "A", "pairs_changed": 0}


def test_stability_skips_rules_and_checks_options(tmp_path, capsys):
    path = write(tmp_path, "signed.csv", "model,x,y,z\na,1,-2,3\nb,2,3,1\nc,3,1,2\n")
    code, out, _ = run(capsys, "stability", path, "--rules", "mean,geomean", "--resamples", 0)
    assert code == 0
    assert "| geomean |" not in out
    assert out.splitlines()[-1].startswith("Skipped geomean: the geometric mean needs")
    for options, message in [
        (["--leave-out", "3"], "cannot leave 3 of 3 benchmark(s) out"),
        (["--leave-out", "one"], "--leave-out takes whole numbers"),
        (["--leave-out", ","], "--leave-out needs at least one number"),
        (["--top", "0"], "--top must be at least 1"),
        (["--resamples", "-1"], "cannot be negative"),
        (["--rules", "median"], "unknown rule(s): median"),
    ]:
        code, _, err = run(capsys, "stability", path, *options)
        assert code == 2, options
        assert message in err, options


def test_stability_skips_the_removal_check_for_two_models(tmp_path, capsys):
    path = write(tmp_path, "pair.csv", "model,x,y\na,1,2\nb,2,1\n")
    code, out, _ = run(capsys, "stability", path, "--rules", "mean", "--resamples", 5)
    assert code == 0
    assert "Remove one model" not in out


AGREE = "model,b1,b2,b3,b4\nm1,4,8,40,1\nm2,3,6,30,2\nm3,2,4,20,3\nm4,1,2,10,4\n"


def test_agreement_markdown_lists_the_least_agreeing_benchmark_first(tmp_path, capsys):
    path = write(tmp_path, "agree.csv", AGREE)
    code, out, err = run(capsys, "agreement", path, "--resamples", 40)
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[2] == (
        "Kendall tau-b of each benchmark with the win-rate ranking of the other benchmarks, "
        "lowest first (95 % bootstrap interval over models, 40 resamples, seed 0):"
    )
    assert lines[4:] == [
        "| benchmark | tau | 95 % interval | models |",
        "|---|---:|---:|---:|",
        "| b4 | -1.000 | -1.000 to -1.000 | 4 |",
        "| b1 | 1.000 | 1.000 to 1.000 | 4 |",
        "| b2 | 1.000 | 1.000 to 1.000 | 4 |",
        "| b3 | 1.000 | 1.000 to 1.000 | 4 |",
    ]


def test_agreement_json_and_errors(tmp_path, capsys):
    path = write(tmp_path, "agree.csv", AGREE)
    code, out, _ = run(
        capsys,
        "agreement",
        path,
        "--rule",
        "zscore",
        "--resamples",
        10,
        "--confidence",
        0.9,
        "--format",
        "json",
    )
    assert code == 0
    payload = json.loads(out)
    assert (payload["rule"], payload["n_resamples"], payload["confidence"]) == ("zscore", 10, 0.9)
    assert payload["benchmarks"][0] == {
        "benchmark": "b4",
        "n_models": 4,
        "tau": -1.0,
        "low": -1.0,
        "high": -1.0,
    }
    single = write(tmp_path, "single.csv", "model,x\na,1\nb,2\n")
    code, _, err = run(capsys, "agreement", single)
    assert (code, err) == (2, "rankstable: error: agreement needs at least two benchmarks\n")
    flat = write(tmp_path, "flat.csv", "model,x,y\na,1,5\nb,2,5\nc,3,5\n")
    code, out, _ = run(capsys, "agreement", flat, "--resamples", 5)
    assert "| x | n/a | n/a to n/a | 3 |" in out
    code, out, _ = run(capsys, "agreement", flat, "--resamples", 5, "--format", "json")
    assert json.loads(out)["benchmarks"][-1]["tau"] is None


# Mean ranks 1.25, 1.75, 3 over four benchmarks; the numbers are worked out in test_friedman.py.
RANKED = "model,b1,b2,b3,b4\nx,9,9,9,5\ny,5,5,5,9\nz,1,1,1,1\n"


def test_cd_markdown_by_hand(tmp_path, capsys):
    path = write(tmp_path, "ranked.csv", RANKED)
    code, out, err = run(capsys, "cd", path)
    assert (code, err) == (0, "")
    assert out.splitlines() == [
        "3 models, 4 benchmarks, 0 missing cells.",
        "",
        "Friedman test: chi-square = 6.500 on 2 degrees of freedom, p = 0.03877. "
        "Iman-Davenport F = 13.000 on 2 and 6 degrees of freedom, p = 0.006592.",
        "Nemenyi critical difference at alpha = 0.05: 1.657 mean ranks.",
        "",
        "| model | mean rank |",
        "|---|---:|",
        "| x | 1.250 |",
        "| y | 1.750 |",
        "| z | 3.000 |",
        "",
        "Groups of models whose mean ranks differ by less than the critical difference:",
        "",
        "- x, y",
        "- y, z",
    ]


def test_cd_json_top_and_plot(tmp_path, capsys):
    path = write(tmp_path, "ranked.csv", RANKED)
    picture = tmp_path / "cd.png"
    code, out, _ = run(
        capsys,
        "cd",
        path,
        "--top",
        2,
        "--alpha",
        0.1,
        "--format",
        "json",
        "--plot",
        picture,
        "--title",
        "two models",
    )
    assert code == 0
    payload = json.loads(out)
    assert (payload["n_models"], payload["alpha"], payload["chi2_df"]) == (2, 0.1, 1)
    assert payload["f_df"] == [1, 3]
    assert [row["model"] for row in payload["mean_ranks"]] == ["x", "y"]
    # x is ahead on three benchmarks of four: mean ranks 1.25 and 1.75, chi2 = 4 * 0.25 = 1
    assert payload["chi2"] == pytest.approx(1)
    # q_0.10 for two models is 1.645: CD = 1.645 * sqrt(2 * 3 / 24)
    assert payload["critical_difference"] == pytest.approx(1.645 * 0.5, abs=1e-3)
    assert payload["groups"] == [["x", "y"]]
    assert payload["plot"] == str(picture)
    assert picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    code, out, _ = run(capsys, "cd", path, "--plot", picture)
    assert out.splitlines()[-1] == f"Diagram written to {picture}."


def test_cd_errors_and_perfect_agreement(tmp_path, capsys):
    holes = write(tmp_path, "holes.csv", HOLES)
    code, _, err = run(capsys, "cd", holes)
    assert code == 2
    assert "needs a complete table" in err
    code, out, _ = run(capsys, "cd", holes, "--missing", "drop-benchmarks")
    assert code == 2 or "1 benchmark" in out
    path = write(tmp_path, "ranked.csv", RANKED)
    code, _, err = run(capsys, "cd", path, "--top", 1)
    assert (code, err) == (2, "rankstable: error: --top must be at least 2\n")
    same = write(tmp_path, "same.csv", "model,b1,b2,b3\nx,3,3,3\ny,2,2,2\nz,1,1,1\n")
    code, out, _ = run(capsys, "cd", same)
    assert "Iman-Davenport F = inf on 2 and 4 degrees of freedom, p = 0." in out
    code, out, _ = run(capsys, "cd", same, "--format", "json")
    assert json.loads(out)["f"] is None
