import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_the_example_script_runs():
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "compare_rules.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "mean           atlas > birch > cedar > dune > ember > flint"
    assert "Kendall tau-b between mean and mean-rank: 0.467" in lines
    # only dropping the benchmark on which atlas scores 95 takes the first place away
    assert "Leave one benchmark out, rule mean: first place changes in 20%" in lines
    assert any(line.startswith("  without algebra    first: birch") for line in lines)


def test_the_fetch_script_documents_itself():
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "fetch_open_llm_leaderboard.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "output" in result.stdout


def run_example(name, *arguments):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *map(str, arguments)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_table_checks_counts_zero_scores_and_their_effect_on_geomean(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("model,x,y\na,0,5\nb,4,0\nc,3,3\nd,2,4\ne,1,9\n", encoding="utf-8")
    lines = run_example("table_checks.py", path)
    assert lines[:5] == [
        "5 models, 2 benchmarks",
        "models with a zero score on at least one benchmark: 2",
        "  per benchmark: x 1, y 1",
        "  models with a geometric mean of zero: 2, sharing rank 4 of 5",
        "Kendall tau-b between geomean and other rules: "
        "all 5 models / the 3 models without a zero score",
    ]
    # Mean ranks of a..e: 4 5 2 2 1; geomean ranks: 4 4 1 3 1. Seven pairs agree, one is tied
    # in the mean and two in the geometric mean: 7 / sqrt(9 * 8). Among c, d, e alone one
    # pair agrees and each ranking ties one pair: 1 / sqrt(2 * 2).
    assert "  mean       0.825 / 0.500" in lines


def test_table_checks_without_zero_scores(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("model,x,y\na,1,5\nb,4,2\n", encoding="utf-8")
    lines = run_example("table_checks.py", path)
    assert lines[1] == "models with a zero score on at least one benchmark: 0"
    assert not any(line.startswith("Kendall") for line in lines)
