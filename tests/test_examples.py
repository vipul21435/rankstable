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
    assert lines[-1] == "models that share all their scores with another model: 0"


def test_table_checks_counts_duplicate_rows(tmp_path):
    path = tmp_path / "scores.csv"
    rows = ["a,1,5", "b,4,2", "c,1,5", "d,4,2", "e,1,5", "f,0,-0.0", "g,-0,0", "h,7,7"]
    path.write_text("model,x,y\n" + "\n".join(rows) + "\n", encoding="utf-8")
    lines = run_example("table_checks.py", path)
    # a, c, e share their scores, so do b, d, and so do f, g, whose zeros differ only in sign
    assert lines[-1] == (
        "models that share all their scores with another model: 7, in 3 groups of 2 to 3"
    )


def test_table_checks_without_zero_scores(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("model,x,y\na,1,5\nb,4,2\n", encoding="utf-8")
    lines = run_example("table_checks.py", path)
    assert lines[1] == "models with a zero score on at least one benchmark: 0"
    assert not any(line.startswith("Kendall") for line in lines)


def test_kemeny_gap_script_reports_the_study_and_a_table():
    lines = run_example(
        "kemeny_gap.py", "--profiles", 6, "--seed", 1, "--table", EXAMPLES / "toy_scores.csv"
    )
    assert lines[0] == "6 random tables of 10 to 12 models and 3 to 12 benchmarks, seed 1"
    assert lines[1].startswith("local search reached the best order in ")
    assert lines[2].startswith("shortfall of the objective: mean ")
    # 15 pairs of models on 5 benchmarks without ties: 75 decided comparisons
    assert lines[3].endswith("toy_scores.csv: 6 models, 75 decided pairwise comparisons")
    kept = [int(line.split(": ")[1].split()[0]) for line in lines[4:7]]
    assert [line.split()[3] for line in lines[4:7]] == ["kemeny", "copeland", "mean-rank"]
    # six models are solved exactly, so no other order keeps more
    assert kept[0] == max(kept) <= 75
