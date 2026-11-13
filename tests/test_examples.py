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
