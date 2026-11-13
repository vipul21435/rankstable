import sys

import matplotlib
import numpy as np
import pytest

from rankstable.friedman import friedman_test
from rankstable.plot import plot_critical_difference
from rankstable.table import ScoreTable


def result_for(k=5, n=8, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(k, n)) + np.arange(k)[:, None]
    models = tuple(f"model {i}" for i in range(k))
    return friedman_test(ScoreTable(models, tuple(f"b{j}" for j in range(n)), scores))


def test_the_diagram_is_written_as_a_png(tmp_path):
    path = tmp_path / "cd.png"
    plot_critical_difference(result_for(), path, title="five models")
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5000


def test_the_format_follows_the_file_suffix(tmp_path):
    path = tmp_path / "cd.svg"
    plot_critical_difference(result_for(k=3), path)
    assert "<svg" in path.read_text(encoding="utf-8")[:400]


def test_plotting_leaves_the_backend_of_the_caller_alone(tmp_path):
    before = matplotlib.get_backend()
    matplotlib.use("svg")
    try:
        plot_critical_difference(result_for(), tmp_path / "cd.png")
        assert matplotlib.get_backend().lower() == "svg"
    finally:
        matplotlib.use(before)


def test_a_missing_matplotlib_is_reported_as_an_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    with pytest.raises(ValueError, match="needs matplotlib"):
        plot_critical_difference(result_for(), tmp_path / "cd.png")
