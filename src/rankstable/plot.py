"""Critical-difference diagram. Needs matplotlib: ``pip install "rankstable[plot]"``."""

from __future__ import annotations

from pathlib import Path

from .friedman import FriedmanResult


def plot_critical_difference(
    result: FriedmanResult, path: str | Path, *, title: str | None = None
) -> None:
    """Draw mean ranks on an axis with the best rank to the right and save the figure.

    A thick bar joins every group of models whose mean ranks span less than the critical
    difference, and a ruler of that length is drawn above the axis.
    """
    try:
        import matplotlib
    except ImportError:
        raise ValueError(
            'the diagram needs matplotlib; install it with pip install "rankstable[plot]"'
        ) from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(result.models)
    order = sorted(range(k), key=lambda i: (result.mean_ranks[i], result.models[i]))
    low, high = 1.0, float(k)

    def x_of(rank: float) -> float:
        return (high - rank) / (high - low)

    groups = result.groups()
    half = (k + 1) // 2
    rows = max(half, k - half)
    bar_space = 0.12 * (len(groups) + 1)
    height = 1.6 + 0.28 * rows + bar_space
    figure, axis = plt.subplots(figsize=(9, height))
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(-(bar_space + 0.28 * rows + 0.2), 0.75)
    axis.axis("off")
    axis.plot([0, 1], [0, 0], color="black", linewidth=1)
    step = max(1, round((k - 1) / 12))
    for tick in range(1, k + 1, step):
        axis.plot([x_of(tick)] * 2, [0, 0.06], color="black", linewidth=1)
        axis.text(x_of(tick), 0.1, str(tick), ha="center", va="bottom", fontsize=9)
    width = result.critical_difference / (high - low)
    axis.plot([0, min(width, 1.3)], [0.45, 0.45], color="black", linewidth=2)
    axis.text(0, 0.5, f"CD = {result.critical_difference:.2f}", ha="left", va="bottom", fontsize=9)
    rank_of = {result.models[i]: float(result.mean_ranks[i]) for i in order}
    for level, group in enumerate(groups, start=1):
        ends = [x_of(rank_of[group[0]]), x_of(rank_of[group[-1]])]
        axis.plot(ends, [-0.12 * level] * 2, color="black", linewidth=3, solid_capstyle="butt")
    for place, index in enumerate(order):
        rank = float(result.mean_ranks[index])
        on_right = place < half
        row = place if on_right else k - 1 - place
        y = -(bar_space + 0.28 * row + 0.1)
        edge = 1.02 if on_right else -0.02
        axis.plot([x_of(rank), x_of(rank), edge], [0, y, y], color="black", linewidth=0.8)
        axis.text(
            edge + (0.01 if on_right else -0.01),
            y,
            f"{result.models[index]} ({rank:.2f})",
            ha="left" if on_right else "right",
            va="center",
            fontsize=9,
        )
    if title:
        axis.set_title(title, fontsize=10)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
