from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

CHART_WIDTH_IN = 5.0
CHART_HEIGHT_IN = 3.1
CHART_DPI = 100

COLORS = {
    "train": "#6366F1",
    "val": "#F97316",
    "map50": "#10B981",
    "map5095": "#059669",
    "precision": "#8B5CF6",
    "recall": "#EC4899",
    "grid": "#E5E7EB",
    "text_muted": "#6B7280",
    "text_dark": "#111827",
    "spine": "#D1D5DB",
    "annotation_bg": "#F9FAFB",
}


def new_figure(*, nrows: int = 1, ncols: int = 1) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(nrows, ncols, figsize=(CHART_WIDTH_IN, CHART_HEIGHT_IN), dpi=CHART_DPI)
    fig.patch.set_facecolor("white")
    return fig, ax


def apply_chart_header(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title("")
    ax.text(
        0.0,
        1.12,
        title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="600",
        color=COLORS["text_dark"],
        va="bottom",
        ha="left",
    )
    ax.text(
        0.0,
        1.04,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["text_muted"],
        va="bottom",
        ha="left",
    )


def apply_modern_axes(ax: Axes, *, y_grid: bool = True, x_grid: bool = False) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.tick_params(axis="both", labelsize=8, colors=COLORS["text_muted"], length=0, pad=6)
    ax.xaxis.label.set_color(COLORS["text_muted"])
    ax.yaxis.label.set_color(COLORS["text_muted"])
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    if y_grid:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color=COLORS["grid"], alpha=0.9)
        ax.set_axisbelow(True)
    if x_grid:
        ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color=COLORS["grid"], alpha=0.9)
        ax.set_axisbelow(True)


def finalize_figure(fig: Figure) -> None:
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.18)
