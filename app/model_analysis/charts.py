from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from app.model_analysis.chart_style import (
    COLORS,
    apply_chart_header,
    apply_modern_axes,
    finalize_figure,
    new_figure,
)


def build_loss_chart(
    loss_history: list[float],
    *,
    val_loss_history: list[float] | None = None,
    val_loss_scalar: float | None = None,
) -> Figure | None:
    if not loss_history:
        return None

    epochs = list(range(1, len(loss_history) + 1))
    fig, ax = new_figure()
    apply_chart_header(
        ax,
        "Loss treningowy vs walidacyjny",
        "Niższa wartość = lepiej",
    )

    ax.plot(epochs, loss_history, label="Train", color=COLORS["train"], linewidth=2.2)

    if val_loss_history:
        val_epochs = list(range(1, len(val_loss_history) + 1))
        ax.plot(val_epochs, val_loss_history, label="Valid", color=COLORS["val"], linewidth=2.2)
    elif val_loss_scalar is not None:
        ax.axhline(
            val_loss_scalar,
            color=COLORS["val"],
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            label=f"Valid (końcowa: {val_loss_scalar:.3f})",
        )
        ax.text(
            0.02,
            0.04,
            "Brak historii val loss — linia przerywana to wartość końcowa.",
            transform=ax.transAxes,
            fontsize=7.5,
            color=COLORS["text_muted"],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": COLORS["annotation_bg"], "edgecolor": "none"},
        )

    ax.set_xlabel("Epoka")
    ax.set_ylabel("Loss")
    apply_modern_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=COLORS["text_muted"])
    finalize_figure(fig)
    return fig


def build_val_map_chart(
    history: list[float],
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    color: str,
    label: str,
) -> Figure | None:
    if not history:
        return None

    epochs = list(range(1, len(history) + 1))
    fig, ax = new_figure()
    apply_chart_header(ax, title, subtitle)
    ax.fill_between(epochs, history, alpha=0.12, color=color)
    ax.plot(epochs, history, label=label, color=color, linewidth=2.2)
    ax.set_xlabel("Epoka")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, min(1.0, max(history) * 1.15 + 0.05))
    apply_modern_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=COLORS["text_muted"])
    finalize_figure(fig)
    return fig


def build_val_map50_chart(map50_history: list[float]) -> Figure | None:
    return build_val_map_chart(
        map50_history,
        title="mAP@0.5 (valid)",
        subtitle="Metryka detekcji na zbiorze walidacyjnym · wyżej = lepiej",
        ylabel="mAP@0.5",
        color=COLORS["map50"],
        label="Valid mAP@0.5",
    )


def build_val_map5095_chart(map5095_history: list[float]) -> Figure | None:
    return build_val_map_chart(
        map5095_history,
        title="mAP@0.5:0.95 (valid)",
        subtitle="Średnie AP po progach IoU 0.5–0.95 · wyżej = lepiej",
        ylabel="mAP@0.5:0.95",
        color=COLORS["map5095"],
        label="Valid mAP@0.5:0.95",
    )


def build_precision_recall_chart(
    precision_history: list[float],
    recall_history: list[float],
) -> Figure | None:
    if not precision_history and not recall_history:
        return None

    length = max(len(precision_history), len(recall_history))
    epochs = list(range(1, length + 1))
    fig, ax = new_figure()
    apply_chart_header(
        ax,
        "Precision i recall (valid)",
        "Metryki detekcji na zbiorze walidacyjnym · wyżej = lepiej",
    )

    if precision_history:
        ax.plot(
            list(range(1, len(precision_history) + 1)),
            precision_history,
            label="Precision",
            color=COLORS["precision"],
            linewidth=2.2,
        )
    if recall_history:
        ax.plot(
            list(range(1, len(recall_history) + 1)),
            recall_history,
            label="Recall",
            color=COLORS["recall"],
            linewidth=2.2,
        )

    ax.set_xlabel("Epoka")
    ax.set_ylabel("Wartość")
    ax.set_xlim(0.5, length + 0.5)
    ax.set_ylim(0.0, 1.05)
    apply_modern_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=COLORS["text_muted"])
    finalize_figure(fig)
    return fig


def as_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None
