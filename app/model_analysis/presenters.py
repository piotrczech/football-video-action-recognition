from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.model_analysis.glossary import format_metric_value, metric_help


@dataclass(frozen=True)
class KpiItem:
    label: str
    value: str
    help_text: str | None


KPI_METRIC_KEYS = (
    "mAP50",
    "mAP5095",
    "precision",
    "recall",
    "loss",
    "val_loss",
    "epochs",
)


def build_kpi_items(metrics: dict[str, Any]) -> list[KpiItem]:
    items: list[KpiItem] = []
    for key in KPI_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        items.append(
            KpiItem(
                label=key,
                value=format_metric_value(key, value),
                help_text=metric_help(key),
            )
        )
    return items


def format_run_context(
    metadata: dict[str, Any],
    *,
    fallback_model: str | None = None,
    fallback_variant: str | None = None,
    fallback_created_at: str | None = None,
) -> str:
    parts = [
        _cell(metadata.get("model", fallback_model)),
        f"variant: {_cell(metadata.get('dataset_variant', fallback_variant))}",
        f"profile: {_cell(metadata.get('profile'))}",
        f"train/valid: {_cell(metadata.get('train_samples'))}/{_cell(metadata.get('valid_samples'))}",
        _cell(metadata.get("created_at_utc", fallback_created_at)),
    ]
    return " · ".join(part for part in parts if part and part != "missing")


def _cell(value: Any, *, empty_label: str = "missing") -> str:
    if value is None:
        return empty_label
    if isinstance(value, str) and not value.strip():
        return empty_label
    return str(value)
