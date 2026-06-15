from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    short: str
    description: str
    interpretation: str

    @property
    def help_text(self) -> str:
        return f"{self.description} {self.interpretation}"


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "loss": MetricDefinition(
        key="loss",
        label="Loss (training)",
        short="Training loss",
        description="Sum of loss components on the training set.",
        interpretation="Lower value means better fit to the training data.",
    ),
    "val_loss": MetricDefinition(
        key="val_loss",
        label="Loss (validation)",
        short="Validation loss",
        description="Loss computed on the validation set.",
        interpretation="Lower value means better generalization.",
    ),
    "mAP50": MetricDefinition(
        key="mAP50",
        label="mAP@0.5 (valid)",
        short="mAP50",
        description="Mean Average Precision @ IoU=0.5 on the validation set.",
        interpretation="Scale 0–1; higher value means better detection. In object detection, mAP is computed on validation data only — there is no train/val equivalent like for loss.",
    ),
    "mAP5095": MetricDefinition(
        key="mAP5095",
        label="mAP@0.5:0.95 (valid)",
        short="mAP50-95",
        description="Average AP over IoU thresholds from 0.5 to 0.95 on the validation set.",
        interpretation="More strict metric than mAP@0.5; higher is better.",
    ),
    "precision": MetricDefinition(
        key="precision",
        label="Precision (valid)",
        short="Precision",
        description="Share of correct detections among all model predictions on the validation set.",
        interpretation="Higher precision means fewer false positives.",
    ),
    "recall": MetricDefinition(
        key="recall",
        label="Recall (valid)",
        short="Recall",
        description="Share of detected objects among all annotated objects on the validation set.",
        interpretation="Higher recall means fewer missed objects.",
    ),
    "epochs": MetricDefinition(
        key="epochs",
        label="Epochs",
        short="Epochs",
        description="Number of training epochs with recorded metric history.",
        interpretation="Context for the learning curves.",
    ),
}


def glossary_rows() -> list[dict[str, str]]:
    return [
        {
            "Metric": definition.label,
            "Description": definition.description,
            "Interpretation": definition.interpretation,
        }
        for definition in METRIC_DEFINITIONS.values()
    ]


def metric_help(key: str) -> str | None:
    definition = METRIC_DEFINITIONS.get(key)
    return definition.help_text if definition is not None else None


_RATIO_METRICS = {"mAP50", "mAP5095", "precision", "recall"}
_LOSS_METRICS = {"loss", "val_loss"}


def format_metric_value(key: str, value: Any) -> str:
    if value is None:
        return "missing"
    if key in _RATIO_METRICS and isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if key in _LOSS_METRICS and isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)
