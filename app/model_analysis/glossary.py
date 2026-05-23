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
        label="Loss (trening)",
        short="Strata treningowa",
        description="Suma składowych funkcji straty na zbiorze treningowym.",
        interpretation="Niższa wartość oznacza lepsze dopasowanie do danych treningowych.",
    ),
    "val_loss": MetricDefinition(
        key="val_loss",
        label="Loss (walidacja)",
        short="Strata walidacyjna",
        description="Funkcja straty liczona na zbiorze walidacyjnym.",
        interpretation="Niższa wartość oznacza lepszą generalizację.",
    ),
    "mAP50": MetricDefinition(
        key="mAP50",
        label="mAP@0.5 (valid)",
        short="mAP50",
        description="Mean Average Precision @ IoU=0.5 na zbiorze walidacyjnym.",
        interpretation="Skala 0–1; wyższa wartość = lepsza detekcja. W detekcji obiektów mAP liczy się tylko na valid — nie ma odpowiednika train/val jak przy loss.",
    ),
    "mAP5095": MetricDefinition(
        key="mAP5095",
        label="mAP@0.5:0.95 (valid)",
        short="mAP50-95",
        description="Średnie AP po progach IoU od 0.5 do 0.95 na zbiorze walidacyjnym.",
        interpretation="Bardziej rygorystyczna metryka niż mAP@0.5; wyższa = lepiej.",
    ),
    "precision": MetricDefinition(
        key="precision",
        label="Precision (valid)",
        short="Precision",
        description="Odsetek trafnych detekcji wśród wszystkich predykcji modelu (valid).",
        interpretation="Wyższa precision = mniej fałszywych alarmów.",
    ),
    "recall": MetricDefinition(
        key="recall",
        label="Recall (valid)",
        short="Recall",
        description="Odsetek wykrytych obiektów względem wszystkich obiektów w annotacjach (valid).",
        interpretation="Wyższy recall = mniej przeoczonych obiektów.",
    ),
    "epochs": MetricDefinition(
        key="epochs",
        label="Epoki",
        short="Epoki",
        description="Liczba epok treningowych z zapisaną historią metryk.",
        interpretation="Kontekst dla krzywych uczenia.",
    ),
}


def glossary_rows() -> list[dict[str, str]]:
    return [
        {
            "Metryka": definition.label,
            "Opis": definition.description,
            "Interpretacja": definition.interpretation,
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
        return "brak"
    if key in _RATIO_METRICS and isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if key in _LOSS_METRICS and isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)
