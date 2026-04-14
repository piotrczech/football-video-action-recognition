from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingRunsConfig:
    """Issue #12/#13 orchestration contract for model-vs-variant experiments."""

    dataset_variants: tuple[str, ...]
    config_path: Path
    project_root: Path


def run_yolo_on_variants(config: TrainingRunsConfig) -> dict[str, str]:
    """Issue #12: run YOLO on base/ball-extended/transformed variants."""
    raise NotImplementedError(
        "TODO(Issue #12): orchestrate YOLO training for comparable variants (base, ball-extended, transformed), "
        "save checkpoint + metadata per run, and keep explicit variant-to-run mapping."
    )


def run_rfdetr_on_variants(config: TrainingRunsConfig) -> dict[str, str]:
    """Issue #13: run RF-DETR on base/ball-extended/transformed variants."""
    raise NotImplementedError(
        "TODO(Issue #13): orchestrate RF-DETR training for comparable variants (base, ball-extended, transformed), "
        "save checkpoint + metadata per run, and keep explicit variant-to-run mapping."
    )
