from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from murawa.data import LoadedSplit
from murawa.settings import (
    DEFAULT_DETECTION_CONFIDENCE,
    IMAGE_SUFFIXES,
    MODELS_METADATA,
    infer_project_root_from_output_dir,
)

__all__ = [
    "IMAGE_SUFFIXES",
    "as_bool",
    "as_float",
    "as_int",
    "as_optional_int",
    "infer_project_root_from_output_dir",
    "load_checkpoint_config",
    "require_mapping",
    "resolve_detection_confidence",
    "sampling_summary_to_dict",
    "seed_everything",
    "validate_image_frame_path",
]


def load_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    run_name = checkpoint_path.parent.name
    project_root = infer_project_root_from_output_dir(checkpoint_path.parent)
    config_path = project_root / MODELS_METADATA / run_name / "config.yaml"
    if not config_path.is_file():
        return {}

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse prediction config '{config_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Saved training config '{config_path}' must contain a mapping at top-level.")
    return payload


def resolve_detection_confidence(checkpoint_path: Path, *, section: str) -> float:
    payload = load_checkpoint_config(checkpoint_path)
    section_cfg = payload.get(section)
    if section_cfg is None:
        return DEFAULT_DETECTION_CONFIDENCE
    if not isinstance(section_cfg, dict):
        raise RuntimeError(
            f"Saved training config for run '{checkpoint_path.parent.name}' has invalid '{section}' section."
        )

    confidence = as_float(
        section_cfg.get("detection_confidence", DEFAULT_DETECTION_CONFIDENCE),
        key="detection_confidence",
        minimum=0.0,
    )
    if confidence > 1.0:
        raise ValueError(
            f"Config value 'detection_confidence' must be <= 1.0, got: {confidence} "
            f"(run={checkpoint_path.parent.name})."
        )
    return confidence


def validate_image_frame_path(frame_path: Path, *, backend_name: str) -> Path:
    resolved_frame = frame_path.resolve()
    if resolved_frame.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(
            f"{backend_name} sampled frame requires an image file. Received: {resolved_frame}"
        )
    if not resolved_frame.is_file():
        raise FileNotFoundError(f"{backend_name} sampled frame does not exist: {resolved_frame}")
    return resolved_frame


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def require_mapping(value: Any, *, key: str, config_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Training config '{config_path}' must include a mapping section '{key}'.")
    return value


def as_int(value: Any, *, key: str, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value '{key}' must be an integer, got: {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"Config value '{key}' must be >= {minimum}, got: {parsed}")
    return parsed


def as_optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    return as_int(value, key=key, minimum=1)


def as_float(value: Any, *, key: str, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value '{key}' must be numeric, got: {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"Config value '{key}' must be >= {minimum}, got: {parsed}")
    return parsed


def as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Config value '{key}' must be boolean, got: {value!r}")


def sampling_summary_to_dict(split: LoadedSplit) -> dict[str, Any]:
    if split.sampling_summary is None:
        return {}
    summary = split.sampling_summary
    return {
        "strategy": summary.strategy,
        "seed": summary.seed,
        "requested_max_samples": summary.requested_max_samples,
        "original_sample_count": summary.original_sample_count,
        "selected_sample_count": summary.selected_sample_count,
        "source_counts": summary.source_counts,
        "density_counts": summary.density_counts,
        "ball_presence_counts": summary.ball_presence_counts,
    }
