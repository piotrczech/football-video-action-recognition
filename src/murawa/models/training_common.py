from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from murawa.data import DataLoaderError, LoadedSplit, load_training_split
from murawa.models.common import require_mapping


@dataclass(frozen=True)
class TrainValidSplits:
    train_split: LoadedSplit
    valid_split: LoadedSplit
    valid_split_source: str


def load_training_config_payload(config_path: Path | None, *, backend_name: str) -> tuple[dict[str, Any], Path]:
    if config_path is None:
        raise ValueError(f"{backend_name} training requires config_path for explicit training settings.")

    cfg_path = config_path.resolve()
    if not cfg_path.exists() or not cfg_path.is_file():
        raise FileNotFoundError(f"Training config file does not exist: {cfg_path}")

    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse training config '{cfg_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Training config '{cfg_path}' must contain a mapping at top-level.")
    return payload, cfg_path


def read_training_sections(payload: dict[str, Any], cfg_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    training_cfg = require_mapping(payload.get("training"), key="training", config_path=cfg_path)
    runtime_cfg = require_mapping(payload.get("runtime"), key="runtime", config_path=cfg_path)
    return training_cfg, runtime_cfg


def load_train_valid_splits(
    *,
    project_root: Path,
    dataset_variant: str,
    max_train_samples: int | None,
    max_valid_samples: int | None,
    sampling_seed: int,
    backend_name: str,
) -> TrainValidSplits:
    try:
        train_split = load_training_split(
            project_root=project_root,
            dataset_variant=dataset_variant,
            split="train",
            max_samples=max_train_samples,
            sampling_seed=sampling_seed,
        )
    except DataLoaderError as exc:
        raise RuntimeError(f"{backend_name} training data loading failed for split='train': {exc}") from exc

    try:
        valid_split = load_training_split(
            project_root=project_root,
            dataset_variant=dataset_variant,
            split="valid",
            max_samples=max_valid_samples,
            sampling_seed=sampling_seed,
        )
        valid_split_source = "valid"
    except DataLoaderError:
        try:
            valid_split = load_training_split(
                project_root=project_root,
                dataset_variant=dataset_variant,
                split="train",
                max_samples=max_valid_samples,
                sampling_seed=sampling_seed,
            )
        except DataLoaderError as exc:
            raise RuntimeError(
                f"{backend_name} validation split fallback failed. Neither 'valid' nor fallback 'train' "
                f"could be loaded: {exc}"
            ) from exc
        valid_split_source = "train"

    return TrainValidSplits(
        train_split=train_split,
        valid_split=valid_split,
        valid_split_source=valid_split_source,
    )
