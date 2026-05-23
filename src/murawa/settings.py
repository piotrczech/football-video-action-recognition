from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
PROJECT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.yaml"

DEFAULT_DATASET_VARIANT = "base"


@lru_cache(maxsize=1)
def _load_project_config() -> dict[str, Any]:
    if not PROJECT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Project config not found: {PROJECT_CONFIG_PATH}")
    payload = yaml.safe_load(PROJECT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid project config format: {PROJECT_CONFIG_PATH}")
    return payload


def _path(key: str) -> Path:
    paths = _load_project_config().get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Project config section 'paths' must be a mapping.")
    value = paths.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Project config paths.{key} must be a non-empty string.")
    return Path(value)


DATA_RAW = _path("data_raw")
DATA_READY = _path("data_ready")
MODELS_CHECKPOINTS = _path("models_checkpoints")
MODELS_METADATA = _path("models_metadata")
OUTPUTS_PREDICTIONS = _path("outputs_predictions")
OUTPUTS_VIDEOS = _path("outputs_videos")

_dataset = _load_project_config().get("dataset", {})
if isinstance(_dataset, dict):
    _default_variant = _dataset.get("default_variant")
    if isinstance(_default_variant, str) and _default_variant.strip():
        DEFAULT_DATASET_VARIANT = _default_variant.strip()

    _splits = _dataset.get("splits")
    SPLITS: tuple[str, ...] = (
        tuple(str(item) for item in _splits)
        if isinstance(_splits, list) and _splits
        else ("train", "valid", "test")
    )

    _image_suffixes = _dataset.get("image_suffixes")
    IMAGE_SUFFIXES: set[str] = (
        {str(item).lower() for item in _image_suffixes}
        if isinstance(_image_suffixes, list) and _image_suffixes
        else {".jpg", ".jpeg", ".png", ".bmp"}
    )

    _video_suffixes = _dataset.get("video_suffixes")
    VIDEO_SUFFIXES: set[str] = (
        {str(item).lower() for item in _video_suffixes}
        if isinstance(_video_suffixes, list) and _video_suffixes
        else {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    )
    _bootstrap = _dataset.get("bootstrap")
else:
    SPLITS = ("train", "valid", "test")
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
    VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    _bootstrap = None

if isinstance(_bootstrap, dict):
    DEFAULT_FRAME_STEP = int(_bootstrap.get("frame_step", 30))
    _split_ratios = _bootstrap.get("split_ratios", {})
    BOOTSTRAP_SPLIT_RATIOS: dict[str, float] = (
        {
            "train": float(_split_ratios.get("train", 0.8)),
            "valid": float(_split_ratios.get("valid", 0.1)),
            "test": float(_split_ratios.get("test", 0.1)),
        }
        if isinstance(_split_ratios, dict)
        else {"train": 0.8, "valid": 0.1, "test": 0.1}
    )
    BOOTSTRAP_VALID_FROM_TRAIN_RATIO = float(_bootstrap.get("valid_from_train_ratio", 0.1))
    _default_resolution = _bootstrap.get("default_resolution", {})
    if isinstance(_default_resolution, dict):
        DEFAULT_FRAME_WIDTH = int(_default_resolution.get("width", 1920))
        DEFAULT_FRAME_HEIGHT = int(_default_resolution.get("height", 1080))
    else:
        DEFAULT_FRAME_WIDTH = 1920
        DEFAULT_FRAME_HEIGHT = 1080
    COLOR_CLUSTER_ITERATIONS = int(_bootstrap.get("color_cluster_iterations", 12))
else:
    DEFAULT_FRAME_STEP = 30
    BOOTSTRAP_SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
    BOOTSTRAP_VALID_FROM_TRAIN_RATIO = 0.1
    DEFAULT_FRAME_WIDTH = 1920
    DEFAULT_FRAME_HEIGHT = 1080
    COLOR_CLUSTER_ITERATIONS = 12

SUPPORTED_IMAGE_EXTENSIONS = IMAGE_SUFFIXES

_classes = _load_project_config().get("detection_classes", {})
if isinstance(_classes, dict) and _classes:
    CATEGORY_ID_TO_NAME = {int(key): str(value) for key, value in _classes.items()}
else:
    CATEGORY_ID_TO_NAME = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

CATEGORY_NAME_TO_ID = {name: idx for idx, name in CATEGORY_ID_TO_NAME.items()}
BALL_CATEGORY_ID = CATEGORY_NAME_TO_ID.get("ball", 3)

_inference = _load_project_config().get("inference", {})
if not isinstance(_inference, dict):
    _inference = {}

DEFAULT_SAMPLE_FPS = int(_inference.get("default_sample_fps", 3))
MIN_SAMPLE_FPS = int(_inference.get("min_sample_fps", 1))
MAX_SAMPLE_FPS = int(_inference.get("max_sample_fps", 24))
MAX_VIDEO_DURATION_SECONDS = float(_inference.get("max_video_duration_seconds", 60.0))
DEFAULT_DETECTION_CONFIDENCE = float(_inference.get("detection_confidence", 0.25))


def resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return PROJECT_ROOT.resolve()


def infer_project_root_from_output_dir(output_dir: Path) -> Path:
    """Infer project root from a checkpoint directory (<root>/models/checkpoints/<run>)."""
    output_dir = output_dir.resolve()
    if len(output_dir.parents) >= 3:
        return output_dir.parents[2]
    return PACKAGE_ROOT.parents[1]


def infer_project_root_from_ready_root(ready_root: Path) -> Path:
    """Infer project root from data/ready path (<root>/data/ready)."""
    ready_root = Path(ready_root).resolve()
    return ready_root.parents[1]


def resolve_variant_dir(
    project_root: Path,
    variant: str | None = None,
    *,
    strict: bool = False,
) -> Path:
    """Return data/ready/<variant>; fallback to base unless strict=True."""
    normalized = (variant or DEFAULT_DATASET_VARIANT).strip() or DEFAULT_DATASET_VARIANT
    root = resolve_project_root(project_root)
    specific = (root / DATA_READY / normalized).resolve()
    if specific.exists() and specific.is_dir():
        return specific
    if strict and normalized != DEFAULT_DATASET_VARIANT:
        raise FileNotFoundError(
            f"Dataset variant '{normalized}' does not exist under '{root / DATA_READY}'."
        )
    return (root / DATA_READY / DEFAULT_DATASET_VARIANT).resolve()


def list_dataset_variants(project_root: Path) -> list[str]:
    ready_root = resolve_project_root(project_root) / DATA_READY
    if not ready_root.exists():
        return [DEFAULT_DATASET_VARIANT]

    variants = sorted(
        path.name for path in ready_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if DEFAULT_DATASET_VARIANT not in variants:
        variants.insert(0, DEFAULT_DATASET_VARIANT)
    return variants or [DEFAULT_DATASET_VARIANT]
