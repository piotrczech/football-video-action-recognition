from pathlib import Path

from murawa.settings import (
    DEFAULT_DATASET_VARIANT,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    resolve_variant_dir,
)


def training_path(project_root: Path, dataset_variant: str) -> Path:
    return resolve_variant_dir(project_root, dataset_variant)


def pick_input(
    project_root: Path,
    mode: str,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> tuple[str, bool]:
    root = training_path(project_root, dataset_variant) / "test"
    if not root.exists():
        return str(root), False

    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")),
        key=lambda path: str(path).lower(),
    )
    if not files:
        return str(root), False

    allowed_suffixes = _suffixes_for_mode(mode)
    if not allowed_suffixes:
        return str(root), False

    candidates = [path for path in files if path.suffix.lower() in allowed_suffixes]
    if not candidates:
        return str(root), False

    return str(candidates[0]), True


def _suffixes_for_mode(mode: str) -> set[str]:
    if mode == "frame":
        return IMAGE_SUFFIXES

    if mode == "match":
        return VIDEO_SUFFIXES

    return set()
