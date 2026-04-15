from pathlib import Path

READY_ROOT = Path("data/ready")
PREDICTIONS_ROOT = Path("outputs/predictions")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def training_path(project_root: Path, dataset_variant: str) -> Path:
    # TODO(Issue #9): finalize explicit variant naming/resolution strategy in data/ready.
    specific = project_root / READY_ROOT / dataset_variant
    if specific.exists():
        return specific
    return project_root / READY_ROOT / "base"


def pick_input(project_root: Path, mode: str, dataset_variant: str = "base") -> tuple[str, bool]:
    root = training_path(project_root, dataset_variant) / "test"
    if not root.exists():
        return str(root), False

    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")),
        key=lambda p: str(p).lower(),
    )
    if not files:
        return str(root), False

    allowed_suffixes = _suffixes_for_mode(mode)
    if not allowed_suffixes:
        return str(root), False

    candidates = [p for p in files if p.suffix.lower() in allowed_suffixes]
    if not candidates:
        return str(root), False

    return str(candidates[0]), True


def _suffixes_for_mode(mode: str) -> set[str]:
    if mode == "image":
        return IMAGE_SUFFIXES

    if mode == "frame":
        return IMAGE_SUFFIXES

    if mode == "video":
        return VIDEO_SUFFIXES

    if mode == "match":
        return VIDEO_SUFFIXES | IMAGE_SUFFIXES

    return set()
