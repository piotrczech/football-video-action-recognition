from pathlib import Path

READY_ROOT = Path("data/ready")
PREDICTIONS_ROOT = Path("outputs/predictions")


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

    files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not files:
        return str(root), False

    if mode == "image":
        for p in files:
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                return str(p), True

    if mode == "video":
        for p in files:
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                return str(p), True

    return str(files[0]), True
