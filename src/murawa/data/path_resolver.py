from pathlib import Path

TRAINING_ROOT = Path("data/training")
TEST_INPUT_ROOT = Path("data/test/base-format")
PREDICTIONS_ROOT = Path("outputs/predictions")


def training_path(project_root: Path, dataset_variant: str) -> Path:
    specific = project_root / TRAINING_ROOT / dataset_variant
    if specific.exists():
        return specific
    return project_root / TRAINING_ROOT / "base-format"


def pick_input(project_root: Path, mode: str) -> tuple[str, bool]:
    root = project_root / TEST_INPUT_ROOT
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
