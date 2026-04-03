import json
import re
from datetime import datetime
from pathlib import Path

CKPT_DIR = Path("models/checkpoints")
META_DIR = Path("models/metadata")
REQUIRED_METADATA = [
    "config.yaml",
    "class_mapping.json",
    "train_metadata.json",
    "metrics_summary.json",
    "dataset_variant.json",
]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_config(src: Path, dst: Path, model: str, dataset_variant: str) -> bool:
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return True

    dst.write_text(
        "\n".join(
            [
                "# Auto-generated fallback config for MVP mock",
                f"model: {model}",
                f"dataset_variant: {dataset_variant}",
                "epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.001",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return False


def make_run_name(model: str, dataset_variant: str, created_at: datetime, tag: str = "auto") -> str:
    return f"{model}_{dataset_variant}_{created_at.strftime('%Y%m%d-%H%M')}_{tag}"


def latest_run(project_root: Path, model: str, dataset_variant: str) -> str:
    prefix = f"{model}_{dataset_variant}_"
    root = project_root / CKPT_DIR
    if not root.exists():
        raise FileNotFoundError("No checkpoints directory found.")

    matches = []
    for item in root.iterdir():
        if not item.is_dir() or not item.name.startswith(prefix):
            continue
        ckpt = project_root / CKPT_DIR / item.name / "model.pt"
        meta_dir = project_root / META_DIR / item.name
        has_required = ckpt.exists() and meta_dir.exists() and all(
            (meta_dir / name).exists() for name in REQUIRED_METADATA
        )
        if not has_required:
            continue
        m = re.match(rf"^{re.escape(prefix)}(\d{{8}}-\d{{4}})_", item.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d-%H%M")
        matches.append((ts, item.name))

    if not matches:
        raise FileNotFoundError(
            f"No valid runs found for model='{model}', dataset_variant='{dataset_variant}'."
        )

    matches.sort(reverse=True)
    return matches[0][1]
