import json
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ArtifactManifest:
    run_name: str
    model: str
    dataset_variant: str
    created_at_utc: str
    checkpoint_file: str = "model.pt"
    required_metadata: tuple[str, ...] = tuple(REQUIRED_METADATA)


@dataclass(frozen=True)
class ArtifactWriteContext:
    run_name: str
    project_root: Path
    checkpoint_path: Path
    metadata_dir: Path


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


class StandardizedArtifactCallback:
    """Issue #14 contract: central callback for standardized checkpoint/metadata persistence."""

    def write_manifest(self, meta_dir: Path, manifest: ArtifactManifest) -> Path:
        raise NotImplementedError(
            "TODO(Issue #14): write run manifest with checkpoint location, resolved config source, "
            "dataset variant and metrics references."
        )

    def validate_contract(self, context: ArtifactWriteContext) -> None:
        raise NotImplementedError(
            "TODO(Issue #14): validate required file layout and naming so cluster outputs are "
            "directly usable by local prediction/integration flow."
        )


def write_artifact_manifest(meta_dir: Path, manifest: ArtifactManifest) -> Path:
    """Compatibility wrapper to callback contract for Issue #14."""
    return StandardizedArtifactCallback().write_manifest(meta_dir=meta_dir, manifest=manifest)


def validate_artifact_contract(project_root: Path, run_name: str) -> None:
    """Compatibility wrapper to callback contract for Issue #14."""
    context = ArtifactWriteContext(
        run_name=run_name,
        project_root=project_root,
        checkpoint_path=project_root / CKPT_DIR / run_name / "model.pt",
        metadata_dir=project_root / META_DIR / run_name,
    )
    StandardizedArtifactCallback().validate_contract(context=context)


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
