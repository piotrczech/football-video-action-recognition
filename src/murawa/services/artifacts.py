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


@dataclass(frozen=True)
class TrainedRunRecord:
    run_name: str
    model: str
    dataset_variant: str
    created_at_utc: str
    run_tag: str
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
    sanitized_tag = sanitize_run_tag(tag)
    if sanitized_tag != "auto":
        return sanitized_tag
    return f"{model}_{dataset_variant}_{created_at.strftime('%Y%m%d-%H%M')}_auto"


def sanitize_run_tag(tag: str | None) -> str:
    if tag is None:
        return "auto"

    cleaned = re.sub(r"[^a-z0-9_-]+", "-", tag.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned or "auto"


class StandardizedArtifactCallback:
    """Issue #14 contract: central callback for standardized checkpoint/metadata persistence."""

    def write_manifest(self, meta_dir: Path, manifest: ArtifactManifest) -> Path:
        meta_dir = meta_dir.resolve()
        meta_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = meta_dir / "manifest.json"
        payload = {
            "schema_version": 1,
            "run_name": manifest.run_name,
            "model": manifest.model,
            "dataset_variant": manifest.dataset_variant,
            "created_at_utc": manifest.created_at_utc,
            "checkpoint": {
                "file": manifest.checkpoint_file,
                "relative_path": str(CKPT_DIR / manifest.run_name / manifest.checkpoint_file),
            },
            "metadata": {
                "directory": str(META_DIR / manifest.run_name),
                "required_files": list(manifest.required_metadata),
                "manifest_file": manifest_path.name,
            },
        }
        write_json(manifest_path, payload)
        return manifest_path

    def validate_contract(self, context: ArtifactWriteContext) -> None:
        project_root = context.project_root.resolve()
        checkpoint_path = context.checkpoint_path.resolve()
        metadata_dir = context.metadata_dir.resolve()
        errors: list[str] = []

        expected_checkpoint = (project_root / CKPT_DIR / context.run_name / "model.pt").resolve()
        expected_metadata = (project_root / META_DIR / context.run_name).resolve()

        if checkpoint_path != expected_checkpoint:
            errors.append(f"checkpoint path should be '{expected_checkpoint}', got '{checkpoint_path}'")
        if metadata_dir != expected_metadata:
            errors.append(f"metadata dir should be '{expected_metadata}', got '{metadata_dir}'")
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            errors.append(f"missing checkpoint file: {checkpoint_path}")
        if metadata_dir.exists() and not metadata_dir.is_dir():
            errors.append(f"metadata path is not a directory: {metadata_dir}")
        if not metadata_dir.exists():
            errors.append(f"missing metadata directory: {metadata_dir}")

        if metadata_dir.exists() and metadata_dir.is_dir():
            for filename in REQUIRED_METADATA:
                path = metadata_dir / filename
                if not path.exists() or not path.is_file():
                    errors.append(f"missing metadata file: {path}")
                    continue
                if filename.endswith(".json"):
                    try:
                        json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        errors.append(f"invalid JSON in metadata file '{path}': {exc}")

            train_metadata = metadata_dir / "train_metadata.json"
            if train_metadata.exists() and train_metadata.is_file():
                try:
                    payload = json.loads(train_metadata.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and payload.get("run_name") != context.run_name:
                    errors.append(
                        "train_metadata.json run_name mismatch: "
                        f"expected '{context.run_name}', got '{payload.get('run_name')}'"
                    )

        if errors:
            bullet_list = "\n- ".join(errors)
            raise RuntimeError(f"Artifact contract validation failed for run '{context.run_name}':\n- {bullet_list}")


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


def list_available_runs(project_root: Path) -> list[TrainedRunRecord]:
    root = project_root / META_DIR
    if not root.exists():
        return []

    runs: list[TrainedRunRecord] = []
    for item in root.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        record = _load_run_record(project_root=project_root, run_name=item.name)
        if record is not None:
            runs.append(record)

    runs.sort(key=_sort_key, reverse=True)
    return runs


def resolve_run(project_root: Path, run_name: str) -> TrainedRunRecord:
    record = _load_run_record(project_root=project_root, run_name=run_name)
    if record is None:
        raise FileNotFoundError(f"No valid run found for run_name='{run_name}'.")
    return record


def latest_run(project_root: Path, model: str, dataset_variant: str) -> str:
    matches = [
        record
        for record in list_available_runs(project_root)
        if record.model == model and record.dataset_variant == dataset_variant
    ]

    if not matches:
        raise FileNotFoundError(
            f"No valid runs found for model='{model}', dataset_variant='{dataset_variant}'."
        )

    return matches[0].run_name


def _load_run_record(project_root: Path, run_name: str) -> TrainedRunRecord | None:
    checkpoint_path = project_root / CKPT_DIR / run_name / "model.pt"
    metadata_dir = project_root / META_DIR / run_name
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        return None
    if not metadata_dir.exists() or not metadata_dir.is_dir():
        return None
    if not all((metadata_dir / name).exists() for name in REQUIRED_METADATA):
        return None

    manifest_path = metadata_dir / "manifest.json"
    train_metadata_path = metadata_dir / "train_metadata.json"
    manifest_payload = _read_json(manifest_path) if manifest_path.exists() else None
    train_payload = _read_json(train_metadata_path)
    payload = {**manifest_payload, **train_payload} if manifest_payload and train_payload else (
        manifest_payload or train_payload
    )
    if not isinstance(payload, dict):
        return None

    model = payload.get("model")
    dataset_variant = payload.get("dataset_variant")
    created_at_utc = payload.get("created_at_utc")
    if not all(isinstance(value, str) and value for value in (model, dataset_variant, created_at_utc)):
        return None

    run_tag = payload.get("run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        run_tag = _infer_run_tag(run_name)

    return TrainedRunRecord(
        run_name=run_name,
        model=model,
        dataset_variant=dataset_variant,
        created_at_utc=created_at_utc,
        run_tag=run_tag,
        checkpoint_path=checkpoint_path.resolve(),
        metadata_dir=metadata_dir.resolve(),
    )


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _infer_run_tag(run_name: str) -> str:
    structured = re.match(r"^[a-z0-9-]+_[a-z0-9-]+_\d{8}-\d{4}_(.+)$", run_name)
    if structured:
        return structured.group(1)
    parts = run_name.split("_")
    return parts[-1] if parts else "auto"


def _sort_key(record: TrainedRunRecord) -> tuple[datetime, str]:
    try:
        created_at = datetime.fromisoformat(record.created_at_utc)
    except ValueError:
        created_at = datetime.min
    return created_at, record.run_name
