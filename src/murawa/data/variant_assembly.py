from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from murawa.data.bootstrap_variant import (
    BootstrapConfig,
    SPLITS,
    SUPPORTED_VARIANTS as BOOTSTRAP_VARIANTS,
    build_bootstrap_variant,
    describe_variant_composition,
)
from murawa.data.variant_image_transforms import (
    TRANSFORM_PIPELINE_NAME,
    TRANSFORM_PIPELINE_OPS,
    apply_lightweight_training_transform,
)

COCO_REQUIRED_KEYS = {"images", "annotations", "categories"}
SUPPORTED_VARIANTS = (
    *BOOTSTRAP_VARIANTS,
    "extended-transformed",
    "extended-only-train-transformed",
)


@dataclass(frozen=True)
class VariantAssemblyConfig:
    selected_root: Path
    final_root: Path
    variant_name: str
    include_ball_extra: bool = False
    enable_transforms: bool = False
    frame_step: int = 30
    force: bool = False


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    source_variant: str
    transforms_enabled: bool = False


def assemble_variant(config: VariantAssemblyConfig) -> Path:
    """Build one named variant in data/ready and write concise metadata."""
    spec = _resolve_variant_spec(config.variant_name)
    project_root = _resolve_project_root(config.final_root)

    if spec.transforms_enabled:
        source_variant_dir = _ensure_source_variant(
            project_root=project_root,
            final_root=config.final_root,
            source_variant=spec.source_variant,
            frame_step=config.frame_step,
            force=config.force,
        )

        target_variant_dir = config.final_root / spec.variant_name
        if target_variant_dir.exists():
            if not config.force:
                _validate_variant_dir(target_variant_dir)
                _write_variant_metadata(
                    variant_dir=target_variant_dir,
                    config=config,
                    spec=spec,
                    project_root=project_root,
                    transform_summary=None,
                )
                return target_variant_dir
            shutil.rmtree(target_variant_dir)

        shutil.copytree(source_variant_dir, target_variant_dir)

        copied_bootstrap_summary = target_variant_dir / "bootstrap_summary.json"
        if copied_bootstrap_summary.exists():
            copied_bootstrap_summary.unlink()

        transform_summary = _augment_train_split_with_transformed_copies(target_variant_dir / "train")
        _validate_variant_dir(target_variant_dir)
        _write_variant_metadata(
            variant_dir=target_variant_dir,
            config=config,
            spec=spec,
            project_root=project_root,
            transform_summary=transform_summary,
        )
        return target_variant_dir

    target_variant_dir = config.final_root / spec.variant_name
    if not target_variant_dir.exists() or config.force:
        build_bootstrap_variant(
            project_root=project_root,
            config=BootstrapConfig(
                output_variant=spec.variant_name,
                frame_step=config.frame_step,
                force=config.force,
            ),
        )

    _validate_variant_dir(target_variant_dir)
    _write_variant_metadata(
        variant_dir=target_variant_dir,
        config=config,
        spec=spec,
        project_root=project_root,
        transform_summary=None,
    )
    return target_variant_dir


def describe_variant(variant_dir: Path) -> dict[str, str]:
    """Produce concise metadata explaining what differentiates this variant."""
    variant_dir = Path(variant_dir)
    spec = _resolve_variant_spec(variant_dir.name)
    split_stats = _read_split_stats(variant_dir)

    composition = describe_variant_composition(spec.source_variant)
    train_sources = list(composition["train"])
    if spec.transforms_enabled:
        train_sources.append("transformed-copies")

    return {
        "variant_name": spec.variant_name,
        "derived_from_variant": spec.source_variant,
        "train_sources": ", ".join(train_sources),
        "valid_sources": ", ".join(composition["valid"]),
        "test_sources": ", ".join(composition["test"]),
        "transforms_enabled": str(spec.transforms_enabled).lower(),
        "transform_scope": "train only" if spec.transforms_enabled else "none",
        "transform_pipeline": TRANSFORM_PIPELINE_NAME if spec.transforms_enabled else "none",
        "transform_ops": " | ".join(TRANSFORM_PIPELINE_OPS) if spec.transforms_enabled else "none",
        "train_images": str(split_stats["train"]["images"]),
        "train_annotations": str(split_stats["train"]["annotations"]),
        "valid_images": str(split_stats["valid"]["images"]),
        "valid_annotations": str(split_stats["valid"]["annotations"]),
        "test_images": str(split_stats["test"]["images"]),
        "test_annotations": str(split_stats["test"]["annotations"]),
    }


def _resolve_variant_spec(variant_name: str) -> VariantSpec:
    if variant_name in BOOTSTRAP_VARIANTS:
        return VariantSpec(variant_name=variant_name, source_variant=variant_name, transforms_enabled=False)

    if variant_name == "extended-transformed":
        return VariantSpec(
            variant_name=variant_name,
            source_variant="extended",
            transforms_enabled=True,
        )

    if variant_name == "extended-only-train-transformed":
        return VariantSpec(
            variant_name=variant_name,
            source_variant="extended-only-train",
            transforms_enabled=True,
        )

    raise ValueError(
        f"Unsupported variant '{variant_name}'. "
        f"Expected one of: {', '.join(SUPPORTED_VARIANTS)}"
    )


def _resolve_project_root(final_root: Path) -> Path:
    final_root = Path(final_root).resolve()
    return final_root.parents[1]

def _to_project_relative_path(path: Path, project_root: Path) -> str:
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path)

def _ensure_source_variant(
    *,
    project_root: Path,
    final_root: Path,
    source_variant: str,
    frame_step: int,
    force: bool,
) -> Path:
    source_variant_dir = Path(final_root) / source_variant

    # Do not rebuild an already prepared source variant here.
    # During full runs, base variants are built first and transformed variants
    # should reuse them instead of wiping their metadata.
    if source_variant_dir.exists():
        return source_variant_dir

    build_bootstrap_variant(
        project_root=project_root,
        config=BootstrapConfig(
            output_variant=source_variant,
            frame_step=frame_step,
            force=force,
        ),
    )
    return source_variant_dir


def _augment_train_split_with_transformed_copies(train_split_dir: Path) -> dict[str, object]:
    annotation_path = train_split_dir / "_annotations.coco.json"
    payload = _load_coco(annotation_path)

    images = payload["images"]
    annotations = payload["annotations"]
    categories = payload["categories"]
    info = payload.get("info", {})

    images_dir = train_split_dir / "images"
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        if isinstance(image_id, int):
            annotations_by_image[image_id].append(ann)

    original_images = list(images)
    next_image_id = max((img["id"] for img in images), default=0) + 1
    next_annotation_id = max((ann["id"] for ann in annotations), default=0) + 1

    added_images = 0
    added_annotations = 0
    skipped_images = 0
    preview_files: list[str] = []

    for image_entry in original_images:
        file_name = image_entry.get("file_name")
        image_id = image_entry.get("id")
        if not isinstance(file_name, str) or not isinstance(image_id, int):
            skipped_images += 1
            continue

        source_path = train_split_dir / file_name
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped_images += 1
            continue

        transformed_image, transform_metadata = apply_lightweight_training_transform(image=image, key=file_name)

        base_name = Path(file_name).name
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        transformed_base_name = f"{stem}__tf{suffix}"
        transformed_rel_path = f"images/{transformed_base_name}"
        transformed_abs_path = images_dir / transformed_base_name

        cv2.imwrite(str(transformed_abs_path), transformed_image)
        height, width = transformed_image.shape[:2]

        images.append(
            {
                "id": next_image_id,
                "file_name": transformed_rel_path,
                "width": width,
                "height": height,
            }
        )

        for ann in annotations_by_image.get(image_id, []):
            duplicated = dict(ann)
            duplicated["id"] = next_annotation_id
            duplicated["image_id"] = next_image_id
            annotations.append(duplicated)
            next_annotation_id += 1
            added_annotations += 1

        if len(preview_files) < 5:
            preview_files.append(f"{transformed_base_name} [{transform_metadata.pipeline}]")

        next_image_id += 1
        added_images += 1

    payload["info"] = {
        **info,
        "derived_train_transforms": {
            "pipeline": TRANSFORM_PIPELINE_NAME,
            "ops": list(TRANSFORM_PIPELINE_OPS),
            "scope": "train only",
            "added_images": added_images,
            "added_annotations": added_annotations,
        },
    }
    annotation_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pipeline": TRANSFORM_PIPELINE_NAME,
        "ops": list(TRANSFORM_PIPELINE_OPS),
        "scope": "train only",
        "added_images": added_images,
        "added_annotations": added_annotations,
        "skipped_images": skipped_images,
        "preview_files": preview_files,
    }


def _write_variant_metadata(
    *,
    variant_dir: Path,
    config: VariantAssemblyConfig,
    spec: VariantSpec,
    project_root: Path,
    transform_summary: dict[str, object] | None,
) -> None:
    description = describe_variant(variant_dir)
    description["selected_root_reference"] = _to_project_relative_path(config.selected_root, project_root)
    description["final_root"] = _to_project_relative_path(config.final_root, project_root)
    description["bootstrap_source_of_truth"] = "true"
    description["frame_step_reference"] = str(config.frame_step)

    if transform_summary is not None:
        description["transformed_images_added_train"] = str(transform_summary["added_images"])
        description["transformed_annotations_added_train"] = str(transform_summary["added_annotations"])

    (variant_dir / "variant_description.json").write_text(
        json.dumps(description, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = _build_variant_summary(
        variant_dir=variant_dir,
        spec=spec,
        config=config,
        project_root=project_root,
        transform_summary=transform_summary,
    )
    (variant_dir / "variant_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_variant_summary(
    *,
    variant_dir: Path,
    spec: VariantSpec,
    config: VariantAssemblyConfig,
    project_root: Path,
    transform_summary: dict[str, object] | None,
) -> dict[str, object]:
    split_stats = _read_split_stats(variant_dir)
    composition = describe_variant_composition(spec.source_variant)

    train_sources = list(composition["train"])
    if spec.transforms_enabled:
        train_sources.append("transformed-copies")

    return {
        "variant": spec.variant_name,
        "derived_from_variant": spec.source_variant,
        "bootstrap_source_of_truth": True,
        "selected_root_reference": _to_project_relative_path(config.selected_root, project_root),
        "final_root": _to_project_relative_path(config.final_root, project_root),
        "frame_step_reference": config.frame_step,
        "transforms": {
            "enabled": spec.transforms_enabled,
            "scope": "train only" if spec.transforms_enabled else "none",
            "pipeline": TRANSFORM_PIPELINE_NAME if spec.transforms_enabled else "none",
            "ops": list(TRANSFORM_PIPELINE_OPS) if spec.transforms_enabled else [],
            "summary": transform_summary or {},
        },
        "splits": {
            "train": {
                "images": split_stats["train"]["images"],
                "annotations": split_stats["train"]["annotations"],
                "sources": train_sources,
            },
            "valid": {
                "images": split_stats["valid"]["images"],
                "annotations": split_stats["valid"]["annotations"],
                "sources": list(composition["valid"]),
            },
            "test": {
                "images": split_stats["test"]["images"],
                "annotations": split_stats["test"]["annotations"],
                "sources": list(composition["test"]),
            },
        },
        "notes": [
            "Split composition follows bootstrap_variant.py as the source of truth.",
            "Transformed variants add lightweight transformed copies only to the train split.",
            "Validation and test remain untransformed for cleaner comparison of model results.",
        ],
    }


def _read_split_stats(variant_dir: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        annotation_path = variant_dir / split / "_annotations.coco.json"
        payload = _load_coco(annotation_path)
        stats[split] = {
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
        }
    return stats


def _validate_variant_dir(variant_dir: Path) -> None:
    if not variant_dir.exists():
        raise FileNotFoundError(f"Variant directory does not exist: {variant_dir}")

    for split in SPLITS:
        split_dir = variant_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        annotation_path = split_dir / "_annotations.coco.json"
        payload = _load_coco(annotation_path)

        for image_entry in payload["images"]:
            file_name = image_entry.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError(f"Invalid `file_name` entry in: {annotation_path}")

            image_path = split_dir / file_name
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Broken image path in variant '{variant_dir.name}', split '{split}': {file_name}"
                )


def _load_coco(annotation_path: Path) -> dict:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not COCO_REQUIRED_KEYS.issubset(payload):
        raise ValueError(
            f"Invalid COCO payload in: {annotation_path}. "
            f"Required keys: {sorted(COCO_REQUIRED_KEYS)}"
        )
    return payload


__all__ = [
    "SUPPORTED_VARIANTS",
    "VariantAssemblyConfig",
    "assemble_variant",
    "describe_variant",
]