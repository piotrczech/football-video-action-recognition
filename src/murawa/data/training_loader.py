from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

READY_ROOT = Path("data/ready")
TRAIN_SPLIT = "train"
KNOWN_SPLITS = ("train", "valid", "test")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class DataLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedAnnotation:
    annotation_id: int
    category_id: int
    category_name: str
    bbox_xywh: tuple[float, float, float, float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class LoadedSample:
    image_id: int
    image_path: Path
    width: int
    height: int
    annotations: tuple[LoadedAnnotation, ...]


@dataclass(frozen=True)
class LoadedSplit:
    dataset_variant: str
    split: str
    available_splits: tuple[str, ...]
    variant_dir: Path
    split_dir: Path
    annotation_path: Path
    class_mapping: dict[int, str]
    samples: tuple[LoadedSample, ...]
    total_images: int
    total_annotations: int


@dataclass(frozen=True)
class SplitSummary:
    split: str
    split_dir: Path
    annotation_path: Path
    total_images: int
    total_annotations: int
    class_mapping: dict[int, str]


@dataclass(frozen=True)
class VariantSummary:
    dataset_variant: str
    variant_dir: Path
    available_splits: tuple[str, ...]
    split_summaries: dict[str, SplitSummary]

    def to_dict(self) -> dict:
        return {
            "dataset_variant": self.dataset_variant,
            "variant_dir": str(self.variant_dir),
            "available_splits": list(self.available_splits),
            "split_summaries": {
                split: {
                    "split_dir": str(summary.split_dir),
                    "annotation_path": str(summary.annotation_path),
                    "total_images": summary.total_images,
                    "total_annotations": summary.total_annotations,
                    "class_mapping": {str(k): v for k, v in summary.class_mapping.items()},
                }
                for split, summary in self.split_summaries.items()
            },
        }


def preprocess_every_nth_frame(frame_step: int) -> None:
    _ = frame_step
    raise NotImplementedError(
        "TODO(Issue #08): use murawa.data.frame_selection.select_n_frames + preprocess_selected_frames."
    )


def load_training_split(
    project_root: Path,
    dataset_variant: str,
    split: str = TRAIN_SPLIT,
    max_samples: int | None = None,
) -> LoadedSplit:
    variant_dir = _resolve_variant_dir(project_root=project_root, dataset_variant=dataset_variant)
    payload, annotation_path, split_dir = _load_coco_payload(variant_dir=variant_dir, split=split)
    available_splits = _validated_available_splits(variant_dir=variant_dir, validated_split=split)
    class_mapping = _build_class_mapping(payload=payload, variant_dir=variant_dir, split=split)
    annotations_by_image = _build_annotations_by_image(
        payload=payload,
        class_mapping=class_mapping,
        variant_dir=variant_dir,
        split=split,
    )

    samples: list[LoadedSample] = []
    for image in payload["images"]:
        image_id = _int_field(image=image, key="id", variant_dir=variant_dir, split=split)
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            _raise_loader_error(variant_dir=variant_dir, split=split, detail="COCO image has empty 'file_name'.")

        image_path = (split_dir / file_name).resolve()
        _ensure_within_split_dir(
            image_path=image_path,
            split_dir=split_dir,
            variant_dir=variant_dir,
            split=split,
            file_name=file_name,
        )
        if not image_path.exists() or not image_path.is_file():
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Image listed in COCO does not exist: '{file_name}'.",
            )
        _validate_image_file(image_path=image_path, variant_dir=variant_dir, split=split)

        width = _int_field(image=image, key="width", variant_dir=variant_dir, split=split)
        height = _int_field(image=image, key="height", variant_dir=variant_dir, split=split)
        if width <= 0 or height <= 0:
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Image '{file_name}' has invalid dimensions width={width}, height={height}.",
            )

        samples.append(
            LoadedSample(
                image_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                annotations=tuple(annotations_by_image.get(image_id, [])),
            )
        )

    if max_samples is not None:
        if max_samples <= 0:
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"'max_samples' must be > 0 when provided, got {max_samples}.",
            )
        samples = samples[:max_samples]

    return LoadedSplit(
        dataset_variant=dataset_variant,
        split=split,
        available_splits=available_splits,
        variant_dir=variant_dir,
        split_dir=split_dir,
        annotation_path=annotation_path,
        class_mapping=class_mapping,
        samples=tuple(samples),
        total_images=len(payload["images"]),
        total_annotations=len(payload["annotations"]),
    )


def summarize_variant(project_root: Path, dataset_variant: str) -> VariantSummary:
    variant_dir = _resolve_variant_dir(project_root=project_root, dataset_variant=dataset_variant)
    split_summaries: dict[str, SplitSummary] = {}

    for split in KNOWN_SPLITS:
        split_dir = variant_dir / split
        if not split_dir.exists():
            continue

        payload, annotation_path, split_dir = _load_coco_payload(variant_dir=variant_dir, split=split)
        class_mapping = _build_class_mapping(payload=payload, variant_dir=variant_dir, split=split)
        split_summaries[split] = SplitSummary(
            split=split,
            split_dir=split_dir,
            annotation_path=annotation_path,
            total_images=len(payload["images"]),
            total_annotations=len(payload["annotations"]),
            class_mapping=class_mapping,
        )

    if TRAIN_SPLIT not in split_summaries:
        _raise_loader_error(
            variant_dir=variant_dir,
            split=TRAIN_SPLIT,
            detail=f"Required split '{TRAIN_SPLIT}' is missing for dataset_variant='{dataset_variant}'.",
        )

    return VariantSummary(
        dataset_variant=dataset_variant,
        variant_dir=variant_dir,
        available_splits=tuple(split_summaries.keys()),
        split_summaries=split_summaries,
    )


def _resolve_variant_dir(project_root: Path, dataset_variant: str) -> Path:
    variant_dir = (project_root / READY_ROOT / dataset_variant).resolve()
    if not variant_dir.exists() or not variant_dir.is_dir():
        _raise_loader_error(
            variant_dir=variant_dir,
            split=TRAIN_SPLIT,
            detail=f"Dataset variant '{dataset_variant}' does not exist.",
        )
    return variant_dir


def _validated_available_splits(variant_dir: Path, validated_split: str) -> tuple[str, ...]:
    available_splits: list[str] = []
    for candidate_split in KNOWN_SPLITS:
        split_dir = variant_dir / candidate_split
        if not split_dir.exists():
            continue
        if candidate_split != validated_split:
            _load_coco_payload(variant_dir=variant_dir, split=candidate_split)
        available_splits.append(candidate_split)

    if validated_split not in available_splits:
        available_splits.insert(0, validated_split)
    return tuple(available_splits)


def _load_coco_payload(variant_dir: Path, split: str) -> tuple[dict, Path, Path]:
    split_dir = variant_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        _raise_loader_error(variant_dir=variant_dir, split=split, detail=f"Split directory '{split}' does not exist.")

    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists() or not annotation_path.is_file():
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"Missing COCO file '{annotation_path.name}'.",
        )

    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"Invalid JSON in '{annotation_path.name}': {exc}",
        )

    if not isinstance(payload, dict):
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"COCO payload in '{annotation_path.name}' must be a JSON object.",
        )

    missing_keys = [key for key in ("images", "annotations", "categories") if key not in payload]
    if missing_keys:
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"COCO payload is missing keys: {','.join(missing_keys)}.",
        )

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail="COCO keys 'images', 'annotations', 'categories' must all be lists.",
        )
    if not images:
        _raise_loader_error(variant_dir=variant_dir, split=split, detail="COCO list 'images' is empty.")
    if not annotations:
        _raise_loader_error(variant_dir=variant_dir, split=split, detail="COCO list 'annotations' is empty.")
    if not categories:
        _raise_loader_error(variant_dir=variant_dir, split=split, detail="COCO list 'categories' is empty.")

    return payload, annotation_path.resolve(), split_dir.resolve()


def _build_class_mapping(payload: dict, variant_dir: Path, split: str) -> dict[int, str]:
    class_mapping: dict[int, str] = {}
    for category in payload["categories"]:
        if not isinstance(category, dict):
            _raise_loader_error(variant_dir=variant_dir, split=split, detail="COCO category entry is not an object.")

        category_id = category.get("id")
        category_name = category.get("name")
        if not isinstance(category_id, int):
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"COCO category id must be int, got: {category_id!r}.",
            )
        if not isinstance(category_name, str) or not category_name.strip():
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"COCO category name for id={category_id} is invalid.",
            )
        class_mapping[category_id] = category_name.strip()
    return class_mapping


def _build_annotations_by_image(
    payload: dict,
    class_mapping: dict[int, str],
    variant_dir: Path,
    split: str,
) -> dict[int, list[LoadedAnnotation]]:
    result: dict[int, list[LoadedAnnotation]] = {}
    for annotation in payload["annotations"]:
        if not isinstance(annotation, dict):
            _raise_loader_error(variant_dir=variant_dir, split=split, detail="COCO annotation entry is not an object.")

        annotation_id = _int_field(
            image=annotation,
            key="id",
            variant_dir=variant_dir,
            split=split,
            object_name="annotation",
        )
        image_id = _int_field(
            image=annotation,
            key="image_id",
            variant_dir=variant_dir,
            split=split,
            object_name="annotation",
        )
        category_id = _int_field(
            image=annotation,
            key="category_id",
            variant_dir=variant_dir,
            split=split,
            object_name="annotation",
        )
        if category_id not in class_mapping:
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Annotation id={annotation_id} references unknown category_id={category_id}.",
            )

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Annotation id={annotation_id} has invalid bbox. Expected [x,y,w,h].",
            )
        try:
            x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Annotation id={annotation_id} has non-numeric bbox values.",
            )
        if w <= 0 or h <= 0:
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Annotation id={annotation_id} has non-positive bbox size w={w}, h={h}.",
            )

        area_raw = annotation.get("area", w * h)
        try:
            area = float(area_raw)
        except (TypeError, ValueError):
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Annotation id={annotation_id} has invalid area: {area_raw!r}.",
            )
        iscrowd = annotation.get("iscrowd", 0)
        if not isinstance(iscrowd, int):
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"Annotation id={annotation_id} has invalid iscrowd: {iscrowd!r}.",
            )

        result.setdefault(image_id, []).append(
            LoadedAnnotation(
                annotation_id=annotation_id,
                category_id=category_id,
                category_name=class_mapping[category_id],
                bbox_xywh=(x, y, w, h),
                area=area,
                iscrowd=iscrowd,
            )
        )

    return result


def _int_field(
    image: dict,
    key: str,
    variant_dir: Path,
    split: str,
    object_name: str = "image",
) -> int:
    value = image.get(key)
    if not isinstance(value, int):
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"COCO {object_name} field '{key}' must be int, got {value!r}.",
        )
    return value


def _validate_image_file(image_path: Path, variant_dir: Path, split: str) -> None:
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"Unsupported image extension in split '{split}': '{image_path.name}'.",
        )
    if image_path.stat().st_size <= 0:
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=f"Image file is empty: '{image_path.name}'.",
        )


def _ensure_within_split_dir(
    image_path: Path,
    split_dir: Path,
    variant_dir: Path,
    split: str,
    file_name: str,
) -> None:
    try:
        image_path.relative_to(split_dir)
    except ValueError:
        _raise_loader_error(
            variant_dir=variant_dir,
            split=split,
            detail=(
                "Image path escapes split directory and is not allowed: "
                f"'{file_name}' -> '{image_path}'."
            ),
        )


def _raise_loader_error(variant_dir: Path, split: str, detail: str) -> None:
    raise DataLoaderError(
        "\n".join(
            [
                f"Data loader error for variant='{variant_dir.name}', split='{split}': {detail}",
                f"Checked path: {variant_dir / split}",
                "Expected format: data/ready/<variant>/<split>/_annotations.coco.json with image files referenced by COCO.",
                "Hint: przygotuj dane do common format i uruchom preprocessing (TODO: Issue #8 - wybór co n-tej klatki).",
            ]
        )
    )


__all__ = [
    "DataLoaderError",
    "LoadedAnnotation",
    "LoadedSample",
    "LoadedSplit",
    "SplitSummary",
    "VariantSummary",
    "load_training_split",
    "summarize_variant",
]
