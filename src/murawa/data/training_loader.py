from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random

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
class SamplingSummary:
    strategy: str
    seed: int | None
    requested_max_samples: int
    original_sample_count: int
    selected_sample_count: int
    source_counts: dict[str, int]
    density_counts: dict[str, int]
    ball_presence_counts: dict[str, int]


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
    sampling_summary: SamplingSummary | None = None


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
    sampling_seed: int | None = None,
    sampling_strategy: str = "deterministic_stratified",
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

    sampling_summary: SamplingSummary | None = None
    if max_samples is not None:
        if max_samples <= 0:
            _raise_loader_error(
                variant_dir=variant_dir,
                split=split,
                detail=f"'max_samples' must be > 0 when provided, got {max_samples}.",
            )
        samples, sampling_summary = _sample_loaded_samples(
            samples=samples,
            max_samples=max_samples,
            sampling_seed=sampling_seed,
            sampling_strategy=sampling_strategy,
        )

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
        sampling_summary=sampling_summary,
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


def _sample_loaded_samples(
    *,
    samples: list[LoadedSample],
    max_samples: int,
    sampling_seed: int | None,
    sampling_strategy: str,
) -> tuple[list[LoadedSample], SamplingSummary]:
    if max_samples >= len(samples):
        selected = list(samples)
        return selected, _build_sampling_summary(
            selected=selected,
            strategy=sampling_strategy,
            seed=sampling_seed,
            max_samples=max_samples,
            original_count=len(samples),
        )

    if sampling_strategy != "deterministic_stratified":
        raise ValueError(
            f"Unsupported sampling_strategy='{sampling_strategy}'. Expected 'deterministic_stratified'."
        )

    rng = random.Random(sampling_seed)
    indexed_samples = list(enumerate(samples))
    buckets: dict[tuple[str, str, str], list[tuple[int, LoadedSample]]] = defaultdict(list)
    for original_index, sample in indexed_samples:
        key = (
            _detect_sample_source(sample),
            _annotation_density_bucket(sample),
            _ball_presence_bucket(sample),
        )
        buckets[key].append((original_index, sample))

    for bucket_samples in buckets.values():
        rng.shuffle(bucket_samples)

    selected_pairs = _allocate_stratified_sample(
        buckets=buckets,
        target_size=max_samples,
        rng=rng,
    )
    selected = [sample for _, sample in selected_pairs]
    summary = _build_sampling_summary(
        selected=selected,
        strategy=sampling_strategy,
        seed=sampling_seed,
        max_samples=max_samples,
        original_count=len(samples),
    )
    return selected, summary


def _allocate_stratified_sample(
    *,
    buckets: dict[tuple[str, str, str], list[tuple[int, LoadedSample]]],
    target_size: int,
    rng: random.Random,
) -> list[tuple[int, LoadedSample]]:
    total_size = sum(len(bucket) for bucket in buckets.values())
    allocations = {key: 0 for key in buckets}
    remainders: list[tuple[float, int, str, str, str]] = []

    for key, bucket_samples in buckets.items():
        raw_target = (len(bucket_samples) * target_size) / total_size
        base_target = min(len(bucket_samples), int(raw_target))
        allocations[key] = base_target
        remainder = raw_target - base_target
        remainders.append((remainder, len(bucket_samples), key[0], key[1], key[2]))

    assigned = sum(allocations.values())
    remainders.sort(reverse=True)
    for _, _, source, density, ball_presence in remainders:
        if assigned >= target_size:
            break
        key = (source, density, ball_presence)
        capacity = len(buckets[key]) - allocations[key]
        if capacity <= 0:
            continue
        allocations[key] += 1
        assigned += 1

    selected: list[tuple[int, LoadedSample]] = []
    leftovers: list[tuple[int, LoadedSample]] = []
    for key, bucket_samples in buckets.items():
        take = allocations[key]
        selected.extend(bucket_samples[:take])
        leftovers.extend(bucket_samples[take:])

    if len(selected) < target_size:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_size - len(selected)])

    return selected[:target_size]


def _build_sampling_summary(
    *,
    selected: list[LoadedSample],
    strategy: str,
    seed: int | None,
    max_samples: int,
    original_count: int,
) -> SamplingSummary:
    source_counts = Counter(_detect_sample_source(sample) for sample in selected)
    density_counts = Counter(_annotation_density_bucket(sample) for sample in selected)
    ball_counts = Counter(_ball_presence_bucket(sample) for sample in selected)
    return SamplingSummary(
        strategy=strategy,
        seed=seed,
        requested_max_samples=max_samples,
        original_sample_count=original_count,
        selected_sample_count=len(selected),
        source_counts=dict(sorted(source_counts.items())),
        density_counts=dict(sorted(density_counts.items())),
        ball_presence_counts=dict(sorted(ball_counts.items())),
    )


def _detect_sample_source(sample: LoadedSample) -> str:
    file_name = sample.image_path.name.lower()
    if file_name.startswith("ballextra_"):
        return "ball-extra"
    if file_name.startswith("soccernet_"):
        return "soccernet"
    return "unknown"


def _annotation_density_bucket(sample: LoadedSample) -> str:
    count = len(sample.annotations)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    return "6+"


def _ball_presence_bucket(sample: LoadedSample) -> str:
    has_ball = any(annotation.category_name == "ball" for annotation in sample.annotations)
    return "ball" if has_ball else "no-ball"


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
    "SamplingSummary",
    "SplitSummary",
    "VariantSummary",
    "load_training_split",
    "summarize_variant",
]
