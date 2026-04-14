from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from murawa.data.bootstrap_variant import (
    BALL_EXTRA_SPLIT_ALIASES,
    CATEGORY_ID_TO_NAME,
    CandidateAnnotation,
    CandidateSample,
    SOCCERNET_SPLIT_ALIASES,
    collect_ball_extra_samples_by_split,
    collect_soccernet_split_samples,
)

SPLITS = ("train", "valid", "test")
COCO_CATEGORIES = [{"id": idx, "name": name} for idx, name in CATEGORY_ID_TO_NAME.items()]
DEFAULT_MAX_DIM = 1280


@dataclass(frozen=True)
class FrameSelectionConfig:
    raw_root: Path
    selected_root: Path
    frame_step: int = 30
    keep_original_resolution: bool = True


def select_n_frames(config: FrameSelectionConfig) -> Path:
    """Issue #08: select every n-th frame from data/raw and write to data/selected."""
    if config.frame_step <= 0:
        raise ValueError("frame_step must be > 0")

    raw_root = config.raw_root
    selected_root = config.selected_root
    selected_root.mkdir(parents=True, exist_ok=True)

    processed_any = False

    soccernet_root = raw_root / "soccernet"
    if soccernet_root.exists():
        soccernet_samples = _collect_available_soccernet_samples(
            soccernet_root=soccernet_root,
            frame_step=config.frame_step,
        )
        if any(soccernet_samples.values()):
            _write_selected_dataset(
                dataset_root=selected_root / "soccernet",
                split_to_samples=soccernet_samples,
                keep_original_resolution=config.keep_original_resolution,
                frame_step=config.frame_step,
                source_name="soccernet",
            )
            processed_any = True

    ball_extra_root = raw_root / "ball-extra"
    if ball_extra_root.exists():
        ball_extra_samples = collect_ball_extra_samples_by_split(ball_extra_root)
        ball_extra_samples = {
            split: sorted(samples, key=lambda s: s.output_file_name)
            for split, samples in ball_extra_samples.items()
            if samples
        }
        if ball_extra_samples:
            _write_selected_dataset(
                dataset_root=selected_root / "ball-extra",
                split_to_samples=ball_extra_samples,
                keep_original_resolution=config.keep_original_resolution,
                frame_step=1,
                source_name="ball-extra",
            )
            processed_any = True

    if not processed_any:
        raise FileNotFoundError(
            f"No supported raw datasets found under: {raw_root}. "
            "Expected at least data/raw/soccernet and/or data/raw/ball-extra."
        )

    return selected_root


def preprocess_selected_frames(selected_root: Path, *, normalize: bool = False) -> Path:
    """Issue #08: run basic preprocessing on selected frames before variant assembly (#09)."""
    selected_root = Path(selected_root)
    if not selected_root.exists():
        raise FileNotFoundError(f"selected_root does not exist: {selected_root}")

    dataset_roots = _discover_dataset_roots(selected_root)
    if not dataset_roots:
        raise FileNotFoundError(
            f"No selected datasets found under: {selected_root}. "
            "Expected split folders or dataset subfolders with train/valid/test."
        )

    for dataset_root in dataset_roots:
        preprocessing_summary: dict[str, object] = {
            "dataset_root": str(dataset_root),
            "normalize": normalize,
            "splits": {},
        }

        for split in SPLITS:
            split_dir = dataset_root / split
            if not split_dir.is_dir():
                continue

            annotation_path = split_dir / "_annotations.coco.json"
            if not annotation_path.exists():
                continue

            payload = _load_json(annotation_path)
            images = payload.get("images", [])
            annotations = payload.get("annotations", [])
            categories = payload.get("categories", COCO_CATEGORIES)
            info = payload.get("info", {})

            annotations_by_image: dict[int, list[dict]] = defaultdict(list)
            for ann in annotations:
                image_id = ann.get("image_id")
                if isinstance(image_id, int):
                    annotations_by_image[image_id].append(ann)

            kept_images: list[dict] = []
            kept_annotations: list[dict] = []
            next_annotation_id = 1
            dropped_images = 0
            dropped_annotations = 0

            for image_entry in images:
                if not isinstance(image_entry, dict):
                    continue

                image_id = image_entry.get("id")
                file_name = image_entry.get("file_name")
                if not isinstance(image_id, int) or not isinstance(file_name, str) or not file_name.strip():
                    dropped_images += 1
                    continue

                image_path = split_dir / file_name
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    dropped_images += 1
                    dropped_annotations += len(annotations_by_image.get(image_id, []))
                    continue

                if normalize:
                    image = _normalize_uint8(image)

                height, width = image.shape[:2]
                cv2.imwrite(str(image_path), image)

                kept_images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                    }
                )

                for ann in annotations_by_image.get(image_id, []):
                    bbox = ann.get("bbox")
                    category_id = ann.get("category_id")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        dropped_annotations += 1
                        continue

                    x, y, w, h = (
                        _safe_float(bbox[0]),
                        _safe_float(bbox[1]),
                        _safe_float(bbox[2]),
                        _safe_float(bbox[3]),
                    )
                    if x is None or y is None or w is None or h is None:
                        dropped_annotations += 1
                        continue

                    clipped_bbox = _clip_bbox(x=x, y=y, w=w, h=h, width=width, height=height)
                    if clipped_bbox is None:
                        dropped_annotations += 1
                        continue

                    bx, by, bw, bh = clipped_bbox
                    kept_annotations.append(
                        {
                            "id": next_annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [bx, by, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    next_annotation_id += 1

            processed_payload = {
                "info": info,
                "images": kept_images,
                "annotations": kept_annotations,
                "categories": categories,
            }
            annotation_path.write_text(
                json.dumps(processed_payload, indent=2),
                encoding="utf-8",
            )

            preprocessing_summary["splits"][split] = {
                "images": len(kept_images),
                "annotations": len(kept_annotations),
                "dropped_images": dropped_images,
                "dropped_annotations": dropped_annotations,
            }

        (dataset_root / "preprocessing_summary.json").write_text(
            json.dumps(preprocessing_summary, indent=2),
            encoding="utf-8",
        )

    return selected_root


def _collect_available_soccernet_samples(
    soccernet_root: Path,
    frame_step: int,
) -> dict[str, list[CandidateSample]]:
    split_to_samples: dict[str, list[CandidateSample]] = {}

    for canonical_split, aliases in SOCCERNET_SPLIT_ALIASES.items():
        resolved_split_dir, resolved_alias = _resolve_existing_split_dir(soccernet_root, aliases)
        if resolved_split_dir is None:
            continue

        split_to_samples[canonical_split] = collect_soccernet_split_samples(
            split_root=resolved_split_dir,
            frame_step=frame_step,
            source_split=resolved_alias or canonical_split,
        )

    return split_to_samples


def _resolve_existing_split_dir(root: Path, aliases: tuple[str, ...]) -> tuple[Path | None, str | None]:
    for alias in aliases:
        candidate = root / alias
        if candidate.exists() and candidate.is_dir():
            return candidate, alias
    return None, None


def _write_selected_dataset(
    dataset_root: Path,
    split_to_samples: dict[str, list[CandidateSample]],
    *,
    keep_original_resolution: bool,
    frame_step: int,
    source_name: str,
) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    dataset_root.mkdir(parents=True, exist_ok=True)

    selection_summary: dict[str, object] = {
        "dataset": source_name,
        "frame_step": frame_step,
        "keep_original_resolution": keep_original_resolution,
        "splits": {},
    }

    for split in SPLITS:
        samples = sorted(split_to_samples.get(split, []), key=lambda s: s.output_file_name)
        if not samples:
            continue

        split_dir = dataset_root / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        next_image_id = 1
        next_annotation_id = 1
        skipped_images = 0
        written_images = 0

        for sample in samples:
            image = cv2.imread(str(sample.src_image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped_images += 1
                continue

            original_height, original_width = image.shape[:2]
            scale = 1.0

            if not keep_original_resolution:
                image, scale = _resize_if_needed(image)

            height, width = image.shape[:2]
            transformed_annotations = _transform_annotations(
                sample.annotations,
                scale=scale,
                width=width,
                height=height,
            )
            if not transformed_annotations:
                skipped_images += 1
                continue

            output_path = images_dir / sample.output_file_name
            cv2.imwrite(str(output_path), image)

            coco_images.append(
                {
                    "id": next_image_id,
                    "file_name": f"images/{sample.output_file_name}",
                    "width": width,
                    "height": height,
                }
            )

            for ann in transformed_annotations:
                x, y, w, h = ann.bbox_xywh
                coco_annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": next_image_id,
                        "category_id": ann.category_id,
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                next_annotation_id += 1

            next_image_id += 1
            written_images += 1

        payload = {
            "info": {
                "description": f"Selected frames dataset for {source_name}, split={split}",
                "frame_step": frame_step,
                "keep_original_resolution": keep_original_resolution,
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": COCO_CATEGORIES,
        }

        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        selection_summary["splits"][split] = {
            "images": written_images,
            "annotations": len(coco_annotations),
            "skipped_images": skipped_images,
            "source_images_seen": len(samples),
        }

    (dataset_root / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2),
        encoding="utf-8",
    )


def _resize_if_needed(image, max_dim: int = DEFAULT_MAX_DIM):
    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dim:
        return image, 1.0

    scale = max_dim / float(current_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _transform_annotations(
    annotations: tuple[CandidateAnnotation, ...],
    *,
    scale: float,
    width: int,
    height: int,
) -> list[CandidateAnnotation]:
    transformed: list[CandidateAnnotation] = []

    for ann in annotations:
        x, y, w, h = ann.bbox_xywh
        clipped_bbox = _clip_bbox(
            x=x * scale,
            y=y * scale,
            w=w * scale,
            h=h * scale,
            width=width,
            height=height,
        )
        if clipped_bbox is None:
            continue

        transformed.append(
            CandidateAnnotation(
                category_id=ann.category_id,
                bbox_xywh=clipped_bbox,
            )
        )

    return transformed


def _clip_bbox(
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if w <= 0 or h <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    return (x1, y1, clipped_w, clipped_h)


def _discover_dataset_roots(selected_root: Path) -> list[Path]:
    roots: list[Path] = []

    if any((selected_root / split).is_dir() for split in SPLITS):
        roots.append(selected_root)

    for child in sorted(selected_root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and any((child / split).is_dir() for split in SPLITS):
            roots.append(child)

    return roots


def _normalize_uint8(image):
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized.astype("uint8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None