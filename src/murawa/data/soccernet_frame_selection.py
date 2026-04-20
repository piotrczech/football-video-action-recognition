from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from murawa.data.bootstrap_variant import (
    CATEGORY_ID_TO_NAME,
    CandidateAnnotation,
    parse_seq_size,
    parse_tracklet_classes,
)

if TYPE_CHECKING:
    from murawa.data.frame_selection import FrameSelectionConfig

SPLITS = ("train", "valid", "test")
SOCCERNET_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val", "challenge"),
    "test": ("test",),
}
COCO_CATEGORIES = [{"id": idx, "name": name} for idx, name in CATEGORY_ID_TO_NAME.items()]
DEFAULT_MAX_DIM = 1280


def select_soccernet_frames(config: FrameSelectionConfig) -> Path:
    if config.frame_step <= 0:
        raise ValueError("frame_step must be > 0")

    raw_soccernet_root = config.raw_root / "soccernet"
    if not raw_soccernet_root.exists():
        raise FileNotFoundError(f"Missing SoccerNet raw root: {raw_soccernet_root}")

    selected_dataset_root = _resolve_selected_soccernet_root(config.selected_root)
    if selected_dataset_root.exists():
        shutil.rmtree(selected_dataset_root)
    selected_dataset_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "dataset": "soccernet",
        "selection_rule": f"every {config.frame_step}-th frame from full img1 sequence",
        "frame_step": config.frame_step,
        "keep_original_resolution": config.keep_original_resolution,
        "scope_notes": [
            "This step applies every-nth-frame selection only to SoccerNet.",
            "ball-extra is intentionally excluded because it is not a full-match source.",
            "Split boundaries are preserved; there is no re-splitting or split mixing here.",
        ],
        "splits": {},
    }

    processed_any_split = False
    for split in SPLITS:
        split_dir, resolved_alias = _resolve_split_dir(raw_soccernet_root, SOCCERNET_SPLIT_ALIASES[split])
        if split_dir is None:
            continue

        split_summary = _write_selected_split(
            split_root=split_dir,
            output_split_dir=selected_dataset_root / split,
            source_split=resolved_alias or split,
            frame_step=config.frame_step,
            keep_original_resolution=config.keep_original_resolution,
        )
        summary["splits"][split] = split_summary
        processed_any_split = True

    if not processed_any_split:
        raise FileNotFoundError(
            f"No SoccerNet splits found under: {raw_soccernet_root}. "
            "Expected train/valid/test or train/challenge/test layout."
        )

    (selected_dataset_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return selected_dataset_root


def preprocess_soccernet_selected_dataset(selected_root: Path, *, normalize: bool = False) -> Path:
    selected_dataset_root = _resolve_selected_soccernet_root(selected_root)
    if not selected_dataset_root.exists():
        raise FileNotFoundError(f"Selected SoccerNet root does not exist: {selected_dataset_root}")

    summary: dict[str, object] = {
        "dataset": "soccernet",
        "normalize": normalize,
        "scope_notes": [
            "Preprocessing stays lightweight on purpose.",
            "No heavy augmentation is applied at this stage.",
            "This step validates image files, refreshes image sizes, and clips invalid bbox coordinates.",
        ],
        "splits": {},
    }

    for split in SPLITS:
        split_dir = selected_dataset_root / split
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
                dropped_images += 1
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

        annotation_path.write_text(
            json.dumps(
                {
                    "info": info,
                    "images": kept_images,
                    "annotations": kept_annotations,
                    "categories": categories,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        summary["splits"][split] = {
            "images": len(kept_images),
            "annotations": len(kept_annotations),
            "dropped_images": dropped_images,
            "dropped_annotations": dropped_annotations,
        }

    (selected_dataset_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return selected_dataset_root


def _write_selected_split(
    *,
    split_root: Path,
    output_split_dir: Path,
    source_split: str,
    frame_step: int,
    keep_original_resolution: bool,
) -> dict[str, int]:
    images_dir = output_split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    next_image_id = 1
    next_annotation_id = 1

    sequence_dirs = sorted([p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("SNMOT-")])
    skipped_images = 0
    annotated_images = 0
    selected_images = 0

    for seq in sequence_dirs:
        seq_image_dir = seq / "img1"
        frame_paths = sorted([p for p in seq_image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
        if not frame_paths:
            continue

        frame_annotations = _load_frame_annotations(seq / "gt" / "gt.txt", seq / "gameinfo.ini")
        selected_frame_paths = frame_paths[::frame_step]

        for frame_path in selected_frame_paths:
            frame_idx = _parse_frame_index(frame_path)
            if frame_idx is None:
                skipped_images += 1
                continue

            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped_images += 1
                continue

            scale = 1.0
            if not keep_original_resolution:
                image, scale = _resize_if_needed(image)

            height, width = image.shape[:2]
            output_name = f"soccernet_{source_split}_{seq.name}_{frame_path.name}"
            output_path = images_dir / output_name
            cv2.imwrite(str(output_path), image)

            coco_images.append(
                {
                    "id": next_image_id,
                    "file_name": f"images/{output_name}",
                    "width": width,
                    "height": height,
                }
            )

            transformed_annotations = _transform_annotations(
                frame_annotations.get(frame_idx, []),
                scale=scale,
                width=width,
                height=height,
            )
            if transformed_annotations:
                annotated_images += 1

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
            selected_images += 1

    output_split_dir.mkdir(parents=True, exist_ok=True)
    (output_split_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "info": {
                    "description": f"Selected SoccerNet frames for split={output_split_dir.name}",
                    "source_split": source_split,
                    "frame_step": frame_step,
                    "keep_original_resolution": keep_original_resolution,
                },
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": COCO_CATEGORIES,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "sequences": len(sequence_dirs),
        "selected_images": selected_images,
        "annotated_images": annotated_images,
        "annotations": len(coco_annotations),
        "skipped_images": skipped_images,
    }


def _load_frame_annotations(gt_path: Path, gameinfo_path: Path) -> dict[int, list[CandidateAnnotation]]:
    if not gt_path.exists():
        return {}

    tracklet_class = parse_tracklet_classes(gameinfo_path)
    annotations_by_frame: dict[int, list[CandidateAnnotation]] = defaultdict(list)

    with gt_path.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue

            frame_idx = _safe_int(row[0])
            tracklet_id = _safe_int(row[1])
            if frame_idx is None or tracklet_id is None:
                continue

            class_id = tracklet_class.get(tracklet_id)
            if class_id is None:
                continue

            x = _safe_float(row[2])
            y = _safe_float(row[3])
            w = _safe_float(row[4])
            h = _safe_float(row[5])
            if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
                continue

            annotations_by_frame[frame_idx].append(
                CandidateAnnotation(category_id=class_id, bbox_xywh=(x, y, w, h))
            )

    return annotations_by_frame


def _resolve_split_dir(root: Path, aliases: tuple[str, ...]) -> tuple[Path | None, str | None]:
    for alias in aliases:
        candidate = root / alias
        if candidate.exists() and candidate.is_dir():
            return candidate, alias
    return None, None


def _resolve_selected_soccernet_root(selected_root: Path) -> Path:
    if selected_root.name == "soccernet":
        return selected_root
    return selected_root / "soccernet"


def _parse_frame_index(frame_path: Path) -> int | None:
    try:
        return int(frame_path.stem)
    except ValueError:
        return None


def _resize_if_needed(image, max_dim: int = DEFAULT_MAX_DIM):
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dim:
        return image, 1.0

    scale = max_dim / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _transform_annotations(
    annotations: list[CandidateAnnotation],
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


def _normalize_uint8(image):
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized.astype("uint8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None