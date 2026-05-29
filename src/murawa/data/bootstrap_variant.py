from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from murawa.settings import (
    BALL_CATEGORY_ID,
    BOOTSTRAP_SPLIT_RATIOS,
    BOOTSTRAP_VALID_FROM_TRAIN_RATIO,
    CATEGORY_ID_TO_NAME,
    CATEGORY_NAME_TO_ID,
    DATA_RAW,
    DATA_READY,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_STEP,
    DEFAULT_FRAME_WIDTH,
    SPLITS,
)

LOGGER = logging.getLogger("bootstrap-base-variant")
SUPPORTED_VARIANTS = ("base", "extended")

SOCCERNET_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}
BALL_EXTRA_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}

VARIANT_BALL_EXTRA_POLICY: dict[str, dict[str, bool]] = {
    "base": {"train": False, "valid": False, "test": False},
    "extended": {"train": True, "valid": True, "test": True},
}

class BootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class BootstrapConfig:
    output_variant: str = "base"
    frame_step: int = DEFAULT_FRAME_STEP
    force: bool = False


@dataclass(frozen=True)
class BootstrapBuildResult:
    variant_dir: Path
    split_counts: dict[str, int]
    used_fallback: bool


@dataclass(frozen=True)
class CandidateAnnotation:
    category_id: int
    bbox_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class CandidateSample:
    source: str
    source_split: str
    src_image_path: Path
    output_file_name: str
    width: int
    height: int
    annotations: tuple[CandidateAnnotation, ...]

    @property
    def has_ball(self) -> bool:
        return any(ann.category_id == BALL_CATEGORY_ID for ann in self.annotations)


def build_bootstrap_variant(project_root: Path, config: BootstrapConfig) -> BootstrapBuildResult:
    _validate_config(config)

    raw_soccernet_root = project_root / DATA_RAW / "soccernet"
    raw_ball_extra_root = project_root / DATA_RAW / "ball-extra"

    soccernet_by_split, used_fallback = collect_soccernet_samples_by_split(
        root=raw_soccernet_root,
        frame_step=config.frame_step,
    )
    split_to_samples = {split: list(soccernet_by_split[split]) for split in SPLITS}

    ball_extra_policy = VARIANT_BALL_EXTRA_POLICY[config.output_variant]
    if any(ball_extra_policy.values()):
        ball_extra_by_split = collect_ball_extra_samples_by_split(root=raw_ball_extra_root)
        for split in SPLITS:
            if not ball_extra_policy[split]:
                continue
            required_samples = ball_extra_by_split[split]
            if not required_samples:
                raise BootstrapError(
                    f"Variant '{config.output_variant}' requires ball-extra split '{split}', "
                    "but no valid samples were found."
                )
            split_to_samples[split].extend(required_samples)

    for split in SPLITS:
        split_to_samples[split] = sorted(split_to_samples[split], key=lambda s: s.output_file_name)
        if not split_to_samples[split]:
            raise BootstrapError(f"Split '{split}' is empty for variant '{config.output_variant}'.")

    variant_dir = project_root / DATA_READY / config.output_variant
    prepare_output_root(variant_dir=variant_dir, force=config.force)
    write_variant(variant_dir=variant_dir, split_to_samples=split_to_samples)
    write_summary(
        variant_dir=variant_dir,
        split_to_samples=split_to_samples,
        frame_step=config.frame_step,
        used_fallback=used_fallback,
        soccernet_split_strategy=describe_soccernet_split_strategy(
            soccernet_by_split=soccernet_by_split,
            used_train_only_fallback=used_fallback,
        ),
        variant_composition=describe_variant_composition(config.output_variant),
    )
    return BootstrapBuildResult(
        variant_dir=variant_dir,
        split_counts={split: len(split_to_samples[split]) for split in SPLITS},
        used_fallback=used_fallback,
    )


def collect_soccernet_samples_by_split(
    root: Path,
    frame_step: int,
) -> tuple[dict[str, list[CandidateSample]], bool]:
    if not root.exists():
        raise BootstrapError(f"SoccerNet path does not exist: {root}")

    train_dir, _ = _resolve_split_dir(root, SOCCERNET_SPLIT_ALIASES["train"])
    valid_dir, valid_alias = _resolve_split_dir(root, SOCCERNET_SPLIT_ALIASES["valid"])
    test_dir, test_alias = _resolve_split_dir(root, SOCCERNET_SPLIT_ALIASES["test"])

    if train_dir is None:
        raise BootstrapError(f"Missing SoccerNet split 'train' under: {root}")

    train_samples = collect_soccernet_split_samples(
        split_root=train_dir,
        frame_step=frame_step,
        source_split="train",
    )

    if valid_dir is not None and test_dir is not None:
        valid_samples = collect_soccernet_split_samples(
            split_root=valid_dir,
            frame_step=frame_step,
            source_split=valid_alias or "valid",
        )
        test_samples = collect_soccernet_split_samples(
            split_root=test_dir,
            frame_step=frame_step,
            source_split=test_alias or "test",
        )
        return {
            "train": train_samples,
            "valid": valid_samples,
            "test": test_samples,
        }, False

    if test_dir is not None:
        test_samples = collect_soccernet_split_samples(
            split_root=test_dir,
            frame_step=frame_step,
            source_split=test_alias or "test",
        )
        train_split, valid_split = split_train_valid_soccernet_samples(train_samples)
        return {
            "train": train_split,
            "valid": valid_split,
            "test": test_samples,
        }, False

    return split_train_only_soccernet_samples(train_samples), True


def split_train_valid_soccernet_samples(
    train_samples: list[CandidateSample],
) -> tuple[list[CandidateSample], list[CandidateSample]]:
    samples = sorted(train_samples, key=lambda s: s.output_file_name)
    total = len(samples)
    if total < 2:
        raise BootstrapError(
            "SoccerNet train/test layout requires at least 2 samples in 'train' "
            "to derive a validation split."
        )

    valid_count = int(total * BOOTSTRAP_VALID_FROM_TRAIN_RATIO)
    if valid_count == 0:
        valid_count = 1

    train_count = total - valid_count
    if train_count <= 0:
        raise BootstrapError("SoccerNet train/test layout produced non-positive train split size.")

    return samples[:train_count], samples[train_count:]


def split_train_only_soccernet_samples(train_samples: list[CandidateSample]) -> dict[str, list[CandidateSample]]:
    samples = sorted(train_samples, key=lambda s: s.output_file_name)
    total = len(samples)
    if total < 3:
        raise BootstrapError(
            "SoccerNet fallback requires at least 3 samples in 'train' to create train/valid/test splits."
        )

    train_ratio = BOOTSTRAP_SPLIT_RATIOS["train"]
    valid_ratio = BOOTSTRAP_SPLIT_RATIOS["valid"]
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - train_count - valid_count

    if valid_count == 0:
        valid_count = 1
        train_count -= 1
    if test_count == 0:
        test_count = 1
        train_count -= 1

    if train_count <= 0:
        raise BootstrapError("SoccerNet fallback produced non-positive train split size.")

    train_split = samples[:train_count]
    valid_split = samples[train_count : train_count + valid_count]
    test_split = samples[train_count + valid_count : train_count + valid_count + test_count]

    return {
        "train": train_split,
        "valid": valid_split,
        "test": test_split,
    }


def collect_soccernet_split_samples(split_root: Path, frame_step: int, source_split: str) -> list[CandidateSample]:
    if not split_root.exists():
        return []

    grouped: dict[str, list[CandidateAnnotation]] = defaultdict(list)
    sample_meta: dict[str, tuple[Path, int, int]] = {}
    sequence_dirs = sorted([p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("SNMOT-")])

    for seq in sequence_dirs:
        tracklet_class = parse_tracklet_classes(seq / "gameinfo.ini")
        width, height = parse_seq_size(seq / "seqinfo.ini")
        gt_path = seq / "gt" / "gt.txt"
        if not gt_path.exists():
            continue

        with gt_path.open(encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 6:
                    continue

                frame_idx = _safe_int(row[0])
                tracklet_id = _safe_int(row[1])
                if frame_idx is None or tracklet_id is None:
                    continue
                if (frame_idx - 1) % frame_step != 0:
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

                src_image = seq / "img1" / f"{frame_idx:06d}.jpg"
                if not src_image.exists():
                    continue

                output_name = f"soccernet_{source_split}_{seq.name}_{frame_idx:06d}.jpg"
                grouped[output_name].append(CandidateAnnotation(category_id=class_id, bbox_xywh=(x, y, w, h)))
                sample_meta[output_name] = (src_image, width, height)

    samples: list[CandidateSample] = []
    for output_name in sorted(grouped.keys()):
        annotations = tuple(grouped[output_name])
        if not annotations:
            continue

        src_image, width, height = sample_meta[output_name]
        samples.append(
            CandidateSample(
                source="soccernet",
                source_split=source_split,
                src_image_path=src_image,
                output_file_name=output_name,
                width=width,
                height=height,
                annotations=annotations,
            )
        )

    return samples


def collect_ball_extra_samples_by_split(root: Path) -> dict[str, list[CandidateSample]]:
    if not root.exists():
        raise BootstrapError(f"ball-extra path does not exist: {root}")

    split_to_samples: dict[str, list[CandidateSample]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        split_dir, split_alias = _resolve_split_dir(root, BALL_EXTRA_SPLIT_ALIASES[split])
        if split_dir is None:
            continue

        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            continue

        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_map: dict[int, int] = {}
        for category in payload.get("categories", []):
            if not isinstance(category, dict) or not isinstance(category.get("id"), int):
                continue
            raw_name = str(category.get("name", "")).strip().lower()
            mapped = _map_ball_extra_category(raw_name)
            if mapped is None:
                continue
            category_map[category["id"]] = mapped

        ann_by_image: dict[int, list[CandidateAnnotation]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            image_id = annotation.get("image_id")
            raw_category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if not isinstance(image_id, int) or raw_category_id not in category_map:
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x, y, w, h = (_safe_float(bbox[0]), _safe_float(bbox[1]), _safe_float(bbox[2]), _safe_float(bbox[3]))
            if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
                continue

            ann_by_image[image_id].append(
                CandidateAnnotation(category_id=category_map[raw_category_id], bbox_xywh=(x, y, w, h))
            )

        collected: list[CandidateSample] = []
        for image in payload.get("images", []):
            if not isinstance(image, dict):
                continue

            image_id = image.get("id")
            file_name = image.get("file_name")
            width = image.get("width")
            height = image.get("height")
            if not isinstance(image_id, int) or not isinstance(file_name, str):
                continue
            if not isinstance(width, int) or not isinstance(height, int):
                continue

            annotations = tuple(ann_by_image.get(image_id, []))
            if not annotations:
                continue

            src_image = split_dir / file_name
            if not src_image.exists():
                continue

            safe_name = file_name.replace("/", "_")
            collected.append(
                CandidateSample(
                    source="ball-extra",
                    source_split=split_alias or split,
                    src_image_path=src_image,
                    output_file_name=f"ballextra_{split}_{safe_name}",
                    width=width,
                    height=height,
                    annotations=annotations,
                )
            )

        split_to_samples[split] = sorted(collected, key=lambda s: s.output_file_name)

    return split_to_samples


def prepare_output_root(variant_dir: Path, force: bool) -> None:
    if variant_dir.exists():
        if not force:
            raise BootstrapError(f"Output variant already exists: {variant_dir}. Use --force to overwrite it.")
        shutil.rmtree(variant_dir)

    for split in SPLITS:
        (variant_dir / split / "images").mkdir(parents=True, exist_ok=True)


def write_variant(variant_dir: Path, split_to_samples: dict[str, list[CandidateSample]]) -> None:
    for split in SPLITS:
        split_dir = variant_dir / split
        images_dir = split_dir / "images"
        coco_images: list[dict] = []
        coco_annotations: list[dict] = []
        next_image_id, next_annotation_id = 1, 1

        for sample in split_to_samples[split]:
            shutil.copy2(sample.src_image_path, images_dir / sample.output_file_name)
            coco_images.append(
                {
                    "id": next_image_id,
                    "file_name": f"images/{sample.output_file_name}",
                    "width": sample.width,
                    "height": sample.height,
                }
            )
            for ann in sample.annotations:
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

        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "info": {"description": f"Stage-2 dataset variant '{variant_dir.name}'."},
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": [{"id": idx, "name": name} for idx, name in CATEGORY_ID_TO_NAME.items()],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def write_summary(
    variant_dir: Path,
    split_to_samples: dict[str, list[CandidateSample]],
    frame_step: int,
    used_fallback: bool,
    soccernet_split_strategy: str,
    variant_composition: dict[str, list[str]],
) -> None:
    summary: dict[str, object] = {
        "variant": variant_dir.name,
        "frame_step": frame_step,
        "used_soccernet_train_only_fallback": used_fallback,
        "soccernet_split_strategy": soccernet_split_strategy,
        "variant_composition": variant_composition,
        "notes": [
            "TODO(Issue #8): Extend this bootstrap with full frame selection and preprocessing pipeline.",
            "TODO(Issue #9): Keep explicit variant naming and traceability in data/ready.",
            "Stage-2 bootstrap preserves the official SoccerNet test split when it is labeled.",
            "When SoccerNet has no labeled valid split, validation is derived from train.",
        ],
        "splits": {},
    }

    for split in SPLITS:
        source_counts: dict[str, int] = defaultdict(int)
        class_counts: dict[str, int] = defaultdict(int)
        source_split_counts: dict[str, int] = defaultdict(int)
        ball_images = 0

        for sample in split_to_samples[split]:
            source_counts[sample.source] += 1
            source_split_counts[f"{sample.source}:{sample.source_split}"] += 1
            if sample.has_ball:
                ball_images += 1
            for ann in sample.annotations:
                class_counts[CATEGORY_ID_TO_NAME[ann.category_id]] += 1

        summary["splits"][split] = {
            "images": len(split_to_samples[split]),
            "ball_images": ball_images,
            "source_image_counts": dict(sorted(source_counts.items())),
            "source_split_image_counts": dict(sorted(source_split_counts.items())),
            "annotation_class_counts": dict(sorted(class_counts.items())),
        }

    (variant_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def describe_variant_composition(variant_name: str) -> dict[str, list[str]]:
    policy = VARIANT_BALL_EXTRA_POLICY[variant_name]
    composition: dict[str, list[str]] = {}
    for split in SPLITS:
        sources = ["soccernet"]
        if policy[split]:
            sources.append("ball-extra")
        composition[split] = sources
    return composition


def describe_soccernet_split_strategy(
    soccernet_by_split: dict[str, list[CandidateSample]],
    used_train_only_fallback: bool,
) -> str:
    if used_train_only_fallback:
        return "train_only_80_10_10"

    valid_source_splits = {
        sample.source_split
        for sample in soccernet_by_split.get("valid", [])
        if sample.source == "soccernet"
    }
    test_source_splits = {
        sample.source_split
        for sample in soccernet_by_split.get("test", [])
        if sample.source == "soccernet"
    }

    if valid_source_splits == {"train"} and test_source_splits == {"test"}:
        return "train_valid_from_train_official_test"

    if valid_source_splits <= {"valid", "val"} and test_source_splits == {"test"}:
        return "provided_train_valid_test"

    return "custom"


def parse_tracklet_classes(gameinfo_path: Path) -> dict[int, int]:
    if not gameinfo_path.exists():
        return {}

    role_map: dict[int, int] = {}
    pattern = re.compile(r"^trackletID_(\d+)\s*=\s*(.+)$")
    for raw_line in gameinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue

        tracklet_id = int(match.group(1))
        descriptor = match.group(2).strip().lower()

        if descriptor.startswith("other;"):
            continue
        if "referee" in descriptor:
            role_map[tracklet_id] = CATEGORY_NAME_TO_ID["referee"]
        elif "goalkeeper" in descriptor or "goalkeepers" in descriptor:
            role_map[tracklet_id] = CATEGORY_NAME_TO_ID["goalkeeper"]
        elif "ball" in descriptor:
            role_map[tracklet_id] = CATEGORY_NAME_TO_ID["ball"]
        elif "player" in descriptor:
            role_map[tracklet_id] = CATEGORY_NAME_TO_ID["player"]

    return role_map


def parse_seq_size(seqinfo_path: Path) -> tuple[int, int]:
    width, height = DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT
    if not seqinfo_path.exists():
        return width, height

    for raw_line in seqinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("imWidth="):
            maybe_width = _safe_int(line.split("=", 1)[1])
            if maybe_width:
                width = maybe_width
        elif line.startswith("imHeight="):
            maybe_height = _safe_int(line.split("=", 1)[1])
            if maybe_height:
                height = maybe_height

    return width, height


def _resolve_split_dir(root: Path, aliases: tuple[str, ...]) -> tuple[Path | None, str | None]:
    for alias in aliases:
        candidate = root / alias
        if candidate.exists() and candidate.is_dir():
            return candidate, alias
    return None, None


def _map_ball_extra_category(raw_name: str) -> int | None:
    if not raw_name:
        return None
    if "goalkeeper" in raw_name:
        return CATEGORY_NAME_TO_ID.get("goalkeeper")
    if "referee" in raw_name:
        return CATEGORY_NAME_TO_ID.get("referee")
    if "ball" in raw_name:
        return CATEGORY_NAME_TO_ID.get("ball")
    if "player" in raw_name:
        return CATEGORY_NAME_TO_ID.get("player")
    return None


def _validate_config(config: BootstrapConfig) -> None:
    if config.output_variant not in SUPPORTED_VARIANTS:
        raise BootstrapError(
            f"Unsupported variant '{config.output_variant}'. Expected one of: {', '.join(SUPPORTED_VARIANTS)}"
        )
    if config.frame_step <= 0:
        raise BootstrapError("frame_step must be > 0.")


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


__all__ = [
    "BootstrapBuildResult",
    "BootstrapConfig",
    "BootstrapError",
    "CandidateAnnotation",
    "CandidateSample",
    "SPLITS",
    "SUPPORTED_VARIANTS",
    "build_bootstrap_variant",
    "describe_variant_composition",
]
