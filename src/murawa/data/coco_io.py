from __future__ import annotations

import json
from pathlib import Path

COCO_REQUIRED_KEYS = frozenset({"images", "annotations", "categories"})


class CocoIOError(RuntimeError):
    """Raised when a COCO annotation file cannot be read or validated."""


def load_coco(annotation_path: Path) -> dict:
    if not annotation_path.exists() or not annotation_path.is_file():
        raise CocoIOError(f"Missing COCO annotation file: {annotation_path}")

    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CocoIOError(f"Invalid JSON in '{annotation_path.name}': {exc}") from exc

    if not isinstance(payload, dict):
        raise CocoIOError(f"COCO payload in '{annotation_path.name}' must be a JSON object.")

    missing_keys = sorted(COCO_REQUIRED_KEYS - payload.keys())
    if missing_keys:
        raise CocoIOError(
            f"COCO payload in '{annotation_path.name}' is missing keys: {', '.join(missing_keys)}."
        )

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise CocoIOError(
            f"COCO payload in '{annotation_path.name}' must contain list fields: "
            "images, annotations, categories."
        )

    return payload


def load_coco_split(variant_dir: Path, split: str) -> tuple[dict, Path, Path]:
    split_dir = variant_dir / split
    if not split_dir.exists() or not split_dir.is_dir():
        raise CocoIOError(f"Split directory '{split}' does not exist under '{variant_dir}'.")

    annotation_path = split_dir / "_annotations.coco.json"
    payload = load_coco(annotation_path)
    return payload, annotation_path, split_dir


def save_coco(annotation_path: Path, payload: dict) -> None:
    if not isinstance(payload, dict) or not COCO_REQUIRED_KEYS.issubset(payload):
        raise CocoIOError(
            f"Cannot save invalid COCO payload to '{annotation_path}'. "
            f"Required keys: {sorted(COCO_REQUIRED_KEYS)}."
        )
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
