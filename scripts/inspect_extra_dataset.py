#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(split_dir: Path) -> list[Path]:
    images = []
    for p in split_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def find_annotation_json(split_dir: Path) -> Path | None:
    json_files = list(split_dir.glob("*.json"))
    if not json_files:
        return None

    preferred = [p for p in json_files if "coco" in p.name.lower() or "annotation" in p.name.lower()]
    if preferred:
        return preferred[0]
    return json_files[0]


def summarize_split(split_dir: Path) -> dict:
    images = find_images(split_dir)
    ann_path = find_annotation_json(split_dir)

    summary = {
        "split": split_dir.name,
        "split_path": str(split_dir),
        "image_count": len(images),
        "annotation_file": str(ann_path) if ann_path else None,
        "annotation_count": 0,
        "category_count": 0,
        "categories": [],
    }

    if ann_path and ann_path.exists():
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        annotations = data.get("annotations", [])

        summary["annotation_count"] = len(annotations)
        summary["category_count"] = len(categories)
        summary["categories"] = [c.get("name", f"id_{c.get('id')}") for c in categories]

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect additional Roboflow COCO dataset.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to dataset root, e.g. data/raw/ball_extra/playersdetection_v2_coco",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    splits = ["train", "valid", "test"]
    result = {
        "dataset_root": str(dataset_root),
        "splits": {},
    }

    for split in splits:
        split_dir = dataset_root / split
        if split_dir.exists():
            result["splits"][split] = summarize_split(split_dir)
        else:
            result["splits"][split] = {
                "split": split,
                "missing": True,
            }

    output_path = dataset_root / "dataset_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())