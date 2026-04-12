#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_KEYS = {"images", "annotations", "categories"}


def find_images(split_dir: Path) -> list[Path]:
    return [p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def find_annotation_json(split_dir: Path) -> Path | None:
    json_files = [p for p in split_dir.rglob("*.json") if p.is_file()]
    if not json_files:
        return None

    preferred = [
        p
        for p in json_files
        if any(token in p.name.lower() for token in ("coco", "annotation", "annotations", "instances"))
    ]
    return preferred[0] if preferred else json_files[0]


def summarize_split(split_dir: Path, split_name: str) -> dict:
    images = find_images(split_dir)
    ann_path = find_annotation_json(split_dir)

    summary = {
        "split": split_name,
        "split_path": str(split_dir),
        "image_count": len(images),
        "annotation_file": str(ann_path) if ann_path else None,
        "annotation_count": 0,
        "category_count": 0,
        "categories": [],
        "is_coco": False,
        "missing_coco_keys": sorted(list(COCO_KEYS)),
    }

    if ann_path and ann_path.exists():
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        annotations = data.get("annotations", [])
        summary["annotation_count"] = len(annotations)
        summary["category_count"] = len(categories)
        summary["categories"] = [c.get("name", f"id_{c.get('id')}") for c in categories]

        present = {k for k in COCO_KEYS if k in data}
        summary["is_coco"] = len(present) == len(COCO_KEYS)
        summary["missing_coco_keys"] = sorted(list(COCO_KEYS - present))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect SoccerNet dataset and verify COCO-like structure.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to extracted dataset root, e.g. data/raw/soccernet/soccernet_tracking_2023_coco",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    result = {
        "dataset_root": str(dataset_root),
        "splits": {},
    }

    for split in args.splits:
        split_dir = dataset_root / split
        if split_dir.exists():
            result["splits"][split] = summarize_split(split_dir, split)
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
