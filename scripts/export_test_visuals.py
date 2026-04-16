#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from murawa.data.training_loader import load_training_split
from murawa.models.factory import normalize_model_name
from murawa.models.yolo import YoloAdapter
from murawa.services.artifacts import CKPT_DIR, latest_run

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export visual predictions for split images.")
    p.add_argument("--model", choices=["yolo", "rf", "rfdetr"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--split", default="test", choices=["test", "valid", "train"])
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = normalize_model_name(args.model)
    if model != "yolo":
        raise SystemExit(
            "Only YOLO visual export is supported in this script for now. Use --model yolo."
        )

    run_name = latest_run(ROOT, "yolo", args.dataset_variant)
    checkpoint = ROOT / CKPT_DIR / run_name / "model.pt"

    adapter = YoloAdapter()
    split = load_training_split(
        ROOT,
        args.dataset_variant,
        split=args.split,
        max_samples=args.max_samples,
    )

    out_dir = ROOT / "outputs" / "predictions_visual" / f"{run_name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_colors = {
        "player": (0, 220, 255),
        "goalkeeper": (0, 200, 120),
        "referee": (255, 80, 0),
        "ball": (0, 0, 255),
    }

    images_with_detections = 0
    all_detections = 0
    saved = 0

    for idx, sample in enumerate(split.samples, start=1):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            continue

        detections = adapter.predict(sample.image_path, checkpoint_path=checkpoint, mode="frame")
        if detections:
            images_with_detections += 1
        all_detections += len(detections)

        for det in detections:
            cls = str(det.get("class", "unknown"))
            conf = float(det.get("confidence", 0.0))
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])]
            color = class_colors.get(cls, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        header = f"run={run_name} | image={idx}/{len(split.samples)} | detections={len(detections)}"
        cv2.putText(image, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = out_dir / f"{sample.image_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), image)
        saved += 1

    summary = {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "images_total": len(split.samples),
        "images_saved": saved,
        "images_with_detections": images_with_detections,
        "images_with_detections_ratio": (
            images_with_detections / len(split.samples)
        )
        if split.samples
        else 0.0,
        "total_detections": all_detections,
        "avg_detections_per_image": (all_detections / len(split.samples)) if split.samples else 0.0,
        "output_dir": str(out_dir),
    }
    summary_path = out_dir / "visual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"run_name={run_name}")
    print(f"output_dir={out_dir}")
    print(f"summary_path={summary_path}")
    print(f"images_saved={saved}")
    print(f"images_with_detections={images_with_detections}")
    print(f"total_detections={all_detections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
