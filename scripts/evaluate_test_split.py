#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from murawa.data.training_loader import load_training_split
from murawa.models.factory import normalize_model_name
from murawa.models.yolo import YoloAdapter
from murawa.services.artifacts import CKPT_DIR, latest_run

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model predictions on a test split.")
    p.add_argument("--model", choices=["yolo", "rf", "rfdetr"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--split", default="test", choices=["test", "valid", "train"])
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def xywh_to_xyxy(xywh: tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def quality_label(f1: float) -> str:
    if f1 >= 0.60:
        return "good"
    if f1 >= 0.35:
        return "moderate"
    return "weak"


def evaluate_yolo(
    dataset_variant: str,
    split_name: str,
    iou_threshold: float,
    max_samples: int | None,
) -> tuple[dict, Path]:
    run_name = latest_run(ROOT, "yolo", dataset_variant)
    checkpoint = ROOT / CKPT_DIR / run_name / "model.pt"

    adapter = YoloAdapter()
    split = load_training_split(ROOT, dataset_variant, split=split_name, max_samples=max_samples)

    class_names = sorted({name for name in split.class_mapping.values()})
    per_class = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_iou_sum": 0.0,
            "matched_count": 0,
        }
        for name in class_names
    }

    image_stats = []

    for sample in split.samples:
        preds = adapter.predict(sample.image_path, checkpoint_path=checkpoint, mode="frame")

        pred_by_class: dict[str, list[dict]] = defaultdict(list)
        for pred in preds:
            pred_by_class[str(pred["class"])].append(
                {
                    "bbox": [float(v) for v in pred["bbox_xyxy"]],
                    "confidence": float(pred["confidence"]),
                }
            )

        gt_by_class: dict[str, list[dict]] = defaultdict(list)
        for ann in sample.annotations:
            gt_by_class[ann.category_name].append({"bbox": xywh_to_xyxy(ann.bbox_xywh)})

        image_tp = 0
        image_fp = 0
        image_fn = 0

        all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())
        for cls in all_classes:
            gts = gt_by_class.get(cls, [])
            pds = pred_by_class.get(cls, [])

            candidate_pairs: list[tuple[float, int, int]] = []
            for gi, gt in enumerate(gts):
                for pi, pd in enumerate(pds):
                    iou = iou_xyxy(gt["bbox"], pd["bbox"])
                    if iou >= iou_threshold:
                        candidate_pairs.append((iou, gi, pi))

            candidate_pairs.sort(key=lambda item: item[0], reverse=True)

            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            for iou, gi, pi in candidate_pairs:
                if gi in matched_gt or pi in matched_pred:
                    continue
                matched_gt.add(gi)
                matched_pred.add(pi)
                if cls in per_class:
                    per_class[cls]["matched_iou_sum"] += iou
                    per_class[cls]["matched_count"] += 1

            tp = len(matched_gt)
            fp = len(pds) - len(matched_pred)
            fn = len(gts) - len(matched_gt)

            if cls not in per_class:
                per_class[cls] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "matched_iou_sum": 0.0,
                    "matched_count": 0,
                }

            per_class[cls]["tp"] += tp
            per_class[cls]["fp"] += fp
            per_class[cls]["fn"] += fn

            image_tp += tp
            image_fp += fp
            image_fn += fn

        image_stats.append(
            {
                "image": str(sample.image_path),
                "pred_count": len(preds),
                "gt_count": len(sample.annotations),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
            }
        )

    overall_tp = sum(v["tp"] for v in per_class.values())
    overall_fp = sum(v["fp"] for v in per_class.values())
    overall_fn = sum(v["fn"] for v in per_class.values())

    precision = safe_div(overall_tp, overall_tp + overall_fp)
    recall = safe_div(overall_tp, overall_tp + overall_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    per_class_report = {}
    for cls, stats in sorted(per_class.items()):
        cls_precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        cls_recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        cls_f1 = safe_div(2 * cls_precision * cls_recall, cls_precision + cls_recall)
        mean_iou = safe_div(stats["matched_iou_sum"], stats["matched_count"])
        per_class_report[cls] = {
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
            "mean_iou_of_tp": mean_iou,
        }

    images_with_any_prediction = sum(1 for s in image_stats if s["pred_count"] > 0)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "yolo",
        "dataset_variant": dataset_variant,
        "split": split_name,
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "evaluation": {
            "iou_threshold": iou_threshold,
            "images_total": len(image_stats),
            "images_with_any_prediction": images_with_any_prediction,
            "images_with_any_prediction_ratio": safe_div(
                images_with_any_prediction, len(image_stats)
            ),
            "overall": {
                "tp": overall_tp,
                "fp": overall_fp,
                "fn": overall_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "quality_label": quality_label(f1),
            },
            "per_class": per_class_report,
        },
    }

    out_dir = ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"yolo_{split_name}_eval_{run_name}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, out_path


def main() -> int:
    args = parse_args()
    model = normalize_model_name(args.model)
    if model != "yolo":
        raise SystemExit(
            "Only YOLO evaluation is supported in this script for now. Use --model yolo."
        )

    report, out_path = evaluate_yolo(
        dataset_variant=args.dataset_variant,
        split_name=args.split,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples,
    )

    overall = report["evaluation"]["overall"]
    print(f"report_path={out_path}")
    print(f"run_name={report['run_name']}")
    print(f"images_total={report['evaluation']['images_total']}")
    print(f"images_with_any_prediction={report['evaluation']['images_with_any_prediction']}")
    print(f"precision={overall['precision']:.6f}")
    print(f"recall={overall['recall']:.6f}")
    print(f"f1={overall['f1']:.6f}")
    print(f"quality_label={overall['quality_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
