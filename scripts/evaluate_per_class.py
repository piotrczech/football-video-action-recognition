#!/usr/bin/env python3
"""
evaluate_per_class.py — Per-class P/R/F1 and COCO mAP evaluation for trained runs.

Runs inference on the validation split of each requested run and writes two sets
of metrics to models/metadata/<run_name>/per_class_metrics.json:

  1. Per-class P/R/F1 at a fixed conf=0.25, IoU=0.5 threshold (greedy matching).
     Used for cross-model F1 comparison — identical pipeline for YOLO and RF-DETR.

  2. COCO-style mAP@0.5 and mAP@0.5:0.95 per class and overall, computed via
     faster-coco-eval (already in requirements.txt). This replaces the framework-
     internal mAP values from metrics_summary.json, which use different NMS
     implementations and confidence thresholds for each model and are therefore
     NOT cross-model comparable. The COCO evaluator sweeps all confidence thresholds
     to build the full P-R curve — conf_threshold is NOT applied here, so the
     result is independent of the conf parameter used for P/R/F1.

Evaluation consistency note (extended variants)
------------------------------------------------
For runs trained on `extended` or `extended-transformed`, the validation split
contains images from two sources:
  - soccernet_*  (SoccerNet frames — fully annotated for all four classes)
  - ballextra_*  (ball-extra frames — ball annotations only; player/goalkeeper/
                  referee are not labelled even where visible)

Including ball-extra images causes spuriously low precision for non-ball classes.
Both the P/R/F1 and COCO mAP calculations therefore run on the soccernet_* subset
only for extended variants. The soccernet image count is the same as in the base
variant, so cross-variant comparisons remain valid.

Usage
-----
# Default — evaluate the four main comparison runs:
python scripts/evaluate_per_class.py

# Specific runs:
python scripts/evaluate_per_class.py --runs yolo-base-full-v2 rfdetr-base-full-v2

# All eight runs:
python scripts/evaluate_per_class.py --all-runs

# Force re-evaluation even if results already exist:
python scripts/evaluate_per_class.py --force

# Quick sanity check (first 50 images only):
python scripts/evaluate_per_class.py --max-images 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

from murawa.data import load_training_split
from murawa.models import build_training_adapter, normalize_model_name
from murawa.services.runtime.artifacts import resolve_run
from murawa.settings import MODELS_METADATA, PROJECT_ROOT

ROOT = PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("evaluate_per_class")


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_RUNS = [
    "yolo-base-full-v2",
    "rfdetr-base-full-v2",
    "yolo-extended-full-v2",
    "rfdetr-extended-full-v2",
]

ALL_RUNS = [
    "yolo-base-full-v2",
    "yolo-base-transformed-v2",
    "yolo-extended-full-v2",
    "yolo-extended-transformed-v2",
    "rfdetr-base-full-v2",
    "rfdetr-base-transformed-v2",
    "rfdetr-extended-full-v2",
    "rfdetr-extended-transformed-v2",
]

# Validation split includes ball-extra images for these variants; exclude them
# so that non-ball class metrics are not deflated by missing annotations.
EXTENDED_VARIANTS = {"extended", "extended-transformed"}

# Filename prefix for ball-extra images (set by bootstrap_base_variant.py)
BALL_EXTRA_PREFIX = "ballextra_"

OUTPUT_FILENAME = "per_class_metrics.json"
SCHEMA_VERSION = 3   # bumped: adds coco_map50 / coco_map5095 fields


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-class P/R/F1 and COCO mAP on the validation split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--runs", nargs="+", default=None, metavar="RUN_NAME")
    p.add_argument("--all-runs", action="store_true",
                   help="Evaluate all eight standard runs.")
    p.add_argument("--iou-threshold", type=float, default=0.5, metavar="FLOAT",
                   help="IoU threshold for P/R/F1 greedy matching (default: 0.5).")
    p.add_argument("--conf-threshold", type=float, default=0.25, metavar="FLOAT",
                   help="Min confidence for P/R/F1 matching (default: 0.25). "
                        "Not used for COCO mAP — that sweeps all thresholds.")
    p.add_argument("--max-images", type=int, default=None, metavar="N",
                   help="Limit to first N images (default: all).")
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate and overwrite existing results.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_ball_extra_image(image_path: Path) -> bool:
    return image_path.name.lower().startswith(BALL_EXTRA_PREFIX)


def box_iou(a: list, b: list) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ── COCO mAP computation ──────────────────────────────────────────────────────

def build_coco_gt(samples, class_mapping: dict[int, str]) -> COCO:
    """
    Build a COCO ground-truth object from LoadedSample instances.

    faster-coco-eval accepts a pre-loaded dataset dict — no JSON file needed.
    """
    # COCO categories: map our internal int IDs → sequential 1-based IDs
    # (COCO convention; we keep our IDs since they already start at 0 or 1)
    categories = [
        {"id": cat_id, "name": name}
        for cat_id, name in class_mapping.items()
    ]
    cat_id_set = set(class_mapping.keys())

    images = []
    annotations = []
    ann_id = 1

    for sample in samples:
        images.append({
            "id": sample.image_id,
            "width": sample.width,
            "height": sample.height,
        })
        for ann in sample.annotations:
            if ann.category_id not in cat_id_set:
                continue
            x, y, w, h = ann.bbox_xywh
            annotations.append({
                "id": ann_id,
                "image_id": sample.image_id,
                "category_id": ann.category_id,
                "bbox": [x, y, w, h],          # COCO format: [x, y, w, h]
                "area": ann.area if ann.area > 0 else float(w * h),
                "iscrowd": ann.iscrowd,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()
    return coco_gt


def build_coco_dt(all_detections: list[tuple], class_name_to_id: dict[str, int]) -> list[dict]:
    """
    Flatten per-image detections into a COCO-format predictions list.

    all_detections: list of (image_id, detections) where detections is the
    raw list returned by adapter.predict_frames for that image.
    """
    dt_list = []
    for image_id, detections in all_detections:
        for det in detections:
            cat_id = class_name_to_id.get(det.get("class"))
            if cat_id is None:
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            dt_list.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],        # COCO format: [x, y, w, h]
                "score": float(det.get("confidence", 0.0)),
            })
    return dt_list


def run_coco_eval(
    coco_gt: COCO,
    dt_list: list[dict],
    class_mapping: dict[int, str],
) -> dict:
    """
    Run faster-coco-eval and return per-class and overall AP50 / AP50-95.

    Returns a dict shaped like:
    {
        "overall": {"mAP50": float, "mAP5095": float},
        "per_class": {
            "player":     {"AP50": float, "AP5095": float},
            "goalkeeper": {"AP50": float, "AP5095": float},
            ...
        }
    }
    """
    if not dt_list:
        # No predictions at all — return zeros
        zero = {"AP50": 0.0, "AP5095": 0.0}
        return {
            "overall": {"mAP50": 0.0, "mAP5095": 0.0},
            "per_class": {name: dict(zero) for name in class_mapping.values()},
        }

    coco_dt = coco_gt.loadRes(dt_list)

    ev = COCOeval_faster(
        coco_gt,
        coco_dt,
        iouType="bbox",
        separate_eval=True,   # needed to access per-category precision tensor
        extra_calc=True,
    )
    # iouThrs defaults to [0.50, 0.55, …, 0.95] — exactly the COCO standard
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # ev.stats: [mAP@.50:.95, mAP@.50, mAP@.75, mAP-S, mAP-M, mAP-L, ...]
    overall_map5095 = float(ev.stats[0])
    overall_map50   = float(ev.stats[1])

    # Per-class AP from the precision tensor
    # Shape: [T=10 IoU thresholds, R=101 recall points, K=num_cats, A=4 area ranges, M=3 max-dets]
    # We want: area=all (index 0), maxDets=100 (index -1)
    prec = ev.eval["precision"]
    cat_ids = list(ev.params.catIds)     # ordered list of category IDs

    per_class = {}
    for ki, cat_id in enumerate(cat_ids):
        cat_name = class_mapping.get(int(cat_id), str(cat_id))

        # AP@0.5: IoU threshold index 0 (0.50)
        p_at_50 = prec[0, :, ki, 0, -1]
        valid_50 = p_at_50[p_at_50 >= 0]
        ap50 = float(np.mean(valid_50)) if len(valid_50) > 0 else 0.0

        # AP@0.5:0.95: mean over all 10 IoU thresholds
        p_all = prec[:, :, ki, 0, -1]
        valid_all = p_all[p_all >= 0]
        ap5095 = float(np.mean(valid_all)) if len(valid_all) > 0 else 0.0

        per_class[cat_name] = {
            "AP50":   round(ap50,   6),
            "AP5095": round(ap5095, 6),
        }

    return {
        "overall": {
            "mAP50":   round(overall_map50,   6),
            "mAP5095": round(overall_map5095, 6),
        },
        "per_class": per_class,
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_run(
    run_name: str,
    *,
    iou_threshold: float,
    conf_threshold: float,
    max_images: int | None,
) -> dict:
    logger.info("[%s] Loading run ...", run_name)
    run = resolve_run(ROOT, run_name)

    dv = run.dataset_variant
    variant_str: str = dv if isinstance(dv, str) else dv.get("dataset_variant", "base")

    logger.info("[%s] Loading validation split (variant=%s) ...", run_name, variant_str)
    split = load_training_split(
        project_root=ROOT,
        dataset_variant=variant_str,
        split="valid",
    )

    class_mapping: dict[int, str] = split.class_mapping          # {cat_id: name}
    class_names:   list[str]      = list(class_mapping.values())
    class_name_to_id = {v: k for k, v in class_mapping.items()}  # {name: cat_id}

    # Filter ball-extra images for extended variants
    all_samples = list(split.samples)
    if variant_str in EXTENDED_VARIANTS:
        soccernet = [s for s in all_samples if not is_ball_extra_image(s.image_path)]
        logger.info(
            "[%s] Extended variant: excluding %d ball-extra images (%d soccernet retained).",
            run_name, len(all_samples) - len(soccernet), len(soccernet),
        )
        all_samples = soccernet

    samples = all_samples[:max_images] if max_images else all_samples

    logger.info(
        "[%s] Evaluating %d images (model=%s, iou=%.2f, conf=%.2f) ...",
        run_name, len(samples), run.model, iou_threshold, conf_threshold,
    )

    adapter = build_training_adapter(normalize_model_name(run.model))
    frame_paths = [s.image_path for s in samples]

    # Collect everything in one inference pass
    # P/R/F1 accumulators (greedy matching at fixed conf + iou threshold)
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    # COCO: accumulate all predictions across all images
    all_detections_for_coco: list[tuple] = []

    t0 = time.perf_counter()

    for idx, (sample, detections) in enumerate(
        zip(samples, adapter.predict_frames(
            frame_paths=frame_paths,
            checkpoint_path=run.checkpoint_path,
        )),
        start=1,
    ):
        if idx % 100 == 0:
            elapsed = time.perf_counter() - t0
            fps = idx / elapsed
            remaining = (len(samples) - idx) / fps if fps > 0 else 0
            logger.info(
                "[%s] %d/%d  (%.1f img/s, ~%.0fs remaining)",
                run_name, idx, len(samples), fps, remaining,
            )

        # ── P/R/F1: greedy matching at fixed thresholds ───────────────────────
        gt_by_class: dict[str, list] = defaultdict(list)
        for ann in sample.annotations:
            cls = class_mapping.get(ann.category_id)
            if cls is None:
                continue
            x, y, w, h = ann.bbox_xywh
            gt_by_class[cls].append([x, y, x + w, y + h])

        pred_by_class: dict[str, list] = defaultdict(list)
        for det in detections:
            if det.get("confidence", 0.0) >= conf_threshold:
                pred_by_class[det["class"]].append(
                    (det["confidence"], det["bbox_xyxy"])
                )

        for cls in class_names:
            gts   = gt_by_class[cls]
            preds = sorted(pred_by_class[cls], key=lambda x: -x[0])
            matched: set[int] = set()
            for conf, pred_box in preds:
                best_iou, best_gi = 0.0, -1
                for gi, gt_box in enumerate(gts):
                    if gi in matched:
                        continue
                    iou = box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_threshold and best_gi >= 0:
                    tp[cls] += 1
                    matched.add(best_gi)
                else:
                    fp[cls] += 1
            fn[cls] += len(gts) - len(matched)

        # ── COCO mAP: collect ALL detections (no conf filter — evaluator handles it) ──
        all_detections_for_coco.append((sample.image_id, detections))

    elapsed_total = time.perf_counter() - t0
    logger.info("[%s] Inference done in %.1fs.", run_name, elapsed_total)

    # ── Compute P/R/F1 ────────────────────────────────────────────────────────
    per_class_prf: dict[str, dict] = {}
    for cls in class_names:
        t = tp[cls]; f = fp[cls]; n = fn[cls]
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall    = t / (t + n) if (t + n) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_prf[cls] = {
            "precision": round(precision, 6),
            "recall":    round(recall,    6),
            "f1":        round(f1,        6),
            "tp": t, "fp": f, "fn": n,
            "support": t + n,
        }
        logger.info(
            "[%s] %-12s  P=%.3f  R=%.3f  F1=%.3f  support=%d",
            run_name, cls, precision, recall, f1, t + n,
        )

    # ── Compute COCO mAP ──────────────────────────────────────────────────────
    logger.info("[%s] Running COCO mAP evaluation ...", run_name)
    coco_gt  = build_coco_gt(samples, class_mapping)
    dt_list  = build_coco_dt(all_detections_for_coco, class_name_to_id)
    coco_result = run_coco_eval(coco_gt, dt_list, class_mapping)

    logger.info(
        "[%s] COCO mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",
        run_name,
        coco_result["overall"]["mAP50"],
        coco_result["overall"]["mAP5095"],
    )
    for cls, ap in coco_result["per_class"].items():
        logger.info(
            "[%s] %-12s  AP@50=%.3f  AP@50:95=%.3f",
            run_name, cls, ap["AP50"], ap["AP5095"],
        )

    # Merge AP values into per_class dict for convenient notebook access
    for cls in class_names:
        ap = coco_result["per_class"].get(cls, {"AP50": 0.0, "AP5095": 0.0})
        per_class_prf[cls]["AP50"]   = ap["AP50"]
        per_class_prf[cls]["AP5095"] = ap["AP5095"]

    return {
        "schema_version":   SCHEMA_VERSION,
        "run_name":         run_name,
        "model":            run.model,
        "dataset_variant":  variant_str,
        "iou_threshold":    iou_threshold,
        "conf_threshold":   conf_threshold,
        "images_evaluated": len(samples),
        "images_total":     len(split.samples),
        "elapsed_seconds":  round(elapsed_total, 1),
        # Cross-model comparable mAP (shared evaluator, full P-R curve sweep)
        "coco_map50":       coco_result["overall"]["mAP50"],
        "coco_map5095":     coco_result["overall"]["mAP5095"],
        # Per-class P/R/F1 + AP
        "per_class":        per_class_prf,
    }


# ── Output helpers ────────────────────────────────────────────────────────────

def output_path(run_name: str) -> Path:
    return ROOT / MODELS_METADATA / run_name / OUTPUT_FILENAME


def already_evaluated(run_name: str) -> bool:
    path = output_path(run_name)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return data.get("schema_version") == SCHEMA_VERSION
    except Exception:
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    runs = ALL_RUNS if args.all_runs else (args.runs or DEFAULT_RUNS)

    logger.info("Runs: %s", runs)
    logger.info(
        "Settings: iou=%.2f  conf=%.2f  max_images=%s  force=%s",
        args.iou_threshold, args.conf_threshold,
        args.max_images or "all", args.force,
    )

    skipped, succeeded, failed = [], [], []

    for run_name in runs:
        if not args.force and already_evaluated(run_name):
            logger.info("[%s] Already at schema v%d — skipping (--force to re-run).",
                        run_name, SCHEMA_VERSION)
            skipped.append(run_name)
            continue

        try:
            result = evaluate_run(
                run_name,
                iou_threshold=args.iou_threshold,
                conf_threshold=args.conf_threshold,
                max_images=args.max_images,
            )
            out = output_path(run_name)
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info("[%s] Written to %s", run_name, out)
            succeeded.append(run_name)

        except FileNotFoundError as exc:
            logger.error("[%s] Run not found: %s", run_name, exc)
            failed.append(run_name)
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", run_name, exc)
            failed.append(run_name)

    logger.info("─" * 60)
    logger.info("Done.  succeeded=%d  skipped=%d  failed=%d",
                len(succeeded), len(skipped), len(failed))
    if failed:
        logger.error("Failed: %s", failed)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
