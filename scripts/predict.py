#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

from murawa.settings import DEFAULT_SAMPLE_FPS, MIN_SAMPLE_FPS, MAX_SAMPLE_FPS, PROJECT_ROOT
from murawa.services.analysis.pipeline import analyze_frame, analyze_match


ROOT = PROJECT_ROOT
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prediction entrypoint.")
    p.add_argument("--model", choices=["yolo", "rfdetr", "rf"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--mode", choices=["frame", "match"], default="match")
    p.add_argument(
        "--input-path",
        default=None,
        help="Optional input path: image for mode=frame, video for mode=match.",
    )
    p.add_argument(
        "--sample-fps",
        type=int,
        default=DEFAULT_SAMPLE_FPS,
        help=f"Frame sampling rate for mode=match. Expected range: {MIN_SAMPLE_FPS}-{MAX_SAMPLE_FPS}.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "frame":
        result = analyze_frame(
            project_root=ROOT,
            model=args.model,
            dataset_variant=args.dataset_variant,
            input_path=args.input_path,
        )
    else:
        result = analyze_match(
            project_root=ROOT,
            model=args.model,
            dataset_variant=args.dataset_variant,
            input_path=args.input_path,
            sample_fps=args.sample_fps,
        )

    if result["status"] != "ok":
        logger.error(result.get("message", "Prediction failed."))
        return 1

    logger.info("mode=%s", result["mode"])
    logger.info("summary=%s", result["summary_path"])
    logger.info("output=%s", result["output_dir"])
    logger.info("detections=%s", result.get("stats", {}).get("total_detections", 0))
    logger.info("preview_assets=%s", len(result.get("preview_assets", [])))
    logger.info("debug_preview_assets=%s", len(result.get("debug_preview_assets", [])))
    if result.get("video_path"):
        logger.info("video=%s", result["video_path"])

    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "model": result["model"],
                "dataset_variant": result["dataset_variant"],
                "resolved_input": result["resolved_input"],
                "summary_path": result["summary_path"],
                "preview_assets": result.get("preview_assets", []),
                "debug_preview_assets": result.get("debug_preview_assets", []),
                "video_path": result.get("video_path", ""),
                "sample_fps": result.get("sample_fps"),
                "sampled_frames": result.get("sampled_frames"),
                "stats": result.get("stats", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
