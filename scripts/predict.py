#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from murawa.services.pipeline import analyze_frame, analyze_match


ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock prediction entrypoint.")
    p.add_argument("--model", choices=["yolo", "rfdetr", "rf"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--mode", choices=["image", "frames", "video"], default="video")
    p.add_argument("--input-path", default=None, help="Optional path to image/video input.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "image":
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
        )

    if result["status"] != "ok":
        logger.error(result.get("message", "Prediction failed."))
        return 1

    logger.info("mode=%s", result["mode"])
    logger.info("summary=%s", result["summary_path"])
    logger.info("output=%s", result["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
