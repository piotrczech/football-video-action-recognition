#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from murawa.data.path_resolver import PREDICTIONS_ROOT, pick_input
from murawa.services.artifacts import CKPT_DIR, META_DIR, latest_run, write_json


ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock prediction entrypoint.")
    p.add_argument("--model", choices=["yolo", "rfdetr"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--mode", choices=["image", "frames", "video"], default="video")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_name = latest_run(ROOT, args.model, args.dataset_variant)

    out_dir = ROOT / PREDICTIONS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    input_target, input_found = pick_input(ROOT, args.mode)

    write_json(
        out_dir / "prediction_summary.json",
        {
            "mock": True,
            "model": args.model,
            "dataset_variant": args.dataset_variant,
            "mode": args.mode,
            "resolved_run_name": run_name,
            "checkpoint_path": str(ROOT / CKPT_DIR / run_name / "model.pt"),
            "metadata_path": str(ROOT / META_DIR / run_name),
            "resolved_input": input_target,
            "input_found": input_found,
            "output_dir": str(out_dir),
        },
    )

    (out_dir / f"{args.mode}_prediction.txt").write_text(
        "Mock prediction output placeholder.\n", encoding="utf-8"
    )

    logger.info("run_name=%s", run_name)
    logger.info("output=%s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
