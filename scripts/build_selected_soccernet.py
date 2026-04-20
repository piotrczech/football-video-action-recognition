#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from murawa.data.frame_selection import (  # noqa: E402
    FrameSelectionConfig,
    preprocess_selected_frames,
    select_n_frames,
)

LOGGER = logging.getLogger("build-selected-soccernet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selected SoccerNet frames dataset in data/selected/soccernet."
    )
    parser.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--selected-root", default=str(ROOT / "data" / "selected"))
    parser.add_argument("--frame-step", type=int, default=30)
    parser.add_argument(
        "--resize-selected",
        action="store_true",
        help="Resize selected images so the longest side is at most 1280.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply lightweight image normalization in preprocessing step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    selected_path = select_n_frames(
        FrameSelectionConfig(
            raw_root=Path(args.raw_root),
            selected_root=Path(args.selected_root),
            frame_step=args.frame_step,
            keep_original_resolution=not args.resize_selected,
        )
    )
    LOGGER.info("Selected dataset created: %s", selected_path)

    processed_path = preprocess_selected_frames(selected_path, normalize=args.normalize)
    LOGGER.info("Preprocessing completed: %s", processed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())