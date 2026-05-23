#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from murawa.data.variant_assembly import (
    SUPPORTED_VARIANTS,
    VariantAssemblyConfig,
    assemble_variant,
    describe_variant,
)
from murawa.settings import DATA_RAW, DATA_READY, DEFAULT_FRAME_STEP, PROJECT_ROOT

LOGGER = logging.getLogger("build-ready-variants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build experiment-ready dataset variants in data/ready."
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default=None,
        help="Build only one variant. Default: build all supported variants.",
    )
    parser.add_argument(
        "--selected-root",
        default=str(PROJECT_ROOT / DATA_RAW),
        help="Bootstrap source root (defaults to data/raw).",
    )
    parser.add_argument(
        "--final-root",
        default=str(PROJECT_ROOT / DATA_READY),
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Reference frame_step passed to bootstrap source variants.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output variants.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    variants_to_build = [args.variant] if args.variant else list(SUPPORTED_VARIANTS)
    LOGGER.info("Preparing variants: %s", ", ".join(variants_to_build))

    for variant_name in variants_to_build:
        variant_dir = assemble_variant(
            VariantAssemblyConfig(
                selected_root=Path(args.selected_root),
                final_root=Path(args.final_root),
                variant_name=variant_name,
                frame_step=args.frame_step,
                force=args.force,
            )
        )
        LOGGER.info("Variant ready: %s", variant_dir)
        LOGGER.info("Description: %s", json.dumps(describe_variant(variant_dir), ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
