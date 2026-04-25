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

from murawa.data.bootstrap_variant import (  # noqa: E402
    BootstrapConfig,
    BootstrapError,
    SUPPORTED_VARIANTS,
    build_bootstrap_variant,
)

LOGGER = logging.getLogger("bootstrap-base-variant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage-2 dataset variants (base, extended, extended-only-train)."
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default=None,
        help="Build only one variant. Default: build all variants.",
    )
    parser.add_argument(
        "--output-variant",
        dest="output_variant_alias",
        choices=SUPPORTED_VARIANTS,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frame-step", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output variant(s).")

    args = parser.parse_args()
    if args.variant and args.output_variant_alias and args.variant != args.output_variant_alias:
        parser.error("Use either --variant or --output-variant with the same value.")

    if args.output_variant_alias and not args.variant:
        args.variant = args.output_variant_alias
        args.used_deprecated_alias = True
    else:
        args.used_deprecated_alias = False

    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.used_deprecated_alias:
        LOGGER.warning("Argument --output-variant is deprecated. Use --variant instead.")

    variants_to_build = [args.variant] if args.variant else list(SUPPORTED_VARIANTS)
    LOGGER.info("Preparing variants: %s", ", ".join(variants_to_build))

    for variant in variants_to_build:
        try:
            result = build_bootstrap_variant(
                project_root=ROOT,
                config=BootstrapConfig(
                    output_variant=variant,
                    frame_step=args.frame_step,
                    force=args.force,
                ),
            )
        except BootstrapError as exc:
            LOGGER.error("Bootstrap failed for variant='%s': %s", variant, exc)
            return 1

        LOGGER.info("Variant bootstrap completed.")
        LOGGER.info("variant=%s", variant)
        LOGGER.info("path=%s", result.variant_dir)
        LOGGER.info("used_fallback=%s", result.used_fallback)
        for split, count in result.split_counts.items():
            LOGGER.info("split=%s images=%d", split, count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
