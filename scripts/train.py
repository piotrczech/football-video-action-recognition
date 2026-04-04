#!/usr/bin/env python3

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import torch

from murawa.data.path_resolver import training_path
from murawa.models import build_model, normalize_model_name
from murawa.services.artifacts import (
    CKPT_DIR,
    META_DIR,
    make_run_name,
    save_config,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "train.yaml"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock training entrypoint.")
    p.add_argument("--model", choices=["yolo", "rfdetr", "rf"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_name = normalize_model_name(args.model)
    created_at = datetime.now(timezone.utc)
    run_name = make_run_name(model_name, args.dataset_variant, created_at)

    ckpt_dir = ROOT / CKPT_DIR / run_name
    meta_dir = ROOT / META_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    model_impl = build_model(model_name)
    train_result = model_impl.train(args.dataset_variant)
    checkpoint_path = ckpt_dir / "model.pt"
    torch.save(
        {"mock": True, "model": model_name, "weights": train_result["weights"]},
        checkpoint_path,
    )

    cfg_path = Path(args.config)
    cfg_found = save_config(cfg_path, meta_dir / "config.yaml", model_name, args.dataset_variant)

    data_path = training_path(ROOT, args.dataset_variant)

    write_json(meta_dir / "class_mapping.json", {"0": "player", "1": "goalkeeper", "2": "referee", "3": "ball"})
    write_json(
        meta_dir / "train_metadata.json",
        {
            "run_name": run_name,
            "model": model_name,
            "framework": "pytorch",
            "created_at_utc": created_at.isoformat(),
            "dataset_variant": args.dataset_variant,
            "resolved_training_path": str(data_path),
            "config_argument": str(cfg_path),
            "config_found": cfg_found,
            "mock_model": model_impl.__class__.__name__,
            "mock_note": train_result["note"],
        },
    )
    metrics = dict(train_result["metrics"])
    metrics["note"] = train_result["note"]
    write_json(meta_dir / "metrics_summary.json", metrics)
    write_json(
        meta_dir / "dataset_variant.json",
        {"dataset_variant": args.dataset_variant, "resolved_training_path": str(data_path)},
    )

    logger.info("run_name=%s", run_name)
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("metadata=%s", meta_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
