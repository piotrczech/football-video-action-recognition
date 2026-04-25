#!/usr/bin/env python3

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from murawa.data import DataLoaderError, load_training_split, summarize_variant
from murawa.models import build_model, build_training_adapter, normalize_model_name
from murawa.services.artifacts import (
    ArtifactManifest,
    ArtifactWriteContext,
    CKPT_DIR,
    META_DIR,
    StandardizedArtifactCallback,
    make_run_name,
    save_config,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_CONFIGS = {
    "quick": ROOT / "configs" / "train.quick.yaml",
    "full": ROOT / "configs" / "train.full.yaml",
    "rf": ROOT / "configs" / "train.rf.yaml",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entrypoint.")
    parser.add_argument("--model", choices=["yolo", "rfdetr", "rf"], required=True)
    parser.add_argument("--dataset-variant", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS.keys()), default="full")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP for YOLO training (useful for ROCm/GPU stability checks).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU device for YOLO training (overrides config device).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = normalize_model_name(args.model)
    config_path = _resolve_profile_config(args.profile)

    use_mock_yolo = model_name == "yolo" and _env_flag("MURAWA_YOLO_MOCK")
    use_real_adapter = model_name == "yolo" and not use_mock_yolo
    artifact_callback = StandardizedArtifactCallback()
    amp_override = False if args.no_amp else None
    device_override = "cpu" if args.force_cpu else None

    try:
        loaded_split = load_training_split(
            project_root=ROOT,
            dataset_variant=args.dataset_variant,
            split="train",
            max_samples=16,
        )
    except DataLoaderError as exc:
        logger.error("Training data validation failed.\n%s", exc)
        return 1

    sampled_annotations = sum(len(sample.annotations) for sample in loaded_split.samples)
    created_at = datetime.now(timezone.utc)
    run_name = make_run_name(model_name, args.dataset_variant, created_at)
    
    ckpt_dir = ROOT / CKPT_DIR / run_name
    meta_dir = ROOT / META_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if model_name in ["rfdetr", "rf_detr", "rf"]:
        model_impl = build_model(model_name)
        train_result = model_impl.train(
            dataset_variant=args.dataset_variant,
            output_dir=ckpt_dir,
            config_path=config_path 
        )
        if train_result is None:
            train_result = {}
        checkpoint_path = ckpt_dir / "model.pt"
        is_mock_run = False
        
    elif use_real_adapter:
        model_impl = build_training_adapter(model_name)
        train_result = model_impl.train(
            args.dataset_variant,
            config_path=config_path,
            output_dir=ckpt_dir,
            artifact_callback=artifact_callback,
            amp=amp_override,
            device=device_override,
        )
        checkpoint_path = Path(
            train_result.get("weights", {}).get("checkpoint_path", str(ckpt_dir / "model.pt"))
        ).resolve()
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            logger.error("YOLO adapter did not produce a valid checkpoint at: %s", checkpoint_path)
            return 1
        is_mock_run = bool(train_result.get("mock", False))
        
    else:
        model_impl = build_model(model_name)
        train_result = model_impl.train(args.dataset_variant)
        if train_result is None:
            train_result = {"weights": {}}
        checkpoint_path = ckpt_dir / "model.pt"
        torch.save(
            {"mock": True, "model": model_name, "weights": train_result.get("weights", {})},
            checkpoint_path,
        )
        is_mock_run = True

    cfg_found = save_config(config_path, meta_dir / "config.yaml", model_name, args.dataset_variant)

    try:
        from murawa.data import build_dataset_preview
        preview_created, preview_result = build_dataset_preview(
            project_root=ROOT,
            dataset_variant=args.dataset_variant,
        )
    except ImportError:
        pass

    train_samples = int(train_result.get("train_samples", len(loaded_split.samples)))
    valid_samples = int(train_result.get("valid_samples", 0))
    backend_name = str(train_result.get("backend", "mock"))
    train_device = str(train_result.get("train_device", "cpu"))
    valid_split_source = str(train_result.get("valid_split_source", "valid"))
    note = str(train_result.get("note", f"{model_name} training finished."))
    train_amp = train_result.get("train_amp")
    if train_amp is None:
        train_amp = None if is_mock_run else (False if args.no_amp else True)
    train_device = "cpu" if args.force_cpu and not is_mock_run else train_device

    try:
        variant_summary_payload = summarize_variant(ROOT, args.dataset_variant).to_dict()
    except DataLoaderError:
        variant_summary_payload = {
            "dataset_variant": args.dataset_variant,
            "resolved_training_path": str(loaded_split.variant_dir),
            "available_splits": list(loaded_split.available_splits),
            "validated_split": loaded_split.split,
            "total_images_in_validated_split": loaded_split.total_images,
            "total_annotations_in_validated_split": loaded_split.total_annotations,
            "class_mapping_source": "coco.categories",
            "class_mapping": {str(k): v for k, v in loaded_split.class_mapping.items()},
        }

    write_json(
        meta_dir / "class_mapping.json",
        {str(k): v for k, v in loaded_split.class_mapping.items()},
    )
    write_json(
        meta_dir / "train_metadata.json",
        {
            "run_name": run_name,
            "model": model_name,
            "framework": "pytorch",
            "backend": backend_name,
            "created_at_utc": created_at.isoformat(),
            "dataset_variant": args.dataset_variant,
            "resolved_training_path": str(loaded_split.variant_dir),
            "profile": args.profile,
            "config_argument": str(config_path),
            "config_found": cfg_found,
            "train_device": train_device,
            "train_amp": train_amp,
            "force_cpu": bool(args.force_cpu),
            "train_samples": train_samples,
            "valid_samples": valid_samples,
            "valid_split_source": valid_split_source,
            "is_mock_run": is_mock_run,
            "model_impl": model_impl.__class__.__name__,
            "mock_note": note if is_mock_run else "",
            "loader_summary": {
                "validated_split": loaded_split.split,
                "sampled_images": train_samples,
                "sampled_annotations": sampled_annotations,
                "total_images_in_split": loaded_split.total_images,
                "total_annotations_in_split": loaded_split.total_annotations,
                "class_mapping_source": "coco.categories",
                "class_mapping": {str(k): v for k, v in loaded_split.class_mapping.items()},
            },
        },
    )

    metrics = dict(train_result.get("metrics", {}))
    metrics["note"] = note
    write_json(meta_dir / "metrics_summary.json", metrics)
    write_json(meta_dir / "dataset_variant.json", variant_summary_payload)

    _run_artifact_callback_hooks(
        artifact_callback=artifact_callback,
        run_name=run_name,
        model_name=model_name,
        dataset_variant=args.dataset_variant,
        created_at=created_at,
        checkpoint_path=checkpoint_path,
        meta_dir=meta_dir,
    )

    logger.info("run_name=%s", run_name)
    logger.info("profile=%s", args.profile)
    logger.info("config=%s", config_path)
    logger.info("device=%s", train_device)
    logger.info("amp=%s", train_amp if train_amp is not None else "n/a")
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("metadata=%s", meta_dir)
    if use_mock_yolo:
        logger.info("MURAWA_YOLO_MOCK is enabled. YOLO mock model was used.")
    return 0


def _resolve_profile_config(profile: str) -> Path:
    config_path = PROFILE_CONFIGS[profile].resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config profile file not found: {config_path}")
    return config_path


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_artifact_callback_hooks(
    artifact_callback: StandardizedArtifactCallback,
    run_name: str,
    model_name: str,
    dataset_variant: str,
    created_at: datetime,
    checkpoint_path: Path,
    meta_dir: Path,
) -> None:
    context = ArtifactWriteContext(
        run_name=run_name,
        project_root=ROOT,
        checkpoint_path=checkpoint_path,
        metadata_dir=meta_dir,
    )
    manifest = ArtifactManifest(
        run_name=run_name,
        model=model_name,
        dataset_variant=dataset_variant,
        created_at_utc=created_at.isoformat(),
        checkpoint_file=checkpoint_path.name,
    )

    try:
        artifact_callback.validate_contract(context=context)
        artifact_callback.write_manifest(meta_dir=meta_dir, manifest=manifest)
    except NotImplementedError:
        logger.warning(
            "StandardizedArtifactCallback is not implemented yet (Issue #14). "
            "Training run is valid; callback artifacts were skipped."
        )
    except Exception as exc:
        raise RuntimeError(f"Artifact callback failed for run '{run_name}': {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
