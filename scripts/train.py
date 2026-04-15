#!/usr/bin/env python3

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

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
DEFAULT_CONFIG = ROOT / "configs" / "train.yaml"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

SPLIT_ORDER = ("train", "valid", "test")
TARGET_ALL_CLASSES = frozenset({"player", "goalkeeper", "referee", "ball"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock training entrypoint + debug preview.")
    p.add_argument("--model", choices=["yolo", "rfdetr", "rf"], required=True)
    p.add_argument("--dataset-variant", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_name = normalize_model_name(args.model)
    # TODO(Issue #12, #13): replace single-run mock call with services.training_runs variant suites.
    use_mock_yolo = model_name == "yolo" and _env_flag("MURAWA_YOLO_MOCK")
    use_real_adapter = model_name == "yolo" and not use_mock_yolo
    artifact_callback = _resolve_artifact_callback()

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

    if use_real_adapter:
        model_impl = build_training_adapter(model_name)
        train_result = model_impl.train(
            args.dataset_variant,
            config_path=Path(args.config),
            output_dir=ckpt_dir,
            artifact_callback=artifact_callback,
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
        checkpoint_path = ckpt_dir / "model.pt"
        torch.save(
            {"mock": True, "model": model_name, "weights": train_result["weights"]},
            checkpoint_path,
        )
        is_mock_run = True

    cfg_path = Path(args.config)
    cfg_found = save_config(cfg_path, meta_dir / "config.yaml", model_name, args.dataset_variant)

    preview_created, preview_result = build_dataset_preview(
        project_root=ROOT,
        dataset_variant=args.dataset_variant,
    )

    train_samples = int(train_result.get("train_samples", len(loaded_split.samples)))
    valid_samples = int(train_result.get("valid_samples", 0))
    backend_name = str(train_result.get("backend", "mock"))
    train_device = str(train_result.get("train_device", "cpu"))
    valid_split_source = str(train_result.get("valid_split_source", "valid"))
    note = str(train_result.get("note", f"{model_name} training finished."))

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
            "config_argument": str(cfg_path),
            "config_found": cfg_found,
            "train_device": train_device,
            "train_samples": train_samples,
            "valid_samples": valid_samples,
            "valid_split_source": valid_split_source,
            "is_mock_run": is_mock_run,
            "model_impl": model_impl.__class__.__name__,
            "mock_note": note if is_mock_run else "",
            "preview_created": preview_created,
            "preview_result": preview_result,
            "preview_path": "",
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
    logger.info("checkpoint=%s", checkpoint_path)
    logger.info("metadata=%s", meta_dir)
    logger.info("preview=%s", preview_result)
    if use_mock_yolo:
        logger.info("MURAWA_YOLO_MOCK is enabled. YOLO mock model was used.")
    return 0


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_artifact_callback() -> StandardizedArtifactCallback | None:
    if not _env_flag("MURAWA_ENABLE_ARTIFACT_CALLBACK"):
        return None
    return StandardizedArtifactCallback()


def _run_artifact_callback_hooks(
    artifact_callback: StandardizedArtifactCallback | None,
    run_name: str,
    model_name: str,
    dataset_variant: str,
    created_at: datetime,
    checkpoint_path: Path,
    meta_dir: Path,
) -> None:
    if artifact_callback is None:
        return

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
    except NotImplementedError as exc:
        raise RuntimeError(
            "MURAWA_ENABLE_ARTIFACT_CALLBACK is enabled, but StandardizedArtifactCallback "
            "is not implemented yet (Issue #14)."
        ) from exc


def build_dataset_preview(
    project_root: Path,
    dataset_variant: str,
) -> tuple[bool, str]:
    cols = 3
    split_payloads = []
    for split in SPLIT_ORDER:
        try:
            loaded = load_training_split(project_root, dataset_variant, split=split, max_samples=None)
        except DataLoaderError as exc:
            logger.warning("Skipping split '%s' in preview: %s", split, exc)
            continue
        split_payloads.append((split, loaded))

    if not split_payloads:
        return False, "no_splits_available"

    row_ball_only = []
    row_all_classes = []

    for split_name, loaded in split_payloads:
        for sample in loaded.samples:
            class_names = {ann.category_name for ann in sample.annotations}
            is_ball_extra = "ballextra_" in sample.image_path.name.lower()
            if is_ball_extra and class_names == {"ball"} and len(row_ball_only) < cols:
                row_ball_only.append((split_name, sample))
            if class_names == TARGET_ALL_CLASSES and len(row_all_classes) < cols:
                row_all_classes.append((split_name, sample))
            if len(row_ball_only) >= cols and len(row_all_classes) >= cols:
                break
        if len(row_ball_only) >= cols and len(row_all_classes) >= cols:
            break

    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows))
    axes = _normalize_axes(axes)

    _fill_preview_row(
        axes=axes,
        row_idx=0,
        cols=cols,
        samples=row_ball_only,
        row_label="ball-extra | only ball",
    )
    _fill_preview_row(
        axes=axes,
        row_idx=1,
        cols=cols,
        samples=row_all_classes,
        row_label="mixed | all classes present",
    )

    fig.suptitle(
        f"Dataset preview ({dataset_variant}) | row1=ball-only | row2=all-classes",
        fontsize=12,
    )
    fig.tight_layout()

    try:
        plt.show()
        plt.close(fig)
        return True, "shown_window"
    except Exception as exc:
        plt.close(fig)
        logger.warning("Preview show failed: %s", exc)
        return False, f"show_failed: {exc}"


def _fill_preview_row(axes, row_idx: int, cols: int, samples, row_label: str) -> None:
    for col_idx in range(cols):
        ax = axes[row_idx][col_idx]
        if col_idx >= len(samples):
            ax.text(
                0.5,
                0.5,
                f"{row_label}\n(no sample)",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        split_name, sample = samples[col_idx]
        try:
            image = plt.imread(str(sample.image_path))
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"{split_name}\n(unreadable)\n{sample.image_path.name}",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
            ax.axis("off")
            logger.warning("Could not read image %s: %s", sample.image_path, exc)
            continue

        title = f"{split_name} | {sample.image_path.name}"
        _draw_sample_with_bboxes(ax=ax, image=image, annotations=sample.annotations, title=title)


def _draw_sample_with_bboxes(ax, image, annotations, title: str) -> None:
    color_by_id = {
        0: "#47DDFF",  # player
        1: "#48C9B0",  # goalkeeper
        2: "#0066FF",  # referee
        3: "#FF0000",  # ball
    }
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

    for ann in annotations:
        x, y, w, h = ann.bbox_xywh
        color = color_by_id.get(ann.category_id, "#FFFFFF")
        rect = Rectangle((x, y), w, h, linewidth=1.8, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x,
            max(0, y - 3),
            ann.category_name,
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
        )


def _normalize_axes(axes):
    try:
        if hasattr(axes, "ndim"):
            if axes.ndim == 1:
                return [list(axes)]
            return [list(row) for row in axes]
    except Exception:
        pass
    return [[axes]]


if __name__ == "__main__":
    raise SystemExit(main())
