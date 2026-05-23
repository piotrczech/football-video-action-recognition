from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
import threading
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import cv2

from murawa.data import LoadedSplit
from murawa.models.rfdetr_support import _first_float

METRICS_CSV_NAMES = ("metrics.csv", "results.csv")
BEST_CHECKPOINT_NAMES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)

def _prepare_coco_dataset(
    train_split: LoadedSplit,
    valid_split: LoadedSplit,
    dataset_root: Path,
    image_size: int,
) -> list[str]:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    category_ids = sorted(train_split.class_mapping.keys())
    category_to_idx = {category_id: idx for idx, category_id in enumerate(category_ids)}
    categories = [
        {
            "id": category_to_idx[category_id],
            "name": train_split.class_mapping[category_id],
            "supercategory": "object",
        }
        for category_id in category_ids
    ]
    class_names = [train_split.class_mapping[category_id] for category_id in category_ids]

    _write_split_as_coco(
        train_split,
        dataset_root / "train",
        category_to_idx,
        categories,
        image_size,
    )
    _write_split_as_coco(
        valid_split,
        dataset_root / "valid",
        category_to_idx,
        categories,
        image_size,
    )
    return class_names


def _write_split_as_coco(
    split: LoadedSplit,
    split_root: Path,
    category_to_idx: dict[int, int],
    categories: list[dict],
    image_size: int,
) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    next_annotation_id = 1

    for sample in split.samples:
        file_name = f"{sample.image_id}_{sample.image_path.name}"
        scale_x, scale_y = _write_resized_image(
            src_path=sample.image_path,
            dst_path=split_root / file_name,
            image_size=image_size,
            source_width=sample.width,
            source_height=sample.height,
        )
        images.append(
            {
                "id": sample.image_id,
                "file_name": file_name,
                "width": image_size,
                "height": image_size,
            }
        )

        for annotation in sample.annotations:
            if annotation.category_id not in category_to_idx:
                continue
            bbox = _scale_bbox_xywh(annotation.bbox_xywh, scale_x=scale_x, scale_y=scale_y)
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue
            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": sample.image_id,
                    "category_id": category_to_idx[annotation.category_id],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": int(annotation.iscrowd),
                    "segmentation": [],
                }
            )
            next_annotation_id += 1

    payload = {
        "info": {"description": f"Murawa RF-DETR {split.dataset_variant}/{split.split}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (split_root / "_annotations.coco.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_resized_image(
    *,
    src_path: Path,
    dst_path: Path,
    image_size: int,
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"RF-DETR could not read training image: {src_path}")

    height, width = image.shape[:2]
    if width != source_width or height != source_height:
        source_width = width
        source_height = height

    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(dst_path), resized):
        raise RuntimeError(f"RF-DETR could not write resized training image: {dst_path}")

    return image_size / float(source_width), image_size / float(source_height)


def _scale_bbox_xywh(
    bbox_xywh: tuple[float, float, float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> list[float]:
    x, y, width, height = bbox_xywh
    return [
        max(0.0, float(x) * scale_x),
        max(0.0, float(y) * scale_y),
        max(0.0, float(width) * scale_x),
        max(0.0, float(height) * scale_y),
    ]


def _build_rfdetr_model(rfdetr_cls, weights: str, variant: str):
    default_weights = {
        "",
        "auto",
        "default",
        f"rf-detr-{variant}",
        f"rf-detr-{variant}.pth",
        f"rfdetr-{variant}",
        f"rfdetr-{variant}.pth",
    }
    if variant == "medium":
        default_weights.update({"rfdetr-m.pt", "rfdetr-m.pth", "rfdetr-m"})
    if variant == "large":
        default_weights.update({"rfdetr-l.pt", "rfdetr-l.pth", "rfdetr-l"})
    if weights.strip().lower() in default_weights:
        return rfdetr_cls()
    return rfdetr_cls(pretrain_weights=weights)


@contextmanager
def _maybe_quiet_backend_logs(quiet: bool, log_path: Path):
    terminal_stdout = sys.stdout

    def emit(message: str) -> None:
        terminal_stdout.write(message + "\n")
        terminal_stdout.flush()

    if not quiet:
        yield emit
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("=== RF-DETR backend log ===\n")
        saved_streams = []
        for logger_name in (
            "",
            "rf-detr",
            "rfdetr",
            "pytorch_lightning",
            "lightning_fabric",
            "transformers",
        ):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                if not hasattr(handler, "setStream"):
                    continue
                try:
                    saved_streams.append((handler, handler.stream))
                    handler.setStream(handle)
                except Exception:
                    continue
        try:
            with (
                warnings.catch_warnings(),
                redirect_stdout(handle),
                redirect_stderr(handle),
            ):
                warnings.filterwarnings(
                    "ignore",
                    message=".*use_return_dict.*deprecated.*",
                    category=Warning,
                )
                yield emit
        finally:
            for handler, stream in saved_streams:
                try:
                    handler.setStream(stream)
                except Exception:
                    continue


@contextmanager
def _rfdetr_progress_monitor(
    *,
    enabled: bool,
    backend_dir: Path,
    total_epochs: int,
    emit,
):
    if not enabled:
        yield
        return

    stop_event = threading.Event()
    printed_epochs: set[int] = set()

    def poll() -> None:
        while not stop_event.is_set():
            _emit_rfdetr_epoch_progress(
                backend_dir=backend_dir,
                total_epochs=total_epochs,
                printed_epochs=printed_epochs,
                emit=emit,
            )
            stop_event.wait(0.5)

    thread = threading.Thread(target=poll, name="rfdetr-progress-monitor", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        _emit_rfdetr_epoch_progress(
            backend_dir=backend_dir,
            total_epochs=total_epochs,
            printed_epochs=printed_epochs,
            emit=emit,
        )


def _emit_rfdetr_epoch_progress(
    *,
    backend_dir: Path,
    total_epochs: int,
    printed_epochs: set[int],
    emit,
) -> None:
    metrics_csv = _find_metrics_csv(backend_dir)
    if metrics_csv is None:
        return

    for summary in _read_rfdetr_epoch_summaries(metrics_csv):
        epoch_idx = summary["epoch_idx"]
        if epoch_idx in printed_epochs:
            continue
        train_loss = summary.get("loss")
        if train_loss is None:
            continue

        printed_epochs.add(epoch_idx)
        parts = [
            f"RF-DETR epoch {epoch_idx + 1}/{total_epochs}",
            f"loss={train_loss:.4f}",
        ]
        val_loss = summary.get("val_loss")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        map50 = summary.get("mAP50")
        if map50 is not None:
            parts.append(f"mAP50={map50:.4f}")
        emit(" | ".join(parts))


def _read_rfdetr_epoch_summaries(metrics_csv: Path) -> list[dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    try:
        with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epoch_value = _first_float(row, ("epoch",))
                if epoch_value is None:
                    continue
                epoch_idx = int(epoch_value)
                summary = summaries.setdefault(epoch_idx, {"epoch_idx": epoch_idx})

                train_loss = _first_float(row, ("train/loss", "train_loss", "loss"))
                if train_loss is not None:
                    summary["loss"] = train_loss

                val_loss = _first_float(row, ("val/loss", "val_loss"))
                if val_loss is not None:
                    summary["val_loss"] = val_loss

                map50 = _first_float(row, ("val/mAP_50", "val/ema_mAP_50", "mAP50"))
                if map50 is not None:
                    summary["mAP50"] = map50
    except (OSError, csv.Error):
        return []

    return [summaries[key] for key in sorted(summaries)]


def _resolve_best_checkpoint(backend_dir: Path) -> Path:
    backend_dir = backend_dir.resolve()
    for name in BEST_CHECKPOINT_NAMES:
        candidate = backend_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate

    candidates: list[Path] = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        candidates.extend(path for path in backend_dir.rglob(pattern) if path.is_file())
    if candidates:
        priority = {name: idx for idx, name in enumerate(BEST_CHECKPOINT_NAMES)}
        candidates.sort(key=lambda p: (priority.get(p.name, 99), -p.stat().st_mtime))
        return candidates[0]

    raise RuntimeError(
        f"RF-DETR training completed, but no checkpoint was found in '{backend_dir}'."
    )


def _extract_training_metrics(backend_dir: Path, fallback_epochs: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"epochs": fallback_epochs, "loss": 0.0, "mAP50": 0.0}
    metrics_csv = _find_metrics_csv(backend_dir)
    if metrics_csv is None:
        return metrics

    loss_history: list[float] = []
    val_history: list[float] = []
    map50_history: list[float] = []
    map5095_history: list[float] = []
    precision_history: list[float] = []
    recall_history: list[float] = []
    map50_last: float | None = None
    map5095_last: float | None = None
    precision_last: float | None = None
    recall_last: float | None = None
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_loss = _first_float(row, ("train/loss", "train_loss", "loss"))
            if train_loss is not None:
                loss_history.append(train_loss)

            val_loss = _first_float(row, ("val/loss", "val_loss"))
            if val_loss is not None:
                val_history.append(val_loss)

            map50 = _first_float(row, ("val/mAP_50", "val/ema_mAP_50", "mAP50"))
            if map50 is not None:
                map50_history.append(map50)
                map50_last = map50

            map5095 = _first_float(
                row,
                ("val/mAP_50_95", "val/mAP_5095", "val/ema_mAP_50_95", "mAP5095", "mAP50-95"),
            )
            if map5095 is not None:
                map5095_history.append(map5095)
                map5095_last = map5095

            precision = _first_float(row, ("val/precision", "precision", "val/Precision"))
            if precision is not None:
                precision_history.append(precision)
                precision_last = precision

            recall = _first_float(row, ("val/recall", "recall", "val/Recall"))
            if recall is not None:
                recall_history.append(recall)
                recall_last = recall

    if loss_history:
        metrics["epochs"] = len(loss_history)
        metrics["loss_history"] = loss_history
        metrics["loss"] = loss_history[-1]
    if val_history:
        metrics["val_loss_history"] = val_history
        metrics["val_loss"] = val_history[-1]
    if map50_history:
        metrics["map50_history"] = map50_history
    if map50_last is not None:
        metrics["mAP50"] = map50_last
    if map5095_history:
        metrics["map5095_history"] = map5095_history
    if map5095_last is not None:
        metrics["mAP5095"] = map5095_last
    if precision_history:
        metrics["precision_history"] = precision_history
    if precision_last is not None:
        metrics["precision"] = precision_last
    if recall_history:
        metrics["recall_history"] = recall_history
    if recall_last is not None:
        metrics["recall"] = recall_last
    return metrics


def _find_metrics_csv(backend_dir: Path) -> Path | None:
    for name in METRICS_CSV_NAMES:
        candidate = backend_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate

    candidates = [path for path in backend_dir.rglob("*.csv") if path.name in METRICS_CSV_NAMES]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


