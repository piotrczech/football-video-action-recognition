from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from typing import Any
import warnings

import cv2
import yaml

from murawa.data import DataLoaderError, LoadedSplit, load_training_split
from murawa.services.artifacts import StandardizedArtifactCallback


@dataclass
class RfDetrAdapter:
    """Real RF-DETR backend wired into the same project contract as YOLO."""

    name: str = "rfdetr"
    backend: str = "roboflow-rfdetr"

    def train(
        self,
        dataset_variant: str,
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        artifact_callback: StandardizedArtifactCallback | None = None,
        device: str | None = None,
    ) -> dict:
        if output_dir is None:
            raise ValueError("RfDetrAdapter.train requires output_dir to persist model checkpoint.")

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        project_root = _resolve_project_root(output_dir=output_dir)
        cfg = _resolve_training_config(config_path)
        if device is not None:
            cfg["device"] = device
        _seed_everything(cfg["seed"])
        rfdetr_cls = _import_rfdetr()

        try:
            train_split = load_training_split(
                project_root=project_root,
                dataset_variant=dataset_variant,
                split="train",
                max_samples=cfg["max_train_samples"],
            )
        except DataLoaderError as exc:
            raise RuntimeError(f"RF-DETR training data loading failed for split='train': {exc}") from exc

        try:
            valid_split = load_training_split(
                project_root=project_root,
                dataset_variant=dataset_variant,
                split="valid",
                max_samples=cfg["max_valid_samples"],
            )
            valid_split_name = "valid"
        except DataLoaderError:
            try:
                valid_split = load_training_split(
                    project_root=project_root,
                    dataset_variant=dataset_variant,
                    split="train",
                    max_samples=cfg["max_valid_samples"],
                )
            except DataLoaderError as exc:
                raise RuntimeError(
                    "RF-DETR validation split fallback failed. Neither 'valid' nor fallback 'train' "
                    f"could be loaded: {exc}"
                ) from exc
            valid_split_name = "train"

        dataset_root = output_dir / "_rfdetr_dataset"
        class_names = _prepare_coco_dataset(
            train_split=train_split,
            valid_split=valid_split,
            dataset_root=dataset_root,
            image_size=cfg["resolution"],
        )

        backend_dir = output_dir / "rfdetr_backend"
        backend_dir.mkdir(parents=True, exist_ok=True)
        backend_log_path = output_dir / "rfdetr_backend.log"
        try:
            with _maybe_quiet_backend_logs(cfg["quiet"], backend_log_path) as emit_progress:
                try:
                    model = _build_rfdetr_model(rfdetr_cls=rfdetr_cls, weights=cfg["weights"])
                except Exception as exc:
                    raise RuntimeError(
                        f"RF-DETR backend initialization failed for weights='{cfg['weights']}': {exc}"
                    ) from exc

                if cfg["quiet"]:
                    emit_progress(
                        f"RF-DETR training for {cfg['epochs']} epochs "
                        f"(details: {backend_log_path})"
                    )
                with _rfdetr_progress_monitor(
                    enabled=cfg["quiet"],
                    backend_dir=backend_dir,
                    total_epochs=cfg["epochs"],
                    emit=emit_progress,
                ):
                    model.train(
                        dataset_dir=str(dataset_root),
                        output_dir=str(backend_dir),
                        epochs=cfg["epochs"],
                        batch_size=cfg["batch_size"],
                        grad_accum_steps=cfg["grad_accum_steps"],
                        lr=cfg["learning_rate"],
                        resolution=cfg["resolution"],
                        device=cfg["device"],
                        checkpoint_interval=cfg["checkpoint_interval"],
                        seed=cfg["seed"],
                        tensorboard=cfg["tensorboard"],
                        multi_scale=cfg["multi_scale"],
                        log_per_class_metrics=cfg["log_per_class_metrics"],
                        aug_config={},
                        progress_bar=None,
                    )
        except Exception as exc:
            log_hint = f" See backend log: {backend_log_path}" if cfg["quiet"] else ""
            raise RuntimeError(f"RF-DETR backend training failed: {exc}.{log_hint}") from exc

        best_checkpoint = _resolve_best_checkpoint(backend_dir)
        checkpoint_path = output_dir / "model.pt"
        shutil.copy2(best_checkpoint, checkpoint_path)

        metrics = _extract_training_metrics(backend_dir=backend_dir, fallback_epochs=cfg["epochs"])
        note = (
            f"RF-DETR adapter trained on {dataset_variant} "
            f"(train={len(train_split.samples)}, val_source={valid_split_name}, val={len(valid_split.samples)})."
        )

        return {
            "weights": {
                "checkpoint_path": str(checkpoint_path),
                "class_names": class_names,
            },
            "metrics": metrics,
            "note": note,
            "mock": False,
            "backend": self.backend,
            "train_device": str(cfg["device"]),
            "train_amp": None,
            "train_samples": len(train_split.samples),
            "valid_samples": len(valid_split.samples),
            "valid_split_source": valid_split_name,
        }

    def predict(
        self,
        input_path: Path,
        *,
        checkpoint_path: Path,
        mode: str,
    ) -> list[dict]:
        supported_modes = {"frame", "match"}
        if mode not in supported_modes:
            raise ValueError(f"Unsupported mode='{mode}'. Expected one of: {sorted(supported_modes)}.")

        input_path = input_path.resolve()
        checkpoint_path = checkpoint_path.resolve()
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Prediction input does not exist or is not a file: {input_path}")
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"RF-DETR checkpoint does not exist: {checkpoint_path}")

        rfdetr_cls = _import_rfdetr()
        try:
            model = rfdetr_cls(pretrain_weights=str(checkpoint_path))
        except Exception as exc:
            raise RuntimeError(f"RF-DETR backend failed to load checkpoint '{checkpoint_path}': {exc}") from exc

        class_mapping = _load_class_mapping(checkpoint_path=checkpoint_path)
        detection_confidence = _resolve_detection_confidence(checkpoint_path=checkpoint_path)
        if mode == "frame":
            if input_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(
                    f"Frame mode requires an image file. Received suffix='{input_path.suffix}'."
                )
            detections = _predict_image(model=model, image=str(input_path), threshold=detection_confidence)
            return _convert_detections_to_frame_schema(detections, class_mapping)

        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(
                f"Match mode requires a video file. Received suffix='{input_path.suffix}', "
                f"expected one of {sorted(VIDEO_SUFFIXES)}."
            )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video input for RF-DETR prediction: {input_path}")

        frame_step = 5
        frame_index = -1
        frame_batches: list[tuple[int, list[dict]]] = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % frame_step != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                try:
                    detections = _predict_image(
                        model=model,
                        image=frame_rgb,
                        threshold=detection_confidence,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"RF-DETR match prediction failed at frame_index={frame_index}: {exc}"
                    ) from exc

                frame_batches.append(
                    (frame_index, _convert_detections_to_frame_schema(detections, class_mapping))
                )
        finally:
            cap.release()

        return _to_match_schema(frame_batches=frame_batches, max_distance_px=55.0)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
METRICS_CSV_NAMES = ("metrics.csv", "results.csv")
RFDETR_MEDIUM_RESOLUTION_BLOCK = 32
BEST_CHECKPOINT_NAMES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)


def _import_rfdetr():
    try:
        from rfdetr import RFDETRMedium

        return RFDETRMedium
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow RF-DETR backend is unavailable. Install dependencies with: pip install rfdetr"
        ) from exc


def _resolve_training_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        raise ValueError("RfDetrAdapter.train requires config_path for explicit training settings.")

    cfg_path = config_path.resolve()
    if not cfg_path.exists() or not cfg_path.is_file():
        raise FileNotFoundError(f"Training config file does not exist: {cfg_path}")

    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse training config '{cfg_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Training config '{cfg_path}' must contain a mapping at top-level.")

    training_cfg = _require_mapping(payload.get("training"), key="training", config_path=cfg_path)
    runtime_cfg = _require_mapping(payload.get("runtime"), key="runtime", config_path=cfg_path)
    rfdetr_cfg = _require_mapping(payload.get("rfdetr"), key="rfdetr", config_path=cfg_path)
    resolution = _as_int(
        rfdetr_cfg.get("resolution", training_cfg.get("resolution", 728)),
        key="resolution",
        minimum=RFDETR_MEDIUM_RESOLUTION_BLOCK,
    )
    if resolution % RFDETR_MEDIUM_RESOLUTION_BLOCK != 0:
        raise ValueError(
            "Config value 'resolution' must be divisible by 32 for RF-DETR Medium "
            "(patch_size=16, num_windows=2), got: "
            f"{resolution}"
        )

    return {
        "epochs": _as_int(
            rfdetr_cfg.get("epochs", training_cfg.get("epochs", 1)),
            key="epochs",
            minimum=1,
        ),
        "batch_size": _as_int(
            rfdetr_cfg.get("batch_size", training_cfg.get("batch_size", 2)),
            key="batch_size",
            minimum=1,
        ),
        "grad_accum_steps": _as_int(
            rfdetr_cfg.get("grad_accum_steps", 4),
            key="grad_accum_steps",
            minimum=1,
        ),
        "learning_rate": _as_float(
            rfdetr_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.0001)),
            key="learning_rate",
            minimum=0.0,
        ),
        "resolution": resolution,
        "device": _as_device(rfdetr_cfg.get("device", "cuda")),
        "checkpoint_interval": _as_int(
            rfdetr_cfg.get("checkpoint_interval", 10),
            key="checkpoint_interval",
            minimum=1,
        ),
        "max_train_samples": _as_optional_int(rfdetr_cfg.get("max_train_samples"), "max_train_samples"),
        "max_valid_samples": _as_optional_int(rfdetr_cfg.get("max_valid_samples"), "max_valid_samples"),
        "seed": _as_int(runtime_cfg.get("seed", 42), key="seed", minimum=0),
        "weights": str(rfdetr_cfg.get("weights", "default")).strip(),
        "tensorboard": _as_bool(rfdetr_cfg.get("tensorboard", True), key="tensorboard"),
        "multi_scale": _as_bool(rfdetr_cfg.get("multi_scale", True), key="multi_scale"),
        "log_per_class_metrics": _as_bool(
            rfdetr_cfg.get("log_per_class_metrics", True),
            key="log_per_class_metrics",
        ),
        "quiet": _as_bool(rfdetr_cfg.get("quiet", False), key="quiet"),
    }


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


def _resolve_project_root(output_dir: Path) -> Path:
    if len(output_dir.parents) >= 3:
        return output_dir.parents[2]
    return Path(__file__).resolve().parents[3]


def _build_rfdetr_model(rfdetr_cls, weights: str):
    default_weights = {"", "auto", "default", "rfdetr-m.pt", "rfdetr-medium"}
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
    val_loss_last: float | None = None
    map50_last: float | None = None
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_loss = _first_float(row, ("train/loss", "train_loss", "loss"))
            if train_loss is not None:
                loss_history.append(train_loss)

            val_loss = _first_float(row, ("val/loss", "val_loss"))
            if val_loss is not None:
                val_loss_last = val_loss

            map50 = _first_float(row, ("val/mAP_50", "val/ema_mAP_50", "mAP50"))
            if map50 is not None:
                map50_last = map50

    if loss_history:
        metrics["epochs"] = len(loss_history)
        metrics["loss_history"] = loss_history
        metrics["loss"] = loss_history[-1]
    if val_loss_last is not None:
        metrics["val_loss"] = val_loss_last
    if map50_last is not None:
        metrics["mAP50"] = map50_last
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


def _predict_image(model, image: Any, threshold: float):
    try:
        detections = model.predict(image, threshold=threshold)
    except Exception as exc:
        raise RuntimeError(f"RF-DETR prediction failed: {exc}") from exc
    if isinstance(detections, list):
        if len(detections) != 1:
            raise RuntimeError(f"RF-DETR returned {len(detections)} detection batches for one input.")
        return detections[0]
    return detections


def _convert_detections_to_frame_schema(detections: Any, class_mapping: dict[int, str]) -> list[dict]:
    xyxy_values = _to_list(getattr(detections, "xyxy", []))
    confidence_values = _to_list(getattr(detections, "confidence", []))
    class_id_values = _to_list(getattr(detections, "class_id", []))
    payload: list[dict] = []

    for idx, coords in enumerate(xyxy_values):
        if len(coords) != 4:
            continue
        class_id = int(class_id_values[idx]) if idx < len(class_id_values) else -1
        confidence = float(confidence_values[idx]) if idx < len(confidence_values) else 0.0
        payload.append(
            {
                "class": _class_name(class_id, class_mapping),
                "confidence": confidence,
                "bbox_xyxy": [int(round(float(value))) for value in coords],
            }
        )
    return payload


def _to_match_schema(frame_batches: list[tuple[int, list[dict]]], max_distance_px: float) -> list[dict]:
    active_tracks: dict[str, list[tuple[int, tuple[float, float]]]] = {}
    next_track_id = 1
    output: list[dict] = []

    for frame_index, detections in frame_batches:
        new_tracks: dict[str, list[tuple[int, tuple[float, float]]]] = {}
        used_track_ids: set[int] = set()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            class_name = det["class"]
            assigned_track_id = None
            best_distance = None

            for track_id, previous_center in active_tracks.get(class_name, []):
                if track_id in used_track_ids:
                    continue
                distance = math.dist(center, previous_center)
                if distance > max_distance_px:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    assigned_track_id = track_id

            if assigned_track_id is None:
                assigned_track_id = next_track_id
                next_track_id += 1

            used_track_ids.add(assigned_track_id)
            new_tracks.setdefault(class_name, []).append((assigned_track_id, center))
            output.append(
                {
                    "frame_index": frame_index,
                    "class": class_name,
                    "confidence": float(det["confidence"]),
                    "track_id": assigned_track_id,
                }
            )
        active_tracks = new_tracks

    return output


def _load_class_mapping(checkpoint_path: Path) -> dict[int, str]:
    run_name = checkpoint_path.parent.name
    project_root = checkpoint_path.parents[3]
    class_mapping_path = project_root / "models" / "metadata" / run_name / "class_mapping.json"
    if not class_mapping_path.exists() or not class_mapping_path.is_file():
        return {}

    try:
        payload = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse class mapping '{class_mapping_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Class mapping '{class_mapping_path}' must contain a JSON object.")
    return {int(key): str(value) for key, value in payload.items()}


def _resolve_detection_confidence(checkpoint_path: Path) -> float:
    default_confidence = 0.25
    run_name = checkpoint_path.parent.name
    project_root = checkpoint_path.parents[3]
    config_path = project_root / "models" / "metadata" / run_name / "config.yaml"
    if not config_path.exists() or not config_path.is_file():
        return default_confidence

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse prediction config '{config_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Saved training config '{config_path}' must contain a mapping at top-level.")

    rfdetr_cfg = payload.get("rfdetr")
    if rfdetr_cfg is None:
        return default_confidence
    if not isinstance(rfdetr_cfg, dict):
        raise RuntimeError(f"Saved training config '{config_path}' has invalid 'rfdetr' section.")

    confidence = _as_float(rfdetr_cfg.get("detection_confidence", default_confidence), key="detection_confidence", minimum=0.0)
    if confidence > 1.0:
        raise ValueError(
            f"Config value 'detection_confidence' must be <= 1.0, got: {confidence} (run={run_name})."
        )
    return confidence


def _class_name(class_id: int, class_mapping: dict[int, str]) -> str:
    if class_id in class_mapping:
        return class_mapping[class_id]

    ordered_names = [class_mapping[key] for key in sorted(class_mapping)]
    if 0 <= class_id < len(ordered_names):
        return ordered_names[class_id]
    return str(class_id)


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _require_mapping(value: Any, *, key: str, config_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Training config '{config_path}' must include a mapping section '{key}'.")
    return value


def _as_int(value: Any, *, key: str, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value '{key}' must be an integer, got: {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"Config value '{key}' must be >= {minimum}, got: {parsed}")
    return parsed


def _as_optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, key=key, minimum=1)


def _as_float(value: Any, *, key: str, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config value '{key}' must be numeric, got: {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"Config value '{key}' must be >= {minimum}, got: {parsed}")
    return parsed


def _as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Config value '{key}' must be boolean, got: {value!r}")


def _as_device(value: Any) -> str:
    parsed = str(value).strip().lower()
    if parsed not in {"cuda", "cpu", "mps"}:
        raise ValueError(f"Config value 'device' must be one of cuda, cpu, mps; got: {value!r}")
    return parsed


def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
