from dataclasses import dataclass
import csv
import importlib
import math
from pathlib import Path
import random
import shutil
from typing import Any

import cv2
import yaml

from murawa.data import DataLoaderError, LoadedSplit, load_training_split
from murawa.services.artifacts import StandardizedArtifactCallback

@dataclass
class RfDetrMockModel:
    name: str = "rfdetr"

    def train(self, dataset_variant: str) -> dict:
        return {
            "weights": {"decoder.layer_0": [0.17, 0.29, 0.44]},
            "metrics": {"epochs": 1, "loss": 0.97, "mAP50": 0.27},
            "note": f"RF-DETR mock trained on {dataset_variant}",
        }

    def predict(self, mode: str) -> list[dict]:
        if mode == "frame":
            return [
                {"class": "player", "confidence": 0.89, "bbox_xyxy": [98, 76, 244, 342]},
                {"class": "referee", "confidence": 0.77, "bbox_xyxy": [301, 91, 355, 292]},
            ]
        return [
            {"frame_index": 20, "class": "player", "confidence": 0.87, "track_id": 8},
            {"frame_index": 21, "class": "referee", "confidence": 0.75, "track_id": 15},
        ]


@dataclass
class RfDetrAdapter:
    """Issue #11: target integration layer for real RF-DETR backend."""

    name: str = "rfdetr"
    backend: str = "roboflow-rfdetr"

    def train(
        self,
        dataset_variant: str,
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        artifact_callback: StandardizedArtifactCallback | None = None,
    ) -> dict:
        if output_dir is None:
            raise ValueError("RfDetrAdapter.train requires output_dir to persist model checkpoint.")

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        project_root = _resolve_project_root(output_dir=output_dir)

        cfg = _resolve_training_config(config_path)
        _seed_everything(cfg["seed"])

        # 1. Load data splits using common pipeline loader
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

        # 2. Dynamically prepare dataset in YOLO format (RF-DETR supports this structure)
        dataset_root = output_dir / "_rfdetr_dataset"
        data_yaml_path, class_names = _prepare_yolo_dataset(
            train_split=train_split,
            valid_split=valid_split,
            dataset_root=dataset_root,
        )

        rfdetr_cls = _import_rfdetr()
        try:
            # We assume cfg["weights"] provides the base model, e.g. "rfdetr-base.pt"
            model = rfdetr_cls(cfg["weights"])
        except Exception as exc:
            raise RuntimeError(
                f"RF-DETR backend initialization failed for weights='{cfg['weights']}': {exc}"
            ) from exc

        run_root = output_dir / "rfdetr_runs"
        run_root.mkdir(parents=True, exist_ok=True)
        
        # 3. Execute training
        try:
            results = model.train(
                data=str(data_yaml_path),
                epochs=cfg["epochs"],
                batch=cfg["batch_size"],
                imgsz=cfg["image_size"],
                device=cfg["device"],
                lr0=cfg["learning_rate"],
                seed=cfg["seed"],
                workers=0,
                project=str(run_root),
                name="train",
                exist_ok=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"RF-DETR backend training failed: {exc}") from exc

        best_checkpoint = _resolve_best_checkpoint(results)
        checkpoint_path = output_dir / "model.pt"
        shutil.copy2(best_checkpoint, checkpoint_path)

        metrics = _extract_training_metrics(results=results, fallback_epochs=cfg["epochs"])
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
            raise FileNotFoundError(f"Prediction input does not exist: {input_path}")
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"RF-DETR checkpoint does not exist: {checkpoint_path}")

        rfdetr_cls = _import_rfdetr()
        try:
            model = rfdetr_cls(str(checkpoint_path))
        except Exception as exc:
            raise RuntimeError(f"RF-DETR backend failed to load checkpoint '{checkpoint_path}': {exc}") from exc

        detection_confidence = _resolve_detection_confidence(checkpoint_path=checkpoint_path)

        if mode == "frame":
            try:
                results = model.predict(
                    source=str(input_path),
                    conf=detection_confidence,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(f"RF-DETR frame prediction failed: {exc}") from exc
            return _convert_results_to_frame_schema(results)

        # Match mode
        if input_path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                results = model.predict(
                    source=str(input_path),
                    conf=detection_confidence,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(f"RF-DETR match prediction failed for image input: {exc}") from exc
            frame_detections = _extract_frame_detections(results)
            return _to_match_schema(frame_batches=[(0, frame_detections)], max_distance_px=55.0)

        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(
                f"Unsupported file suffix '{input_path.suffix}' for mode='match'. "
                f"Expected image ({sorted(IMAGE_SUFFIXES)}) or video ({sorted(VIDEO_SUFFIXES)})."
            )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video input for RF-DETR prediction: {input_path}")

        frame_step = 5
        frame_index = -1
        frame_batches: list[tuple[int, list[dict]]] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % frame_step != 0:
                    continue
                try:
                    batch_results = model.predict(
                        source=frame,
                        conf=detection_confidence,
                        verbose=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"RF-DETR match prediction failed at frame_index={frame_index}: {exc}"
                    ) from exc

                detections = _extract_frame_detections(batch_results)
                frame_batches.append((frame_index, detections))
        finally:
            cap.release()

        return _to_match_schema(frame_batches=frame_batches, max_distance_px=55.0)


# Replicated utility functions to keep adapters fully symmetric and self-contained

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TRAIN_LOSS_COLUMNS = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
VAL_LOSS_COLUMNS = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
MAP50_COLUMNS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")

def _import_rfdetr():
    try:
        from rfdetr import RFDETR
        return RFDETR
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow RF-DETR backend is unavailable. Install dependencies with: pip install rfdetr"
        ) from exc

def _resolve_training_config(config_path: Path | None) -> dict[str, Any]:
    cfg_path = config_path.resolve() if config_path is not None else None
    raw_cfg: dict[str, Any] = {}
    if cfg_path is not None and cfg_path.exists():
        try:
            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not parse training config '{cfg_path}': {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Training config '{cfg_path}' must contain a mapping at top-level.")
        raw_cfg = payload

    training_cfg = raw_cfg.get("training") if isinstance(raw_cfg.get("training"), dict) else {}
    runtime_cfg = raw_cfg.get("runtime") if isinstance(raw_cfg.get("runtime"), dict) else {}
    rfdetr_cfg = raw_cfg.get("rfdetr") if isinstance(raw_cfg.get("rfdetr"), dict) else {}

    return {
        "epochs": _as_int(rfdetr_cfg.get("epochs", training_cfg.get("epochs", 1)), key="epochs", minimum=1),
        "batch_size": _as_int(rfdetr_cfg.get("batch_size", training_cfg.get("batch_size", 2)), key="batch_size", minimum=1),
        "learning_rate": _as_float(rfdetr_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.001)), key="learning_rate", minimum=0.0),
        "image_size": _as_int(rfdetr_cfg.get("image_size", 320), key="image_size", minimum=64),
        "device": str(rfdetr_cfg.get("device", "cpu")),
        "max_train_samples": _as_optional_int(rfdetr_cfg.get("max_train_samples", 128), "max_train_samples"),
        "max_valid_samples": _as_optional_int(rfdetr_cfg.get("max_valid_samples", 64), "max_valid_samples"),
        "seed": _as_int(runtime_cfg.get("seed", 42), key="seed", minimum=0),
        "weights": str(rfdetr_cfg.get("weights", "rfdetr-base.pt")),
    }

def _resolve_project_root(output_dir: Path) -> Path:
    if len(output_dir.parents) >= 3:
        return output_dir.parents[2]
    return Path(__file__).resolve().parents[3]

def _resolve_detection_confidence(checkpoint_path: Path) -> float:
    default_confidence = 0.25
    try:
        run_name = checkpoint_path.parent.name
        project_root = checkpoint_path.parents[3]
        config_path = project_root / "models" / "metadata" / run_name / "config.yaml"
        if not config_path.exists() or not config_path.is_file():
            return default_confidence

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default_confidence

        rfdetr_cfg = payload.get("rfdetr")
        if not isinstance(rfdetr_cfg, dict):
            return default_confidence

        raw = rfdetr_cfg.get("detection_confidence", default_confidence)
        confidence = _as_float(raw, key="detection_confidence", minimum=0.0)
        return confidence if confidence <= 1.0 else default_confidence
    except Exception:
        return default_confidence

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

def _prepare_yolo_dataset(
    train_split: LoadedSplit,
    valid_split: LoadedSplit,
    dataset_root: Path,
) -> tuple[Path, list[str]]:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    sorted_category_ids = sorted(train_split.class_mapping.keys())
    category_to_idx = {category_id: idx for idx, category_id in enumerate(sorted_category_ids)}
    class_names = [train_split.class_mapping[category_id] for category_id in sorted_category_ids]

    _write_split_as_yolo(train_split, dataset_root / "train", category_to_idx)
    _write_split_as_yolo(valid_split, dataset_root / "valid", category_to_idx)

    data_yaml = dataset_root / "data.yaml"
    yaml_lines = [
        f"path: {dataset_root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "names:",
    ]
    for idx, class_name in enumerate(class_names):
        yaml_lines.append(f"  {idx}: {class_name}")
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return data_yaml, class_names

def _write_split_as_yolo(split: LoadedSplit, split_root: Path, category_to_idx: dict[int, int]) -> None:
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in split.samples:
        dst_stem = f"{sample.image_id}_{sample.image_path.stem}"
        dst_image = images_dir / f"{dst_stem}{sample.image_path.suffix.lower()}"
        dst_label = labels_dir / f"{dst_stem}.txt"

        shutil.copy2(sample.image_path, dst_image)

        label_lines: list[str] = []
        for ann in sample.annotations:
            if ann.category_id not in category_to_idx:
                continue
            x, y, w, h = ann.bbox_xywh
            if w <= 0 or h <= 0:
                continue
            x_center = (x + (w / 2.0)) / sample.width
            y_center = (y + (h / 2.0)) / sample.height
            w_norm = w / sample.width
            h_norm = h / sample.height

            x_center = _clip_unit_interval(x_center)
            y_center = _clip_unit_interval(y_center)
            w_norm = _clip_unit_interval(w_norm)
            h_norm = _clip_unit_interval(h_norm)
            if w_norm <= 0.0 or h_norm <= 0.0:
                continue

            label_lines.append(f"{category_to_idx[ann.category_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        dst_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

def _resolve_best_checkpoint(results: Any) -> Path:
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("RF-DETR training completed but no save_dir was returned.")

    save_dir_path = Path(str(save_dir)).resolve()
    best = save_dir_path / "weights" / "best.pt"
    if best.exists() and best.is_file():
        return best

    last = save_dir_path / "weights" / "last.pt"
    if last.exists() and last.is_file():
        return last

    raise RuntimeError(f"RF-DETR training completed but checkpoint was not found in '{save_dir_path / 'weights'}'.")

def _extract_training_metrics(results: Any, fallback_epochs: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"epochs": fallback_epochs, "loss": 0.0, "mAP50": 0.0}
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        return metrics

    results_csv = Path(str(save_dir)) / "results.csv"
    if not results_csv.exists():
        return metrics

    loss_history: list[float] = []
    val_history: list[float] = []
    map50_last: float | None = None

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_losses = [_read_float(row.get(col)) for col in TRAIN_LOSS_COLUMNS]
            train_loss = sum(v for v in train_losses if v is not None)
            if train_loss > 0:
                loss_history.append(train_loss)

            val_losses = [_read_float(row.get(col)) for col in VAL_LOSS_COLUMNS]
            val_loss = sum(v for v in val_losses if v is not None)
            if val_loss > 0:
                val_history.append(val_loss)

            for col in MAP50_COLUMNS:
                map50_value = _read_float(row.get(col))
                if map50_value is not None:
                    map50_last = map50_value
                    break

    if loss_history:
        metrics["epochs"] = len(loss_history)
        metrics["loss_history"] = loss_history
        metrics["loss"] = loss_history[-1]
    if val_history:
        metrics["val_loss"] = val_history[-1]
    if map50_last is not None:
        metrics["mAP50"] = map50_last
    return metrics

def _convert_results_to_frame_schema(results: Any) -> list[dict]:
    detections = _extract_frame_detections(results)
    return [{"class": det["class"], "confidence": det["confidence"], "bbox_xyxy": det["bbox_xyxy"]} for det in detections]

def _extract_frame_detections(results: Any) -> list[dict]:
    payload: list[dict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []
        confs = boxes.conf.tolist() if hasattr(boxes, "conf") else []
        classes = boxes.cls.tolist() if hasattr(boxes, "cls") else []
        names = getattr(result, "names", {})

        for idx, coords in enumerate(xyxy):
            if len(coords) != 4:
                continue
            class_idx = int(classes[idx]) if idx < len(classes) else -1
            if isinstance(names, dict):
                class_name = names.get(class_idx, str(class_idx))
            elif isinstance(names, (list, tuple)) and 0 <= class_idx < len(names):
                class_name = names[class_idx]
            else:
                class_name = str(class_idx)
            confidence = float(confs[idx]) if idx < len(confs) else 0.0
            payload.append({"class": str(class_name), "confidence": confidence, "bbox_xyxy": [int(round(c)) for c in coords]})
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
            candidates = active_tracks.get(class_name, [])
            best_distance = None

            for track_id, previous_center in candidates:
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
            output.append({"frame_index": frame_index, "class": class_name, "confidence": float(det["confidence"]), "track_id": assigned_track_id})

        active_tracks = new_tracks
    return output

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

def _clip_unit_interval(value: float) -> float:
    if value < 0.0: return 0.0
    if value > 1.0: return 1.0
    return value

def _read_float(value: Any) -> float | None:
    try:
        if value is None or value == "": return None
        return float(value)
    except (TypeError, ValueError):
        return None