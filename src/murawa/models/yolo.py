import csv
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import random
import shutil
from typing import Any

import yaml

from murawa.data import LoadedSplit
from murawa.models.training_common import (
    load_train_valid_splits,
    load_training_config_payload,
    read_training_sections,
)
from murawa.models.common import (
    IMAGE_SUFFIXES,
    as_float,
    as_int,
    as_optional_int,
    require_mapping,
    resolve_detection_confidence,
    infer_project_root_from_output_dir,
    sampling_summary_to_dict,
    seed_everything,
    validate_image_frame_path,
)
from murawa.services.runtime.artifacts import StandardizedArtifactCallback


@dataclass
class YoloAdapter:
    """Issue #10: target integration layer for real YOLO backend."""

    name: str = "yolo"
    backend: str = "ultralytics-yolo"

    def train(
        self,
        dataset_variant: str,
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        artifact_callback: StandardizedArtifactCallback | None = None,
        amp: bool | None = None,
        device: str | None = None,
    ) -> dict:
        if output_dir is None:
            raise ValueError("YoloAdapter.train requires output_dir to persist model checkpoint.")

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        project_root = infer_project_root_from_output_dir(output_dir)

        cfg = _resolve_training_config(config_path)
        if amp is not None:
            cfg["amp"] = amp
        if device is not None:
            cfg["device"] = device
        seed_everything(cfg["seed"])

        splits = load_train_valid_splits(
            project_root=project_root,
            dataset_variant=dataset_variant,
            max_train_samples=cfg["max_train_samples"],
            max_valid_samples=cfg["max_valid_samples"],
            sampling_seed=cfg["seed"],
            backend_name="YOLO",
        )
        train_split = splits.train_split
        valid_split = splits.valid_split
        valid_split_name = splits.valid_split_source

        dataset_root = project_root / "_yolo_ds" / output_dir.name
        data_yaml_path, class_names = _prepare_ultralytics_dataset(
            train_split=train_split,
            valid_split=valid_split,
            dataset_root=dataset_root,
        )

        yolo_cls = _import_ultralytics_yolo()
        _patch_ultralytics_if_polars_missing()
        try:
            model = yolo_cls(cfg["weights"])
        except Exception as exc:
            raise RuntimeError(
                f"YOLO backend initialization failed for weights='{cfg['weights']}': {exc}"
            ) from exc

        run_root = output_dir / "ultralytics_runs"
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            train_kwargs: dict[str, Any] = {
                "data": str(data_yaml_path),
                "epochs": cfg["epochs"],
                "batch": cfg["batch_size"],
                "imgsz": cfg["image_size"],
                "lr0": cfg["learning_rate"],
                "seed": cfg["seed"],
                "workers": 0,
                "project": str(run_root),
                "name": "train",
                "exist_ok": True,
                "verbose": False,
                "plots": False,
                "pretrained": True,
                "amp": cfg["amp"],
            }
            if cfg["device"] is not None:
                train_kwargs["device"] = cfg["device"]

            results = model.train(**train_kwargs)
        except Exception as exc:
            raise RuntimeError(f"YOLO backend training failed: {exc}") from exc

        best_checkpoint = _resolve_best_checkpoint(results)
        checkpoint_path = output_dir / "model.pt"
        shutil.copy2(best_checkpoint, checkpoint_path)

        metrics = _extract_training_metrics(results=results, fallback_epochs=cfg["epochs"])
        note = (
            f"YOLO adapter trained on {dataset_variant} "
            f"(train={len(train_split.samples)}, val_source={valid_split_name}, val={len(valid_split.samples)})."
        )

        return {
            "weights": {
                "checkpoint_path": str(checkpoint_path),
                "class_names": class_names,
            },
            "metrics": metrics,
            "note": note,
            "backend": self.backend,
            "train_device": str(cfg["device"]) if cfg["device"] is not None else "auto",
            "train_amp": bool(cfg["amp"]),
            "train_samples": len(train_split.samples),
            "valid_samples": len(valid_split.samples),
            "valid_split_source": valid_split_name,
            "train_sampling_summary": sampling_summary_to_dict(train_split),
            "valid_sampling_summary": sampling_summary_to_dict(valid_split),
        }

    def predict(
        self,
        input_path: Path,
        *,
        checkpoint_path: Path,
        mode: str,
    ) -> list[dict]:
        if mode != "frame":
            raise ValueError("YoloAdapter.predict supports only mode='frame'.")

        frame_batches = self.predict_frames(
            frame_paths=[input_path],
            checkpoint_path=checkpoint_path,
        )
        return frame_batches[0]

    def predict_frames(
        self,
        frame_paths: list[Path],
        *,
        checkpoint_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[dict]]:
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint does not exist: {checkpoint_path}")

        yolo_cls = _import_ultralytics_yolo()
        try:
            model = yolo_cls(str(checkpoint_path))
        except Exception as exc:
            raise RuntimeError(
                f"YOLO backend failed to load checkpoint '{checkpoint_path}': {exc}"
            ) from exc

        detection_confidence = resolve_detection_confidence(checkpoint_path=checkpoint_path, section="yolo")
        frame_batches: list[list[dict]] = []
        total_frames = len(frame_paths)

        for frame_number, frame_path in enumerate(frame_paths, start=1):
            resolved_frame = validate_image_frame_path(frame_path, backend_name="YOLO")

            try:
                results = model.predict(
                    source=str(resolved_frame),
                    conf=detection_confidence,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"YOLO sampled-frame prediction failed for '{resolved_frame.name}': {exc}"
                ) from exc

            frame_batches.append(_convert_results_to_frame_schema(results))
            if progress_callback is not None:
                progress_callback(frame_number, total_frames)

        return frame_batches


TRAIN_LOSS_COLUMNS = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
VAL_LOSS_COLUMNS = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
MAP50_COLUMNS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
PRECISION_COLUMN = "metrics/precision(B)"
RECALL_COLUMN = "metrics/recall(B)"
MAP5095_COLUMN = "metrics/mAP50-95(B)"


def _resolve_training_config(config_path: Path | None) -> dict[str, Any]:
    payload, cfg_path = load_training_config_payload(config_path, backend_name="YoloAdapter")
    training_cfg, runtime_cfg = read_training_sections(payload, cfg_path)
    yolo_cfg = require_mapping(payload.get("yolo"), key="yolo", config_path=cfg_path)

    return {
        "epochs": as_int(yolo_cfg.get("epochs", training_cfg.get("epochs", 1)), key="epochs", minimum=1),
        "batch_size": as_int(
            yolo_cfg.get("batch_size", training_cfg.get("batch_size", 2)),
            key="batch_size",
            minimum=1,
        ),
        "learning_rate": as_float(
            yolo_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.001)),
            key="learning_rate",
            minimum=0.0,
        ),
        "amp": _as_bool(yolo_cfg.get("amp", True), key="amp"),
        "image_size": as_int(yolo_cfg.get("image_size", 320), key="image_size", minimum=64),
        "device": _as_optional_device(yolo_cfg.get("device")),
        "max_train_samples": as_optional_int(yolo_cfg.get("max_train_samples", None), "max_train_samples"),
        "max_valid_samples": as_optional_int(yolo_cfg.get("max_valid_samples", None), "max_valid_samples"),
        "seed": as_int(runtime_cfg.get("seed", 42), key="seed", minimum=0),
        "weights": str(yolo_cfg.get("weights", "yolov8n.pt")),
    }


def _prepare_ultralytics_dataset(
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
        dst_stem = f"{sample.image_id}_{_compact_stem(sample.image_path.stem)}"
        dst_image = images_dir / f"{dst_stem}{sample.image_path.suffix.lower()}"
        dst_label = labels_dir / f"{dst_stem}.txt"

        try:
            shutil.copy2(sample.image_path, dst_image)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to copy image. source='{sample.image_path}' target='{dst_image}'"
            ) from exc

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

            label_lines.append(
                f"{category_to_idx[ann.category_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        dst_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def _compact_stem(stem: str, max_len: int = 80) -> str:
    if len(stem) <= max_len:
        return stem
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:10]
    prefix_len = max_len - len(digest) - 1
    return f"{stem[:prefix_len]}-{digest}"


def _resolve_best_checkpoint(results: Any) -> Path:
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("YOLO training completed but no save_dir was returned by backend.")

    save_dir_path = Path(str(save_dir)).resolve()
    best = save_dir_path / "weights" / "best.pt"
    if best.exists() and best.is_file():
        return best

    last = save_dir_path / "weights" / "last.pt"
    if last.exists() and last.is_file():
        return last

    raise RuntimeError(
        f"YOLO training completed but checkpoint was not found in '{save_dir_path / 'weights'}'."
    )


def _extract_training_metrics(results: Any, fallback_epochs: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "epochs": fallback_epochs,
        "loss": 0.0,
        "mAP50": 0.0,
    }

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        return metrics

    results_csv = Path(str(save_dir)) / "results.csv"
    if not results_csv.exists():
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

            map50_value = _read_float(row.get("metrics/mAP50(B)"))
            if map50_value is not None:
                map50_history.append(map50_value)
                map50_last = map50_value
            else:
                for col in MAP50_COLUMNS:
                    fallback_map50 = _read_float(row.get(col))
                    if fallback_map50 is not None:
                        map50_last = fallback_map50
                        break

            map5095_value = _read_float(row.get(MAP5095_COLUMN))
            if map5095_value is not None:
                map5095_history.append(map5095_value)
                map5095_last = map5095_value

            precision_value = _read_float(row.get(PRECISION_COLUMN))
            if precision_value is not None:
                precision_history.append(precision_value)
                precision_last = precision_value

            recall_value = _read_float(row.get(RECALL_COLUMN))
            if recall_value is not None:
                recall_history.append(recall_value)
                recall_last = recall_value

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


def _convert_results_to_frame_schema(results: Any) -> list[dict]:
    detections = _extract_frame_detections(results)
    payload: list[dict] = []
    for det in detections:
        payload.append(
            {
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox_xyxy": det["bbox_xyxy"],
            }
        )
    return payload


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
            payload.append(
                {
                    "class": str(class_name),
                    "confidence": confidence,
                    "bbox_xyxy": [int(round(c)) for c in coords],
                }
            )
    return payload


def _import_ultralytics_yolo():
    try:
        module = importlib.import_module("ultralytics")
        yolo_cls = getattr(module, "YOLO")
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics YOLO backend is unavailable. Install dependencies with: pip install ultralytics"
        ) from exc
    return yolo_cls


def _patch_ultralytics_if_polars_missing() -> None:
    if importlib.util.find_spec("polars") is not None:
        return

    try:
        trainer_module = importlib.import_module("ultralytics.engine.trainer")
    except Exception:
        return

    base_trainer = getattr(trainer_module, "BaseTrainer", None)
    if base_trainer is None or getattr(base_trainer, "_murawa_polars_patch", False):
        return

    def _read_results_csv_without_polars(self):
        # Keep training/checkpoint save working when optional polars is unavailable.
        return {}

    base_trainer.read_results_csv = _read_results_csv_without_polars
    base_trainer._murawa_polars_patch = True


def _as_optional_device(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config value '{key}' must be a boolean, got: {value!r}")


def _clip_unit_interval(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _read_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
