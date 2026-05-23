from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from murawa.models.common import load_checkpoint_config
from murawa.settings import MODELS_METADATA, infer_project_root_from_output_dir

from murawa.models.rfdetr_support import (
    _as_rfdetr_variant,
    _class_name,
    _import_rfdetr,
    _to_list,
)

def _load_prediction_model(checkpoint_path: Path):
    variant = _resolve_prediction_variant(checkpoint_path=checkpoint_path)
    rfdetr_cls = _import_rfdetr(variant)
    try:
        return rfdetr_cls(pretrain_weights=str(checkpoint_path))
    except Exception as exc:
        raise RuntimeError(
            f"RF-DETR backend failed to load checkpoint '{checkpoint_path}' "
            f"for variant='{variant}': {exc}"
        ) from exc


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


def _read_frame_image_rgb(input_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read frame image for RF-DETR prediction: {input_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


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


def _load_class_mapping(checkpoint_path: Path) -> dict[int, str]:
    run_name = checkpoint_path.parent.name
    project_root = infer_project_root_from_output_dir(checkpoint_path.parent)
    class_mapping_path = project_root / MODELS_METADATA / run_name / "class_mapping.json"
    if not class_mapping_path.exists() or not class_mapping_path.is_file():
        return {}

    try:
        payload = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse class mapping '{class_mapping_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Class mapping '{class_mapping_path}' must contain a JSON object.")
    return {int(key): str(value) for key, value in payload.items()}


def _resolve_prediction_variant(checkpoint_path: Path) -> str:
    rfdetr_cfg = _load_rfdetr_config_for_checkpoint(checkpoint_path)
    return _as_rfdetr_variant(rfdetr_cfg.get("variant", "medium"))


def _load_rfdetr_config_for_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    payload = load_checkpoint_config(checkpoint_path)
    rfdetr_cfg = payload.get("rfdetr")
    if rfdetr_cfg is None:
        return {}
    if not isinstance(rfdetr_cfg, dict):
        run_name = checkpoint_path.parent.name
        raise RuntimeError(f"Saved training config for run '{run_name}' has invalid 'rfdetr' section.")
    return rfdetr_cfg

