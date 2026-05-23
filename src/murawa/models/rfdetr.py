from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import shutil
from pathlib import Path
from typing import Any

from murawa.models.common import (
    as_bool,
    as_float,
    as_int,
    as_optional_int,
    infer_project_root_from_output_dir,
    require_mapping,
    resolve_detection_confidence,
    sampling_summary_to_dict,
    seed_everything,
    validate_image_frame_path,
)
from murawa.models.rfdetr_inference import (
    _convert_detections_to_frame_schema,
    _load_class_mapping,
    _load_prediction_model,
    _predict_image,
    _read_frame_image_rgb,
)
from murawa.models.rfdetr_support import (
    RFDETR_DEFAULT_RESOLUTIONS,
    RFDETR_RESOLUTION_BLOCK,
    _as_device,
    _as_rfdetr_variant,
    _import_rfdetr,
)
from murawa.models.rfdetr_training import (
    _build_rfdetr_model,
    _extract_training_metrics,
    _maybe_quiet_backend_logs,
    _prepare_coco_dataset,
    _resolve_best_checkpoint,
    _rfdetr_progress_monitor,
)
from murawa.models.training_common import (
    load_train_valid_splits,
    load_training_config_payload,
    read_training_sections,
)
from murawa.services.runtime.artifacts import StandardizedArtifactCallback

logger = logging.getLogger(__name__)


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
        project_root = infer_project_root_from_output_dir(output_dir)
        cfg = _resolve_training_config(config_path)
        if device is not None:
            cfg["device"] = device
        seed_everything(cfg["seed"])
        rfdetr_cls = _import_rfdetr(cfg["variant"])
        logger.info(
            "RF-DETR config: variant=%s resolution=%s batch_size=%s "
            "grad_accum_steps=%s effective_batch_size=%s device=%s",
            cfg["variant"],
            cfg["resolution"],
            cfg["batch_size"],
            cfg["grad_accum_steps"],
            cfg["effective_batch_size"],
            cfg["device"],
        )

        splits = load_train_valid_splits(
            project_root=project_root,
            dataset_variant=dataset_variant,
            max_train_samples=cfg["max_train_samples"],
            max_valid_samples=cfg["max_valid_samples"],
            sampling_seed=cfg["seed"],
            backend_name="RF-DETR",
        )
        train_split = splits.train_split
        valid_split = splits.valid_split
        valid_split_name = splits.valid_split_source

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
                    model = _build_rfdetr_model(
                        rfdetr_cls=rfdetr_cls,
                        weights=cfg["weights"],
                        variant=cfg["variant"],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "RF-DETR backend initialization failed for "
                        f"variant='{cfg['variant']}', weights='{cfg['weights']}': {exc}"
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
            "backend": self.backend,
            "rfdetr_variant": cfg["variant"],
            "train_device": str(cfg["device"]),
            "train_amp": None,
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
            raise ValueError("RfDetrAdapter.predict supports only mode='frame'.")

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
            raise FileNotFoundError(f"RF-DETR checkpoint does not exist: {checkpoint_path}")

        model = _load_prediction_model(checkpoint_path=checkpoint_path)

        class_mapping = _load_class_mapping(checkpoint_path=checkpoint_path)
        detection_confidence = resolve_detection_confidence(checkpoint_path=checkpoint_path, section="rfdetr")
        frame_batches: list[list[dict]] = []
        total_frames = len(frame_paths)

        for frame_number, frame_path in enumerate(frame_paths, start=1):
            resolved_frame = validate_image_frame_path(frame_path, backend_name="RF-DETR")

            frame_rgb = _read_frame_image_rgb(resolved_frame)
            detections = _predict_image(
                model=model,
                image=frame_rgb,
                threshold=detection_confidence,
            )
            frame_batches.append(_convert_detections_to_frame_schema(detections, class_mapping))
            if progress_callback is not None:
                progress_callback(frame_number, total_frames)

        return frame_batches


def _resolve_training_config(config_path: Path | None) -> dict[str, Any]:
    payload, cfg_path = load_training_config_payload(config_path, backend_name="RfDetrAdapter")
    training_cfg, runtime_cfg = read_training_sections(payload, cfg_path)
    rfdetr_cfg = require_mapping(payload.get("rfdetr"), key="rfdetr", config_path=cfg_path)
    variant = _as_rfdetr_variant(rfdetr_cfg.get("variant", "medium"))
    resolution = as_int(
        rfdetr_cfg.get(
            "resolution",
            training_cfg.get("resolution", RFDETR_DEFAULT_RESOLUTIONS[variant]),
        ),
        key="resolution",
        minimum=RFDETR_RESOLUTION_BLOCK,
    )
    if resolution % RFDETR_RESOLUTION_BLOCK != 0:
        raise ValueError(
            "Config value 'resolution' must be divisible by 32 for RF-DETR "
            "(patch_size=16, num_windows=2), got: "
            f"{resolution}"
        )
    batch_size = as_int(
        rfdetr_cfg.get("batch_size", training_cfg.get("batch_size", 2)),
        key="batch_size",
        minimum=1,
    )
    grad_accum_steps = as_int(
        rfdetr_cfg.get("grad_accum_steps", 4),
        key="grad_accum_steps",
        minimum=1,
    )

    return {
        "variant": variant,
        "epochs": as_int(
            rfdetr_cfg.get("epochs", training_cfg.get("epochs", 1)),
            key="epochs",
            minimum=1,
        ),
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": as_float(
            rfdetr_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.0001)),
            key="learning_rate",
            minimum=0.0,
        ),
        "resolution": resolution,
        "device": _as_device(rfdetr_cfg.get("device", "cuda")),
        "checkpoint_interval": as_int(
            rfdetr_cfg.get("checkpoint_interval", 10),
            key="checkpoint_interval",
            minimum=1,
        ),
        "max_train_samples": as_optional_int(rfdetr_cfg.get("max_train_samples"), "max_train_samples"),
        "max_valid_samples": as_optional_int(rfdetr_cfg.get("max_valid_samples"), "max_valid_samples"),
        "seed": as_int(runtime_cfg.get("seed", 42), key="seed", minimum=0),
        "weights": str(rfdetr_cfg.get("weights", "default")).strip(),
        "tensorboard": as_bool(rfdetr_cfg.get("tensorboard", True), key="tensorboard"),
        "multi_scale": as_bool(rfdetr_cfg.get("multi_scale", True), key="multi_scale"),
        "log_per_class_metrics": as_bool(
            rfdetr_cfg.get("log_per_class_metrics", True),
            key="log_per_class_metrics",
        ),
        "quiet": as_bool(rfdetr_cfg.get("quiet", False), key="quiet"),
    }


__all__ = ["RfDetrAdapter", "_extract_training_metrics"]
