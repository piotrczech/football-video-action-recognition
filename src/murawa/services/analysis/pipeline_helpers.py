from pathlib import Path

from murawa.data.path_resolver import pick_input
from murawa.services.runtime.video_processing import DEFAULT_SAMPLE_FPS, ProgressCallback
from murawa.services.vision.team_assignment_helpers import normalize_class_name


def _make_base_payload(mode: str, model: str, dataset_variant: str) -> dict:
    return {
        "status": "error",
        "mode": mode,
        "model": model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": "",
        "resolved_input": "",
        "output_dir": "",
        "summary_path": "",
        "preview_path": "",
        "preview_assets": [],
        "debug_preview_assets": [],
        "video_path": "",
        "video_metadata": {},
        "sample_fps": DEFAULT_SAMPLE_FPS,
        "sampled_frames": 0,
        "stats": {},
        "tracking": {},
        "render_options": {},
        "team_assignment": {},
        "minimap_entities": [],
        "detections": [],
    }


def _select_primary_ball_detections(detections: list[dict]) -> tuple[list[dict], dict]:
    ball_indexes: list[int] = []

    for idx, detection in enumerate(detections):
        if normalize_class_name(detection.get("class")) == "ball":
            ball_indexes.append(idx)

    if len(ball_indexes) <= 1:
        return detections, {
            "enabled": True,
            "policy": "keep_highest_confidence_ball_per_frame",
            "raw_detection_count": len(detections),
            "filtered_detection_count": len(detections),
            "ball_candidates_before": len(ball_indexes),
            "ball_candidates_after": len(ball_indexes),
        }

    best_ball_idx = max(
        ball_indexes,
        key=lambda idx: float(detections[idx].get("confidence", 0.0)),
    )

    filtered = [
        detection
        for idx, detection in enumerate(detections)
        if normalize_class_name(detection.get("class")) != "ball" or idx == best_ball_idx
    ]

    return filtered, {
        "enabled": True,
        "policy": "keep_highest_confidence_ball_per_frame",
        "raw_detection_count": len(detections),
        "filtered_detection_count": len(filtered),
        "ball_candidates_before": len(ball_indexes),
        "ball_candidates_after": 1,
    }


def _build_detection_stats(detections: list[dict]) -> dict:
    by_class: dict[str, int] = {}
    confidences: list[float] = []
    track_ids: set[int] = set()
    tracked_detections = 0

    for detection in detections:
        class_name = str(detection.get("class", "unknown"))
        by_class[class_name] = by_class.get(class_name, 0) + 1

        confidence = detection.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

        track_id = detection.get("track_id")
        if isinstance(track_id, int) and not isinstance(track_id, bool):
            track_ids.add(track_id)
            tracked_detections += 1

    return {
        "total_detections": len(detections),
        "classes": by_class,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "track_count": len(track_ids),
        "tracked_detections": tracked_detections,
    }


def _resolve_input(
    project_root: Path, mode: str, dataset_variant: str, input_path: str | None
) -> tuple[str, bool]:
    if input_path:
        uploaded = Path(input_path).resolve()
        return str(uploaded), uploaded.exists()

    return pick_input(project_root, mode, dataset_variant)


def _emit_inference_progress(
    *,
    progress_callback: ProgressCallback | None,
    completed: int,
    total: int,
) -> None:
    _emit_progress(
        progress_callback,
        "inference",
        completed / max(1, total),
        f"Running model inference ({completed}/{total} frames).",
    )


def _emit_tracking_progress(
    *,
    progress_callback: ProgressCallback | None,
    completed: int,
    total: int,
    progress_start: float,
    progress_width: float,
    message: str,
) -> None:
    _emit_progress(
        progress_callback,
        "tracking",
        progress_start + progress_width * completed / max(1, total),
        f"{message} ({completed}/{total} frames).",
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, min(1.0, max(0.0, progress)), message)
