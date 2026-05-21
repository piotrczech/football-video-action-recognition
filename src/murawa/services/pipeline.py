from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np

from murawa.data.path_resolver import IMAGE_SUFFIXES, PREDICTIONS_ROOT, VIDEO_SUFFIXES, pick_input
from murawa.models import build_training_adapter, normalize_model_name
from murawa.services.artifacts import latest_run, resolve_run, write_json
from murawa.services.video_processing import (
    DEFAULT_SAMPLE_FPS,
    ProgressCallback,
    SampledFrame,
    VideoProcessingError,
    extract_sampled_frames,
    validate_sample_fps,
    validate_video_input,
)
from murawa.vision.team_assignment import (
    PLAYER_CLASSES,
    REFEREE_CLASSES,
    assign_teams_to_frame,
    count_player_teams,
)
from murawa.vision.team_assignment_helpers import (
    crop_jersey_region,
    normalize_class_name,
    read_bbox_xyxy,
    team_preview_color_bgr,
)
from murawa.vision.tracking import (
    PRIMARY_BALL_MEMORY_SECONDS,
    TRACK_SMOOTHING_WINDOW_SECONDS,
    build_tracking_summary,
    select_primary_ball_batches,
    smooth_tracked_batches,
    track_frame_batches,
)


def analyze_frame(
    project_root: Path, model: str, dataset_variant: str, input_path: str | None = None
) -> dict:
    return _run_analysis(project_root, model, dataset_variant, mode="frame", input_path=input_path)


def analyze_match(
    project_root: Path,
    model: str,
    dataset_variant: str,
    input_path: str | None = None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    return _run_analysis(
        project_root,
        model,
        dataset_variant,
        mode="match",
        input_path=input_path,
        sample_fps=sample_fps,
        show_boxes=show_boxes,
        show_confidence=show_confidence,
        progress_callback=progress_callback,
    )


def analyze_frame_run(project_root: Path, run_name: str, input_path: str | None = None) -> dict:
    return _run_analysis_for_run(project_root, run_name, mode="frame", input_path=input_path)


def analyze_match_run(
    project_root: Path,
    run_name: str,
    input_path: str | None = None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    return _run_analysis_for_run(
        project_root,
        run_name,
        mode="match",
        input_path=input_path,
        sample_fps=sample_fps,
        show_boxes=show_boxes,
        show_confidence=show_confidence,
        progress_callback=progress_callback,
    )


def _run_analysis(
    project_root: Path,
    model: str,
    dataset_variant: str,
    mode: str,
    input_path: str | None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    normalized_model = normalize_model_name(model)
    base_payload = _make_base_payload(
        mode=mode,
        model=normalized_model,
        dataset_variant=dataset_variant,
    )

    try:
        run_name = latest_run(project_root, normalized_model, dataset_variant)
    except FileNotFoundError:
        base_payload["status"] = "missing_run"
        base_payload["message"] = (
            "Brak gotowego runu/checkpointu. Najpierw uruchom trening, np.: "
            f"python scripts/train.py --model {normalized_model} "
            f"--dataset-variant {dataset_variant}"
        )
        return base_payload

    return _run_analysis_for_run(
        project_root=project_root,
        run_name=run_name,
        mode=mode,
        input_path=input_path,
        sample_fps=sample_fps,
        show_boxes=show_boxes,
        show_confidence=show_confidence,
        progress_callback=progress_callback,
        fallback_payload=base_payload,
    )


def _run_analysis_for_run(
    project_root: Path,
    run_name: str,
    mode: str,
    input_path: str | None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
    fallback_payload: dict | None = None,
) -> dict:
    try:
        run = resolve_run(project_root, run_name)
    except FileNotFoundError:
        payload = fallback_payload or _make_base_payload(mode=mode, model="", dataset_variant="")
        payload["status"] = "missing_run"
        payload["message"] = (
            "Brak gotowego runu/checkpointu dla wybranego modelu. "
            f"Nie znaleziono kompletnego runu: {run_name}"
        )
        return payload

    normalized_model = normalize_model_name(run.model)
    dataset_variant = run.dataset_variant
    base_payload = _make_base_payload(
        mode=mode,
        model=normalized_model,
        dataset_variant=dataset_variant,
    )
    base_payload["resolved_run_name"] = run.run_name

    if mode == "match":
        return _run_match_video_analysis(
            project_root=project_root,
            run=run,
            normalized_model=normalized_model,
            dataset_variant=dataset_variant,
            input_path=input_path,
            sample_fps=sample_fps,
            show_boxes=show_boxes,
            show_confidence=show_confidence,
            progress_callback=progress_callback,
            base_payload=base_payload,
        )

    out_dir = project_root / PREDICTIONS_ROOT / run.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input, input_found = _resolve_input(project_root, mode, dataset_variant, input_path)
    checkpoint_path = run.checkpoint_path

    if not input_found:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Could not find a valid image input file for {normalized_model}. "
            "Provide --input-path or ensure the file exists in the test directory."
        )
        return base_payload

    resolved_path = Path(resolved_input) if resolved_input else None

    if resolved_path is None or not resolved_path.exists() or not resolved_path.is_file():
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Input path for {normalized_model} backend must point to a file. "
            f"Received: {resolved_input}"
        )
        return base_payload

    suffix = resolved_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Frame mode requires an image file. Received suffix='{suffix}', "
            f"expected one of {sorted(IMAGE_SUFFIXES)}."
        )
        return base_payload

    frame_image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if frame_image_bgr is None:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Could not read frame image: {resolved_input}"
        return base_payload

    try:
        model_instance = build_training_adapter(normalized_model)
        detections = model_instance.predict(
            input_path=resolved_path,
            checkpoint_path=checkpoint_path,
            mode="frame",
        )
    except Exception as exc:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Prediction backend failed for model='{normalized_model}': {exc}"
        return base_payload

    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / f"{mode}_prediction.txt"

    assignment_result = assign_teams_to_frame(
        image_bgr=frame_image_bgr,
        detections=detections,
    )
    detections = assignment_result.detections
    team_assignment = assignment_result.summary
    minimap_entities = assignment_result.minimap_entities

    stats = _build_detection_stats(detections=detections)

    preview_assets = _write_preview_assets(
        detections=detections,
        out_dir=out_dir,
        team_assignment=team_assignment,
        frame_image_bgr=frame_image_bgr.copy(),
    )
    debug_preview_assets = _write_team_assignment_debug_preview(
        frame_image_bgr=frame_image_bgr,
        detections=detections,
        out_dir=out_dir,
    )

    payload = {
        "status": "ok",
        "mode": mode,
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": run.run_name,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(run.metadata_dir),
        "resolved_input": resolved_input,
        "input_found": input_found,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "preview_assets": preview_assets,
        "debug_preview_assets": debug_preview_assets,
        "stats": stats,
        "team_assignment": team_assignment,
        "minimap_entities": minimap_entities,
        "detections": detections,
    }
    write_json(summary_path, payload)

    preview_path.write_text(
        "\n".join(
            [
                f"mode={mode}",
                f"model={normalized_model}",
                f"dataset_variant={dataset_variant}",
                f"resolved_input={resolved_input}",
                f"detections={stats.get('total_detections', 0)}",
                f"classes={stats.get('classes', {})}",
                f"preview_assets={len(preview_assets)}",
                f"debug_preview_assets={len(debug_preview_assets)}",
                "Real adapter output.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _run_match_video_analysis(
    *,
    project_root: Path,
    run,
    normalized_model: str,
    dataset_variant: str,
    input_path: str | None,
    sample_fps: int,
    show_boxes: bool,
    show_confidence: bool,
    progress_callback: ProgressCallback | None,
    base_payload: dict,
) -> dict:
    base_payload["sample_fps"] = sample_fps
    resolved_input, input_found = _resolve_input(project_root, "match", dataset_variant, input_path)
    base_payload["resolved_input"] = resolved_input
    base_payload["input_found"] = input_found

    if not input_found:
        base_payload["status"] = "error"
        base_payload["message"] = (
            "Could not find a valid video input. Upload a video file or pass --input-path."
        )
        return base_payload

    resolved_path = Path(resolved_input)
    try:
        parsed_sample_fps = validate_sample_fps(sample_fps)
        _emit_progress(
            progress_callback,
            "validate",
            0.0,
            "Validating video input.",
        )
        metadata = validate_video_input(resolved_path, allowed_suffixes=VIDEO_SUFFIXES)
        _emit_progress(
            progress_callback,
            "validate",
            1.0,
            "Video input is ready.",
        )
    except VideoProcessingError as exc:
        base_payload["status"] = "error"
        base_payload["message"] = str(exc)
        return base_payload

    analysis_id = _make_match_analysis_id()
    out_dir = project_root / PREDICTIONS_ROOT / run.run_name / analysis_id
    videos_dir = project_root / "outputs" / "videos"
    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / "match_prediction.txt"
    video_path = videos_dir / f"{analysis_id}.webm"

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    try:
        with TemporaryDirectory(prefix="murawa-match-") as temp_dir:
            sampled_frames = extract_sampled_frames(
                resolved_path,
                output_dir=Path(temp_dir) / "frames",
                metadata=metadata,
                sample_fps=parsed_sample_fps,
                progress_callback=progress_callback,
            )

            model_instance = build_training_adapter(normalized_model)
            frame_batches = model_instance.predict_frames(
                frame_paths=[frame.path for frame in sampled_frames],
                checkpoint_path=run.checkpoint_path,
                progress_callback=lambda completed, total: _emit_inference_progress(
                    progress_callback=progress_callback,
                    completed=completed,
                    total=total,
                ),
            )
            if len(frame_batches) != len(sampled_frames):
                raise RuntimeError(
                    "Prediction backend returned a different number of frame batches "
                    f"({len(frame_batches)}) than sampled frames ({len(sampled_frames)})."
                )

            timeline_batches = _add_sample_timeline_batches(
                sampled_frames=sampled_frames,
                frame_batches=frame_batches,
            )
            _emit_progress(
                progress_callback,
                "tracking",
                0.0,
                "Tracking detections across sampled frames.",
            )
            tracked_batches = track_frame_batches(
                timeline_batches,
                progress_callback=lambda completed, total: _emit_tracking_progress(
                    progress_callback=progress_callback,
                    completed=completed,
                    total=total,
                    progress_start=0.0,
                    progress_width=0.45,
                    message="Tracking detections",
                ),
            )
            team_batches, team_summaries = _assign_teams_to_match_frames(
                sampled_frames=sampled_frames,
                frame_batches=tracked_batches,
                progress_callback=progress_callback,
            )
            _emit_progress(
                progress_callback,
                "tracking",
                0.92,
                "Smoothing tracked labels.",
            )
            smoothed_batches = smooth_tracked_batches(
                team_batches,
                window_seconds=TRACK_SMOOTHING_WINDOW_SECONDS,
            )
            selected_batches = select_primary_ball_batches(
                smoothed_batches,
                memory_seconds=PRIMARY_BALL_MEMORY_SECONDS,
            )
            team_summaries = [
                _team_overlay_summary(summary=summary, detections=detections)
                for summary, detections in zip(team_summaries, selected_batches, strict=True)
            ]
            _emit_progress(
                progress_callback,
                "tracking",
                1.0,
                "Tracking postprocessing is ready.",
            )

            detections = _write_annotated_match_video(
                video_path=video_path,
                sampled_frames=sampled_frames,
                frame_batches=selected_batches,
                team_summaries=team_summaries,
                sample_fps=parsed_sample_fps,
                show_boxes=show_boxes,
                show_confidence=show_confidence,
                progress_callback=progress_callback,
            )
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        base_payload["status"] = "error"
        base_payload["output_dir"] = str(out_dir)
        base_payload["video_path"] = str(video_path)
        base_payload["message"] = f"Match video analysis failed: {exc}"
        return base_payload

    stats = _build_detection_stats(detections=detections)
    stats["sampled_frames"] = len(sampled_frames)
    stats["team_counts"] = count_player_teams(detections)
    tracking = build_tracking_summary(
        detections,
        window_seconds=TRACK_SMOOTHING_WINDOW_SECONDS,
    )
    tracking["primary_ball"] = {
        "enabled": True,
        "max_per_frame": 1,
        "memory_seconds": PRIMARY_BALL_MEMORY_SECONDS,
    }

    payload = {
        "status": "ok",
        "mode": "match",
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": run.run_name,
        "analysis_id": analysis_id,
        "checkpoint_path": str(run.checkpoint_path),
        "metadata_path": str(run.metadata_dir),
        "resolved_input": resolved_input,
        "input_found": input_found,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "preview_assets": [],
        "debug_preview_assets": [],
        "video_path": str(video_path),
        "video_metadata": metadata.as_dict(),
        "sample_fps": parsed_sample_fps,
        "sampled_frames": len(sampled_frames),
        "sampled_frame_timeline": [_sampled_frame_timeline_item(frame) for frame in sampled_frames],
        "stats": stats,
        "tracking": tracking,
        "render_options": {
            "show_boxes": bool(show_boxes),
            "show_confidence": bool(show_confidence),
        },
        "team_assignment": {
            "enabled": True,
            "method": "jersey_color_kmeans_mvp_per_sampled_frame",
            "team_counts": stats["team_counts"],
        },
        "minimap_entities": [],
        "detections": detections,
    }
    write_json(summary_path, payload)
    preview_path.write_text(
        "\n".join(
            [
                "mode=match",
                f"model={normalized_model}",
                f"dataset_variant={dataset_variant}",
                f"resolved_input={resolved_input}",
                f"sample_fps={parsed_sample_fps}",
                f"sampled_frames={len(sampled_frames)}",
                f"detections={stats.get('total_detections', 0)}",
                f"tracks={tracking.get('track_count', 0)}",
                f"video_path={video_path}",
                "Sampled-frame video analysis output.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _emit_progress(
        progress_callback,
        "save",
        1.0,
        "Saved video analysis output.",
    )
    return payload


def _write_annotated_match_video(
    *,
    video_path: Path,
    sampled_frames: list[SampledFrame],
    frame_batches: list[list[dict]],
    team_summaries: list[dict],
    sample_fps: int,
    show_boxes: bool,
    show_confidence: bool,
    progress_callback: ProgressCallback | None,
) -> list[dict]:
    writer = None
    frame_size: tuple[int, int] | None = None
    all_detections: list[dict] = []

    try:
        for render_index, (sampled_frame, frame_detections, team_summary) in enumerate(
            zip(sampled_frames, frame_batches, team_summaries, strict=True),
            start=1,
        ):
            frame_image_bgr = cv2.imread(str(sampled_frame.path), cv2.IMREAD_COLOR)
            if frame_image_bgr is None:
                raise RuntimeError(f"Could not read sampled frame: {sampled_frame.path}")

            height, width = frame_image_bgr.shape[:2]
            if writer is None:
                frame_size = (width, height)
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"VP80"),
                    float(sample_fps),
                    frame_size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {video_path}")
            elif frame_size is not None and frame_size != (width, height):
                frame_image_bgr = cv2.resize(frame_image_bgr, frame_size)

            _draw_frame_overlay(
                frame_image_bgr=frame_image_bgr,
                detections=frame_detections,
                team_assignment=team_summary,
                show_boxes=show_boxes,
                show_confidence=show_confidence,
                show_tracking_markers=True,
                status_text=(
                    f"t={sampled_frame.timestamp_seconds:.2f}s "
                    f"source_frame={sampled_frame.source_frame_index}"
                ),
            )
            writer.write(frame_image_bgr)
            all_detections.extend(frame_detections)
            _emit_progress(
                progress_callback,
                "render",
                render_index / max(1, len(sampled_frames)),
                f"Rendering output video ({render_index}/{len(sampled_frames)} frames).",
            )
    finally:
        if writer is not None:
            writer.release()

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"Output video was not written: {video_path}")
    return all_detections


def _add_sample_timeline_batches(
    *,
    sampled_frames: list[SampledFrame],
    frame_batches: list[list[dict]],
) -> list[list[dict]]:
    return [
        [_add_sample_timeline(detection, sampled_frame) for detection in frame_detections]
        for sampled_frame, frame_detections in zip(sampled_frames, frame_batches, strict=True)
    ]


def _assign_teams_to_match_frames(
    *,
    sampled_frames: list[SampledFrame],
    frame_batches: list[list[dict]],
    progress_callback: ProgressCallback | None,
) -> tuple[list[list[dict]], list[dict]]:
    assigned_batches: list[list[dict]] = []
    team_summaries: list[dict] = []

    for batch_index, (sampled_frame, frame_detections) in enumerate(
        zip(sampled_frames, frame_batches, strict=True),
        start=1,
    ):
        frame_image_bgr = cv2.imread(str(sampled_frame.path), cv2.IMREAD_COLOR)
        if frame_image_bgr is None:
            raise RuntimeError(f"Could not read sampled frame: {sampled_frame.path}")

        assignment_result = assign_teams_to_frame(
            image_bgr=frame_image_bgr,
            detections=frame_detections,
        )
        assigned_batches.append(assignment_result.detections)
        team_summaries.append(assignment_result.summary)
        _emit_tracking_progress(
            progress_callback=progress_callback,
            completed=batch_index,
            total=len(sampled_frames),
            progress_start=0.45,
            progress_width=0.45,
            message="Assigning teams",
        )

    return assigned_batches, team_summaries


def _team_overlay_summary(*, summary: dict, detections: list[dict]) -> dict:
    enriched = dict(summary)
    enriched["team_counts"] = count_player_teams(detections)
    return enriched


def _add_sample_timeline(detection: dict, sampled_frame: SampledFrame) -> dict:
    enriched = dict(detection)
    enriched["sample_index"] = sampled_frame.sample_index
    enriched["frame_index"] = sampled_frame.source_frame_index
    enriched["timestamp_seconds"] = round(sampled_frame.timestamp_seconds, 4)
    return enriched


def _sampled_frame_timeline_item(sampled_frame: SampledFrame) -> dict:
    return {
        "sample_index": sampled_frame.sample_index,
        "source_frame_index": sampled_frame.source_frame_index,
        "timestamp_seconds": round(sampled_frame.timestamp_seconds, 4),
    }


def _make_match_analysis_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"match_{timestamp}_{uuid4().hex[:8]}"


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


def _write_preview_assets(
    detections: list[dict],
    out_dir: Path,
    team_assignment: dict,
    frame_image_bgr: np.ndarray,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _write_frame_preview(
            frame_image_bgr=frame_image_bgr,
            detections=detections,
            preview_dir=preview_dir,
            team_assignment=team_assignment,
        )
    except Exception:
        return []


def _write_frame_preview(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    preview_dir: Path,
    team_assignment: dict,
) -> list[str]:
    _draw_frame_overlay(
        frame_image_bgr=frame_image_bgr,
        detections=detections,
        team_assignment=team_assignment,
    )

    preview_path = preview_dir / "frame_preview.jpg"
    if not cv2.imwrite(str(preview_path), frame_image_bgr):
        return []

    return [str(preview_path)]


def _draw_frame_overlay(
    *,
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    team_assignment: dict,
    status_text: str = "",
    show_boxes: bool = True,
    show_confidence: bool = True,
    show_tracking_markers: bool = False,
) -> None:
    for detection in detections:
        bbox = read_bbox_xyxy(detection)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        team = str(detection.get("team", "unknown"))
        color = team_preview_color_bgr(team)
        track_id = _read_detection_track_id(detection)

        if show_tracking_markers:
            marker_geometry = _draw_tracking_marker(
                frame_image_bgr=frame_image_bgr,
                bbox=bbox,
                color=color,
            )
            _draw_tracking_marker_label(
                frame_image_bgr=frame_image_bgr,
                bbox=bbox,
                marker_geometry=marker_geometry,
                label=_build_tracking_marker_label(
                    detection=detection,
                    track_id=track_id,
                    show_confidence=show_confidence,
                ),
            )

        if not show_boxes:
            continue

        cv2.rectangle(frame_image_bgr, (x1, y1), (x2, y2), color, 2)
        if not show_tracking_markers:
            _draw_shadowed_text(
                image_bgr=frame_image_bgr,
                text=_build_bbox_label(detection=detection, show_confidence=show_confidence),
                origin=(x1, max(15, y1 - 6)),
                font_scale=0.5,
                color=color,
                thickness=2,
            )

    legend_bottom = _draw_team_legend(
        image_bgr=frame_image_bgr,
        team_counts=team_assignment.get("team_counts", {}),
        team_colors_bgr=team_assignment.get("team_colors_bgr", {}),
    )

    cv2.putText(
        frame_image_bgr,
        f"detections={len(detections)}",
        (10, legend_bottom + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if status_text:
        image_height = frame_image_bgr.shape[0]
        cv2.putText(
            frame_image_bgr,
            status_text,
            (10, max(28, image_height - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def _draw_tracking_marker(
    *,
    frame_image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> tuple[int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    image_height, image_width = frame_image_bgr.shape[:2]

    radius_x = max(8, min(44, int(round(box_width * 0.48))))
    radius_y = max(4, min(14, int(round(box_height * 0.12))))
    center_x = max(radius_x, min(image_width - radius_x - 1, (x1 + x2) // 2))
    center_y = min(image_height - 2, y2 + radius_y + 2)

    backing = frame_image_bgr.copy()
    cv2.ellipse(
        backing,
        (center_x, center_y),
        (radius_x + 3, radius_y + 3),
        0,
        0,
        180,
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(backing, 0.56, frame_image_bgr, 0.44, 0, frame_image_bgr)

    overlay = frame_image_bgr.copy()
    cv2.ellipse(
        overlay,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        180,
        color,
        -1,
    )
    cv2.addWeighted(overlay, 0.62, frame_image_bgr, 0.38, 0, frame_image_bgr)
    cv2.ellipse(
        frame_image_bgr,
        (center_x, center_y),
        (radius_x + 2, radius_y + 2),
        0,
        0,
        180,
        (0, 0, 0),
        5,
    )
    cv2.ellipse(
        frame_image_bgr,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        180,
        color,
        3,
    )
    return center_x, center_y, radius_y


def _draw_tracking_marker_label(
    *,
    frame_image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker_geometry: tuple[int, int, int],
    label: str,
) -> None:
    if not label:
        return

    center_x, center_y, radius_y = marker_geometry
    font_scale = 0.68
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    image_height, image_width = frame_image_bgr.shape[:2]
    x = max(4, min(image_width - text_width - 4, center_x - text_width // 2))
    y = center_y + radius_y + text_height + 10
    if y + baseline > image_height - 4:
        y = max(text_height + 6, bbox[1] - 10)

    _draw_shadowed_text(
        image_bgr=frame_image_bgr,
        text=label,
        origin=(x, y),
        font_scale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
    )


def _draw_shadowed_text(
    *,
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    shadow_origin = (origin[0] + 2, origin[1] + 2)
    cv2.putText(
        image_bgr,
        text,
        shadow_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _build_tracking_marker_label(
    *,
    detection: dict,
    track_id: int | None,
    show_confidence: bool,
) -> str:
    label = _display_team_label(detection.get("team"))
    if label is None:
        label = _display_class_label(detection.get("class"))
    if track_id is not None:
        label = f"{label} #{track_id}"
    confidence_text = _confidence_text(detection=detection, show_confidence=show_confidence)
    return f"{label}{confidence_text}"


def _build_bbox_label(*, detection: dict, show_confidence: bool) -> str:
    label = _display_class_label(detection.get("class"))
    label = f"{label}{_confidence_text(detection=detection, show_confidence=show_confidence)}"
    team_label = _display_team_label(detection.get("team"))
    if team_label is not None:
        label = f"{label} {team_label}"
    return label


def _confidence_text(*, detection: dict, show_confidence: bool) -> str:
    confidence = detection.get("confidence")
    if show_confidence and isinstance(confidence, (int, float)):
        return f" {float(confidence):.2f}"
    return ""


def _display_class_label(value: object) -> str:
    class_name = normalize_class_name(value) or "object"
    return class_name.replace("_", " ").title()


def _display_team_label(value: object) -> str | None:
    labels = {
        "team_a": "TeamA",
        "team_b": "TeamB",
        "referee": "Referee",
    }
    return labels.get(str(value))


def _read_detection_track_id(detection: dict) -> int | None:
    track_id = detection.get("track_id")
    if isinstance(track_id, int) and not isinstance(track_id, bool):
        return track_id
    return None


def _draw_team_legend(
    image_bgr: np.ndarray,
    team_counts: object,
    team_colors_bgr: object,
) -> int:
    counts = team_counts if isinstance(team_counts, dict) else {}
    team_colors = team_colors_bgr if isinstance(team_colors_bgr, dict) else {}
    teams = ["team_a", "team_b", "referee"]

    x = 10
    y = 10
    row_height = 24
    width = 170
    height = 12 + row_height * len(teams)
    bottom = y + height

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + width, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image_bgr, 0.45, 0, image_bgr)

    for row_idx, team in enumerate(teams):
        row_y = y + 22 + row_idx * row_height
        swatch_color = _legend_color_bgr(team, team_colors)
        cv2.rectangle(image_bgr, (x + 10, row_y - 13), (x + 26, row_y + 3), swatch_color, -1)
        cv2.rectangle(image_bgr, (x + 10, row_y - 13), (x + 26, row_y + 3), (255, 255, 255), 1)
        label = f"{team} {int(counts.get(team, 0))}"
        cv2.putText(
            image_bgr,
            label,
            (x + 34, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

    return bottom


def _legend_color_bgr(team: str, team_colors_bgr: dict) -> tuple[int, int, int]:
    color = team_colors_bgr.get(team)
    if isinstance(color, list) and len(color) == 3:
        try:
            return tuple(int(value) for value in color)
        except (TypeError, ValueError):
            pass
    return team_preview_color_bgr(team)


def _write_team_assignment_debug_preview(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    out_dir: Path,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    debug_image = _build_team_assignment_crop_sheet(frame_image_bgr, detections)
    debug_path = preview_dir / "team_assignment_crops.jpg"
    if not cv2.imwrite(str(debug_path), debug_image):
        return []
    return [str(debug_path)]


def _build_team_assignment_crop_sheet(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    tile_width = 170
    tile_height = 116
    crop_size = 64
    columns = 4
    padding = 10

    debug_items = _team_assignment_debug_items(frame_image_bgr, detections)
    if not debug_items:
        image = np.full((90, 380, 3), 32, dtype=np.uint8)
        cv2.putText(
            image,
            "no team assignment crops",
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
        )
        return image

    rows = int(np.ceil(len(debug_items) / columns))
    sheet_height = padding + rows * tile_height
    sheet_width = padding + columns * tile_width
    sheet = np.full((sheet_height, sheet_width, 3), 32, dtype=np.uint8)

    for idx, item in enumerate(debug_items):
        col = idx % columns
        row = idx // columns
        x = padding + col * tile_width
        y = padding + row * tile_height

        cv2.rectangle(sheet, (x, y), (x + tile_width - 8, y + tile_height - 8), (58, 58, 58), -1)
        crop = cv2.resize(item["crop"], (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        sheet[y + 8 : y + 8 + crop_size, x + 8 : x + 8 + crop_size] = crop

        label = item["team"]
        confidence = item.get("team_confidence")
        if isinstance(confidence, (int, float)):
            label = f"{label} {float(confidence):.2f}"
        cv2.putText(
            sheet,
            label,
            (x + 8, y + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
        )
        cv2.putText(
            sheet,
            item["class_name"],
            (x + 8, y + 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
        )

        jersey_color = item.get("jersey_color_bgr")
        if jersey_color is not None:
            cv2.rectangle(sheet, (x + 82, y + 12), (x + 126, y + 38), jersey_color, -1)
            cv2.rectangle(sheet, (x + 82, y + 12), (x + 126, y + 38), (255, 255, 255), 1)

    return sheet


def _team_assignment_debug_items(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
) -> list[dict]:
    items: list[dict] = []

    for det in detections:
        class_name = normalize_class_name(det.get("class"))
        if class_name not in PLAYER_CLASSES and class_name not in REFEREE_CLASSES:
            continue

        bbox = read_bbox_xyxy(det)
        if bbox is None:
            continue

        crop = crop_jersey_region(frame_image_bgr, bbox)
        if crop.size == 0:
            continue

        jersey_color = _read_color_bgr(det.get("jersey_color_bgr"))
        items.append(
            {
                "class_name": class_name,
                "team": str(det.get("team", "unknown")),
                "team_confidence": det.get("team_confidence"),
                "jersey_color_bgr": jersey_color,
                "crop": crop,
            }
        )

    return items


def _read_color_bgr(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return tuple(int(channel) for channel in value)
    except (TypeError, ValueError):
        return None


def _resolve_input(
    project_root: Path, mode: str, dataset_variant: str, input_path: str | None
) -> tuple[str, bool]:
    if input_path:
        uploaded = Path(input_path).resolve()
        return str(uploaded), uploaded.exists()

    return pick_input(project_root, mode, dataset_variant)
