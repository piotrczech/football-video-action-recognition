from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2

from murawa.data.path_resolver import IMAGE_SUFFIXES, PREDICTIONS_ROOT, VIDEO_SUFFIXES, pick_input
from murawa.models import build_training_adapter, normalize_model_name
from murawa.services.analysis.clip_teams import (
    stabilize_match_team_assignments_with_clip_prototypes,
    team_summary_with_clip_prototypes,
)
from murawa.services.rendering.overlay import (
    team_overlay_summary,
    write_annotated_match_video,
    write_preview_assets,
    write_team_assignment_debug_preview,
)
from murawa.services.runtime.artifacts import latest_run, resolve_run, write_json
from murawa.services.runtime.video_processing import (
    DEFAULT_SAMPLE_FPS,
    ProgressCallback,
    SampledFrame,
    VideoProcessingError,
    extract_sampled_frames,
    validate_sample_fps,
    validate_video_input,
)
from murawa.services.vision.team_assignment import assign_teams_to_frame, count_player_teams
from murawa.services.vision.team_assignment_helpers import normalize_class_name
from murawa.services.vision.tracking import (
    PRIMARY_BALL_MEMORY_SECONDS,
    TRACK_SMOOTHING_WINDOW_SECONDS,
    build_tracking_summary,
    select_primary_ball_batches,
    smooth_tracked_batches,
    track_frame_batches,
)
from murawa.settings import OUTPUTS_VIDEOS


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
    # --- 1. Resolve run and validate input ---
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
            "No trained run or checkpoint found. Run training first, e.g.: "
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
    # --- 1. Resolve run and validate input ---
    try:
        run = resolve_run(project_root, run_name)
    except FileNotFoundError:
        payload = fallback_payload or _make_base_payload(mode=mode, model="", dataset_variant="")
        payload["status"] = "missing_run"
        payload["message"] = (
            "No trained run or checkpoint found for the selected model. "
            f"Could not find a complete run: {run_name}"
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

    # --- 2. Frame / video extraction ---
    frame_image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if frame_image_bgr is None:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Could not read frame image: {resolved_input}"
        return base_payload

    # --- 3. Model inference ---
    try:
        model_instance = build_training_adapter(normalized_model)
        detections = model_instance.predict(
            input_path=resolved_path,
            checkpoint_path=checkpoint_path,
            mode="frame",
        )
        detections, primary_ball_filter = _select_primary_ball_detections(detections)
    except Exception as exc:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Prediction backend failed for model='{normalized_model}': {exc}"
        return base_payload

    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / f"{mode}_prediction.txt"

    # --- 5. Team assignment ---
    assignment_result = assign_teams_to_frame(
        image_bgr=frame_image_bgr,
        detections=detections,
    )
    detections = assignment_result.detections
    team_assignment = assignment_result.summary
    minimap_entities = assignment_result.minimap_entities

    stats = _build_detection_stats(detections=detections)

    # --- 7. Render outputs and persist artifacts ---
    preview_assets = write_preview_assets(
        detections=detections,
        out_dir=out_dir,
        team_assignment=team_assignment,
        frame_image_bgr=frame_image_bgr.copy(),
    )
    debug_preview_assets = write_team_assignment_debug_preview(
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
        "primary_ball_filter": primary_ball_filter,
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
    # --- 1. Resolve run and validate input ---
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
    videos_dir = project_root / OUTPUTS_VIDEOS
    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / "match_prediction.txt"
    preview_video_path = videos_dir / f"{analysis_id}.webm"
    download_video_path = videos_dir / f"{analysis_id}.mp4"

    video_path = preview_video_path

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    try:
        with TemporaryDirectory(prefix="murawa-match-") as temp_dir:
            # --- 2. Frame / video extraction ---
            sampled_frames = extract_sampled_frames(
                resolved_path,
                output_dir=Path(temp_dir) / "frames",
                metadata=metadata,
                sample_fps=parsed_sample_fps,
                progress_callback=progress_callback,
            )

            # --- 3. Model inference ---
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

            # --- 4. Tracking ---
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

            # --- 5. Team assignment ---
            team_batches, team_summaries = _assign_teams_to_match_frames(
                sampled_frames=sampled_frames,
                frame_batches=tracked_batches,
                progress_callback=progress_callback,
            )

            # --- 6. Clip-level team stabilization ---
            team_batches, clip_team_summary = stabilize_match_team_assignments_with_clip_prototypes(
                team_batches
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
                team_overlay_summary(
                    summary=team_summary_with_clip_prototypes(
                        summary=summary,
                        clip_team_summary=clip_team_summary,
                    ),
                    detections=detections,
                )
                for summary, detections in zip(team_summaries, selected_batches, strict=True)
            ]
            _emit_progress(
                progress_callback,
                "tracking",
                1.0,
                "Tracking postprocessing is ready.",
            )

            # --- 7. Render outputs and persist artifacts ---
            detections = write_annotated_match_video(
                video_path=preview_video_path,
                sampled_frames=sampled_frames,
                frame_batches=selected_batches,
                team_summaries=team_summaries,
                sample_fps=parsed_sample_fps,
                show_boxes=show_boxes,
                show_confidence=show_confidence,
                progress_callback=progress_callback,
            )

            write_annotated_match_video(
                video_path=download_video_path,
                sampled_frames=sampled_frames,
                frame_batches=selected_batches,
                team_summaries=team_summaries,
                sample_fps=parsed_sample_fps,
                show_boxes=show_boxes,
                show_confidence=show_confidence,
                progress_callback=None,
            )
    except Exception as exc:
        preview_video_path.unlink(missing_ok=True)
        download_video_path.unlink(missing_ok=True)
        base_payload["status"] = "error"
        base_payload["output_dir"] = str(out_dir)
        base_payload["video_path"] = str(preview_video_path)
        base_payload["download_video_path"] = str(download_video_path)
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
        "video_path": str(preview_video_path),
        "preview_video_path": str(preview_video_path),
        "download_video_path": str(download_video_path),
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
            "method": "clip_level_jersey_color_prototypes_with_track_smoothing",
            "frame_assignment_method": "jersey_color_kmeans_mvp_per_sampled_frame",
            "scope": "whole_clip",
            "team_counts": stats["team_counts"],
            "clip_level": clip_team_summary,
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
                f"video_path={preview_video_path}",
                f"download_video_path={download_video_path}",
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
