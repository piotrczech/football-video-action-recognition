from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2

from murawa.models import build_training_adapter
from murawa.services.analysis.clip_teams import (
    stabilize_match_team_assignments_with_clip_prototypes,
    team_summary_with_clip_prototypes,
)
from murawa.services.analysis.pipeline_helpers import (
    _build_detection_stats,
    _emit_inference_progress,
    _emit_progress,
    _emit_tracking_progress,
    _resolve_input,
)
from murawa.services.rendering.overlay import (
    team_overlay_summary,
    write_annotated_match_video,
)
from murawa.services.runtime.artifacts import write_json
from murawa.services.runtime.video_processing import (
    ProgressCallback,
    SampledFrame,
    VideoProcessingError,
    extract_sampled_frames,
    validate_sample_fps,
    validate_video_input,
)
from murawa.services.vision.team_assignment import assign_teams_to_frame, count_player_teams
from murawa.services.vision.tracking import (
    PRIMARY_BALL_MEMORY_SECONDS,
    TRACK_SMOOTHING_WINDOW_SECONDS,
    build_tracking_summary,
    select_primary_ball_batches,
    smooth_tracked_batches,
    track_frame_batches,
)
from murawa.data.path_resolver import VIDEO_SUFFIXES
from murawa.services.runtime.video_processing import DEFAULT_SAMPLE_FPS
from murawa.settings import OUTPUTS_PREDICTIONS, OUTPUTS_VIDEOS


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
    from murawa.services.analysis.frame_analysis import _run_analysis

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


def analyze_match_run(
    project_root: Path,
    run_name: str,
    input_path: str | None = None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    from murawa.services.analysis.frame_analysis import _run_analysis_for_run

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
    out_dir = project_root / OUTPUTS_PREDICTIONS / run.run_name / analysis_id
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
