from __future__ import annotations

from pathlib import Path

import cv2

from murawa.services.rendering.overlay_draw import draw_frame_overlay
from murawa.services.runtime.video_processing import ProgressCallback, SampledFrame


def write_annotated_match_video(
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
    return _write_annotated_match_video(
        video_path=video_path,
        sampled_frames=sampled_frames,
        frame_batches=frame_batches,
        team_summaries=team_summaries,
        sample_fps=sample_fps,
        show_boxes=show_boxes,
        show_confidence=show_confidence,
        progress_callback=progress_callback,
    )


def _video_writer_fourcc(video_path: Path) -> int:
    if video_path.suffix.lower() == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter_fourcc(*"VP80")


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
                    _video_writer_fourcc(video_path),
                    float(sample_fps),
                    frame_size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {video_path}")
            elif frame_size is not None and frame_size != (width, height):
                frame_image_bgr = cv2.resize(frame_image_bgr, frame_size)

            draw_frame_overlay(
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
            _emit_render_progress(
                progress_callback=progress_callback,
                render_index=render_index,
                total_frames=len(sampled_frames),
            )
    finally:
        if writer is not None:
            writer.release()

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"Output video was not written: {video_path}")
    return all_detections


def _emit_render_progress(
    *,
    progress_callback: ProgressCallback | None,
    render_index: int,
    total_frames: int,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        "render",
        render_index / max(1, total_frames),
        f"Rendering output video ({render_index}/{total_frames} frames).",
    )
