from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from murawa.settings import (
    DEFAULT_SAMPLE_FPS,
    MAX_SAMPLE_FPS,
    MAX_VIDEO_DURATION_SECONDS,
    MIN_SAMPLE_FPS,
)

ProgressCallback = Callable[[str, float, str], None]


class VideoProcessingError(RuntimeError):
    """Raised when uploaded video input cannot be prepared for analysis."""


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float

    def as_dict(self) -> dict:
        return {
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": round(self.duration_seconds, 4),
        }


@dataclass(frozen=True)
class SampledFrame:
    sample_index: int
    source_frame_index: int
    timestamp_seconds: float
    path: Path


def validate_sample_fps(sample_fps: int) -> int:
    if isinstance(sample_fps, bool):
        raise VideoProcessingError(
            f"Frame sampling rate must be an integer from {MIN_SAMPLE_FPS} to {MAX_SAMPLE_FPS} FPS."
        )

    try:
        parsed = int(sample_fps)
    except (TypeError, ValueError) as exc:
        message = (
            f"Frame sampling rate must be an integer from {MIN_SAMPLE_FPS} "
            f"to {MAX_SAMPLE_FPS} FPS."
        )
        raise VideoProcessingError(message) from exc

    if parsed < MIN_SAMPLE_FPS or parsed > MAX_SAMPLE_FPS:
        raise VideoProcessingError(
            f"Frame sampling rate must be between {MIN_SAMPLE_FPS} and {MAX_SAMPLE_FPS} FPS."
        )
    return parsed


def validate_video_input(
    input_path: Path,
    *,
    allowed_suffixes: set[str],
    max_duration_seconds: float = MAX_VIDEO_DURATION_SECONDS,
) -> VideoMetadata:
    resolved = input_path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise VideoProcessingError(f"Video input must point to an existing file: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix not in allowed_suffixes:
        raise VideoProcessingError(
            f"Video input suffix '{suffix}' is not supported. "
            f"Expected one of {sorted(allowed_suffixes)}."
        )

    if resolved.stat().st_size <= 0:
        raise VideoProcessingError("Uploaded video file is empty.")

    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        raise VideoProcessingError("Uploaded video could not be opened by OpenCV.")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ok, first_frame = cap.read()
    finally:
        cap.release()

    if not ok or first_frame is None or first_frame.size == 0:
        raise VideoProcessingError("Uploaded video is not decodable into frames.")

    if fps <= 0:
        raise VideoProcessingError("Uploaded video has invalid FPS metadata.")
    if frame_count <= 0:
        raise VideoProcessingError("Uploaded video has invalid frame count metadata.")

    if width <= 0 or height <= 0:
        height, width = first_frame.shape[:2]
    if width <= 0 or height <= 0:
        raise VideoProcessingError("Uploaded video has invalid frame dimensions.")

    duration_seconds = frame_count / fps
    if duration_seconds <= 0:
        raise VideoProcessingError("Uploaded video has invalid duration metadata.")
    if duration_seconds > max_duration_seconds:
        raise VideoProcessingError(
            f"Uploaded video is {duration_seconds:.1f} s long. "
            f"Current limit is {max_duration_seconds:.0f} s."
        )

    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def extract_sampled_frames(
    input_path: Path,
    *,
    output_dir: Path,
    metadata: VideoMetadata,
    sample_fps: int,
    progress_callback: ProgressCallback | None = None,
) -> list[SampledFrame]:
    parsed_sample_fps = validate_sample_fps(sample_fps)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise VideoProcessingError("Video input could not be reopened for frame extraction.")

    selected_frames: list[SampledFrame] = []
    target_interval_seconds = 1.0 / parsed_sample_fps
    next_target_seconds = 0.0
    frame_index = -1
    progress_stride = max(1, metadata.frame_count // 100)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            timestamp_seconds = frame_index / metadata.fps
            if timestamp_seconds + 1e-9 < next_target_seconds:
                _emit_extraction_progress(
                    progress_callback=progress_callback,
                    frame_index=frame_index,
                    frame_count=metadata.frame_count,
                    selected_count=len(selected_frames),
                    stride=progress_stride,
                )
                continue

            frame_path = output_dir / f"sample_{len(selected_frames):06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise VideoProcessingError(f"Could not save sampled frame: {frame_path}")

            selected_frames.append(
                SampledFrame(
                    sample_index=len(selected_frames),
                    source_frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    path=frame_path,
                )
            )
            while next_target_seconds <= timestamp_seconds + 1e-9:
                next_target_seconds += target_interval_seconds

            _emit_extraction_progress(
                progress_callback=progress_callback,
                frame_index=frame_index,
                frame_count=metadata.frame_count,
                selected_count=len(selected_frames),
                stride=progress_stride,
            )
    finally:
        cap.release()

    if not selected_frames:
        raise VideoProcessingError("Video sampling did not produce any frames.")

    if progress_callback is not None:
        progress_callback(
            "extract",
            1.0,
            f"Prepared {len(selected_frames)} sampled frames.",
        )
    return selected_frames


def _emit_extraction_progress(
    *,
    progress_callback: ProgressCallback | None,
    frame_index: int,
    frame_count: int,
    selected_count: int,
    stride: int,
) -> None:
    if progress_callback is None:
        return
    if frame_index % stride != 0 and frame_index + 1 < frame_count:
        return

    progress = min(1.0, max(0.0, (frame_index + 1) / max(1, frame_count)))
    progress_callback(
        "extract",
        progress,
        f"Sampling video frames ({selected_count} ready).",
    )
