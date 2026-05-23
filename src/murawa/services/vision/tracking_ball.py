from __future__ import annotations

from typing import Any

from murawa.services.vision.team_assignment_helpers import normalize_class_name

PRIMARY_BALL_MEMORY_SECONDS = 2.0
_BALL_CLASSES = {"ball"}

DetectionBatch = list[dict[str, Any]]


def select_primary_ball_batches(
    frame_batches: list[DetectionBatch],
    *,
    memory_seconds: float = PRIMARY_BALL_MEMORY_SECONDS,
) -> list[DetectionBatch]:
    """Keep one rendered ball per frame and reduce short ball flickering.

    If the detector misses the ball for a very short time, reuse the last
    selected ball position as a temporary memory detection.
    """
    if memory_seconds < 0:
        raise ValueError("Primary ball memory window must be non-negative.")

    selected_track_id: int | None = None
    selected_timestamp: float | None = None
    selected_ball_detection: dict[str, Any] | None = None
    selected_batches: list[DetectionBatch] = []

    for batch in frame_batches:
        ball_candidates: list[tuple[int, dict[str, Any]]] = []
        for detection_index, detection in enumerate(batch):
            if _is_ball_detection(detection):
                ball_candidates.append((detection_index, detection))

        batch_timestamp = _batch_timestamp_seconds(batch)

        if not ball_candidates:
            memory_ball = _make_memory_ball_detection(
                selected_ball_detection=selected_ball_detection,
                selected_timestamp=selected_timestamp,
                current_timestamp=batch_timestamp,
                memory_seconds=memory_seconds,
            )
            if memory_ball is not None:
                selected_batches.append([dict(detection) for detection in batch] + [memory_ball])
            else:
                selected_batches.append([dict(detection) for detection in batch])
            continue

        if len(ball_candidates) == 1:
            selected_index, selected_detection = ball_candidates[0]
        else:
            selected_index, selected_detection = _select_primary_ball_candidate(
                ball_candidates=ball_candidates,
                selected_track_id=selected_track_id,
                selected_timestamp=selected_timestamp,
                memory_seconds=memory_seconds,
            )

        selected_batches.append(
            [
                dict(detection)
                for detection_index, detection in enumerate(batch)
                if not _is_ball_detection(detection) or detection_index == selected_index
            ]
        )

        selected_track_id, selected_timestamp = _remember_selected_ball(
            detection=selected_detection,
            fallback_track_id=selected_track_id,
            fallback_timestamp=selected_timestamp,
        )
        selected_ball_detection = dict(selected_detection)

    return selected_batches


def _select_primary_ball_candidate(
    *,
    ball_candidates: list[tuple[int, dict[str, Any]]],
    selected_track_id: int | None,
    selected_timestamp: float | None,
    memory_seconds: float,
) -> tuple[int, dict[str, Any]]:
    recent_track_candidates = [
        candidate
        for candidate in ball_candidates
        if _continues_selected_ball_track(
            detection=candidate[1],
            selected_track_id=selected_track_id,
            selected_timestamp=selected_timestamp,
            memory_seconds=memory_seconds,
        )
    ]
    candidates = recent_track_candidates or ball_candidates
    return max(candidates, key=lambda candidate: _read_confidence(candidate[1].get("confidence")))


def _continues_selected_ball_track(
    *,
    detection: dict[str, Any],
    selected_track_id: int | None,
    selected_timestamp: float | None,
    memory_seconds: float,
) -> bool:
    track_id = _read_track_id(detection.get("track_id"))
    timestamp = _read_timestamp(detection.get("timestamp_seconds"))
    if (
        track_id is None
        or selected_track_id is None
        or timestamp is None
        or selected_timestamp is None
    ):
        return False
    return track_id == selected_track_id and timestamp - selected_timestamp <= memory_seconds


def _remember_selected_ball(
    *,
    detection: dict[str, Any],
    fallback_track_id: int | None,
    fallback_timestamp: float | None,
) -> tuple[int | None, float | None]:
    track_id = _read_track_id(detection.get("track_id"))
    timestamp = _read_timestamp(detection.get("timestamp_seconds"))
    if track_id is None or timestamp is None:
        return fallback_track_id, fallback_timestamp
    return track_id, timestamp


def _batch_timestamp_seconds(batch: DetectionBatch) -> float | None:
    for detection in batch:
        timestamp = _read_timestamp(detection.get("timestamp_seconds"))
        if timestamp is not None:
            return timestamp
    return None


def _make_memory_ball_detection(
    *,
    selected_ball_detection: dict[str, Any] | None,
    selected_timestamp: float | None,
    current_timestamp: float | None,
    memory_seconds: float,
) -> dict[str, Any] | None:
    if (
        selected_ball_detection is None
        or selected_timestamp is None
        or current_timestamp is None
    ):
        return None

    elapsed = current_timestamp - selected_timestamp
    if elapsed < 0 or elapsed > memory_seconds:
        return None

    memory_ball = dict(selected_ball_detection)
    memory_ball["timestamp_seconds"] = round(current_timestamp, 4)
    memory_ball["ball_memory"] = True
    memory_ball["ball_memory_age_seconds"] = round(float(elapsed), 4)

    confidence = _read_confidence(memory_ball.get("confidence"))
    memory_ball["confidence"] = round(max(0.01, confidence * 0.65), 4)

    return memory_ball


def _is_ball_detection(detection: dict[str, Any]) -> bool:
    return normalize_class_name(detection.get("class")) in _BALL_CLASSES


def _read_track_id(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _read_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
