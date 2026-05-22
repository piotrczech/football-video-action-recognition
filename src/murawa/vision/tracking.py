from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv

from murawa.vision.team_assignment_helpers import normalize_class_name, read_bbox_xyxy

TRACKING_METHOD = "bytetrack"
TRACK_SMOOTHING_WINDOW_SECONDS = 15.0
PRIMARY_BALL_MEMORY_SECONDS = 2.0
_TRACK_MATCH_MIN_IOU = 0.05
_BALL_CLASSES = {"ball"}
_TEAM_LABELS = {"team_a", "team_b"}


DetectionBatch = list[dict[str, Any]]
TrackingProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class _TrackingInput:
    detections: sv.Detections
    source_indexes: list[int]


@dataclass(frozen=True)
class _TrackObservation:
    timestamp_seconds: float
    class_name: str
    team: str | None


def track_frame_batches(
    frame_batches: list[DetectionBatch],
    *,
    tracker: Any | None = None,
    progress_callback: TrackingProgressCallback | None = None,
) -> list[DetectionBatch]:
    """Attach ByteTrack IDs to frame detections without dropping detector output."""
    byte_tracker = tracker if tracker is not None else _build_bytetrack_tracker()
    class_ids: dict[str, int] = {}
    tracked_batches: list[DetectionBatch] = []
    total_batches = len(frame_batches)

    for batch_index, batch in enumerate(frame_batches, start=1):
        enriched = [dict(detection, track_id=None) for detection in batch]
        tracking_input = _build_tracking_input(batch=batch, class_ids=class_ids)
        tracked = byte_tracker.update(tracking_input.detections)
        _copy_tracker_ids(
            detections=enriched,
            tracking_input=tracking_input,
            tracked=tracked,
        )
        tracked_batches.append(enriched)
        if progress_callback is not None:
            progress_callback(batch_index, total_batches)

    return tracked_batches


def smooth_tracked_batches(
    frame_batches: list[DetectionBatch],
    *,
    window_seconds: float = TRACK_SMOOTHING_WINDOW_SECONDS,
    position_alpha: float = 0.65,
) -> list[DetectionBatch]:
    """Smooth class/team labels and bbox positions using tracking information.

    Team labels are stabilized offline using all sampled frames from the clip.
    This reduces team flickering and avoids locking a wrong early label.
    """
    if window_seconds <= 0:
        raise ValueError("Tracking smoothing window must be positive.")
    if not 0.0 < position_alpha <= 1.0:
        raise ValueError("Position smoothing alpha must be in (0, 1].")

    global_team_by_track, global_team_confidence_by_track = _global_team_votes_by_track(
        frame_batches
    )

    history: dict[int, deque[_TrackObservation]] = defaultdict(deque)
    previous_bbox_by_track: dict[int, np.ndarray] = {}
    smoothed_batches: list[DetectionBatch] = []

    for batch in frame_batches:
        smoothed_batch: DetectionBatch = []

        for detection in batch:
            enriched = dict(detection)
            track_id = _read_track_id(enriched.get("track_id"))
            timestamp_seconds = _read_timestamp(enriched.get("timestamp_seconds"))

            if track_id is None or timestamp_seconds is None:
                smoothed_batch.append(enriched)
                continue

            observations = history[track_id]
            _discard_stale_observations(
                observations=observations,
                timestamp_seconds=timestamp_seconds,
                window_seconds=window_seconds,
            )

            raw_team = _read_team_label(enriched.get("team"))

            observations.append(
                _TrackObservation(
                    timestamp_seconds=timestamp_seconds,
                    class_name=str(enriched.get("class", "")),
                    team=raw_team,
                )
            )

            final_class = _strict_majority(
                observation.class_name
                for observation in observations
                if normalize_class_name(observation.class_name)
            )
            if final_class is not None:
                enriched["class"] = final_class

            # Offline team stabilization:
            # use all team votes for this track from the whole clip.
            global_team = global_team_by_track.get(track_id)
            if global_team is not None:
                enriched["team_raw"] = raw_team
                enriched["team"] = global_team
                enriched["team_smoothed"] = True
                enriched["team_smoothing"] = "global_track_majority"
                enriched["team_confidence"] = round(
                    global_team_confidence_by_track.get(track_id, 0.0),
                    4,
                )
            else:
                # Fallback for very short/weak tracks.
                team_votes = [
                    observation.team
                    for observation in observations
                    if observation.team in _TEAM_LABELS
                ]
                final_team = _strict_majority(team_votes)
                if final_team is not None:
                    enriched["team_raw"] = raw_team
                    enriched["team"] = final_team
                    enriched["team_smoothed"] = True
                    enriched["team_smoothing"] = "local_window_majority"
                    enriched["team_confidence"] = round(
                        team_votes.count(final_team) / len(team_votes),
                        4,
                    )

            bbox = read_bbox_xyxy(enriched)
            if bbox is not None:
                current_bbox = np.asarray(bbox, dtype=np.float32)
                previous_bbox = previous_bbox_by_track.get(track_id)

                if previous_bbox is not None:
                    smoothed_bbox = (
                        position_alpha * current_bbox
                        + (1.0 - position_alpha) * previous_bbox
                    )
                    enriched["bbox_xyxy"] = [
                        int(round(float(value))) for value in smoothed_bbox.tolist()
                    ]
                    enriched["bbox_smoothed"] = True
                    previous_bbox_by_track[track_id] = smoothed_bbox
                else:
                    previous_bbox_by_track[track_id] = current_bbox

            smoothed_batch.append(enriched)

        smoothed_batches.append(smoothed_batch)

    return smoothed_batches


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


def build_tracking_summary(
    detections: Iterable[dict[str, Any]],
    *,
    window_seconds: float = TRACK_SMOOTHING_WINDOW_SECONDS,
) -> dict[str, Any]:
    track_ids: set[int] = set()
    tracked_detections = 0
    for detection in detections:
        track_id = _read_track_id(detection.get("track_id"))
        if track_id is None:
            continue
        track_ids.add(track_id)
        tracked_detections += 1
    return {
        "enabled": True,
        "method": TRACKING_METHOD,
        "tracked_classes": "all",
        "track_count": len(track_ids),
        "tracked_detections": tracked_detections,
        "smoothing": {
            "enabled": True,
            "window_seconds": float(window_seconds),
            "labels": ["class", "team"],
        },
    }


def _build_bytetrack_tracker() -> Any:
    try:
        from trackers import ByteTrackTracker
    except ImportError as exc:
        raise RuntimeError(
            "ByteTrack backend is unavailable. Install dependencies with: pip install trackers"
        ) from exc
    return ByteTrackTracker()


def _build_tracking_input(batch: DetectionBatch, class_ids: dict[str, int]) -> _TrackingInput:
    xyxy: list[list[float]] = []
    confidences: list[float] = []
    detection_class_ids: list[int] = []
    source_indexes: list[int] = []

    for source_index, detection in enumerate(batch):
        bbox = read_bbox_xyxy(detection)
        if bbox is None:
            continue

        class_key = normalize_class_name(detection.get("class")) or "unknown"
        if class_key not in class_ids:
            class_ids[class_key] = len(class_ids)

        source_indexes.append(source_index)
        xyxy.append([float(value) for value in bbox])
        confidences.append(_read_confidence(detection.get("confidence")))
        detection_class_ids.append(class_ids[class_key])

    detections = sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32).reshape((-1, 4)),
        confidence=np.asarray(confidences, dtype=np.float32),
        class_id=np.asarray(detection_class_ids, dtype=int),
        data={"source_index": np.asarray(source_indexes, dtype=int)},
    )
    return _TrackingInput(detections=detections, source_indexes=source_indexes)


def _copy_tracker_ids(
    *,
    detections: DetectionBatch,
    tracking_input: _TrackingInput,
    tracked: sv.Detections,
) -> None:
    tracker_ids = _tracker_ids(tracked)
    if len(tracker_ids) == 0:
        return

    if _copy_tracker_ids_from_data(detections=detections, tracked=tracked, tracker_ids=tracker_ids):
        return

    original_boxes = np.asarray(tracking_input.detections.xyxy, dtype=np.float32).reshape((-1, 4))
    tracked_boxes = np.asarray(getattr(tracked, "xyxy", []), dtype=np.float32).reshape((-1, 4))
    if len(original_boxes) == 0 or len(tracked_boxes) == 0:
        return

    candidate_matches: list[tuple[float, int, int]] = []
    for tracked_index, tracked_box in enumerate(tracked_boxes):
        if tracked_index >= len(tracker_ids) or tracker_ids[tracked_index] is None:
            continue
        for original_index, original_box in enumerate(original_boxes):
            candidate_matches.append(
                (_bbox_iou(tracked_box, original_box), tracked_index, original_index)
            )

    claimed_tracked_indexes: set[int] = set()
    claimed_original_indexes: set[int] = set()
    for iou, tracked_index, original_index in sorted(candidate_matches, reverse=True):
        if iou < _TRACK_MATCH_MIN_IOU:
            break
        if tracked_index in claimed_tracked_indexes or original_index in claimed_original_indexes:
            continue

        source_index = tracking_input.source_indexes[original_index]
        detections[source_index]["track_id"] = tracker_ids[tracked_index]
        claimed_tracked_indexes.add(tracked_index)
        claimed_original_indexes.add(original_index)


def _copy_tracker_ids_from_data(
    *,
    detections: DetectionBatch,
    tracked: sv.Detections,
    tracker_ids: list[int | None],
) -> bool:
    data = getattr(tracked, "data", {})
    if not isinstance(data, dict):
        return False
    source_indexes = data.get("source_index")
    if source_indexes is None:
        return False

    indexes = np.asarray(source_indexes).reshape((-1,)).tolist()
    if len(indexes) != len(tracker_ids):
        return False

    copied = False
    for source_index, track_id in zip(indexes, tracker_ids, strict=True):
        if track_id is None:
            continue
        try:
            parsed_index = int(source_index)
        except (TypeError, ValueError):
            continue
        if 0 <= parsed_index < len(detections):
            detections[parsed_index]["track_id"] = track_id
            copied = True
    return copied


def _tracker_ids(tracked: sv.Detections) -> list[int | None]:
    raw_tracker_ids = getattr(tracked, "tracker_id", None)
    if raw_tracker_ids is None:
        return []
    return [_read_track_id(track_id) for track_id in np.asarray(raw_tracker_ids).reshape((-1,))]


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


def _read_team_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value in _TEAM_LABELS else None


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


def _discard_stale_observations(
    *,
    observations: deque[_TrackObservation],
    timestamp_seconds: float,
    window_seconds: float,
) -> None:
    cutoff = timestamp_seconds - window_seconds
    while observations and observations[0].timestamp_seconds < cutoff:
        observations.popleft()

def _global_team_votes_by_track(
    frame_batches: list[DetectionBatch],
    *,
    min_observations: int = 2,
    min_winner_share: float = 0.55,
) -> tuple[dict[int, str], dict[int, float]]:
    """Choose one stable team label per track using all sampled frames.

    This is intentionally offline: the whole clip is already processed,
    so later frames can fix an incorrect early team assignment.
    """
    votes_by_track: dict[int, Counter[str]] = defaultdict(Counter)
    observations_by_track: dict[int, int] = defaultdict(int)

    for batch in frame_batches:
        for detection in batch:
            track_id = _read_track_id(detection.get("track_id"))
            team = _read_team_label(detection.get("team"))

            if track_id is None or team not in _TEAM_LABELS:
                continue

            weight = _read_confidence(detection.get("team_confidence"))
            weight = max(0.10, min(1.0, weight))

            votes_by_track[track_id][team] += weight
            observations_by_track[track_id] += 1

    final_team_by_track: dict[int, str] = {}
    confidence_by_track: dict[int, float] = {}

    for track_id, votes in votes_by_track.items():
        if observations_by_track[track_id] < min_observations:
            continue

        winner, winner_score = votes.most_common(1)[0]
        total_score = sum(votes.values())
        if total_score <= 0:
            continue

        winner_share = winner_score / total_score
        if winner_share < min_winner_share:
            continue

        final_team_by_track[track_id] = winner
        confidence_by_track[track_id] = float(winner_share)

    return final_team_by_track, confidence_by_track

def _strict_majority(values: Iterable[str | None]) -> str | None:
    cleaned = [value for value in values if isinstance(value, str) and value]
    if not cleaned:
        return None

    winner, winner_count = Counter(cleaned).most_common(1)[0]
    if winner_count <= len(cleaned) / 2:
        return None
    return winner


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection_x1 = max(float(first[0]), float(second[0]))
    intersection_y1 = max(float(first[1]), float(second[1]))
    intersection_x2 = min(float(first[2]), float(second[2]))
    intersection_y2 = min(float(first[3]), float(second[3]))
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0.0:
        return 0.0

    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union_area = first_area + second_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area
