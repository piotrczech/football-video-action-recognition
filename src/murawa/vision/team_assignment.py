from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from murawa.vision.team_assignment_helpers import (
    MIN_PLAYER_BOX_HEIGHT_PX,
    MIN_PLAYER_BOX_WIDTH_PX,
    MIN_PLAYER_CONFIDENCE,
    cluster_players_by_jersey_color,
    crop_jersey_region,
    estimate_jersey_color_bgr,
    normalize_class_name,
    read_bbox_xyxy,
)


PLAYER_CLASSES = {"player", "goalkeeper"}
REFEREE_CLASSES = {"referee"}


@dataclass(frozen=True)
class TeamAssignmentResult:
    detections: list[dict[str, Any]]
    summary: dict[str, Any]
    minimap_entities: list[dict[str, Any]]


def assign_teams_to_frame(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> TeamAssignmentResult:
    """Assign player detections to teams using simple jersey-color clustering."""
    enriched = [dict(det) for det in detections]
    players_with_jersey_color: list[tuple[int, np.ndarray]] = []

    for det_idx, det in enumerate(enriched):
        class_name = normalize_class_name(det.get("class"))

        if class_name in REFEREE_CLASSES:
            det["team"] = "referee"
            det["team_confidence"] = 1.0
            continue

        if class_name not in PLAYER_CLASSES:
            continue

        bbox = read_bbox_xyxy(det)
        if bbox is None:
            det["team"] = "unknown"
            det["team_confidence"] = 0.0
            continue

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        confidence = float(det.get("confidence", 0.0))

        if (
            box_width < MIN_PLAYER_BOX_WIDTH_PX
            or box_height < MIN_PLAYER_BOX_HEIGHT_PX
            or confidence < MIN_PLAYER_CONFIDENCE
        ):
            det["team"] = "unknown"
            det["team_confidence"] = 0.0
            continue

        jersey_crop = crop_jersey_region(image_bgr, bbox)
        jersey_color = estimate_jersey_color_bgr(jersey_crop)
        if jersey_color is None:
            det["team"] = "unknown"
            det["team_confidence"] = 0.0
            continue

        det["jersey_color_bgr"] = [int(value) for value in jersey_color.tolist()]
        players_with_jersey_color.append((det_idx, jersey_color.astype(np.float32)))

    team_assignments = cluster_players_by_jersey_color(players_with_jersey_color)

    for det_idx, team, confidence in team_assignments:
        enriched[det_idx]["team"] = team
        enriched[det_idx]["team_confidence"] = round(float(confidence), 4)

    enriched = _smooth_team_by_track_id(enriched)
    minimap_entities = _build_minimap_entities(enriched)

    summary = {
        "enabled": True,
        "method": "jersey_color_kmeans_mvp",
        "player_classes": sorted(PLAYER_CLASSES),
        "referee_classes": sorted(REFEREE_CLASSES),
        "parameters": {
            "min_player_box_width_px": MIN_PLAYER_BOX_WIDTH_PX,
            "min_player_box_height_px": MIN_PLAYER_BOX_HEIGHT_PX,
            "min_player_confidence": MIN_PLAYER_CONFIDENCE,
        },
        "team_counts": _count_player_teams(enriched),
        "minimap_entities": len(minimap_entities),
        "notes": [
            "Team assignment is heuristic and based on jersey color crops.",
            "Referee is kept separately when the model predicts class='referee'.",
            "Track smoothing is applied when track_id is available.",
        ],
    }

    return TeamAssignmentResult(
        detections=enriched,
        summary=summary,
        minimap_entities=minimap_entities,
    )


def _smooth_team_by_track_id(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    votes_by_track: dict[int, Counter[str]] = defaultdict(Counter)

    for det in detections:
        track_id = det.get("track_id")
        team = det.get("team")
        if isinstance(track_id, int) and isinstance(team, str) and team.startswith("team_"):
            votes_by_track[track_id][team] += 1

    if not votes_by_track:
        return detections

    smoothed = [dict(det) for det in detections]
    for det in smoothed:
        track_id = det.get("track_id")
        if not isinstance(track_id, int) or track_id not in votes_by_track:
            continue

        team_votes = votes_by_track[track_id]
        final_team, final_votes = team_votes.most_common(1)[0]
        total_votes = sum(team_votes.values())
        previous_team = det.get("team")

        if previous_team != final_team:
            det["team_raw"] = previous_team
            det["team_confidence_raw"] = det.get("team_confidence")
            det["team_smoothed"] = True

        det["team"] = final_team
        det["team_confidence"] = round(final_votes / total_votes, 4)

    return smoothed


def _count_player_teams(detections: list[dict[str, Any]]) -> dict[str, int]:
    team_counts: Counter[str] = Counter()

    for det in detections:
        class_name = normalize_class_name(det.get("class"))
        if class_name not in PLAYER_CLASSES and class_name not in REFEREE_CLASSES:
            continue
        team_counts[str(det.get("team", "unknown"))] += 1

    return dict(sorted(team_counts.items()))


def _build_minimap_entities(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    for det in detections:
        class_name = normalize_class_name(det.get("class"))
        if class_name not in PLAYER_CLASSES and class_name not in REFEREE_CLASSES:
            continue

        bbox = read_bbox_xyxy(det)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        entity = {
            "class": class_name,
            "team": det.get("team", "unknown"),
            "bbox_xyxy": [x1, y1, x2, y2],
            "center_xy": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
            "confidence": det.get("confidence"),
        }

        if "track_id" in det:
            entity["track_id"] = det["track_id"]
        if "frame_index" in det:
            entity["frame_index"] = det["frame_index"]

        entities.append(entity)

    return entities
