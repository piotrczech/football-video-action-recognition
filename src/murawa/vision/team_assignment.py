from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PLAYER_CLASSES = {"player", "goalkeeper"}
REFEREE_CLASSES = {"referee"}


@dataclass(frozen=True)
class TeamAssignmentResult:
    detections: list[dict[str, Any]]
    summary: dict[str, Any]
    minimap_entities: list[dict[str, Any]]


def assign_teams_to_frame(image_path: Path, detections: list[dict[str, Any]]) -> TeamAssignmentResult:
    """Assign player detections to teams using jersey colors.

    This is an MVP heuristic intended for post-processing model predictions.
    It does not train any new model. It uses bbox crops, estimates jersey color,
    clusters players into two teams, and keeps referee/unknown separately.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return TeamAssignmentResult(
            detections=detections,
            summary={
                "enabled": False,
                "reason": f"Could not read image: {image_path}",
                "team_counts": {},
            },
            minimap_entities=[],
        )

    enriched = [dict(det) for det in detections]
    player_items: list[tuple[int, np.ndarray]] = []

    for idx, det in enumerate(enriched):
        class_name = _normalize_class(det.get("class"))

        if class_name in REFEREE_CLASSES:
            det["team"] = "referee"
            det["team_confidence"] = 1.0
            continue

        if class_name not in PLAYER_CLASSES:
            det["team"] = "not_applicable"
            det["team_confidence"] = 1.0
            continue

        bbox = _read_bbox(det)
        if bbox is None:
            det["team"] = "unknown"
            det["team_confidence"] = 0.0
            continue

        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        confidence = float(det.get("confidence", 0.0))

        # Very small / low-confidence partial detections usually hurt color clustering.
        if box_w < 15 or box_h < 35 or confidence < 0.45:
            det["team"] = "unknown"
            det["team_confidence"] = 0.0
            continue

        jersey_crop = _crop_jersey_region(image, bbox)
        color = _estimate_jersey_color_bgr(jersey_crop)
        if color is None:
            det["team"] = "unknown"
            det["team_confidence"] = 0.0
            continue

        det["jersey_color_bgr"] = [int(v) for v in color.tolist()]
        player_items.append((idx, color.astype(np.float32)))

    team_labels = _cluster_two_teams(player_items)

    for idx, label, confidence in team_labels:
        enriched[idx]["team"] = label
        enriched[idx]["team_confidence"] = round(float(confidence), 4)

    enriched = _smooth_by_track_id(enriched)
    minimap_entities = _build_minimap_entities(enriched)

    team_counts = Counter(str(det.get("team", "unknown")) for det in enriched)
    summary = {
        "enabled": True,
        "method": "jersey_color_kmeans_mvp",
        "player_classes": sorted(PLAYER_CLASSES),
        "referee_classes": sorted(REFEREE_CLASSES),
        "team_counts": dict(sorted(team_counts.items())),
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


def _normalize_class(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _read_bbox(det: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = det.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _crop_jersey_region(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # Focus on torso/jersey area:
    # - avoid head/top noise,
    # - avoid legs and grass,
    # - use central part of bbox.
    crop_x1 = x1 + int(0.20 * box_w)
    crop_x2 = x2 - int(0.20 * box_w)
    crop_y1 = y1 + int(0.20 * box_h)
    crop_y2 = y1 + int(0.62 * box_h)

    crop_x1 = max(0, min(width - 1, crop_x1))
    crop_x2 = max(crop_x1 + 1, min(width, crop_x2))
    crop_y1 = max(0, min(height - 1, crop_y1))
    crop_y2 = max(crop_y1 + 1, min(height, crop_y2))

    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def _estimate_jersey_color_bgr(crop: np.ndarray) -> np.ndarray | None:
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    # Remove grass-like pixels and very dark/gray pixels.
    # OpenCV hue range is 0..179. Grass usually sits around 35..90.
    grass_mask = (h >= 35) & (h <= 90) & (s >= 35)
    informative_mask = (s >= 35) & (v >= 45) & (~grass_mask)

    pixels = crop[informative_mask]
    if len(pixels) < 8:
        # Fallback: use non-green pixels with weaker constraints.
        fallback_mask = (v >= 35) & (~grass_mask)
        pixels = crop[fallback_mask]

    if len(pixels) == 0:
        return None

    # Median is more stable than mean for small crops and compression artifacts.
    return np.median(pixels.reshape(-1, 3), axis=0).astype(np.uint8)


def _cluster_two_teams(player_items: list[tuple[int, np.ndarray]]) -> list[tuple[int, str, float]]:
    if not player_items:
        return []

    if len(player_items) == 1:
        idx, _ = player_items[0]
        return [(idx, "team_a", 1.0)]

    indexes = [idx for idx, _ in player_items]
    colors_bgr = np.stack([color for _, color in player_items]).astype(np.float32)

    # Use LAB because it is more color-distance friendly than raw BGR.
    colors_lab = _bgr_array_to_lab(colors_bgr)

    centers, assignments = _two_means(colors_lab)

    counts = Counter(assignments.tolist())
    if len(counts) < 2:
        return [(idx, "unknown", 0.0) for idx in indexes]

    # Stable naming: larger cluster = team_a, smaller cluster = team_b.
    ordered_clusters = [cluster for cluster, _ in counts.most_common()]
    cluster_to_team = {
        ordered_clusters[0]: "team_a",
        ordered_clusters[1]: "team_b",
    }

    output: list[tuple[int, str, float]] = []

    for row_idx, det_idx in enumerate(indexes):
        cluster = int(assignments[row_idx])
        team = cluster_to_team.get(cluster, "unknown")

        if team == "unknown":
            output.append((det_idx, "unknown", 0.0))
            continue

        own_dist = float(np.linalg.norm(colors_lab[row_idx] - centers[cluster]))
        other_cluster = 1 - cluster
        other_dist = float(np.linalg.norm(colors_lab[row_idx] - centers[other_cluster]))

        confidence = other_dist / (own_dist + other_dist + 1e-6)
        confidence = max(0.0, min(1.0, confidence))

        output.append((det_idx, team, confidence))

    return output


def _bgr_array_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    colors = colors_bgr.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(colors, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def _two_means(points: np.ndarray, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        centers = np.vstack([points[0], points[0]])
        assignments = np.zeros(len(points), dtype=np.int32)
        return centers, assignments

    # Deterministic initialization: farthest pair.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.stack([points[first], points[second]]).astype(np.float32)

    assignments = np.zeros(len(points), dtype=np.int32)
    for _ in range(iterations):
        dist_to_centers = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        assignments = np.argmin(dist_to_centers, axis=1).astype(np.int32)

        for cluster in (0, 1):
            mask = assignments == cluster
            if np.any(mask):
                centers[cluster] = points[mask].mean(axis=0)

    return centers, assignments


def _smooth_by_track_id(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    votes: dict[int, Counter[str]] = defaultdict(Counter)

    for det in detections:
        track_id = det.get("track_id")
        team = det.get("team")
        if isinstance(track_id, int) and isinstance(team, str) and team.startswith("team_"):
            votes[track_id][team] += 1

    if not votes:
        return detections

    smoothed = [dict(det) for det in detections]
    for det in smoothed:
        track_id = det.get("track_id")
        if not isinstance(track_id, int) or track_id not in votes:
            continue
        det["team_raw"] = det.get("team")
        det["team"] = votes[track_id].most_common(1)[0][0]
        det["team_smoothed"] = True

    return smoothed


def _build_minimap_entities(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    for det in detections:
        class_name = _normalize_class(det.get("class"))
        if class_name not in PLAYER_CLASSES and class_name not in REFEREE_CLASSES:
            continue

        bbox = _read_bbox(det)
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


def team_color_bgr(team: str) -> tuple[int, int, int]:
    colors = {
        "team_a": (255, 80, 80),
        "team_b": (80, 180, 255),
        "referee": (80, 255, 255),
        "unknown": (220, 220, 220),
        "not_applicable": (0, 220, 255),
    }
    return colors.get(str(team), colors["unknown"])