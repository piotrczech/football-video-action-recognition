from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np


MIN_PLAYER_BOX_WIDTH_PX = 15
MIN_PLAYER_BOX_HEIGHT_PX = 35
MIN_PLAYER_CONFIDENCE = 0.45

JERSEY_CROP_X_MARGIN_RATIO = 0.20
JERSEY_CROP_Y_START_RATIO = 0.20
JERSEY_CROP_Y_END_RATIO = 0.62

GRASS_HUE_MIN = 35
GRASS_HUE_MAX = 90
MIN_GRASS_SATURATION = 35
MIN_JERSEY_SATURATION = 35
MIN_JERSEY_VALUE = 45
MIN_FALLBACK_VALUE = 35
MIN_INFORMATIVE_PIXELS = 8

COLOR_CLUSTER_ITERATIONS = 12


def normalize_class_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def read_bbox_xyxy(det: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = det.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_jersey_region(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    crop_x1 = x1 + int(JERSEY_CROP_X_MARGIN_RATIO * box_width)
    crop_x2 = x2 - int(JERSEY_CROP_X_MARGIN_RATIO * box_width)
    crop_y1 = y1 + int(JERSEY_CROP_Y_START_RATIO * box_height)
    crop_y2 = y1 + int(JERSEY_CROP_Y_END_RATIO * box_height)

    crop_x1 = max(0, min(width - 1, crop_x1))
    crop_x2 = max(crop_x1 + 1, min(width, crop_x2))
    crop_y1 = max(0, min(height - 1, crop_y1))
    crop_y2 = max(crop_y1 + 1, min(height, crop_y2))

    return image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]


def estimate_jersey_color_bgr(jersey_crop: np.ndarray) -> np.ndarray | None:
    if jersey_crop.size == 0:
        return None

    hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    grass_pixels = (
        (hue >= GRASS_HUE_MIN)
        & (hue <= GRASS_HUE_MAX)
        & (saturation >= MIN_GRASS_SATURATION)
    )
    jersey_pixels = (
        (saturation >= MIN_JERSEY_SATURATION)
        & (value >= MIN_JERSEY_VALUE)
        & (~grass_pixels)
    )

    pixels = jersey_crop[jersey_pixels]
    if len(pixels) < MIN_INFORMATIVE_PIXELS:
        pixels = jersey_crop[(value >= MIN_FALLBACK_VALUE) & (~grass_pixels)]

    if len(pixels) == 0:
        return None

    return np.median(pixels.reshape(-1, 3), axis=0).astype(np.uint8)


def cluster_players_by_jersey_color(
    players_with_jersey_color: list[tuple[int, np.ndarray]],
) -> list[tuple[int, str, float]]:
    if not players_with_jersey_color:
        return []

    if len(players_with_jersey_color) == 1:
        det_idx, _ = players_with_jersey_color[0]
        return [(det_idx, "team_a", 1.0)]

    indexes = [det_idx for det_idx, _ in players_with_jersey_color]
    colors_bgr = np.stack([color for _, color in players_with_jersey_color]).astype(np.float32)
    colors_lab = _bgr_array_to_lab(colors_bgr)

    centers, color_groups = _cluster_two_color_groups(colors_lab)
    counts = Counter(color_groups.tolist())
    if len(counts) < 2:
        return [(det_idx, "unknown", 0.0) for det_idx in indexes]

    ordered_groups = [group for group, _ in counts.most_common()]
    group_to_team = {
        ordered_groups[0]: "team_a",
        ordered_groups[1]: "team_b",
    }

    assignments: list[tuple[int, str, float]] = []
    for row_idx, det_idx in enumerate(indexes):
        group = int(color_groups[row_idx])
        team = group_to_team.get(group, "unknown")
        if team == "unknown":
            assignments.append((det_idx, "unknown", 0.0))
            continue

        own_distance = float(np.linalg.norm(colors_lab[row_idx] - centers[group]))
        other_group = 1 - group
        other_distance = float(np.linalg.norm(colors_lab[row_idx] - centers[other_group]))
        confidence = other_distance / (own_distance + other_distance + 1e-6)

        assignments.append((det_idx, team, max(0.0, min(1.0, confidence))))

    return assignments


def team_preview_color_bgr(team: str) -> tuple[int, int, int]:
    colors = {
        "team_a": (255, 80, 80),
        "team_b": (80, 180, 255),
        "referee": (80, 255, 255),
        "unknown": (220, 220, 220),
    }
    return colors.get(str(team), colors["unknown"])


def _bgr_array_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    colors = colors_bgr.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(colors, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def _cluster_two_color_groups(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        centers = np.vstack([points[0], points[0]])
        color_groups = np.zeros(len(points), dtype=np.int32)
        return centers, color_groups

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.stack([points[first], points[second]]).astype(np.float32)

    color_groups = np.zeros(len(points), dtype=np.int32)
    for _ in range(COLOR_CLUSTER_ITERATIONS):
        distance_to_centers = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        color_groups = np.argmin(distance_to_centers, axis=1).astype(np.int32)

        for group in (0, 1):
            group_mask = color_groups == group
            if np.any(group_mask):
                centers[group] = points[group_mask].mean(axis=0)

    return centers, color_groups
