from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from murawa.services.rendering.overlay_draw import (
    build_tracking_marker_label,
    draw_frame_overlay,
    team_overlay_summary,
)
from murawa.services.rendering.overlay_video import write_annotated_match_video
from murawa.services.vision.team_assignment import PLAYER_CLASSES, REFEREE_CLASSES
from murawa.services.vision.team_assignment_helpers import (
    crop_jersey_region,
    normalize_class_name,
    read_bbox_xyxy,
)


def write_preview_assets(
    detections: list[dict],
    out_dir: Path,
    team_assignment: dict,
    frame_image_bgr: np.ndarray,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        draw_frame_overlay(
            frame_image_bgr=frame_image_bgr,
            detections=detections,
            team_assignment=team_assignment,
        )

        preview_path = preview_dir / "frame_preview.jpg"
        if not cv2.imwrite(str(preview_path), frame_image_bgr):
            return []

        return [str(preview_path)]
    except Exception:
        return []


def write_team_assignment_debug_preview(
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


__all__ = [
    "build_tracking_marker_label",
    "team_overlay_summary",
    "write_annotated_match_video",
    "write_preview_assets",
    "write_team_assignment_debug_preview",
]
