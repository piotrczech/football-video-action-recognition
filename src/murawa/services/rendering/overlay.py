from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from murawa.services.runtime.video_processing import ProgressCallback, SampledFrame
from murawa.services.vision.team_assignment import PLAYER_CLASSES, REFEREE_CLASSES, count_player_teams
from murawa.services.vision.team_assignment_helpers import (
    crop_jersey_region,
    normalize_class_name,
    read_bbox_xyxy,
    team_preview_color_bgr,
)


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


def write_preview_assets(
    detections: list[dict],
    out_dir: Path,
    team_assignment: dict,
    frame_image_bgr: np.ndarray,
) -> list[str]:
    return _write_preview_assets(
        detections=detections,
        out_dir=out_dir,
        team_assignment=team_assignment,
        frame_image_bgr=frame_image_bgr,
    )


def write_team_assignment_debug_preview(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    out_dir: Path,
) -> list[str]:
    return _write_team_assignment_debug_preview(
        frame_image_bgr=frame_image_bgr,
        detections=detections,
        out_dir=out_dir,
    )


def team_overlay_summary(*, summary: dict, detections: list[dict]) -> dict:
    return _team_overlay_summary(summary=summary, detections=detections)


def build_tracking_marker_label(
    *,
    detection: dict,
    track_id: int | None,
    show_confidence: bool,
) -> str:
    return _build_tracking_marker_label(
        detection=detection,
        track_id=track_id,
        show_confidence=show_confidence,
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

            _draw_frame_overlay(
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


def _write_preview_assets(
    detections: list[dict],
    out_dir: Path,
    team_assignment: dict,
    frame_image_bgr: np.ndarray,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _write_frame_preview(
            frame_image_bgr=frame_image_bgr,
            detections=detections,
            preview_dir=preview_dir,
            team_assignment=team_assignment,
        )
    except Exception:
        return []


def _write_frame_preview(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    preview_dir: Path,
    team_assignment: dict,
) -> list[str]:
    _draw_frame_overlay(
        frame_image_bgr=frame_image_bgr,
        detections=detections,
        team_assignment=team_assignment,
    )

    preview_path = preview_dir / "frame_preview.jpg"
    if not cv2.imwrite(str(preview_path), frame_image_bgr):
        return []

    return [str(preview_path)]


def _draw_frame_overlay(
    *,
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    team_assignment: dict,
    status_text: str = "",
    show_boxes: bool = True,
    show_confidence: bool = True,
    show_tracking_markers: bool = False,
) -> None:
    for detection in detections:
        bbox = read_bbox_xyxy(detection)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        team = str(detection.get("team", "unknown"))
        color = team_preview_color_bgr(team)
        track_id = _read_detection_track_id(detection)

        if show_tracking_markers:
            marker_geometry = _draw_tracking_marker(
                frame_image_bgr=frame_image_bgr,
                bbox=bbox,
                color=color,
            )
            _draw_tracking_marker_label(
                frame_image_bgr=frame_image_bgr,
                bbox=bbox,
                marker_geometry=marker_geometry,
                label=_build_tracking_marker_label(
                    detection=detection,
                    track_id=track_id,
                    show_confidence=show_confidence,
                ),
            )

        if not show_boxes:
            continue

        cv2.rectangle(frame_image_bgr, (x1, y1), (x2, y2), color, 2)
        if not show_tracking_markers:
            _draw_shadowed_text(
                image_bgr=frame_image_bgr,
                text=_build_bbox_label(detection=detection, show_confidence=show_confidence),
                origin=(x1, max(15, y1 - 6)),
                font_scale=0.5,
                color=color,
                thickness=2,
            )

    legend_bottom = _draw_team_legend(
        image_bgr=frame_image_bgr,
        team_counts=team_assignment.get("team_counts", {}),
        team_colors_bgr=team_assignment.get("team_colors_bgr", {}),
    )

    cv2.putText(
        frame_image_bgr,
        f"detections={len(detections)}",
        (10, legend_bottom + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if status_text:
        image_height = frame_image_bgr.shape[0]
        cv2.putText(
            frame_image_bgr,
            status_text,
            (10, max(28, image_height - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def _draw_tracking_marker(
    *,
    frame_image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> tuple[int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    image_height, image_width = frame_image_bgr.shape[:2]

    radius_x = max(8, min(44, int(round(box_width * 0.48))))
    radius_y = max(4, min(14, int(round(box_height * 0.12))))
    center_x = max(radius_x, min(image_width - radius_x - 1, (x1 + x2) // 2))
    center_y = min(image_height - 2, y2 + radius_y + 2)

    backing = frame_image_bgr.copy()
    cv2.ellipse(
        backing,
        (center_x, center_y),
        (radius_x + 3, radius_y + 3),
        0,
        0,
        180,
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(backing, 0.56, frame_image_bgr, 0.44, 0, frame_image_bgr)

    overlay = frame_image_bgr.copy()
    cv2.ellipse(
        overlay,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        180,
        color,
        -1,
    )
    cv2.addWeighted(overlay, 0.62, frame_image_bgr, 0.38, 0, frame_image_bgr)
    cv2.ellipse(
        frame_image_bgr,
        (center_x, center_y),
        (radius_x + 2, radius_y + 2),
        0,
        0,
        180,
        (0, 0, 0),
        5,
    )
    cv2.ellipse(
        frame_image_bgr,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        180,
        color,
        3,
    )
    return center_x, center_y, radius_y


def _draw_tracking_marker_label(
    *,
    frame_image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker_geometry: tuple[int, int, int],
    label: str,
) -> None:
    if not label:
        return

    center_x, center_y, radius_y = marker_geometry
    font_scale = 0.68
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    image_height, image_width = frame_image_bgr.shape[:2]
    x = max(4, min(image_width - text_width - 4, center_x - text_width // 2))
    y = center_y + radius_y + text_height + 10
    if y + baseline > image_height - 4:
        y = max(text_height + 6, bbox[1] - 10)

    _draw_shadowed_text(
        image_bgr=frame_image_bgr,
        text=label,
        origin=(x, y),
        font_scale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
    )


def _draw_shadowed_text(
    *,
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    shadow_origin = (origin[0] + 2, origin[1] + 2)
    cv2.putText(
        image_bgr,
        text,
        shadow_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _build_tracking_marker_label(
    *,
    detection: dict,
    track_id: int | None,
    show_confidence: bool,
) -> str:
    label = _display_team_label(detection.get("team"))
    if label is None:
        label = _display_class_label(detection.get("class"))
    if track_id is not None:
        label = f"{label} #{track_id}"
    confidence_text = _confidence_text(detection=detection, show_confidence=show_confidence)
    return f"{label}{confidence_text}"


def _build_bbox_label(*, detection: dict, show_confidence: bool) -> str:
    label = _display_class_label(detection.get("class"))
    label = f"{label}{_confidence_text(detection=detection, show_confidence=show_confidence)}"
    team_label = _display_team_label(detection.get("team"))
    if team_label is not None:
        label = f"{label} {team_label}"
    return label


def _confidence_text(*, detection: dict, show_confidence: bool) -> str:
    confidence = detection.get("confidence")
    if show_confidence and isinstance(confidence, (int, float)):
        return f" {float(confidence):.2f}"
    return ""


def _display_class_label(value: object) -> str:
    class_name = normalize_class_name(value) or "object"
    return class_name.replace("_", " ").title()


def _display_team_label(value: object) -> str | None:
    labels = {
        "team_a": "TeamA",
        "team_b": "TeamB",
        "referee": "Referee",
    }
    return labels.get(str(value))


def _read_detection_track_id(detection: dict) -> int | None:
    track_id = detection.get("track_id")
    if isinstance(track_id, int) and not isinstance(track_id, bool):
        return track_id
    return None


def _draw_team_legend(
    image_bgr: np.ndarray,
    team_counts: object,
    team_colors_bgr: object,
) -> int:
    counts = team_counts if isinstance(team_counts, dict) else {}
    team_colors = team_colors_bgr if isinstance(team_colors_bgr, dict) else {}
    teams = ["team_a", "team_b", "referee"]

    x = 10
    y = 10
    row_height = 24
    width = 170
    height = 12 + row_height * len(teams)
    bottom = y + height

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + width, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image_bgr, 0.45, 0, image_bgr)

    for row_idx, team in enumerate(teams):
        row_y = y + 22 + row_idx * row_height
        swatch_color = _legend_color_bgr(team, team_colors)
        cv2.rectangle(image_bgr, (x + 10, row_y - 13), (x + 26, row_y + 3), swatch_color, -1)
        cv2.rectangle(image_bgr, (x + 10, row_y - 13), (x + 26, row_y + 3), (255, 255, 255), 1)
        label = f"{team} {int(counts.get(team, 0))}"
        cv2.putText(
            image_bgr,
            label,
            (x + 34, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

    return bottom


def _legend_color_bgr(team: str, team_colors_bgr: dict) -> tuple[int, int, int]:
    color = team_colors_bgr.get(team)
    if isinstance(color, list) and len(color) == 3:
        try:
            return tuple(int(value) for value in color)
        except (TypeError, ValueError):
            pass
    return team_preview_color_bgr(team)


def _write_team_assignment_debug_preview(
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


def _team_overlay_summary(*, summary: dict, detections: list[dict]) -> dict:
    enriched = dict(summary)
    enriched["team_counts"] = count_player_teams(detections)
    return enriched
