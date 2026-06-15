from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from app.ui_common import PREVIEW_MEDIA_WIDTH, format_cell, show_counts


def show_result(
    result: dict,
    show_debug_assets: bool = False,
    input_image: bytes | str | Path | None = None,
) -> None:
    if result["status"] == "missing_run":
        st.info(result["message"])
        return

    if result["status"] != "ok":
        st.error(result.get("message", "Analysis failed."))
        with st.expander("Error details", expanded=False):
            st.json(result)
        return

    if result.get("mode") == "frame":
        _show_frame_result(
            result=result,
            show_debug_assets=show_debug_assets,
            input_image=input_image,
        )
        return

    st.success("Processing completed.")
    with st.expander("Raw result", expanded=False):
        st.json(result)


def _show_preview_image(preview_file: Path) -> None:
    with st.container(width=PREVIEW_MEDIA_WIDTH):
        st.image(
            str(preview_file),
            caption=preview_file.name,
            width="stretch",
        )

def _show_frame_result(
    *,
    result: dict,
    show_debug_assets: bool,
    input_image: bytes | str | Path | None,
) -> None:
    st.success("Frame analysis completed.")

    stats = _mapping(result.get("stats"))
    team_assignment = _mapping(result.get("team_assignment"))
    detections = _list_of_dicts(result.get("detections"))
    minimap_entities = _list_of_dicts(result.get("minimap_entities"))

    _show_frame_metrics(
        stats=stats,
        team_assignment=team_assignment,
        minimap_entities=minimap_entities,
    )

    st.divider()
    _show_frame_previews(result=result, input_image=input_image)

    st.divider()
    summary_tab, team_tab, minimap_tab, raw_tab = st.tabs(
        ["Summary", "Teams", "Minimap", "Raw JSON"]
    )

    with summary_tab:
        cols = st.columns(2)
        with cols[0]:
            show_counts("Detection classes", stats.get("classes", {}), "Class")
        with cols[1]:
            _show_primary_ball_filter(result)

        st.markdown("**Detections**")
        rows = _detection_rows(detections)
        if rows:
            st.dataframe(rows, hide_index=True, height="content", width="stretch")
        else:
            st.caption("No detections to show.")

    with team_tab:
        cols = st.columns(2)
        with cols[0]:
            show_counts("Team summary", team_assignment.get("team_counts", {}), "Label")
        with cols[1]:
            _show_team_colors(team_assignment)

        st.markdown("**Jersey color feature visualization**")
        _show_team_feature_scatter(detections)

        if show_debug_assets:
            st.markdown("**Jersey crop debug previews**")
            for asset in _list_of_strings(result.get("debug_preview_assets"))[:3]:
                debug_file = Path(asset)
                if debug_file.exists():
                    _show_preview_image(debug_file)

    with minimap_tab:
        _show_minimap_entities(minimap_entities)

    with raw_tab:
        st.json(result)


def _show_frame_metrics(
    *,
    stats: dict[str, Any],
    team_assignment: dict[str, Any],
    minimap_entities: list[dict[str, Any]],
) -> None:
    class_counts = _mapping(stats.get("classes"))
    team_counts = _mapping(team_assignment.get("team_counts"))

    metrics = st.columns(5)
    metrics[0].metric("Detections", _count_value(stats.get("total_detections")))
    metrics[1].metric("Classes", str(len(class_counts)))
    metrics[2].metric("Avg. confidence", _confidence_value(stats.get("mean_confidence")))
    metrics[3].metric("Teams / role", str(len(team_counts)))
    metrics[4].metric("Minimap entities", str(len(minimap_entities)))


def _show_frame_previews(
    *,
    result: dict,
    input_image: bytes | str | Path | None,
) -> None:
    left, right = st.columns(2)

    with left:
        st.markdown("**Input frame**")
        if input_image is not None:
            st.image(input_image, caption="Input", width="stretch")
        else:
            resolved_input = _safe_path(result.get("resolved_input"))
            if resolved_input is not None and resolved_input.exists():
                st.image(str(resolved_input), caption=resolved_input.name, width="stretch")
            else:
                st.caption("No input preview available.")

    with right:
        st.markdown("**Analysis result**")
        preview_file = _first_existing_path(_list_of_strings(result.get("preview_assets")))
        if preview_file is not None:
            st.image(str(preview_file), caption=preview_file.name, width="stretch")
        else:
            st.caption("No generated preview available.")


def _show_primary_ball_filter(result: dict) -> None:
    primary_ball = _mapping(result.get("primary_ball_filter"))
    st.markdown("**Ball filtering**")
    if not primary_ball:
        st.caption("No data.")
        return

    st.table(
        [
            {"Field": "Enabled", "Value": format_cell(primary_ball.get("enabled"))},
            {"Field": "Policy", "Value": format_cell(primary_ball.get("policy"))},
            {"Field": "Detections before", "Value": format_cell(primary_ball.get("raw_detection_count"))},
            {"Field": "Detections after", "Value": format_cell(primary_ball.get("filtered_detection_count"))},
            {"Field": "Ball candidates after", "Value": format_cell(primary_ball.get("ball_candidates_after"))},
        ]
    )


def _show_team_colors(team_assignment: dict[str, Any]) -> None:
    colors = _mapping(team_assignment.get("team_colors_bgr"))
    st.markdown("**Team colors**")
    if not colors:
        st.caption("No data o kolorach.")
        return

    rows = []
    for team, color in sorted(colors.items()):
        if not isinstance(color, list) or len(color) != 3:
            continue
        b, g, r = [int(v) for v in color]
        rows.append(
            {
                "Label": team,
                "BGR": f"[{b}, {g}, {r}]",
                "RGB": f"[{r}, {g}, {b}]",
            }
        )

    if rows:
        st.table(rows)
    else:
        st.caption("No valid colors.")


def _show_team_feature_scatter(detections: list[dict[str, Any]]) -> None:
    rows = _team_feature_rows(detections)
    if len(rows) < 2:
        st.caption("Not enough players with jersey color data to show the chart.")
        return

    projection_name = rows[0].get("projection", "2D projection")

    st.caption(
        f"Auxiliary visualization: `{projection_name}`. "
        "UMAP is used if `umap-learn` is available; otherwise a PCA fallback is used."
    )

    spec = {
        "data": {"values": rows},
        "mark": {"type": "circle", "size": 180, "opacity": 0.9},
        "encoding": {
            "x": {
                "field": "x",
                "type": "quantitative",
                "title": "feature 1",
                "scale": {"zero": False},
            },
            "y": {
                "field": "y",
                "type": "quantitative",
                "title": "feature 2",
                "scale": {"zero": False},
            },
            "color": {
                "field": "team",
                "type": "nominal",
                "title": "team",
            },
            "shape": {
                "field": "class",
                "type": "nominal",
                "title": "class",
            },
            "tooltip": [
                {"field": "class", "type": "nominal"},
                {"field": "team", "type": "nominal"},
                {"field": "confidence", "type": "quantitative"},
                {"field": "jersey_color_bgr", "type": "nominal"},
                {"field": "x", "type": "quantitative"},
                {"field": "y", "type": "quantitative"},
            ],
        },
        "height": 360,
    }

    st.vega_lite_chart(spec=spec, width="stretch")


def _show_minimap_entities(minimap_entities: list[dict[str, Any]]) -> None:
    if not minimap_entities:
        st.caption("No data minimapy.")
        return

    rows = []
    for entity in minimap_entities:
        rows.append(
            {
                "class": entity.get("class", "unknown"),
                "team": entity.get("team", "unknown"),
                "center_xy": entity.get("center_xy"),
                "bbox_xyxy": entity.get("bbox_xyxy"),
                "confidence": _round_float(entity.get("confidence")),
                "track_id": entity.get("track_id", ""),
                "frame_index": entity.get("frame_index", ""),
            }
        )

    st.dataframe(rows, hide_index=True, height="content", width="stretch")


def _team_feature_rows(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features: list[list[float]] = []
    metadata: list[dict[str, Any]] = []

    for det in detections:
        color = det.get("jersey_color_bgr")
        if not isinstance(color, list) or len(color) != 3:
            continue

        try:
            b, g, r = [float(channel) for channel in color]
        except (TypeError, ValueError):
            continue

        brightness = (r + g + b) / 3.0
        red_blue_delta = r - b
        saturation_proxy = max(r, g, b) - min(r, g, b)

        features.append([r, g, b, brightness, red_blue_delta, saturation_proxy])
        metadata.append(
            {
                "class": str(det.get("class", "unknown")),
                "team": str(det.get("team", "unknown")),
                "confidence": _round_float(det.get("confidence")),
                "jersey_color_bgr": str([int(b), int(g), int(r)]),
            }
        )

    if not features:
        return []

    coords, method = _project_2d(np.array(features, dtype=np.float32))
    return [
        {
            **metadata[idx],
            "x": round(float(coords[idx, 0]), 4),
            "y": round(float(coords[idx, 1]), 4),
            "projection": method,
        }
        for idx in range(len(metadata))
    ]


def _project_2d(features: np.ndarray) -> tuple[np.ndarray, str]:
    if len(features) == 1:
        return np.zeros((1, 2), dtype=np.float32), "single-point projection"

    try:
        import umap  # type: ignore

        if len(features) >= 4:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(8, len(features) - 1),
                min_dist=0.05,
                random_state=42,
            )
            return reducer.fit_transform(features).astype(np.float32), "UMAP"
    except Exception:
        pass

    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    coords = centered @ components

    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1), dtype=coords.dtype)])

    return coords[:, :2].astype(np.float32), "PCA fallback"


def _detection_rows(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, det in enumerate(detections[:80], start=1):
        rows.append(
            {
                "#": idx,
                "class": det.get("class", "unknown"),
                "team": det.get("team", ""),
                "confidence": _round_float(det.get("confidence")),
                "track_id": det.get("track_id", ""),
                "bbox_xyxy": det.get("bbox_xyxy", ""),
                "ball_memory": det.get("ball_memory", ""),
            }
        )
    return rows


def _first_existing_path(paths: list[str]) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def _safe_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _list_of_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _count_value(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value))
    return "0"


def _confidence_value(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return "0.000"


def _round_float(value: object) -> float | str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 4)
    return ""