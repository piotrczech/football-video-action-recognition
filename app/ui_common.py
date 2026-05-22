from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import streamlit as st

from murawa.services.artifacts import TrainedRunRecord, list_available_runs

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_MEDIA_WIDTH = 960


def dataset_variants() -> list[str]:
    ready_root = ROOT / "data" / "ready"
    if not ready_root.exists():
        return ["base"]

    variants = sorted(
        p.name for p in ready_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if "base" not in variants:
        variants.insert(0, "base")
    return variants or ["base"]


def trained_runs() -> list[TrainedRunRecord]:
    return list_available_runs(ROOT)


def format_run_label(run: TrainedRunRecord) -> str:
    if run.run_tag != "auto":
        return run.run_tag

    try:
        created_at = datetime.fromisoformat(run.created_at_utc).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        created_at = run.created_at_utc
    return f"{run.model} | {run.dataset_variant} | {created_at}"


@contextmanager
def temporary_upload_path(uploaded_file):
    if uploaded_file is None:
        yield None
        return

    suffix = Path(uploaded_file.name).suffix or ".bin"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name

    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


def show_counts(title: str, counts: object, key_label: str) -> None:
    st.markdown(f"**{title}**")
    if not isinstance(counts, dict) or not counts:
        st.caption("Brak danych.")
        return
    st.table([{key_label: key, "Liczba": value} for key, value in sorted(counts.items())])


def video_mime_type(video_path: Path) -> str:
    if video_path.suffix.lower() == ".webm":
        return "video/webm"
    return "video/mp4"


def show_result(
    result: dict,
    show_debug_assets: bool = False,
    input_image: bytes | str | Path | None = None,
) -> None:
    if result["status"] == "missing_run":
        st.info(result["message"])
        return

    if result["status"] != "ok":
        st.error(result.get("message", "Analiza zakończyła się błędem."))
        with st.expander("Szczegóły błędu", expanded=False):
            st.json(result)
        return

    if result.get("mode") == "frame":
        _show_frame_result(
            result=result,
            show_debug_assets=show_debug_assets,
            input_image=input_image,
        )
        return

    st.success("Przetwarzanie zakończone.")
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
    st.success("Analiza klatki zakończona.")

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
        ["Podsumowanie", "Drużyny", "Minimapa", "Raw JSON"]
    )

    with summary_tab:
        cols = st.columns(2)
        with cols[0]:
            show_counts("Klasy detekcji", stats.get("classes", {}), "Klasa")
        with cols[1]:
            _show_primary_ball_filter(result)

        st.markdown("**Detekcje**")
        rows = _detection_rows(detections)
        if rows:
            st.dataframe(rows, hide_index=True, height="content", use_container_width=True)
        else:
            st.caption("Brak detekcji do pokazania.")

    with team_tab:
        cols = st.columns(2)
        with cols[0]:
            show_counts("Podsumowanie drużyn", team_assignment.get("team_counts", {}), "Etykieta")
        with cols[1]:
            _show_team_colors(team_assignment)

        st.markdown("**Wizualizacja cech kolorów koszulek**")
        _show_team_feature_scatter(detections)

        if show_debug_assets:
            st.markdown("**Debug cropów koszulek**")
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
    metrics[0].metric("Detekcje", _count_value(stats.get("total_detections")))
    metrics[1].metric("Klasy", str(len(class_counts)))
    metrics[2].metric("Śr. confidence", _confidence_value(stats.get("mean_confidence")))
    metrics[3].metric("Drużyny / role", str(len(team_counts)))
    metrics[4].metric("Minimap entities", str(len(minimap_entities)))


def _show_frame_previews(
    *,
    result: dict,
    input_image: bytes | str | Path | None,
) -> None:
    left, right = st.columns(2)

    with left:
        st.markdown("**Klatka wejściowa**")
        if input_image is not None:
            st.image(input_image, caption="Input", width="stretch")
        else:
            resolved_input = _safe_path(result.get("resolved_input"))
            if resolved_input is not None and resolved_input.exists():
                st.image(str(resolved_input), caption=resolved_input.name, width="stretch")
            else:
                st.caption("Brak dostępnego podglądu wejścia.")

    with right:
        st.markdown("**Wynik analizy**")
        preview_file = _first_existing_path(_list_of_strings(result.get("preview_assets")))
        if preview_file is not None:
            st.image(str(preview_file), caption=preview_file.name, width="stretch")
        else:
            st.caption("Brak wygenerowanego preview.")


def _show_primary_ball_filter(result: dict) -> None:
    primary_ball = _mapping(result.get("primary_ball_filter"))
    st.markdown("**Filtrowanie piłki**")
    if not primary_ball:
        st.caption("Brak danych.")
        return

    st.table(
        [
            {"Pole": "Włączone", "Wartość": str(primary_ball.get("enabled", False))},
            {"Pole": "Polityka", "Wartość": str(primary_ball.get("policy", "brak"))},
            {"Pole": "Detekcje przed", "Wartość": _count_value(primary_ball.get("raw_detection_count"))},
            {"Pole": "Detekcje po", "Wartość": _count_value(primary_ball.get("filtered_detection_count"))},
            {"Pole": "Piłki po", "Wartość": _count_value(primary_ball.get("ball_candidates_after"))},
        ]
    )


def _show_team_colors(team_assignment: dict[str, Any]) -> None:
    colors = _mapping(team_assignment.get("team_colors_bgr"))
    st.markdown("**Kolory drużyn**")
    if not colors:
        st.caption("Brak danych o kolorach.")
        return

    rows = []
    for team, color in sorted(colors.items()):
        if not isinstance(color, list) or len(color) != 3:
            continue
        b, g, r = [int(v) for v in color]
        rows.append(
            {
                "Etykieta": team,
                "BGR": f"[{b}, {g}, {r}]",
                "RGB": f"[{r}, {g}, {b}]",
            }
        )

    if rows:
        st.table(rows)
    else:
        st.caption("Brak poprawnych kolorów.")


def _show_team_feature_scatter(detections: list[dict[str, Any]]) -> None:
    rows = _team_feature_rows(detections)
    if len(rows) < 2:
        st.caption("Za mało zawodników z kolorem koszulki, żeby pokazać wykres.")
        return

    projection_name = rows[0].get("projection", "2D projection")

    st.caption(
        f"Wizualizacja pomocnicza: `{projection_name}`. "
        "UMAP jest używany, jeżeli `umap-learn` jest dostępny; inaczej używany jest fallback PCA."
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

    st.vega_lite_chart(spec=spec, use_container_width=True)


def _show_minimap_entities(minimap_entities: list[dict[str, Any]]) -> None:
    if not minimap_entities:
        st.caption("Brak danych minimapy.")
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

    st.dataframe(rows, hide_index=True, height="content", use_container_width=True)


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