from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st
from app.ui_common import PREVIEW_MEDIA_WIDTH, ROOT, show_counts, video_mime_type

from murawa.services.runtime.saved_match_analyses import SavedMatchAnalysis, list_saved_match_analyses


def render() -> None:
    st.subheader("Przeglądaj analizy")

    analyses = list_saved_match_analyses(project_root=ROOT)
    if not analyses:
        st.info("Brak zapisanych analiz meczowych w `outputs/videos`.")
        return

    table_state = st.dataframe(
        _analysis_rows(analyses),
        hide_index=True,
        height="content",
        on_select="rerun",
        selection_mode="single-row-required",
        key="saved_match_analyses",
    )
    selected_rows = table_state.selection.rows
    selected_index = selected_rows[0] if selected_rows else 0
    _show_analysis(_selected_analysis(analyses, selected_index))


def _show_analysis(analysis: SavedMatchAnalysis) -> None:
    preview_path = analysis.preview_path
    download_path = analysis.download_path

    if preview_path.exists():
        st.video(
            str(preview_path),
            format=video_mime_type(preview_path),
            width=PREVIEW_MEDIA_WIDTH,
        )
    else:
        st.warning("Podgląd wideo (WebM) nie jest już dostępny na dysku.")

    actions = st.columns([1, 2])
    with actions[0]:
        if download_path is not None and download_path.exists():
            st.download_button(
                "Pobierz MP4",
                data=download_path.read_bytes(),
                file_name=download_path.name,
                mime=video_mime_type(download_path),
                key=f"download_{analysis.analysis_id}_mp4",
            )
        else:
            st.caption("Plik MP4 do pobrania nie jest dostępny.")
    with actions[1]:
        st.caption(f"ID analizy: `{analysis.analysis_id}`")
        st.caption(f"Podgląd: `{preview_path}`")
        if download_path is not None:
            st.caption(f"Pobieranie: `{download_path}`")

    if analysis.summary is None:
        st.info("Metadane tej analizy nie są dostępne.")
        return

    _show_summary(analysis.summary)


def _show_summary(summary: dict[str, Any]) -> None:
    model = _string_value(summary.get("model"))
    run_name = _string_value(summary.get("resolved_run_name"))
    st.caption(f"Model: `{model}` | Run: `{run_name}`")

    metadata = _mapping(summary.get("video_metadata"))
    stats = _mapping(summary.get("stats"))
    tracking = _mapping(summary.get("tracking"))
    metrics = st.columns(5)
    metrics[0].metric("Długość klipu", _duration_value(metadata.get("duration_seconds")))
    metrics[1].metric("Próbkowanie", _fps_value(summary.get("sample_fps")))
    metrics[2].metric("Klatki analizy", _count_value(summary.get("sampled_frames")))
    metrics[3].metric("Detekcje", _count_value(stats.get("total_detections")))
    metrics[4].metric("Tracki", _count_value(stats.get("track_count")))
    if tracking.get("enabled"):
        st.caption(f"Tracking: `{_string_value(tracking.get('method'))}`")

    details = st.columns(2)
    with details[0]:
        show_counts("Klasy detekcji", stats.get("classes"), "Klasa")
    with details[1]:
        show_counts("Drużyny", stats.get("team_counts"), "Etykieta")


def _analysis_rows(analyses: list[SavedMatchAnalysis]) -> list[dict[str, str]]:
    return [_analysis_row(analysis) for analysis in analyses]


def _selected_analysis(
    analyses: list[SavedMatchAnalysis],
    selected_index: int,
) -> SavedMatchAnalysis:
    if 0 <= selected_index < len(analyses):
        return analyses[selected_index]
    return analyses[0]


def _analysis_row(analysis: SavedMatchAnalysis) -> dict[str, str]:
    summary = analysis.summary or {}
    metadata = _mapping(summary.get("video_metadata"))
    stats = _mapping(summary.get("stats"))
    return {
        "Wideo": analysis.preview_path.name,
        "Zapisane": _modified_at_value(analysis.modified_at_ns),
        "Model": _string_value(summary.get("model")),
        "Run": _string_value(summary.get("resolved_run_name")),
        "Długość": _duration_value(metadata.get("duration_seconds")),
        "Klatki": _count_value(summary.get("sampled_frames")),
        "Detekcje": _count_value(stats.get("total_detections")),
    }


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _string_value(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return "Brak danych"


def _duration_value(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.1f} s"
    return "Brak danych"


def _fps_value(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value} FPS"
    return "Brak danych"


def _count_value(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value))
    return "Brak danych"


def _modified_at_value(modified_at_ns: int) -> str:
    return datetime.fromtimestamp(modified_at_ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M")
