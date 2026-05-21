from pathlib import Path

import streamlit as st
from ui_common import (
    PREVIEW_MEDIA_WIDTH,
    ROOT,
    format_run_label,
    show_counts,
    temporary_upload_path,
    trained_runs,
    video_mime_type,
)

from murawa.services.pipeline import analyze_match_run

PROGRESS_RANGES = {
    "validate": (0.0, 0.08),
    "extract": (0.08, 0.25),
    "inference": (0.33, 0.45),
    "tracking": (0.78, 0.10),
    "render": (0.88, 0.10),
    "save": (0.98, 0.02),
}


def render() -> None:
    st.subheader("Analizuj mecz")
    runs = trained_runs()
    if not runs:
        st.info(
            "Brak gotowych runów do analizy. Najpierw uruchom trening, np.: "
            "`python scripts/train.py --model yolo --dataset-variant base --profile quick`"
        )
        return

    selected_run = st.selectbox(
        "Wytrenowany model",
        options=runs,
        format_func=format_run_label,
        key="match_run_name",
    )
    uploaded = st.file_uploader(
        "Wgraj klip meczu",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="match_upload",
    )
    sample_fps = st.slider(
        "Klatki analizowane na sekundę",
        min_value=1,
        max_value=24,
        value=3,
        key="match_sample_fps",
    )
    overlay_controls = st.columns(2)
    with overlay_controls[0]:
        show_boxes = st.checkbox(
            "Ramki detekcji",
            value=True,
            key="match_show_boxes",
        )
    with overlay_controls[1]:
        show_confidence = st.checkbox(
            "Confidence w podpisach",
            value=False,
            key="match_show_confidence",
        )
    st.caption("Obsługiwane są klipy demo do 60 sekund.")

    run_clicked = st.button(
        "Uruchom analizę meczu",
        key="run_match",
        disabled=uploaded is None,
        type="primary",
    )
    if run_clicked:
        progress_bar = st.progress(0.0)
        status_line = st.empty()

        def on_progress(stage: str, progress: float, message: str) -> None:
            start, width = PROGRESS_RANGES.get(stage, (0.0, 1.0))
            progress_bar.progress(min(1.0, start + width * progress))
            status_line.caption(message)

        with temporary_upload_path(uploaded) as upload_path:
            st.session_state["match_last_result"] = analyze_match_run(
                project_root=ROOT,
                run_name=selected_run.run_name,
                input_path=upload_path,
                sample_fps=sample_fps,
                show_boxes=show_boxes,
                show_confidence=show_confidence,
                progress_callback=on_progress,
            )

    result = st.session_state.get("match_last_result")
    if isinstance(result, dict):
        _show_match_result(result)


def _show_match_result(result: dict) -> None:
    if result.get("status") == "missing_run":
        st.info(result.get("message", "Nie znaleziono wybranego runu."))
        return
    if result.get("status") != "ok":
        st.error(result.get("message", "Analiza klipu zakończyła się błędem."))
        return

    video_path = Path(result.get("video_path", ""))
    st.success("Analiza klipu zakończona.")

    metadata = result.get("video_metadata", {})
    stats = result.get("stats", {})
    tracking = result.get("tracking", {})
    tracking_info = tracking if isinstance(tracking, dict) else {}
    duration = float(metadata.get("duration_seconds", 0.0))
    input_fps = float(metadata.get("fps", 0.0))
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))

    metrics = st.columns(5)
    metrics[0].metric("Długość klipu", f"{duration:.1f} s")
    metrics[1].metric("Próbkowanie", f"{result.get('sample_fps', 0)} FPS")
    metrics[2].metric("Klatki analizy", str(result.get("sampled_frames", 0)))
    metrics[3].metric("Detekcje", str(stats.get("total_detections", 0)))
    metrics[4].metric("Tracki", str(stats.get("track_count", tracking_info.get("track_count", 0))))
    if tracking_info.get("enabled"):
        st.caption(f"Tracking: `{tracking_info.get('method', 'unknown')}`")

    if video_path.exists():
        st.video(
            str(video_path),
            format=video_mime_type(video_path),
            width=PREVIEW_MEDIA_WIDTH,
        )
        actions = st.columns([1, 2])
        with actions[0]:
            st.download_button(
                "Pobierz wynik",
                data=video_path.read_bytes(),
                file_name=video_path.name,
                mime=video_mime_type(video_path),
            )
        with actions[1]:
            st.caption(f"Zapisano: `{video_path}`")
    else:
        st.warning("Wideo wynikowe nie jest już dostępne na dysku.")

    details = st.columns(3)
    with details[0]:
        st.markdown("**Klip wejściowy**")
        st.caption(
            f"{width} x {height} px | {input_fps:.2f} FPS | "
            f"{int(metadata.get('frame_count', 0))} klatek"
        )
    with details[1]:
        show_counts("Klasy detekcji", stats.get("classes", {}), "Klasa")
    with details[2]:
        show_counts("Drużyny", stats.get("team_counts", {}), "Etykieta")


if __name__ == "__main__":
    render()
