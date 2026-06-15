from pathlib import Path

import streamlit as st
from app.ui_common import (
    PREVIEW_MEDIA_WIDTH,
    ROOT,
    render_run_selector,
    show_counts,
    temporary_upload_path,
    video_mime_type,
)

from murawa.settings import (
    DEFAULT_SAMPLE_FPS,
    MAX_SAMPLE_FPS,
    MAX_VIDEO_DURATION_SECONDS,
    MIN_SAMPLE_FPS,
)
from murawa.services.analysis.pipeline import analyze_match_run

PROGRESS_RANGES = {
    "validate": (0.0, 0.08),
    "extract": (0.08, 0.25),
    "inference": (0.33, 0.45),
    "tracking": (0.78, 0.10),
    "render": (0.88, 0.10),
    "save": (0.98, 0.02),
}


def render() -> None:
    st.subheader("Match analysis")
    selected_run = render_run_selector(key="match_run_name")
    if selected_run is None:
        return

    uploaded = st.file_uploader(
        "Upload match clip",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="match_upload",
    )
    sample_fps = st.slider(
        "Analyzed frames per second",
        min_value=MIN_SAMPLE_FPS,
        max_value=MAX_SAMPLE_FPS,
        value=DEFAULT_SAMPLE_FPS,
        key="match_sample_fps",
    )
    overlay_controls = st.columns(2)
    with overlay_controls[0]:
        show_boxes = st.checkbox(
            "Detection boxes",
            value=True,
            key="match_show_boxes",
        )
    with overlay_controls[1]:
        show_confidence = st.checkbox(
            "Show confidence in labels",
            value=False,
            key="match_show_confidence",
        )
    st.caption(f"Demo clips up to {int(MAX_VIDEO_DURATION_SECONDS)} seconds are supported.")

    run_clicked = st.button(
        "Run match analysis",
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
        st.info(result.get("message", "Selected run was not found."))
        return
    if result.get("status") != "ok":
        st.error(result.get("message", "Clip analysis failed."))
        return

    video_path = Path(result.get("preview_video_path") or result.get("video_path", ""))
    download_video_path = Path(result.get("download_video_path") or result.get("video_path", ""))
    st.success("Clip analysis completed.")

    metadata = result.get("video_metadata", {})
    stats = result.get("stats", {})
    tracking = result.get("tracking", {})
    tracking_info = tracking if isinstance(tracking, dict) else {}
    duration = float(metadata.get("duration_seconds", 0.0))
    input_fps = float(metadata.get("fps", 0.0))
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))

    metrics = st.columns(5)
    metrics[0].metric("Clip duration", f"{duration:.1f} s")
    metrics[1].metric("Sampling", f"{result.get('sample_fps', 0)} FPS")
    metrics[2].metric("Analyzed frames", str(result.get("sampled_frames", 0)))
    metrics[3].metric("Detections", str(stats.get("total_detections", 0)))
    metrics[4].metric("Tracks", str(stats.get("track_count", tracking_info.get("track_count", 0))))
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
            download_path = download_video_path if download_video_path.exists() else video_path

            st.download_button(
                "Download output video",
                data=download_path.read_bytes(),
                file_name=download_path.name,
                mime=video_mime_type(download_path),
            )
        with actions[1]:
            st.caption(f"Preview: `{video_path}`")
            if download_video_path.exists():
                st.caption(f"Download: `{download_video_path}`")
    else:
        st.warning("Output video is no longer available on disk.")

    details = st.columns(3)
    with details[0]:
        st.markdown("**Input clip**")
        st.caption(
            f"{width} x {height} px | {input_fps:.2f} FPS | "
            f"{int(metadata.get('frame_count', 0))} frames"
        )
    with details[1]:
        show_counts("Detection classes", stats.get("classes", {}), "Class")
    with details[2]:
        show_counts("Teams", stats.get("team_counts", {}), "Label")
    with st.expander("Raw summary", expanded=False):
        st.json(result)