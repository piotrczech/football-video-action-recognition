import streamlit as st

from app.result_view import show_result
from app.ui_common import ROOT, render_run_selector, temporary_upload_path
from murawa.services.analysis.pipeline import analyze_frame_run


def render() -> None:
    st.subheader("Analizuj klatkę")
    selected_run = render_run_selector(key="frame_run_name")
    if selected_run is None:
        return

    uploaded = st.file_uploader(
        "Wgraj klatkę (opcjonalnie)", type=["jpg", "jpeg", "png", "bmp"], key="frame_upload"
    )
    show_team_debug = st.checkbox(
        "Wyświetl podgląd rozpoznawania drużyn",
        value=False,
        key="frame_show_team_debug",
    )

    if st.button("Uruchom analizę klatki", key="run_frame"):
        with temporary_upload_path(uploaded) as upload_path:
            st.session_state["frame_last_result"] = analyze_frame_run(
                project_root=ROOT,
                run_name=selected_run.run_name,
                input_path=upload_path,
            )

    result = st.session_state.get("frame_last_result")
    if isinstance(result, dict):
        input_preview = uploaded.getvalue() if uploaded is not None else None
        show_result(
            result,
            show_debug_assets=show_team_debug,
            input_image=input_preview,
        )
