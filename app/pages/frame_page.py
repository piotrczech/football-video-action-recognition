import streamlit as st

from murawa.services.pipeline import analyze_frame_run
from ui_common import ROOT, format_run_label, show_result, temporary_upload_path, trained_runs


def render() -> None:
    st.subheader("Analizuj klatkę")
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
        key="frame_run_name",
    )
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


if __name__ == "__main__":
    render()
