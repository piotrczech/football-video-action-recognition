import streamlit as st

from murawa.services.pipeline import analyze_frame_run
from ui_common import ROOT, format_run_label, save_upload, show_result, trained_runs


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

    if st.button("Uruchom analizę klatki", key="run_frame"):
        result = analyze_frame_run(
            project_root=ROOT,
            run_name=selected_run.run_name,
            input_path=save_upload(uploaded),
        )
        show_result(result)


if __name__ == "__main__":
    render()
