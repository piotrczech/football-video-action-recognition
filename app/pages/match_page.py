import streamlit as st

from murawa.services.pipeline import analyze_match_run
from ui_common import ROOT, format_run_label, save_upload, show_result, trained_runs


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
        "Wgraj nagranie meczu (opcjonalnie)",
        type=["mp4", "avi", "mov", "mkv"],
        key="match_upload",
    )

    if st.button("Uruchom analizę meczu", key="run_match"):
        result = analyze_match_run(
            project_root=ROOT,
            run_name=selected_run.run_name,
            input_path=save_upload(uploaded),
        )
        show_result(result)


if __name__ == "__main__":
    render()
