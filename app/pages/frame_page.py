import streamlit as st

from murawa.services.pipeline import analyze_frame
from ui_common import MODELS, ROOT, dataset_variants, save_upload, show_result


def render() -> None:
    st.subheader("Analizuj klatkę")
    model = st.selectbox("Model", MODELS, key="frame_model")
    dataset_variant = st.selectbox("Wariant danych", dataset_variants(), key="frame_variant")
    uploaded = st.file_uploader(
        "Wgraj klatkę (opcjonalnie)", type=["jpg", "jpeg", "png", "bmp"], key="frame_upload"
    )

    if st.button("Uruchom analizę klatki", key="run_frame"):
        result = analyze_frame(
            project_root=ROOT,
            model=model,
            dataset_variant=dataset_variant,
            input_path=save_upload(uploaded),
        )
        show_result(result)


if __name__ == "__main__":
    render()
