import streamlit as st

from murawa.services.pipeline import analyze_match
from ui_common import MODELS, ROOT, dataset_variants, save_upload, show_result


def render() -> None:
    st.subheader("Analizuj mecz")
    model = st.selectbox("Model", MODELS, key="match_model")
    dataset_variant = st.selectbox("Wariant danych", dataset_variants(), key="match_variant")
    uploaded = st.file_uploader(
        "Wgraj nagranie meczu (opcjonalnie)",
        type=["mp4", "avi", "mov", "mkv"],
        key="match_upload",
    )

    if st.button("Uruchom analizę meczu", key="run_match"):
        result = analyze_match(
            project_root=ROOT,
            model=model,
            dataset_variant=dataset_variant,
            input_path=save_upload(uploaded),
        )
        show_result(result)


if __name__ == "__main__":
    render()
