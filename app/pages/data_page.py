from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle

from murawa.data import DataLoaderError, load_training_split, summarize_variant
from ui_common import ROOT, dataset_variants

SPLITS = ("train", "valid", "test")
COLOR_BY_CLASS = {
    "player": "#47DDFF",
    "goalkeeper": "#48C9B0",
    "referee": "#0066FF",
    "ball": "#FF0000",
}


def render() -> None:
    st.subheader("Przegląd danych")

    variant = st.selectbox("Wariant danych", dataset_variants(), key="data_view_variant")
    split = st.selectbox("Split", SPLITS, key="data_view_split")
    max_examples = st.slider("Liczba przykładów do podglądu", min_value=1, max_value=12, value=6)
    only_with_ball = st.checkbox("Pokaż tylko próbki z piłką", value=False, key="data_view_ball_only")

    try:
        summary = summarize_variant(project_root=ROOT, dataset_variant=variant)
    except DataLoaderError as exc:
        st.error(f"Nie można wczytać podsumowania wariantu: {exc}")
        return

    if split not in summary.available_splits:
        st.warning(f"Split '{split}' nie istnieje dla wariantu '{variant}'.")
        return

    try:
        loaded = load_training_split(
            project_root=ROOT,
            dataset_variant=variant,
            split=split,
            max_samples=None,
        )
    except DataLoaderError as exc:
        st.error(f"Nie można wczytać splitu '{split}': {exc}")
        return

    _render_stats(loaded)

    preview_samples = list(loaded.samples)
    if only_with_ball:
        preview_samples = [sample for sample in preview_samples if _sample_has_ball(sample)]
    preview_samples = preview_samples[:max_examples]

    if not preview_samples:
        st.info("Brak próbek do podglądu dla wybranych filtrów.")
        return

    cols = st.columns(2)
    for idx, sample in enumerate(preview_samples):
        with cols[idx % 2]:
            st.caption(sample.image_path.name)
            fig = _render_sample_preview(sample)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def _render_stats(loaded) -> None:
    class_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    ball_images = 0

    for sample in loaded.samples:
        source_counter[_detect_source(sample.image_path.name)] += 1
        has_ball = False
        for ann in sample.annotations:
            class_counter[ann.category_name] += 1
            if ann.category_name == "ball":
                has_ball = True
        if has_ball:
            ball_images += 1

    st.markdown("**Statystyki splitu**")
    st.json(
        {
            "dataset_variant": loaded.dataset_variant,
            "split": loaded.split,
            "total_images": loaded.total_images,
            "total_annotations": loaded.total_annotations,
            "ball_images": ball_images,
            "class_counts": dict(sorted(class_counter.items())),
            "source_image_counts": dict(sorted(source_counter.items())),
            "class_mapping": {str(k): v for k, v in loaded.class_mapping.items()},
        }
    )


def _render_sample_preview(sample):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    try:
        image = plt.imread(str(sample.image_path))
        ax.imshow(image)
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Nie można odczytać obrazu\n{sample.image_path.name}\n{exc}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.axis("off")
        return fig

    for ann in sample.annotations:
        x, y, w, h = ann.bbox_xywh
        color = COLOR_BY_CLASS.get(ann.category_name, "#FFFFFF")
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.8, edgecolor=color, facecolor="none"))
        ax.text(
            x,
            max(0, y - 3),
            ann.category_name,
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
        )

    ax.axis("off")
    return fig


def _sample_has_ball(sample) -> bool:
    return any(ann.category_name == "ball" for ann in sample.annotations)


def _detect_source(file_name: str) -> str:
    lowered = file_name.lower()
    if lowered.startswith("ballextra_"):
        return "ball-extra"
    if lowered.startswith("soccernet_"):
        return "soccernet"
    return "unknown"


if __name__ == "__main__":
    render()
