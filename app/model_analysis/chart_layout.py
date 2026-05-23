from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

CHART_MAX_WIDTH_PX = 500


def inject_chart_layout_css() -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="stPyplotChart"] {{
            max-width: {CHART_MAX_WIDTH_PX}px;
        }}
        div[data-testid="stPyplotChart"] img {{
            max-width: {CHART_MAX_WIDTH_PX}px;
            width: 100%;
            height: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chart_grid(
    charts: list[tuple[str, Callable[[], Any | None]]],
    *,
    section_title: str | None = None,
) -> None:
    """Render chart builders in a responsive grid (1 col narrow, up to 2 side-by-side)."""
    available = [(title, builder) for title, builder in charts if builder is not None]
    if not available:
        return

    inject_chart_layout_css()
    if section_title:
        st.markdown(f"**{section_title}**")

    if len(available) == 1:
        _show_figure(available[0][1])
        return

    for row_start in range(0, len(available), 2):
        row_items = available[row_start : row_start + 2]
        columns = st.columns(len(row_items), gap="large")
        for column, (_, builder) in zip(columns, row_items, strict=True):
            with column:
                _show_figure(builder)


def render_chart(builder: Callable[[], Any | None]) -> None:
    inject_chart_layout_css()
    _show_figure(builder)


def _show_figure(builder: Callable[[], Any | None]) -> None:
    fig = builder()
    if fig is None:
        st.caption("Brak danych do wykresu.")
        return
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
