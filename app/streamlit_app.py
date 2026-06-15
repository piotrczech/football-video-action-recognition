import sys
from pathlib import Path

import streamlit as st

# Streamlit adds `app/` to sys.path; ensure repo root and `src/` are visible.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from app.views.analyses_page import render as render_analyses_page
from app.views.data_page import render as render_data_page
from app.views.frame_page import render as render_frame_page
from app.views.match_page import render as render_match_page
from app.views.models_page import render as render_models_page


def main() -> None:
    st.set_page_config(page_title="Murawa", layout="wide")
    st.title("Murawa")
    st.caption("Frame and match clip analysis.")

    selected_view = st.sidebar.radio(
        "View",
        options=[
            "Frame analysis",
            "Match analysis",
            "Saved analyses",
            "Data overview",
            "Model analysis",
        ],
    )

    if selected_view == "Frame analysis":
        render_frame_page()
        return
    if selected_view == "Match analysis":
        render_match_page()
        return
    if selected_view == "Saved analyses":
        render_analyses_page()
        return
    if selected_view == "Model analysis":
        render_models_page()
        return

    render_data_page()


if __name__ == "__main__":
    main()
