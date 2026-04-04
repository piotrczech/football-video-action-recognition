import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pages.frame_page import render as render_frame_page
from pages.match_page import render as render_match_page


def main() -> None:
    st.set_page_config(page_title="Murawa MVP", layout="centered")
    st.title("Murawa MVP")
    st.caption("Szkielet integracyjny: upload -> dummy pipeline -> wynik")

    selected_view = st.sidebar.radio(
        "Widok",
        options=["Analizuj klatkę", "Analizuj mecz"],
    )

    if selected_view == "Analizuj klatkę":
        render_frame_page()
        return

    render_match_page()


if __name__ == "__main__":
    main()
