import streamlit as st

from views.analyses_page import render as render_analyses_page
from views.data_page import render as render_data_page
from views.frame_page import render as render_frame_page
from views.match_page import render as render_match_page
from views.models_page import render as render_models_page


def main() -> None:
    st.set_page_config(page_title="Murawa", layout="wide")
    st.title("Murawa")
    st.caption("Analiza klatek i klipów meczowych.")

    selected_view = st.sidebar.radio(
        "Widok",
        options=[
            "Analizuj klatkę",
            "Analizuj mecz",
            "Przeglądaj analizy",
            "Przegląd danych",
            "Analiza modeli",
        ],
    )

    if selected_view == "Analizuj klatkę":
        render_frame_page()
        return
    if selected_view == "Analizuj mecz":
        render_match_page()
        return
    if selected_view == "Przeglądaj analizy":
        render_analyses_page()
        return
    if selected_view == "Analiza modeli":
        render_models_page()
        return

    render_data_page()


if __name__ == "__main__":
    main()
