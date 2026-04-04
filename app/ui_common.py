from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolo", "rf"]


def dataset_variants() -> list[str]:
    training_root = ROOT / "data" / "training"
    if not training_root.exists():
        return ["base-format"]

    variants = sorted(
        p.name for p in training_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if "base-format" not in variants:
        variants.insert(0, "base-format")
    return variants or ["base-format"]


def save_upload(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".bin"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def show_result(result: dict) -> None:
    if result["status"] == "missing_run":
        st.info(result["message"])
        return

    if result["status"] != "ok":
        st.error(result.get("message", "Analiza zakończyła się błędem."))
        return

    st.success("Przetwarzanie zakończone.")
    st.json(result)

    preview_path = Path(result.get("preview_path", ""))
    if preview_path.exists():
        st.text_area("Podgląd wyniku", preview_path.read_text(encoding="utf-8"), height=160)
