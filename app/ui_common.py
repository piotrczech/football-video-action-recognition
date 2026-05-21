from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from murawa.services.artifacts import TrainedRunRecord, list_available_runs

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_MEDIA_WIDTH = 960


def dataset_variants() -> list[str]:
    ready_root = ROOT / "data" / "ready"
    if not ready_root.exists():
        return ["base"]

    variants = sorted(
        p.name for p in ready_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if "base" not in variants:
        variants.insert(0, "base")
    return variants or ["base"]


def trained_runs() -> list[TrainedRunRecord]:
    return list_available_runs(ROOT)


def format_run_label(run: TrainedRunRecord) -> str:
    if run.run_tag != "auto":
        return run.run_tag

    try:
        created_at = datetime.fromisoformat(run.created_at_utc).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        created_at = run.created_at_utc
    return f"{run.model} | {run.dataset_variant} | {created_at}"


@contextmanager
def temporary_upload_path(uploaded_file):
    if uploaded_file is None:
        yield None
        return

    suffix = Path(uploaded_file.name).suffix or ".bin"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name

    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


def show_counts(title: str, counts: object, key_label: str) -> None:
    st.markdown(f"**{title}**")
    if not isinstance(counts, dict) or not counts:
        st.caption("Brak danych.")
        return
    st.table([{key_label: key, "Liczba": value} for key, value in sorted(counts.items())])


def video_mime_type(video_path: Path) -> str:
    if video_path.suffix.lower() == ".webm":
        return "video/webm"
    return "video/mp4"


def show_result(result: dict, show_debug_assets: bool = False) -> None:
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

    preview_assets = result.get("preview_assets", [])
    for asset in preview_assets[:3]:
        preview_file = Path(asset)
        if preview_file.exists():
            st.image(str(preview_file), caption=preview_file.name, width=PREVIEW_MEDIA_WIDTH)

    if not show_debug_assets:
        return

    debug_assets = result.get("debug_preview_assets", [])
    for asset in debug_assets[:3]:
        debug_file = Path(asset)
        if debug_file.exists():
            st.image(str(debug_file), caption=debug_file.name, width=PREVIEW_MEDIA_WIDTH)
