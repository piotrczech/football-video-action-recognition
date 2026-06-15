from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from murawa.services.runtime.artifacts import TrainedRunRecord, list_available_runs
from murawa.settings import PROJECT_ROOT, list_dataset_variants

PREVIEW_MEDIA_WIDTH = 960
ROOT = PROJECT_ROOT


def dataset_variants() -> list[str]:
    return list_dataset_variants(PROJECT_ROOT)


def trained_runs() -> list[TrainedRunRecord]:
    return list_available_runs(PROJECT_ROOT)


def format_run_label(run: TrainedRunRecord) -> str:
    if run.run_tag != "auto":
        return run.run_tag

    try:
        created_at = datetime.fromisoformat(run.created_at_utc).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        created_at = run.created_at_utc
    return f"{run.model} | {run.dataset_variant} | {created_at}"


def render_run_selector(
    *,
    key: str,
    empty_message: str | None = None,
) -> TrainedRunRecord | None:
    runs = trained_runs()
    if not runs:
        st.info(
            empty_message
            or (
                "No trained runs available. Run training first, for example: "
                "`python scripts/train.py --model yolo --dataset-variant base --profile quick`"
            )
        )
        return None

    return st.selectbox(
        "Trained model",
        options=runs,
        format_func=format_run_label,
        key=key,
    )


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
        st.caption("No data.")
        return
    st.table(
        [{key_label: str(key), "Count": str(value)} for key, value in sorted(counts.items())]
    )


def video_mime_type(video_path: Path) -> str:
    if video_path.suffix.lower() == ".webm":
        return "video/webm"
    return "video/mp4"


def format_cell(value: object, *, empty_label: str = "missing") -> str:
    if value is None:
        return empty_label
    if isinstance(value, str) and not value.strip():
        return empty_label
    return str(value)
