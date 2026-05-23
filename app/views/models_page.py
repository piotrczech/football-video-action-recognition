from __future__ import annotations

import streamlit as st

from model_analysis.chart_layout import render_chart_grid
from model_analysis.charts import (
    as_float_list,
    build_loss_chart,
    build_precision_recall_chart,
    build_val_map50_chart,
    build_val_map5095_chart,
)
from model_analysis.glossary import METRIC_DEFINITIONS, glossary_rows
from model_analysis.presenters import build_kpi_items, format_run_context
from murawa.services import load_run_metrics
from ui_common import ROOT, format_run_label, trained_runs


def render() -> None:
    st.subheader("Analiza modeli")
    runs = trained_runs()
    if not runs:
        st.info(
            "Brak gotowych runów treningowych. Najpierw uruchom trening, np.: "
            "`python scripts/train.py --model yolo --dataset-variant base --profile quick`"
        )
        return

    selected_run = st.selectbox(
        "Wytrenowany model",
        options=runs,
        format_func=format_run_label,
        key="models_run_name",
    )

    try:
        payload = load_run_metrics(ROOT, selected_run.run_name)
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    metrics = payload.get("metrics_summary", {})
    if not isinstance(metrics, dict):
        metrics = {}
    metadata = payload.get("train_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    context = format_run_context(
        metadata,
        fallback_model=selected_run.model,
        fallback_variant=selected_run.dataset_variant,
        fallback_created_at=selected_run.created_at_utc,
    )
    if context:
        st.caption(context)

    _render_kpi_section(metrics)
    _render_training_curves(metrics)
    _render_glossary(metrics)


def _render_kpi_section(metrics: dict) -> None:
    kpi_items = build_kpi_items(metrics)
    if not kpi_items:
        st.caption("Brak metryk końcowych w metadanych.")
        return

    st.markdown("**Metryki końcowe**")
    columns = st.columns(min(len(kpi_items), 4))
    for index, item in enumerate(kpi_items):
        definition = METRIC_DEFINITIONS.get(item.label)
        label = definition.label if definition is not None else item.label
        columns[index % len(columns)].metric(label, item.value, help=item.help_text)


def _render_training_curves(metrics: dict) -> None:
    loss_history = as_float_list(metrics.get("loss_history"))
    val_loss_history = as_float_list(metrics.get("val_loss_history"))
    map50_history = as_float_list(metrics.get("map50_history"))
    map5095_history = as_float_list(metrics.get("map5095_history"))
    precision_history = as_float_list(metrics.get("precision_history"))
    recall_history = as_float_list(metrics.get("recall_history"))
    val_loss_scalar = metrics.get("val_loss")
    val_loss_scalar = float(val_loss_scalar) if val_loss_scalar is not None else None

    chart_builders: list[tuple[str, object]] = []

    if loss_history:
        chart_builders.append(
            (
                "loss",
                lambda: build_loss_chart(
                    loss_history,
                    val_loss_history=val_loss_history,
                    val_loss_scalar=val_loss_scalar,
                ),
            )
        )

    if map50_history:
        chart_builders.append(("map50", lambda: build_val_map50_chart(map50_history)))

    if map5095_history:
        chart_builders.append(("map5095", lambda: build_val_map5095_chart(map5095_history)))

    if precision_history or recall_history:
        chart_builders.append(
            (
                "precision_recall",
                lambda: build_precision_recall_chart(precision_history or [], recall_history or []),
            )
        )

    if not chart_builders:
        st.caption("Brak historii metryk treningowych w metadanych.")
        return

    render_chart_grid(chart_builders, section_title="Krzywe treningu")


def _render_glossary(metrics: dict) -> None:
    with st.expander("Co oznaczają metryki?"):
        st.table(glossary_rows())
        if not as_float_list(metrics.get("map50_history")):
            st.caption(
                "mAP, precision i recall są liczone wyłącznie na zbiorze walidacyjnym — "
                "backend detekcji nie raportuje ich odpowiedników train w trakcie uczenia."
            )
