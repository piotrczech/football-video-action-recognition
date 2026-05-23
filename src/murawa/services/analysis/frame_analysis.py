from pathlib import Path

import cv2

from murawa.data.path_resolver import IMAGE_SUFFIXES
from murawa.models import build_training_adapter, normalize_model_name
from murawa.services.analysis.pipeline_helpers import (
    _build_detection_stats,
    _make_base_payload,
    _resolve_input,
    _select_primary_ball_detections,
)
from murawa.services.rendering.overlay import (
    write_preview_assets,
    write_team_assignment_debug_preview,
)
from murawa.services.runtime.artifacts import latest_run, resolve_run, write_json
from murawa.services.runtime.video_processing import (
    DEFAULT_SAMPLE_FPS,
    ProgressCallback,
)
from murawa.services.vision.team_assignment import assign_teams_to_frame
from murawa.settings import OUTPUTS_PREDICTIONS


def analyze_frame(
    project_root: Path, model: str, dataset_variant: str, input_path: str | None = None
) -> dict:
    return _run_analysis(project_root, model, dataset_variant, mode="frame", input_path=input_path)


def analyze_frame_run(project_root: Path, run_name: str, input_path: str | None = None) -> dict:
    return _run_analysis_for_run(project_root, run_name, mode="frame", input_path=input_path)


def _run_analysis(
    project_root: Path,
    model: str,
    dataset_variant: str,
    mode: str,
    input_path: str | None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    # --- 1. Resolve run and validate input ---
    normalized_model = normalize_model_name(model)
    base_payload = _make_base_payload(
        mode=mode,
        model=normalized_model,
        dataset_variant=dataset_variant,
    )

    try:
        run_name = latest_run(project_root, normalized_model, dataset_variant)
    except FileNotFoundError:
        base_payload["status"] = "missing_run"
        base_payload["message"] = (
            "No trained run or checkpoint found. Run training first, e.g.: "
            f"python scripts/train.py --model {normalized_model} "
            f"--dataset-variant {dataset_variant}"
        )
        return base_payload

    return _run_analysis_for_run(
        project_root=project_root,
        run_name=run_name,
        mode=mode,
        input_path=input_path,
        sample_fps=sample_fps,
        show_boxes=show_boxes,
        show_confidence=show_confidence,
        progress_callback=progress_callback,
        fallback_payload=base_payload,
    )


def _run_analysis_for_run(
    project_root: Path,
    run_name: str,
    mode: str,
    input_path: str | None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    show_boxes: bool = True,
    show_confidence: bool = False,
    progress_callback: ProgressCallback | None = None,
    fallback_payload: dict | None = None,
) -> dict:
    # --- 1. Resolve run and validate input ---
    try:
        run = resolve_run(project_root, run_name)
    except FileNotFoundError:
        payload = fallback_payload or _make_base_payload(mode=mode, model="", dataset_variant="")
        payload["status"] = "missing_run"
        payload["message"] = (
            "No trained run or checkpoint found for the selected model. "
            f"Could not find a complete run: {run_name}"
        )
        return payload

    normalized_model = normalize_model_name(run.model)
    dataset_variant = run.dataset_variant
    base_payload = _make_base_payload(
        mode=mode,
        model=normalized_model,
        dataset_variant=dataset_variant,
    )
    base_payload["resolved_run_name"] = run.run_name

    if mode == "match":
        from murawa.services.analysis.match_analysis import _run_match_video_analysis

        return _run_match_video_analysis(
            project_root=project_root,
            run=run,
            normalized_model=normalized_model,
            dataset_variant=dataset_variant,
            input_path=input_path,
            sample_fps=sample_fps,
            show_boxes=show_boxes,
            show_confidence=show_confidence,
            progress_callback=progress_callback,
            base_payload=base_payload,
        )

    out_dir = project_root / OUTPUTS_PREDICTIONS / run.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input, input_found = _resolve_input(project_root, mode, dataset_variant, input_path)
    checkpoint_path = run.checkpoint_path

    if not input_found:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Could not find a valid image input file for {normalized_model}. "
            "Provide --input-path or ensure the file exists in the test directory."
        )
        return base_payload

    resolved_path = Path(resolved_input) if resolved_input else None

    if resolved_path is None or not resolved_path.exists() or not resolved_path.is_file():
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Input path for {normalized_model} backend must point to a file. "
            f"Received: {resolved_input}"
        )
        return base_payload

    suffix = resolved_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Frame mode requires an image file. Received suffix='{suffix}', "
            f"expected one of {sorted(IMAGE_SUFFIXES)}."
        )
        return base_payload

    # --- 2. Frame / video extraction ---
    frame_image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if frame_image_bgr is None:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Could not read frame image: {resolved_input}"
        return base_payload

    # --- 3. Model inference ---
    try:
        model_instance = build_training_adapter(normalized_model)
        detections = model_instance.predict(
            input_path=resolved_path,
            checkpoint_path=checkpoint_path,
            mode="frame",
        )
        detections, primary_ball_filter = _select_primary_ball_detections(detections)
    except Exception as exc:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Prediction backend failed for model='{normalized_model}': {exc}"
        return base_payload

    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / f"{mode}_prediction.txt"

    # --- 5. Team assignment ---
    assignment_result = assign_teams_to_frame(
        image_bgr=frame_image_bgr,
        detections=detections,
    )
    detections = assignment_result.detections
    team_assignment = assignment_result.summary
    minimap_entities = assignment_result.minimap_entities

    stats = _build_detection_stats(detections=detections)

    # --- 7. Render outputs and persist artifacts ---
    preview_assets = write_preview_assets(
        detections=detections,
        out_dir=out_dir,
        team_assignment=team_assignment,
        frame_image_bgr=frame_image_bgr.copy(),
    )
    debug_preview_assets = write_team_assignment_debug_preview(
        frame_image_bgr=frame_image_bgr,
        detections=detections,
        out_dir=out_dir,
    )

    payload = {
        "status": "ok",
        "mode": mode,
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": run.run_name,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(run.metadata_dir),
        "resolved_input": resolved_input,
        "input_found": input_found,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "preview_assets": preview_assets,
        "debug_preview_assets": debug_preview_assets,
        "stats": stats,
        "team_assignment": team_assignment,
        "primary_ball_filter": primary_ball_filter,
        "minimap_entities": minimap_entities,
        "detections": detections,
    }
    write_json(summary_path, payload)

    preview_path.write_text(
        "\n".join(
            [
                f"mode={mode}",
                f"model={normalized_model}",
                f"dataset_variant={dataset_variant}",
                f"resolved_input={resolved_input}",
                f"detections={stats.get('total_detections', 0)}",
                f"classes={stats.get('classes', {})}",
                f"preview_assets={len(preview_assets)}",
                f"debug_preview_assets={len(debug_preview_assets)}",
                "Real adapter output.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload
