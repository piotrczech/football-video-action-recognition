from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np

from murawa.data.path_resolver import IMAGE_SUFFIXES, PREDICTIONS_ROOT, VIDEO_SUFFIXES, pick_input
from murawa.models import build_training_adapter, normalize_model_name
from murawa.services.artifacts import latest_run, resolve_run, write_json
from murawa.services.video_processing import (
    DEFAULT_SAMPLE_FPS,
    ProgressCallback,
    SampledFrame,
    VideoProcessingError,
    extract_sampled_frames,
    validate_sample_fps,
    validate_video_input,
)
from murawa.vision.team_assignment import assign_teams_to_frame, count_player_teams
from murawa.vision.team_assignment_helpers import (
    crop_jersey_region,
    normalize_class_name,
    read_bbox_xyxy,
    team_preview_color_bgr,
)


def analyze_frame(
    project_root: Path, model: str, dataset_variant: str, input_path: str | None = None
) -> dict:
    return _run_analysis(project_root, model, dataset_variant, mode="frame", input_path=input_path)


def analyze_match(
    project_root: Path,
    model: str,
    dataset_variant: str,
    input_path: str | None = None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    return _run_analysis(
        project_root,
        model,
        dataset_variant,
        mode="match",
        input_path=input_path,
        sample_fps=sample_fps,
        progress_callback=progress_callback,
    )


def analyze_frame_run(project_root: Path, run_name: str, input_path: str | None = None) -> dict:
    return _run_analysis_for_run(project_root, run_name, mode="frame", input_path=input_path)


def analyze_match_run(
    project_root: Path,
    run_name: str,
    input_path: str | None = None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    return _run_analysis_for_run(
        project_root,
        run_name,
        mode="match",
        input_path=input_path,
        sample_fps=sample_fps,
        progress_callback=progress_callback,
    )


def _run_analysis(
    project_root: Path,
    model: str,
    dataset_variant: str,
    mode: str,
    input_path: str | None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    normalized_model = normalize_model_name(model)
    base_payload = _make_base_payload(mode=mode, model=normalized_model, dataset_variant=dataset_variant)

    try:
        run_name = latest_run(project_root, normalized_model, dataset_variant)
    except FileNotFoundError:
        base_payload["status"] = "missing_run"
        base_payload["message"] = (
            "Brak gotowego runu/checkpointu. Najpierw uruchom trening, np.: "
            f"python scripts/train.py --model {normalized_model} --dataset-variant {dataset_variant}"
        )
        return base_payload

    return _run_analysis_for_run(
        project_root=project_root,
        run_name=run_name,
        mode=mode,
        input_path=input_path,
        sample_fps=sample_fps,
        progress_callback=progress_callback,
        fallback_payload=base_payload,
    )


def _run_analysis_for_run(
    project_root: Path,
    run_name: str,
    mode: str,
    input_path: str | None,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    progress_callback: ProgressCallback | None = None,
    fallback_payload: dict | None = None,
) -> dict:
    try:
        run = resolve_run(project_root, run_name)
    except FileNotFoundError:
        payload = fallback_payload or _make_base_payload(mode=mode, model="", dataset_variant="")
        payload["status"] = "missing_run"
        payload["message"] = (
            "Brak gotowego runu/checkpointu dla wybranego modelu. "
            f"Nie znaleziono kompletnego runu: {run_name}"
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
        return _run_match_video_analysis(
            project_root=project_root,
            run=run,
            normalized_model=normalized_model,
            dataset_variant=dataset_variant,
            input_path=input_path,
            sample_fps=sample_fps,
            progress_callback=progress_callback,
            base_payload=base_payload,
        )

    out_dir = project_root / PREDICTIONS_ROOT / run.run_name
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

    frame_image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if frame_image_bgr is None:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Could not read frame image: {resolved_input}"
        return base_payload

    try:
        model_instance = build_training_adapter(normalized_model)
        detections = model_instance.predict(
            input_path=resolved_path,
            checkpoint_path=checkpoint_path,
            mode="frame",
        )
    except Exception as exc:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Prediction backend failed for model='{normalized_model}': {exc}"
        return base_payload

    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / f"{mode}_prediction.txt"

    assignment_result = assign_teams_to_frame(
        image_bgr=frame_image_bgr,
        detections=detections,
    )
    detections = assignment_result.detections
    team_assignment = assignment_result.summary
    minimap_entities = assignment_result.minimap_entities

    stats = _build_detection_stats(detections=detections)

    preview_assets = _write_preview_assets(
        detections=detections,
        out_dir=out_dir,
        team_assignment=team_assignment,
        frame_image_bgr=frame_image_bgr.copy(),
    )
    debug_preview_assets = _write_team_assignment_debug_preview(
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


def _run_match_video_analysis(
    *,
    project_root: Path,
    run,
    normalized_model: str,
    dataset_variant: str,
    input_path: str | None,
    sample_fps: int,
    progress_callback: ProgressCallback | None,
    base_payload: dict,
) -> dict:
    base_payload["sample_fps"] = sample_fps
    resolved_input, input_found = _resolve_input(project_root, "match", dataset_variant, input_path)
    base_payload["resolved_input"] = resolved_input
    base_payload["input_found"] = input_found

    if not input_found:
        base_payload["status"] = "error"
        base_payload["message"] = (
            "Could not find a valid video input. Upload a video file or pass --input-path."
        )
        return base_payload

    resolved_path = Path(resolved_input)
    try:
        parsed_sample_fps = validate_sample_fps(sample_fps)
        _emit_progress(
            progress_callback,
            "validate",
            0.0,
            "Validating video input.",
        )
        metadata = validate_video_input(resolved_path, allowed_suffixes=VIDEO_SUFFIXES)
        _emit_progress(
            progress_callback,
            "validate",
            1.0,
            "Video input is ready.",
        )
    except VideoProcessingError as exc:
        base_payload["status"] = "error"
        base_payload["message"] = str(exc)
        return base_payload

    analysis_id = _make_match_analysis_id()
    out_dir = project_root / PREDICTIONS_ROOT / run.run_name / analysis_id
    videos_dir = project_root / "outputs" / "videos"
    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / "match_prediction.txt"
    video_path = videos_dir / f"{analysis_id}.webm"

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    try:
        with TemporaryDirectory(prefix="murawa-match-") as temp_dir:
            sampled_frames = extract_sampled_frames(
                resolved_path,
                output_dir=Path(temp_dir) / "frames",
                metadata=metadata,
                sample_fps=parsed_sample_fps,
                progress_callback=progress_callback,
            )

            model_instance = build_training_adapter(normalized_model)
            frame_batches = model_instance.predict_frames(
                frame_paths=[frame.path for frame in sampled_frames],
                checkpoint_path=run.checkpoint_path,
                progress_callback=lambda completed, total: _emit_inference_progress(
                    progress_callback=progress_callback,
                    completed=completed,
                    total=total,
                ),
            )
            if len(frame_batches) != len(sampled_frames):
                raise RuntimeError(
                    "Prediction backend returned a different number of frame batches "
                    f"({len(frame_batches)}) than sampled frames ({len(sampled_frames)})."
                )

            detections = _write_annotated_match_video(
                video_path=video_path,
                sampled_frames=sampled_frames,
                frame_batches=frame_batches,
                sample_fps=parsed_sample_fps,
                progress_callback=progress_callback,
            )
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        base_payload["status"] = "error"
        base_payload["output_dir"] = str(out_dir)
        base_payload["video_path"] = str(video_path)
        base_payload["message"] = f"Match video analysis failed: {exc}"
        return base_payload

    stats = _build_detection_stats(detections=detections)
    stats["sampled_frames"] = len(sampled_frames)
    stats["team_counts"] = count_player_teams(detections)

    payload = {
        "status": "ok",
        "mode": "match",
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": run.run_name,
        "analysis_id": analysis_id,
        "checkpoint_path": str(run.checkpoint_path),
        "metadata_path": str(run.metadata_dir),
        "resolved_input": resolved_input,
        "input_found": input_found,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "preview_assets": [],
        "debug_preview_assets": [],
        "video_path": str(video_path),
        "video_metadata": metadata.as_dict(),
        "sample_fps": parsed_sample_fps,
        "sampled_frames": len(sampled_frames),
        "sampled_frame_timeline": [_sampled_frame_timeline_item(frame) for frame in sampled_frames],
        "stats": stats,
        "team_assignment": {
            "enabled": True,
            "method": "jersey_color_kmeans_mvp_per_sampled_frame",
            "team_counts": stats["team_counts"],
        },
        "minimap_entities": [],
        "detections": detections,
    }
    write_json(summary_path, payload)
    preview_path.write_text(
        "\n".join(
            [
                "mode=match",
                f"model={normalized_model}",
                f"dataset_variant={dataset_variant}",
                f"resolved_input={resolved_input}",
                f"sample_fps={parsed_sample_fps}",
                f"sampled_frames={len(sampled_frames)}",
                f"detections={stats.get('total_detections', 0)}",
                f"video_path={video_path}",
                "Sampled-frame video analysis output.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _emit_progress(
        progress_callback,
        "save",
        1.0,
        "Saved video analysis output.",
    )
    return payload


def _write_annotated_match_video(
    *,
    video_path: Path,
    sampled_frames: list[SampledFrame],
    frame_batches: list[list[dict]],
    sample_fps: int,
    progress_callback: ProgressCallback | None,
) -> list[dict]:
    writer = None
    frame_size: tuple[int, int] | None = None
    all_detections: list[dict] = []

    try:
        for render_index, (sampled_frame, frame_detections) in enumerate(
            zip(sampled_frames, frame_batches, strict=True),
            start=1,
        ):
            frame_image_bgr = cv2.imread(str(sampled_frame.path), cv2.IMREAD_COLOR)
            if frame_image_bgr is None:
                raise RuntimeError(f"Could not read sampled frame: {sampled_frame.path}")

            height, width = frame_image_bgr.shape[:2]
            if writer is None:
                frame_size = (width, height)
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"VP80"),
                    float(sample_fps),
                    frame_size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {video_path}")
            elif frame_size is not None and frame_size != (width, height):
                frame_image_bgr = cv2.resize(frame_image_bgr, frame_size)

            timeline_detections = [
                _add_sample_timeline(detection, sampled_frame) for detection in frame_detections
            ]
            assignment_result = assign_teams_to_frame(
                image_bgr=frame_image_bgr,
                detections=timeline_detections,
            )
            _draw_frame_overlay(
                frame_image_bgr=frame_image_bgr,
                detections=assignment_result.detections,
                team_assignment=assignment_result.summary,
                status_text=(
                    f"t={sampled_frame.timestamp_seconds:.2f}s "
                    f"source_frame={sampled_frame.source_frame_index}"
                ),
            )
            writer.write(frame_image_bgr)
            all_detections.extend(assignment_result.detections)
            _emit_progress(
                progress_callback,
                "render",
                render_index / max(1, len(sampled_frames)),
                f"Rendering output video ({render_index}/{len(sampled_frames)} frames).",
            )
    finally:
        if writer is not None:
            writer.release()

    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"Output video was not written: {video_path}")
    return all_detections


def _add_sample_timeline(detection: dict, sampled_frame: SampledFrame) -> dict:
    enriched = dict(detection)
    enriched["sample_index"] = sampled_frame.sample_index
    enriched["frame_index"] = sampled_frame.source_frame_index
    enriched["timestamp_seconds"] = round(sampled_frame.timestamp_seconds, 4)
    return enriched


def _sampled_frame_timeline_item(sampled_frame: SampledFrame) -> dict:
    return {
        "sample_index": sampled_frame.sample_index,
        "source_frame_index": sampled_frame.source_frame_index,
        "timestamp_seconds": round(sampled_frame.timestamp_seconds, 4),
    }


def _make_match_analysis_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"match_{timestamp}_{uuid4().hex[:8]}"


def _emit_inference_progress(
    *,
    progress_callback: ProgressCallback | None,
    completed: int,
    total: int,
) -> None:
    _emit_progress(
        progress_callback,
        "inference",
        completed / max(1, total),
        f"Running model inference ({completed}/{total} frames).",
    )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, min(1.0, max(0.0, progress)), message)


def _make_base_payload(mode: str, model: str, dataset_variant: str) -> dict:
    return {
        "status": "error",
        "mode": mode,
        "model": model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": "",
        "resolved_input": "",
        "output_dir": "",
        "summary_path": "",
        "preview_path": "",
        "preview_assets": [],
        "debug_preview_assets": [],
        "video_path": "",
        "video_metadata": {},
        "sample_fps": DEFAULT_SAMPLE_FPS,
        "sampled_frames": 0,
        "stats": {},
        "team_assignment": {},
        "minimap_entities": [],
        "detections": [],
    }


def _build_detection_stats(detections: list[dict]) -> dict:
    by_class: dict[str, int] = {}
    confidences: list[float] = []

    for detection in detections:
        class_name = str(detection.get("class", "unknown"))
        by_class[class_name] = by_class.get(class_name, 0) + 1

        confidence = detection.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    return {
        "total_detections": len(detections),
        "classes": by_class,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
    }


def _write_preview_assets(
    detections: list[dict],
    out_dir: Path,
    team_assignment: dict,
    frame_image_bgr: np.ndarray,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _write_frame_preview(
            frame_image_bgr=frame_image_bgr,
            detections=detections,
            preview_dir=preview_dir,
            team_assignment=team_assignment,
        )
    except Exception:
        return []


def _write_frame_preview(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    preview_dir: Path,
    team_assignment: dict,
) -> list[str]:
    _draw_frame_overlay(
        frame_image_bgr=frame_image_bgr,
        detections=detections,
        team_assignment=team_assignment,
    )

    preview_path = preview_dir / "frame_preview.jpg"
    if not cv2.imwrite(str(preview_path), frame_image_bgr):
        return []

    return [str(preview_path)]


def _draw_frame_overlay(
    *,
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    team_assignment: dict,
    status_text: str = "",
) -> None:
    for detection in detections:
        bbox = read_bbox_xyxy(detection)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        class_name = str(detection.get("class", "unknown"))
        confidence = detection.get("confidence")
        confidence_text = f" {float(confidence):.2f}" if isinstance(confidence, (int, float)) else ""

        team = str(detection.get("team", "unknown"))
        color = team_preview_color_bgr(team)
        label = f"{class_name}{confidence_text}"
        if team != "unknown":
            label = f"{label} {team}"

        cv2.rectangle(frame_image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_image_bgr,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    legend_bottom = _draw_team_legend(
        image_bgr=frame_image_bgr,
        team_counts=team_assignment.get("team_counts", {}),
        team_colors_bgr=team_assignment.get("team_colors_bgr", {}),
    )

    cv2.putText(
        frame_image_bgr,
        f"detections={len(detections)}",
        (10, legend_bottom + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if status_text:
        image_height = frame_image_bgr.shape[0]
        cv2.putText(
            frame_image_bgr,
            status_text,
            (10, max(28, image_height - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def _draw_team_legend(
    image_bgr: np.ndarray,
    team_counts: object,
    team_colors_bgr: object,
) -> int:
    counts = team_counts if isinstance(team_counts, dict) else {}
    team_colors = team_colors_bgr if isinstance(team_colors_bgr, dict) else {}
    teams = ["team_a", "team_b", "referee"]

    x = 10
    y = 10
    row_height = 24
    width = 170
    height = 12 + row_height * len(teams)
    bottom = y + height

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x + width, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image_bgr, 0.45, 0, image_bgr)

    for row_idx, team in enumerate(teams):
        row_y = y + 22 + row_idx * row_height
        swatch_color = _legend_color_bgr(team, team_colors)
        cv2.rectangle(image_bgr, (x + 10, row_y - 13), (x + 26, row_y + 3), swatch_color, -1)
        cv2.rectangle(image_bgr, (x + 10, row_y - 13), (x + 26, row_y + 3), (255, 255, 255), 1)
        label = f"{team} {int(counts.get(team, 0))}"
        cv2.putText(
            image_bgr,
            label,
            (x + 34, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

    return bottom


def _legend_color_bgr(team: str, team_colors_bgr: dict) -> tuple[int, int, int]:
    color = team_colors_bgr.get(team)
    if isinstance(color, list) and len(color) == 3:
        try:
            return tuple(int(value) for value in color)
        except (TypeError, ValueError):
            pass
    return team_preview_color_bgr(team)


def _write_team_assignment_debug_preview(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
    out_dir: Path,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    debug_image = _build_team_assignment_crop_sheet(frame_image_bgr, detections)
    debug_path = preview_dir / "team_assignment_crops.jpg"
    if not cv2.imwrite(str(debug_path), debug_image):
        return []
    return [str(debug_path)]


def _build_team_assignment_crop_sheet(frame_image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    tile_width = 170
    tile_height = 116
    crop_size = 64
    columns = 4
    padding = 10

    debug_items = _team_assignment_debug_items(frame_image_bgr, detections)
    if not debug_items:
        image = np.full((90, 380, 3), 32, dtype=np.uint8)
        cv2.putText(
            image,
            "no team assignment crops",
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
        )
        return image

    rows = int(np.ceil(len(debug_items) / columns))
    sheet_height = padding + rows * tile_height
    sheet_width = padding + columns * tile_width
    sheet = np.full((sheet_height, sheet_width, 3), 32, dtype=np.uint8)

    for idx, item in enumerate(debug_items):
        col = idx % columns
        row = idx // columns
        x = padding + col * tile_width
        y = padding + row * tile_height

        cv2.rectangle(sheet, (x, y), (x + tile_width - 8, y + tile_height - 8), (58, 58, 58), -1)
        crop = cv2.resize(item["crop"], (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        sheet[y + 8 : y + 8 + crop_size, x + 8 : x + 8 + crop_size] = crop

        label = item["team"]
        confidence = item.get("team_confidence")
        if isinstance(confidence, (int, float)):
            label = f"{label} {float(confidence):.2f}"
        cv2.putText(
            sheet,
            label,
            (x + 8, y + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
        )
        cv2.putText(
            sheet,
            item["class_name"],
            (x + 8, y + 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
        )

        jersey_color = item.get("jersey_color_bgr")
        if jersey_color is not None:
            cv2.rectangle(sheet, (x + 82, y + 12), (x + 126, y + 38), jersey_color, -1)
            cv2.rectangle(sheet, (x + 82, y + 12), (x + 126, y + 38), (255, 255, 255), 1)

    return sheet


def _team_assignment_debug_items(
    frame_image_bgr: np.ndarray,
    detections: list[dict],
) -> list[dict]:
    items: list[dict] = []

    for det in detections:
        class_name = normalize_class_name(det.get("class"))
        if class_name not in PLAYER_CLASSES and class_name not in REFEREE_CLASSES:
            continue

        bbox = read_bbox_xyxy(det)
        if bbox is None:
            continue

        crop = crop_jersey_region(frame_image_bgr, bbox)
        if crop.size == 0:
            continue

        jersey_color = _read_color_bgr(det.get("jersey_color_bgr"))
        items.append(
            {
                "class_name": class_name,
                "team": str(det.get("team", "unknown")),
                "team_confidence": det.get("team_confidence"),
                "jersey_color_bgr": jersey_color,
                "crop": crop,
            }
        )

    return items


def _read_color_bgr(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return tuple(int(channel) for channel in value)
    except (TypeError, ValueError):
        return None


def _resolve_input(
    project_root: Path, mode: str, dataset_variant: str, input_path: str | None
) -> tuple[str, bool]:
    if input_path:
        uploaded = Path(input_path).resolve()
        return str(uploaded), uploaded.exists()

    return pick_input(project_root, mode, dataset_variant)
