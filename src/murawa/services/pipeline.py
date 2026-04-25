import os
from pathlib import Path

import cv2

from murawa.data.path_resolver import IMAGE_SUFFIXES, PREDICTIONS_ROOT, VIDEO_SUFFIXES, pick_input
from murawa.models import build_model, build_training_adapter, normalize_model_name
from murawa.services.artifacts import CKPT_DIR, META_DIR, latest_run, write_json


def analyze_frame(
    project_root: Path, model: str, dataset_variant: str, input_path: str | None = None
) -> dict:
    return _run_analysis(project_root, model, dataset_variant, mode="frame", input_path=input_path)


def analyze_match(
    project_root: Path, model: str, dataset_variant: str, input_path: str | None = None
) -> dict:
    return _run_analysis(project_root, model, dataset_variant, mode="match", input_path=input_path)


def _run_analysis(
    project_root: Path, model: str, dataset_variant: str, mode: str, input_path: str | None
) -> dict:
    normalized_model = normalize_model_name(model)
    base_payload = {
        "status": "error",
        "mode": mode,
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_input": "",
        "output_dir": "",
        "summary_path": "",
        "preview_path": "",
        "preview_assets": [],
        "stats": {},
        "detections": [],
    }

    try:
        run_name = latest_run(project_root, normalized_model, dataset_variant)
    except FileNotFoundError:
        base_payload["status"] = "missing_run"
        base_payload["message"] = (
            "Brak gotowego runu/checkpointu. Najpierw uruchom trening, np.: "
            f"python scripts/train.py --model {normalized_model} --dataset-variant {dataset_variant}"
        )
        return base_payload

    out_dir = project_root / PREDICTIONS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input, input_found = _resolve_input(project_root, mode, dataset_variant, input_path)
    checkpoint_path = (project_root / CKPT_DIR / run_name / "model.pt").resolve()

    # Shared flags for inference mode
    use_real_yolo = normalized_model == "yolo" and not _env_flag("MURAWA_YOLO_MOCK")
    use_rf_detr = normalized_model in ["rfdetr", "rf_detr", "rf"]
    use_real_inference = use_real_yolo or use_rf_detr

    if use_real_inference and not input_found:
        expected = "image" if mode == "frame" else "video"
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            f"Could not find a valid input file ({expected}) for {normalized_model}. "
            "Provide --input-path or ensure the file exists in the test directory."
        )
        return base_payload

    resolved_path = Path(resolved_input) if resolved_input else None
    
    if use_real_inference:
        if resolved_path is None or not resolved_path.exists() or not resolved_path.is_file():
            base_payload["status"] = "error"
            base_payload["resolved_input"] = resolved_input
            base_payload["message"] = (
                f"Input path for {normalized_model} backend must point to a file. "
                f"Received: {resolved_input}"
            )
            return base_payload

        suffix = resolved_path.suffix.lower()
        if mode == "frame" and suffix not in IMAGE_SUFFIXES:
            base_payload["status"] = "error"
            base_payload["resolved_input"] = resolved_input
            base_payload["message"] = (
                f"Frame mode requires an image file. Received suffix='{suffix}', "
                f"expected one of {sorted(IMAGE_SUFFIXES)}."
            )
            return base_payload

        if mode == "match" and suffix not in VIDEO_SUFFIXES:
            base_payload["status"] = "error"
            base_payload["resolved_input"] = resolved_input
            base_payload["message"] = (
                f"Match mode requires a video file. Received suffix='{suffix}', "
                f"expected one of {sorted(VIDEO_SUFFIXES)}."
            )
            return base_payload

    try:
        if use_rf_detr:
            model_instance = build_model(normalized_model)
            detections = model_instance.predict(
                input_path=resolved_path,
                checkpoint_path=checkpoint_path,
                mode=mode
            )
            is_mock = False
        elif use_real_yolo:
            detections = build_training_adapter(normalized_model).predict(
                input_path=resolved_path,
                checkpoint_path=checkpoint_path,
                mode=mode,
            )
            is_mock = False
        else:
            detections = build_model(normalized_model).predict(mode)
            is_mock = True
    except Exception as exc:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = f"Prediction backend failed for model='{normalized_model}': {exc}"
        return base_payload

    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / f"{mode}_prediction.txt"

    stats = _build_detection_stats(detections=detections, mode=mode)
    preview_assets: list[str] = []
    
    if use_real_inference and resolved_path is not None:
        preview_assets = _write_preview_assets(
            input_path=resolved_path,
            detections=detections,
            mode=mode,
            out_dir=out_dir,
        )

    payload = {
        "status": "ok",
        "mock": is_mock,
        "mode": mode,
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(project_root / META_DIR / run_name),
        "resolved_input": resolved_input,
        "input_found": input_found,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "preview_assets": preview_assets,
        "stats": stats,
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
                "Mock prediction output placeholder." if is_mock else "Real adapter output.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _build_detection_stats(detections: list[dict], mode: str) -> dict:
    by_class: dict[str, int] = {}
    confidences: list[float] = []

    for detection in detections:
        class_name = str(detection.get("class", "unknown"))
        by_class[class_name] = by_class.get(class_name, 0) + 1

        confidence = detection.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    stats = {
        "total_detections": len(detections),
        "classes": by_class,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
    }

    if mode == "match":
        frame_indexes = {
            int(detection["frame_index"])
            for detection in detections
            if isinstance(detection.get("frame_index"), int)
        }
        track_ids = {
            int(detection["track_id"])
            for detection in detections
            if isinstance(detection.get("track_id"), int)
        }
        stats["frames_with_detections"] = len(frame_indexes)
        stats["unique_track_ids"] = len(track_ids)

    return stats


def _write_preview_assets(
    input_path: Path,
    detections: list[dict],
    mode: str,
    out_dir: Path,
) -> list[str]:
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "frame":
            return _write_frame_preview(input_path=input_path, detections=detections, preview_dir=preview_dir)
        if mode == "match":
            return _write_match_preview(input_path=input_path, detections=detections, preview_dir=preview_dir)
    except Exception:
        return []

    return []


def _write_frame_preview(input_path: Path, detections: list[dict], preview_dir: Path) -> list[str]:
    image = cv2.imread(str(input_path))
    if image is None:
        return []

    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue

        class_name = str(detection.get("class", "unknown"))
        confidence = detection.get("confidence")
        confidence_text = f" {float(confidence):.2f}" if isinstance(confidence, (int, float)) else ""

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            image,
            f"{class_name}{confidence_text}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            2,
        )

    cv2.putText(
        image,
        f"detections={len(detections)}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    preview_path = preview_dir / "frame_preview.jpg"
    if not cv2.imwrite(str(preview_path), image):
        return []

    return [str(preview_path)]


def _write_match_preview(input_path: Path, detections: list[dict], preview_dir: Path) -> list[str]:
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        return []

    frame_counts: dict[int, int] = {}
    for detection in detections:
        frame_index = detection.get("frame_index")
        if isinstance(frame_index, int):
            frame_counts[frame_index] = frame_counts.get(frame_index, 0) + 1

    target_frames = sorted(frame_counts.keys())[:3]
    if not target_frames:
        target_frames = [0]

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return []

    assets: list[str] = []
    wanted = set(target_frames)
    max_frame = max(target_frames)
    frame_index = -1

    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index not in wanted:
                if frame_index > max_frame:
                    break
                continue

            count = frame_counts.get(frame_index, 0)
            cv2.putText(
                frame,
                f"frame={frame_index} detections={count}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            out_path = preview_dir / f"match_preview_{frame_index:06d}.jpg"
            if cv2.imwrite(str(out_path), frame):
                assets.append(str(out_path))
            wanted.remove(frame_index)
    finally:
        cap.release()

    return assets


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_input(
    project_root: Path, mode: str, dataset_variant: str, input_path: str | None
) -> tuple[str, bool]:
    if input_path:
        uploaded = Path(input_path).resolve()
        return str(uploaded), uploaded.exists()

    return pick_input(project_root, mode, dataset_variant)
