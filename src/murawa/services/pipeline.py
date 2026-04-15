import os
from pathlib import Path

from murawa.data.path_resolver import PREDICTIONS_ROOT, pick_input
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
        "detections": [],
    }

    try:
        run_name = latest_run(project_root, normalized_model, dataset_variant)
    except FileNotFoundError:
        base_payload["status"] = "missing_run"
        base_payload["message"] = (
            "Brak gotowego runu/checkpointu. Najpierw uruchom trening mock, np.: "
            f"python scripts/train.py --model {normalized_model} --dataset-variant {dataset_variant}"
        )
        return base_payload

    out_dir = project_root / PREDICTIONS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input, input_found = _resolve_input(project_root, mode, dataset_variant, input_path)
    checkpoint_path = (project_root / CKPT_DIR / run_name / "model.pt").resolve()

    use_mock_yolo = normalized_model == "yolo" and _env_flag("MURAWA_YOLO_MOCK")
    use_real_yolo_adapter = normalized_model == "yolo" and not use_mock_yolo
    if use_real_yolo_adapter and not input_found:
        base_payload["status"] = "error"
        base_payload["resolved_input"] = resolved_input
        base_payload["message"] = (
            "Nie znaleziono poprawnego pliku wejściowego dla realnego backendu YOLO. "
            "Podaj --input-path lub upewnij się, że w katalogu test istnieją pliki media."
        )
        return base_payload

    try:
        if use_real_yolo_adapter:
            detections = build_training_adapter(normalized_model).predict(
                input_path=Path(resolved_input),
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
                f"detections={len(detections)}",
                "Mock prediction output placeholder." if is_mock else "Real adapter output.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_input(
    project_root: Path, mode: str, dataset_variant: str, input_path: str | None
) -> tuple[str, bool]:
    if input_path:
        uploaded = Path(input_path)
        return str(uploaded), uploaded.exists()

    fallback_mode = "image" if mode == "frame" else "video"
    return pick_input(project_root, fallback_mode, dataset_variant)
