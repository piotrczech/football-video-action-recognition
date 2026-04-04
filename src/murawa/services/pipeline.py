from pathlib import Path

from murawa.data.path_resolver import PREDICTIONS_ROOT, pick_input
from murawa.models import build_model, normalize_model_name
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

    resolved_input, input_found = _resolve_input(project_root, mode, input_path)
    detections = build_model(normalized_model).predict(mode)
    summary_path = out_dir / "prediction_summary.json"
    preview_path = out_dir / f"{mode}_prediction.txt"

    payload = {
        "status": "ok",
        "mock": True,
        "mode": mode,
        "model": normalized_model,
        "dataset_variant": dataset_variant,
        "resolved_run_name": run_name,
        "checkpoint_path": str(project_root / CKPT_DIR / run_name / "model.pt"),
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
                "Mock prediction output placeholder.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _resolve_input(project_root: Path, mode: str, input_path: str | None) -> tuple[str, bool]:
    if input_path:
        uploaded = Path(input_path)
        return str(uploaded), uploaded.exists()

    fallback_mode = "image" if mode == "frame" else "video"
    return pick_input(project_root, fallback_mode)
