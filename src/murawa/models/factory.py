from murawa.models.rfdetr import RfDetrAdapter
from murawa.models.yolo import YoloAdapter

SUPPORTED_MODELS = {"yolo", "rfdetr"}


def normalize_model_name(model: str) -> str:
    normalized = model.strip().lower()
    if normalized == "rf":
        return "rfdetr"
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Expected one of: yolo, rfdetr, rf.")
    return normalized


def build_training_adapter(model: str):
    normalized = normalize_model_name(model)
    if normalized == "yolo":
        return YoloAdapter()
    return RfDetrAdapter()
