from murawa.models.rfdetr import RfDetrMockModel
from murawa.models.yolo import YoloMockModel

SUPPORTED_MODELS = {"yolo", "rfdetr"}


def normalize_model_name(model: str) -> str:
    normalized = model.strip().lower()
    if normalized == "rf":
        return "rfdetr"
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Expected one of: yolo, rfdetr, rf.")
    return normalized


def build_model(model: str):
    normalized = normalize_model_name(model)
    if normalized == "yolo":
        return YoloMockModel()
    return RfDetrMockModel()
