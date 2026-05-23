from __future__ import annotations

from typing import Any

RFDETR_RESOLUTION_BLOCK = 32
RFDETR_DEFAULT_RESOLUTIONS = {
    "medium": 576,
    "large": 704,
}
RFDETR_VARIANT_ALIASES = {
    "medium": "medium",
    "rf-detr-medium": "medium",
    "rfdetr-medium": "medium",
    "rfdetr-m": "medium",
    "m": "medium",
    "large": "large",
    "rf-detr-large": "large",
    "rfdetr-large": "large",
    "rfdetr-l": "large",
    "l": "large",
}


def _import_rfdetr(variant: str):
    try:
        from rfdetr import RFDETRLarge, RFDETRMedium

        rfdetr_classes = {
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        return rfdetr_classes[variant]
    except ImportError as exc:
        raise RuntimeError(
            "Roboflow RF-DETR backend is unavailable. "
            'Install dependencies with: pip install "rfdetr[train]>=1.6.5"'
        ) from exc
    except KeyError as exc:
        raise ValueError(
            f"Unsupported RF-DETR variant='{variant}'. Expected one of: "
            f"{sorted(RFDETR_DEFAULT_RESOLUTIONS)}."
        ) from exc


def _as_rfdetr_variant(value: Any) -> str:
    normalized = str(value).strip().lower()
    try:
        return RFDETR_VARIANT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Config value 'variant' must be one of {sorted(RFDETR_DEFAULT_RESOLUTIONS)}, "
            f"got: {value!r}"
        ) from exc


def _as_device(value: Any) -> str:
    parsed = str(value).strip().lower()
    if parsed not in {"cuda", "cpu", "mps"}:
        raise ValueError(f"Config value 'device' must be one of cuda, cpu, mps; got: {value!r}")
    return parsed


def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _class_name(class_id: int, class_mapping: dict[int, str]) -> str:
    if class_id in class_mapping:
        return class_mapping[class_id]

    ordered_names = [class_mapping[key] for key in sorted(class_mapping)]
    if 0 <= class_id < len(ordered_names):
        return ordered_names[class_id]
    return str(class_id)


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)

