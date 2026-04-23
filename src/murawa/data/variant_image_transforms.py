from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

TRANSFORM_PIPELINE_NAME = "lightweight_color_scale_noise"
TRANSFORM_PIPELINE_OPS = (
    "light brightness/contrast/saturation shift",
    "downscale-upscale resolution simulation",
    "gaussian noise",
)


@dataclass(frozen=True)
class TransformMetadata:
    pipeline: str
    brightness_beta: int
    contrast_alpha: float
    saturation_scale: float
    downscale_factor: float
    gaussian_sigma: float


def apply_lightweight_training_transform(image: np.ndarray, key: str) -> tuple[np.ndarray, TransformMetadata]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    brightness_beta = _pick_int(digest[0:2], low=-10, high=10)
    contrast_alpha = 1.0 + (_pick_int(digest[2:4], low=-8, high=8) / 100.0)
    saturation_scale = 1.0 + (_pick_int(digest[4:6], low=-10, high=10) / 100.0)
    downscale_factor = 0.75 + ((int.from_bytes(digest[6:8], "big") % 16) / 100.0)
    gaussian_sigma = 4.0 + float(int.from_bytes(digest[8:10], "big") % 5)

    transformed = _apply_brightness_contrast(image, alpha=contrast_alpha, beta=brightness_beta)
    transformed = _apply_saturation_shift(transformed, saturation_scale=saturation_scale)
    transformed = _apply_downscale_upscale(transformed, factor=downscale_factor)
    transformed = _apply_gaussian_noise(transformed, sigma=gaussian_sigma, seed=int.from_bytes(digest[10:18], "big"))

    metadata = TransformMetadata(
        pipeline=TRANSFORM_PIPELINE_NAME,
        brightness_beta=brightness_beta,
        contrast_alpha=contrast_alpha,
        saturation_scale=saturation_scale,
        downscale_factor=downscale_factor,
        gaussian_sigma=gaussian_sigma,
    )
    return transformed, metadata


def _pick_int(raw: bytes, *, low: int, high: int) -> int:
    span = high - low + 1
    return low + (int.from_bytes(raw, "big") % span)


def _apply_brightness_contrast(image: np.ndarray, *, alpha: float, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _apply_saturation_shift(image: np.ndarray, *, saturation_scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return shifted


def _apply_downscale_upscale(image: np.ndarray, *, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    small_width = max(1, int(round(width * factor)))
    small_height = max(1, int(round(height * factor)))

    reduced = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    return restored


def _apply_gaussian_noise(image: np.ndarray, *, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)