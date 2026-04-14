from dataclasses import dataclass
from pathlib import Path

from murawa.services.artifacts import StandardizedArtifactCallback


@dataclass
class YoloMockModel:
    name: str = "yolo"

    def train(self, dataset_variant: str) -> dict:
        return {
            "weights": {"head.cls": [0.21, 0.37, 0.11]},
            "metrics": {"epochs": 1, "loss": 1.04, "mAP50": 0.23},
            "note": f"YOLO mock trained on {dataset_variant}",
        }

    def predict(self, mode: str) -> list[dict]:
        if mode == "frame":
            return [
                {"class": "player", "confidence": 0.91, "bbox_xyxy": [120, 80, 260, 360]},
                {"class": "ball", "confidence": 0.74, "bbox_xyxy": [342, 210, 358, 226]},
            ]
        return [
            {"frame_index": 10, "class": "player", "confidence": 0.88, "track_id": 4},
            {"frame_index": 11, "class": "ball", "confidence": 0.72, "track_id": 99},
        ]


@dataclass
class YoloAdapter:
    """Issue #10: target integration layer for real YOLO backend."""

    name: str = "yolo"

    def train(
        self,
        dataset_variant: str,
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        artifact_callback: StandardizedArtifactCallback | None = None,
    ) -> dict:
        raise NotImplementedError(
            "TODO(Issue #10): implement YOLO training adapter compatible with shared project flow "
            "(shared dataset format, shared config handling, shared metadata/checkpoint outputs)."
        )

    def predict(
        self,
        input_path: Path,
        *,
        checkpoint_path: Path,
        mode: str,
    ) -> list[dict]:
        raise NotImplementedError(
            "TODO(Issue #10): implement YOLO inference adapter for frame/match modes "
            "with the same output schema expected by common pipeline services."
        )
