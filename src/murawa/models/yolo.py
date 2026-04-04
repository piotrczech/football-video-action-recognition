from dataclasses import dataclass


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
