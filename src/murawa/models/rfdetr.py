from dataclasses import dataclass


@dataclass
class RfDetrMockModel:
    name: str = "rfdetr"

    def train(self, dataset_variant: str) -> dict:
        return {
            "weights": {"decoder.layer_0": [0.17, 0.29, 0.44]},
            "metrics": {"epochs": 1, "loss": 0.97, "mAP50": 0.27},
            "note": f"RF-DETR mock trained on {dataset_variant}",
        }

    def predict(self, mode: str) -> list[dict]:
        if mode == "frame":
            return [
                {"class": "player", "confidence": 0.89, "bbox_xyxy": [98, 76, 244, 342]},
                {"class": "referee", "confidence": 0.77, "bbox_xyxy": [301, 91, 355, 292]},
            ]
        return [
            {"frame_index": 20, "class": "player", "confidence": 0.87, "track_id": 8},
            {"frame_index": 21, "class": "referee", "confidence": 0.75, "track_id": 15},
        ]
