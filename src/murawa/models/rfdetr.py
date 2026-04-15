import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    from rfdetr import RFDETR
except ImportError:
    raise ImportError("The 'rfdetr' package is required. Install it using: pip install rfdetr")

from murawa.services.artifacts import StandardizedArtifactCallback


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


@dataclass
class RfDetrAdapter:
    """Issue #11: Target integration layer for real RF-DETR backend."""

    name: str = "rfdetr"
    device: str = "cuda"
    # 'rfdetr-base.pt' or 'rfdetr-large.pt' are standard base models provided by Roboflow
    base_model_name: str = "rfdetr-base.pt"

    def train(
        self,
        dataset_variant: str,
        *,
        config_path: Path | None = None,
        output_dir: Path | None = None,
        artifact_callback: StandardizedArtifactCallback | None = None,
    ) -> dict:
        """
        Runs RF-DETR training and reports results to the shared interface.
        """
        dataset_path = Path("data/selected") / dataset_variant
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset variant not found: {dataset_path}")

        # Ensure output directory exists based on project structure
        out_dir = output_dir or Path("models/checkpoints") / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize the base RF-DETR model
        model = RFDETR(self.base_model_name)

        # 2. Setup training arguments (compatible with shared config format)
        # We assume the dataset has a standard 'data.yaml' describing classes and splits
        train_args = {
            "data": str(dataset_path / "data.yaml"),
            "project": str(out_dir),
            "name": "run",
            "device": self.device
        }

        # Override defaults if a specific config file was provided
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                custom_cfg = yaml.safe_load(f)
                train_args.update(custom_cfg)

        # 3. Execute the training routine using the RFDETR backend
        results = model.train(**train_args)

        # 4. Extract generated artifacts and metrics
        # The exact properties depend on the library version, standard properties assumed
        best_weights_path = Path(results.save_dir) / "weights" / "best.pt"
        
        raw_metrics = {
            "epochs": getattr(results, "epochs_completed", 0),
            "loss": getattr(results, "loss", 0.0),
            "mAP50": getattr(results, "map50", 0.0)
        }

        # 5. Report via standard callback hook for pipeline integration
        if artifact_callback:
            artifact_callback.save_weights(best_weights_path)
            artifact_callback.save_metrics(raw_metrics)
            artifact_callback.save_metadata({
                "model_name": self.name,
                "dataset_variant": dataset_variant,
                "config_used": str(config_path) if config_path else "None"
            })

        return {
            "weights_path": str(best_weights_path),
            "metrics": raw_metrics,
            "note": f"{self.name} training completed for variant {dataset_variant}",
        }

    def predict(
        self,
        input_path: Path,
        *,
        checkpoint_path: Path,
        mode: str,
    ) -> list[dict]:
        """
        Runs inference using loaded weights, returning a list of detections
        formatted exactly as defined by the project contract.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {checkpoint_path}")

        # 1. Load the fine-tuned RF-DETR model and move to appropriate device
        model = RFDETR(str(checkpoint_path))
        model.to(self.device)

        predictions_output: List[Dict[str, Any]] = []

        # 2. Execute inference based on the requested mode
        if mode == "frame":
            # Image mode inference without tracking
            results = model.predict(source=str(input_path), save=False)
            
            for r in results:
                # Map bounding boxes to project dictionary contract
                for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    predictions_output.append({
                        "class": model.names[int(cls_id)],
                        "confidence": float(conf),
                        "bbox_xyxy": box.tolist()
                    })

        elif mode in ["match", "video"]:
            # Video mode leverages the internal tracker (e.g., ByteTrack)
            results = model.track(source=str(input_path), save=False, tracker="bytetrack.yaml")
            
            frame_idx = 0
            for r in results:
                # Extract tracker IDs if available, else fallback to empty iterable
                track_ids = r.boxes.id if r.boxes.id is not None else []
                
                # Zip ensures we only extract valid mapped data
                for box, cls_id, conf, trk_id in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf, track_ids):
                    predictions_output.append({
                        "frame_index": frame_idx,
                        "class": model.names[int(cls_id)],
                        "confidence": float(conf),
                        "track_id": int(trk_id),
                        "bbox_xyxy": box.tolist()
                    })
                frame_idx += 1
        else:
            raise ValueError(f"Model {self.name} does not support prediction mode: {mode}")

        return predictions_output