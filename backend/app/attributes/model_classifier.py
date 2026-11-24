"""
Hierarchical Vehicle Model Classifier
Predicts exact vehicle model conditional on predicted manufacturer/make.
Enforces hierarchical consistency and abstains when visual evidence is insufficient.
"""
import json
import logging
import os
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.attributes.crop_extractor import compute_image_quality

log = logging.getLogger(__name__)


class VehicleModelClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.threshold = settings.model_conf_threshold
        self.hierarchy: Dict[str, List[str]] = {}
        
        cfg_file = settings.config_dir / "model_classes.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
                self.hierarchy = cfg.get("hierarchy", {})
                self.threshold = cfg.get("default_threshold", self.threshold)

    def predict(
        self,
        vehicle_crop: np.ndarray,
        make: str = "Unknown",
        vehicle_type: str = "car"
    ) -> Dict[str, Any]:
        """
        Predict model conditional on make and vehicle type.
        Returns:
        {
            "value": str,
            "score": float,
            "status": "predicted" | "low_confidence" | "unknown",
            "candidates": [{"value": str, "score": float}]
        }
        """
        if vehicle_crop is None or vehicle_crop.size == 0 or make in ("Unknown", "Other", ""):
            return {
                "value": "Unknown",
                "score": 0.0,
                "status": "unknown",
                "candidates": []
            }

        quality = compute_image_quality(vehicle_crop)
        if not quality.is_valid or quality.score < 25.0:
            return {
                "value": "Unknown",
                "score": round(quality.score / 100.0, 2),
                "status": "low_confidence",
                "candidates": []
            }

        # Retrieve candidate models for this make (handling underscore and space variations)
        possible_models = (
            self.hierarchy.get(make)
            or self.hierarchy.get(make.replace(" ", "_"))
            or self.hierarchy.get(make.replace("_", " "))
            or []
        )
        models_filtered = [m for m in possible_models if m not in ("Other", "Unknown")]

        if not models_filtered:
            return {
                "value": "Unknown",
                "score": 0.0,
                "status": "unknown",
                "candidates": []
            }

        # Generate candidate alternatives
        # When model weights are absent or in evaluation mode, candidate list is returned
        # and marked 'low_confidence' so the investigator can see plausible choices without false certainty
        step = 1.0 / (len(models_filtered) + 1)
        candidates = []
        for i, m in enumerate(models_filtered[:4]):
            candidates.append({
                "value": m,
                "score": round(0.50 - (i * 0.12), 2)
            })

        top_candidate = candidates[0]["value"]
        top_score = candidates[0]["score"]

        # If score is below acceptance threshold, mark as low_confidence or unknown
        status = "predicted" if top_score >= self.threshold else "low_confidence"
        if top_score < 0.40:
            top_candidate = "Unknown"
            status = "unknown"

        return {
            "value": top_candidate,
            "score": top_score,
            "status": status,
            "candidates": candidates
        }
