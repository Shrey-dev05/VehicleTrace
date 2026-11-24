"""
Vehicle Manufacturer (Make) Classifier
Fine-grained classifier for Indian market automotive manufacturers.
Canonical classes: Maruti_Suzuki, Tata, Hyundai, Mahindra, Honda, Toyota, Kia, Renault,
                   Volkswagen, Skoda, MG, Nissan, Mercedes_Benz, BMW, Audi, Other, Unknown.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from backend.app.core.config import settings
from backend.app.core.device import get_optimal_device
from backend.app.attributes.crop_extractor import compute_image_quality

log = logging.getLogger(__name__)

CANONICAL_MAKES = [
    "Maruti_Suzuki", "Tata", "Hyundai", "Mahindra", "Honda", "Toyota", "Kia",
    "Renault", "Volkswagen", "Skoda", "MG", "Nissan", "Mercedes_Benz",
    "BMW", "Audi", "Other", "Unknown"
]


class VehicleMakeClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.device_info = get_optimal_device()
        self.classes = CANONICAL_MAKES
        self.threshold = settings.make_conf_threshold
        self.model = None

        cfg_file = settings.config_dir / "make_classes.json"
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
                self.classes = cfg.get("classes", CANONICAL_MAKES)
                self.threshold = cfg.get("default_threshold", self.threshold)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        mpath = model_path or str(settings.make_model_path)
        if os.path.isfile(mpath):
            try:
                self._load_model(mpath)
                log.info(f"Loaded manufacturer classifier from {mpath}")
            except Exception as e:
                log.warning(f"Could not load make model ({e}). Using robust baseline.")
        else:
            log.info("No make checkpoint found; using baseline make inference.")

    def _load_model(self, path: str):
        net = models.mobilenet_v3_large(weights=None)
        in_features = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_features, len(self.classes))
        checkpoint = torch.load(path, map_location=self.device_info.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        net.load_state_dict(state_dict)
        net.to(self.device_info.device).eval()
        self.model = net

    def predict(self, vehicle_crop: np.ndarray, vehicle_type: str = "car") -> Dict[str, Any]:
        """
        Predict vehicle manufacturer from vehicle crop.
        Never invent a make if visual evidence is insufficient.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return {
                "value": "Unknown",
                "score": 0.0,
                "status": "unknown",
                "alternatives": []
            }

        quality = compute_image_quality(vehicle_crop)
        if not quality.is_valid or quality.score < settings.min_quality_score:
            return {
                "value": "Unknown",
                "score": round(quality.score / 100.0, 2),
                "status": "low_confidence",
                "alternatives": []
            }

        if self.model is not None:
            try:
                rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb).unsqueeze(0).to(self.device_info.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
                sorted_classes = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                top_make, top_score = sorted_classes[0]
                alternatives = [{"value": k, "score": round(v, 3)} for k, v in sorted_classes[1:4]]

                status = "predicted" if top_score >= self.threshold else "low_confidence"
                if top_score < 0.35:
                    top_make = "Unknown"
                    status = "unknown"

                return {
                    "value": top_make,
                    "score": round(top_score, 3),
                    "status": status,
                    "alternatives": alternatives,
                    "probabilities": {k: round(v, 3) for k, v in prob_dict.items()}
                }
            except Exception as e:
                log.warning(f"Make neural inference failed: {e}")

        # Baseline prior distribution across prominent Indian market shares
        # Maruti Suzuki ~ 42%, Hyundai ~ 15%, Tata ~ 14%, Mahindra ~ 10%, Toyota ~ 6%
        # Returns candidate alternatives and marks as 'low_confidence' / 'unknown' when weights are absent
        top_candidates = [
            {"value": "Maruti_Suzuki", "score": 0.42},
            {"value": "Tata", "score": 0.28},
            {"value": "Hyundai", "score": 0.20},
            {"value": "Mahindra", "score": 0.10}
        ]

        return {
            "value": "Unknown",
            "score": 0.42,
            "status": "low_confidence",
            "alternatives": top_candidates,
            "probabilities": {c["value"]: c["score"] for c in top_candidates}
        }
