"""
Vehicle Colour Classifier — High-Accuracy CIE L*a*b* Paint Panel Analysis
Isolates painted vehicle panels, excludes glass/tires/reflections,
runs K-Means clustering in Lab color space, and applies perceptual Delta-E classification.
"""
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from backend.app.core.config import settings
from backend.app.attributes.crop_extractor import extract_vehicle_body_roi, compute_image_quality

log = logging.getLogger(__name__)

CANONICAL_COLOURS = [
    "white", "black", "silver_grey", "red", "blue",
    "green", "yellow", "brown", "orange", "other", "unknown"
]


def normalize_colour_name(raw_name: str) -> str:
    """Normalize raw colour description to one of the canonical colour classes."""
    if not raw_name:
        return "unknown"
    s = raw_name.lower().replace("-", " ").replace("_", " ").strip()
    if any(k in s for k in ("white", "pearl", "ivory")):
        return "white"
    if any(k in s for k in ("black", "charcoal", "ebony")):
        return "black"
    if any(k in s for k in ("silver", "grey", "gray", "gunmetal", "metallic", "titanium")):
        return "silver_grey"
    if any(k in s for k in ("red", "maroon", "crimson", "burgundy", "ruby", "scarlet")):
        return "red"
    if any(k in s for k in ("blue", "navy", "cyan", "azure", "teal", "cobalt")):
        return "blue"
    if any(k in s for k in ("green", "olive", "emerald")):
        return "green"
    if any(k in s for k in ("yellow", "gold", "golden", "amber")):
        return "yellow"
    if any(k in s for k in ("brown", "bronze", "chocolate", "beige", "tan", "copper")):
        return "brown"
    if any(k in s for k in ("orange", "rust")):
        return "orange"
    if s in CANONICAL_COLOURS:
        return s
    return "other"


# Calibrated automotive paint reference centroids in CIE L*a*b* space
# (L: 0..100, a: -128..127, b: -128..127)
COLOR_CENTROIDS = {
    "white": [
        {"L": 92.0, "a": 0.0, "b": 2.0, "name": "pure_white"},
        {"L": 84.0, "a": -1.0, "b": 6.0, "name": "pearl_white"}
    ],
    "black": [
        {"L": 10.0, "a": 0.0, "b": 0.0, "name": "deep_black"},
        {"L": 18.0, "a": 1.0, "b": -1.0, "name": "carbon_black"}
    ],
    "silver_grey": [
        {"L": 72.0, "a": -0.5, "b": -1.0, "name": "metallic_silver"},
        {"L": 52.0, "a": 0.0, "b": 0.0, "name": "gunmetal_grey"},
        {"L": 38.0, "a": -0.5, "b": 1.0, "name": "slate_grey"}
    ],
    "red": [
        {"L": 45.0, "a": 64.0, "b": 48.0, "name": "bright_red"},
        {"L": 30.0, "a": 48.0, "b": 28.0, "name": "maroon"}
    ],
    "blue": [
        {"L": 40.0, "a": 15.0, "b": -54.0, "name": "royal_blue"},
        {"L": 22.0, "a": 8.0, "b": -34.0, "name": "navy_blue"},
        {"L": 66.0, "a": -14.0, "b": -34.0, "name": "sky_blue"}
    ],
    "green": [
        {"L": 34.0, "a": -36.0, "b": 18.0, "name": "forest_green"},
        {"L": 58.0, "a": -52.0, "b": 46.0, "name": "bright_green"}
    ],
    "yellow": [
        {"L": 82.0, "a": -6.0, "b": 80.0, "name": "commercial_yellow"},
        {"L": 72.0, "a": 12.0, "b": 72.0, "name": "gold_yellow"}
    ],
    "orange": [
        {"L": 60.0, "a": 46.0, "b": 64.0, "name": "vibrant_orange"},
        {"L": 46.0, "a": 38.0, "b": 48.0, "name": "burnt_orange"}
    ],
    "brown": [
        {"L": 32.0, "a": 18.0, "b": 28.0, "name": "chocolate_brown"},
        {"L": 46.0, "a": 16.0, "b": 34.0, "name": "bronze_tan"}
    ]
}


def delta_e(lab1: Tuple[float, float, float], lab2: Dict[str, float]) -> float:
    """Euclidean Delta-E distance in CIE L*a*b* space."""
    dL = lab1[0] - lab2["L"]
    da = lab1[1] - lab2["a"]
    db = lab1[2] - lab2["b"]
    return float(np.sqrt(dL * dL + da * da + db * db))


class VehicleColourClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.classes = CANONICAL_COLOURS
        self.threshold = settings.colour_conf_threshold

    def predict(self, vehicle_crop: np.ndarray) -> Dict[str, Any]:
        """
        Predict vehicle body color with high accuracy.
        1. Isolates vehicle body panels (eliminating wheels, road, glass).
        2. Segments paint pixels using lightness and saturation thresholds.
        3. Identifies the dominant body paint mode via K-Means in Lab space.
        4. Matches against calibrated automotive paint centroids using CIE Delta-E.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return {
                "value": "unknown",
                "score": 0.0,
                "status": "unknown",
                "alternatives": [],
                "probabilities": {}
            }

        quality = compute_image_quality(vehicle_crop)
        if not quality.is_valid or quality.score < 12.0:
            return {
                "value": "unknown",
                "score": round(quality.score / 100.0, 2),
                "status": "low_confidence",
                "alternatives": [],
                "probabilities": {}
            }

        # 1. Isolate central body panels
        body_roi = extract_vehicle_body_roi(vehicle_crop)
        h, w = body_roi.shape[:2]

        # 2. Extract paint candidate pixels
        # Convert to HSV and Lab
        hsv = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)
        lab_img = cv2.cvtColor(body_roi, cv2.COLOR_BGR2LAB)

        v_chan = hsv[:, :, 2]
        s_chan = hsv[:, :, 1]

        # Exclude deep undercarriage shadow (v < 22) and specular glares (v > 248 with s < 18)
        valid_mask = (v_chan >= 22) & ~((v_chan > 248) & (s_chan < 18))
        
        valid_lab_pixels = lab_img[valid_mask]
        if len(valid_lab_pixels) < 50:
            valid_lab_pixels = lab_img.reshape(-1, 3)

        # Scale OpenCV Lab (0..255) to standard CIE Lab (L: 0..100, a: -128..127, b: -128..127)
        float_lab = np.zeros((len(valid_lab_pixels), 3), dtype=np.float32)
        float_lab[:, 0] = valid_lab_pixels[:, 0].astype(np.float32) * 100.0 / 255.0
        float_lab[:, 1] = valid_lab_pixels[:, 1].astype(np.float32) - 128.0
        float_lab[:, 2] = valid_lab_pixels[:, 2].astype(np.float32) - 128.0

        # 3. K-Means clustering (K=4) to separate true paint from trim/reflections
        k = min(4, len(float_lab))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
        _, labels, centers = cv2.kmeans(float_lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        counts = np.bincount(labels.flatten())
        
        # Sort clusters by size
        cluster_order = np.argsort(counts)[::-1]

        # Pick dominant paint cluster (preferring clusters that aren't near-zero dark trim if chromatic paint exists)
        dominant_center = centers[cluster_order[0]]
        
        # If the largest cluster is deep dark (<15) but a secondary cluster is substantial (>25% size) and chromatic,
        # it is often body paint with heavy tinted glass/shadow
        if dominant_center[0] < 18.0 and len(cluster_order) > 1:
            second_center = centers[cluster_order[1]]
            second_chroma = np.sqrt(second_center[1]**2 + second_center[2]**2)
            if second_chroma > 18.0 and (counts[cluster_order[1]] / counts[cluster_order[0]]) > 0.25:
                dominant_center = second_center

        L_val, a_val, b_val = float(dominant_center[0]), float(dominant_center[1]), float(dominant_center[2])
        chroma = float(np.sqrt(a_val * a_val + b_val * b_val))

        # 4. Score distances to reference automotive paint centroids
        category_distances = {}
        for cat, centroids in COLOR_CENTROIDS.items():
            min_d = min(delta_e((L_val, a_val, b_val), c) for c in centroids)
            category_distances[cat] = min_d

        # Discrimination rules for chromatic vs achromatic
        # If chroma is very low (<= 14), vehicle is strictly white, silver_grey, or black
        if chroma <= 14.0:
            category_distances["red"] += 50.0
            category_distances["blue"] += 50.0
            category_distances["green"] += 50.0
            category_distances["yellow"] += 50.0
            category_distances["orange"] += 50.0
            category_distances["brown"] += 50.0

            if L_val >= 78.0:
                category_distances["white"] = max(0.0, category_distances["white"] * 0.4)
            elif L_val <= 26.0:
                category_distances["black"] = max(0.0, category_distances["black"] * 0.4)
            else:
                category_distances["silver_grey"] = max(0.0, category_distances["silver_grey"] * 0.4)
        else:
            # If chroma is strong (> 20), penalize achromatic classes
            category_distances["white"] += 35.0
            category_distances["silver_grey"] += 35.0
            if chroma > 28.0:
                category_distances["black"] += 30.0

        # Convert distances into calibrated probabilities using negative exponential
        inv_dists = {cat: np.exp(-d / 18.0) for cat, d in category_distances.items()}
        total_inv = sum(inv_dists.values()) or 1.0
        probs = {cat: float(v / total_inv) for cat, v in inv_dists.items()}

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_color, top_score = sorted_probs[0]
        alternatives = [{"value": k, "score": round(v, 3)} for k, v in sorted_probs[1:4]]

        status = "predicted" if top_score >= 0.40 else "low_confidence"

        return {
            "value": top_color,
            "score": round(top_score, 3),
            "status": status,
            "alternatives": alternatives,
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "paint_metrics": {
                "L": round(L_val, 1),
                "a": round(a_val, 1),
                "b": round(b_val, 1),
                "chroma": round(chroma, 1)
            }
        }
