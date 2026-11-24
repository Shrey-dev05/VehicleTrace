"""
Multi-Frame Attribute Fusion
Aggregates frame observations per tracked vehicle using quality-weighted probability voting.
Preserves frame-level provenance and detects evidentiary conflicts.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

log = logging.getLogger(__name__)


class AttributeFusionEngine:
    def __init__(self, min_frame_separation_ms: float = 150.0):
        self.min_frame_separation_ms = min_frame_separation_ms

    def fuse_track_attributes(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        observations: list of frame-level observation dicts containing:
        - frame_timestamp_ms: float
        - quality_score: float
        - vehicle_crop_path: str
        - plate_crop_path: str
        - colour: {"value": str, "score": float, "probabilities": dict}
        - make: {"value": str, "score": float, "probabilities": dict}
        - model: {"value": str, "score": float, "candidates": list}
        - vehicle_type: {"value": str, "score": float}
        - plate: {"text": str, "score": float, "unknown_positions": list}
        """
        if not observations:
            return self._empty_fusion()

        # Filter out near-zero quality or corrupted frames
        valid_obs = [o for o in observations if o.get("quality_score", 0.0) >= 10.0]
        if not valid_obs:
            valid_obs = observations

        # Pick best crops based on quality score
        best_vehicle_crop = max(valid_obs, key=lambda x: x.get("quality_score", 0.0)).get("vehicle_crop_path")
        best_plate_crop = None
        for o in sorted(valid_obs, key=lambda x: x.get("quality_score", 0.0), reverse=True):
            if o.get("plate_crop_path"):
                best_plate_crop = o.get("plate_crop_path")
                break

        # 1. Fuse Colour across frames
        colour_weights = defaultdict(float)
        colour_counts = defaultdict(int)
        for o in valid_obs:
            c = o.get("colour", {})
            val = c.get("value", "unknown")
            q = max(0.1, o.get("quality_score", 10.0) / 100.0)
            score = c.get("score", 0.5)
            weight = q * score
            colour_weights[val] += weight
            colour_counts[val] += 1

        total_cw = sum(colour_weights.values()) or 1.0
        sorted_colours = sorted(
            [(k, v / total_cw) for k, v in colour_weights.items() if k != "unknown"],
            key=lambda x: x[1],
            reverse=True
        )
        if sorted_colours:
            fused_colour = sorted_colours[0][0]
            fused_colour_score = round(sorted_colours[0][1], 3)
            colour_alts = [{"value": k, "score": round(v, 3)} for k, v in sorted_colours[1:4]]
        else:
            fused_colour = "unknown"
            fused_colour_score = 0.0
            colour_alts = []

        # 2. Fuse Make across frames
        make_weights = defaultdict(float)
        for o in valid_obs:
            m = o.get("make", {})
            val = m.get("value", "Unknown")
            q = max(0.1, o.get("quality_score", 10.0) / 100.0)
            score = m.get("score", 0.5)
            make_weights[val] += q * score

        total_mw = sum(make_weights.values()) or 1.0
        sorted_makes = sorted(
            [(k, v / total_mw) for k, v in make_weights.items() if k not in ("Unknown", "Other")],
            key=lambda x: x[1],
            reverse=True
        )
        if sorted_makes:
            fused_make = sorted_makes[0][0]
            fused_make_score = round(sorted_makes[0][1], 3)
            make_alts = [{"value": k, "score": round(v, 3)} for k, v in sorted_makes[1:4]]
        else:
            fused_make = "Unknown"
            fused_make_score = 0.0
            make_alts = []

        # 3. Fuse Vehicle Type
        type_weights = defaultdict(float)
        for o in valid_obs:
            t = o.get("vehicle_type", {})
            val = t.get("value", "car")
            q = max(0.1, o.get("quality_score", 10.0) / 100.0)
            type_weights[val] += q

        total_tw = sum(type_weights.values()) or 1.0
        sorted_types = sorted([(k, v / total_tw) for k, v in type_weights.items()], key=lambda x: x[1], reverse=True)
        fused_type = sorted_types[0][0] if sorted_types else "car"
        fused_type_score = round(sorted_types[0][1], 3) if sorted_types else 0.5

        # 4. Fuse Model
        model_weights = defaultdict(float)
        for o in valid_obs:
            m = o.get("model", {})
            val = m.get("value", "Unknown")
            if val not in ("Unknown", "Other"):
                model_weights[val] += m.get("score", 0.5)

        total_modw = sum(model_weights.values()) or 1.0
        sorted_models = sorted([(k, v / total_modw) for k, v in model_weights.items()], key=lambda x: x[1], reverse=True)
        if sorted_models:
            fused_model = sorted_models[0][0]
            fused_model_score = round(sorted_models[0][1], 3)
            model_cands = [{"value": k, "score": round(v, 3)} for k, v in sorted_models[1:4]]
        else:
            fused_model = "Unknown"
            fused_model_score = 0.0
            model_cands = []

        # 5. Determine Overall Fusion Status
        # Conflict detection: if top 2 predictions have very close high scores
        is_conflict = False
        if len(sorted_colours) >= 2 and (sorted_colours[0][1] - sorted_colours[1][1] < 0.12) and (sorted_colours[0][1] > 0.35):
            is_conflict = True

        if is_conflict:
            fusion_status = "conflict"
        elif fused_colour_score >= 0.55 and fused_make_score >= 0.50:
            fusion_status = "predicted"
        elif fused_colour != "unknown" or fused_make != "Unknown":
            fusion_status = "low_confidence"
        else:
            fusion_status = "unknown"

        timestamps = [o.get("frame_timestamp_ms", 0.0) for o in valid_obs]

        return {
            "fused_type": fused_type,
            "fused_type_score": fused_type_score,
            "fused_colour": fused_colour,
            "fused_colour_score": fused_colour_score,
            "colour_alternatives": colour_alts,
            "fused_make": fused_make,
            "fused_make_score": fused_make_score,
            "make_alternatives": make_alts,
            "fused_model": fused_model,
            "fused_model_score": fused_model_score,
            "model_candidates": model_cands,
            "fusion_status": fusion_status,
            "frames_used": len(valid_obs),
            "timestamps_ms": timestamps,
            "best_vehicle_crop_path": best_vehicle_crop,
            "best_plate_crop_path": best_plate_crop
        }

    def _empty_fusion(self) -> Dict[str, Any]:
        return {
            "fused_type": "car",
            "fused_type_score": 0.0,
            "fused_colour": "unknown",
            "fused_colour_score": 0.0,
            "colour_alternatives": [],
            "fused_make": "Unknown",
            "fused_make_score": 0.0,
            "make_alternatives": [],
            "fused_model": "Unknown",
            "fused_model_score": 0.0,
            "model_candidates": [],
            "fusion_status": "unknown",
            "frames_used": 0,
            "timestamps_ms": [],
            "best_vehicle_crop_path": None,
            "best_plate_crop_path": None
        }
