"""
Tests for Multi-Frame Evidence Fusion
"""
import pytest
from backend.app.attributes.attribute_fusion import AttributeFusionEngine
from backend.app.recognition.plate_fusion import PlateFusionEngine


def test_attribute_fusion_dominant_voting():
    engine = AttributeFusionEngine()
    obs = [
        {
            "frame_timestamp_ms": 100.0,
            "quality_score": 80.0,
            "vehicle_crop_path": "crop_1.jpg",
            "colour": {"value": "white", "score": 0.85},
            "make": {"value": "Tata", "score": 0.80},
            "model": {"value": "Nexon", "score": 0.70},
            "vehicle_type": {"value": "SUV", "score": 0.90}
        },
        {
            "frame_timestamp_ms": 300.0,
            "quality_score": 75.0,
            "vehicle_crop_path": "crop_2.jpg",
            "colour": {"value": "white", "score": 0.80},
            "make": {"value": "Tata", "score": 0.82},
            "model": {"value": "Nexon", "score": 0.65},
            "vehicle_type": {"value": "SUV", "score": 0.90}
        },
        {
            "frame_timestamp_ms": 500.0,
            "quality_score": 20.0,  # Low quality anomaly
            "vehicle_crop_path": "crop_3.jpg",
            "colour": {"value": "black", "score": 0.40},
            "make": {"value": "Unknown", "score": 0.20},
            "model": {"value": "Unknown", "score": 0.10},
            "vehicle_type": {"value": "car", "score": 0.50}
        }
    ]

    fused = engine.fuse_track_attributes(obs)
    assert fused["fused_colour"] == "white"
    assert fused["fused_make"] == "Tata"
    assert fused["fused_model"] == "Nexon"
    assert fused["fused_type"] == "SUV"
    assert fused["frames_used"] == 3
    assert fused["best_vehicle_crop_path"] == "crop_1.jpg"


def test_plate_temporal_fusion():
    engine = PlateFusionEngine()
    reads = [
        {"text": "DL01AB1234", "confidence": 0.80, "quality_score": 70.0},
        {"text": "DL01AB1234", "confidence": 0.85, "quality_score": 75.0},
        {"text": "DL01AB123?", "confidence": 0.50, "quality_score": 40.0}
    ]

    fused = engine.fuse_plate_reads(reads)
    assert fused["fused_plate"] == "DL01AB1234"
    assert fused["plate_status"] == "clean"
    assert len(fused["unknown_positions"]) == 0
