"""
Tests for Crop Extractor and Quality Metrics
"""
import numpy as np
import pytest
from backend.app.attributes.crop_extractor import (
    extract_vehicle_crop, extract_plate_crop, extract_vehicle_body_roi, compute_image_quality
)


def test_vehicle_crop_extraction():
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[100:300, 150:450] = 200
    bbox = [150, 100, 450, 300]

    crop, adj_box = extract_vehicle_crop(dummy_frame, bbox, expand_ratio=0.05)
    assert crop is not None
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
    # Coordinates should be expanded safely within frame bounds
    assert adj_box[0] <= bbox[0]
    assert adj_box[1] <= bbox[1]
    assert adj_box[2] >= bbox[2]
    assert adj_box[3] >= bbox[3]


def test_plate_crop_extraction():
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    plate_bbox = [200, 250, 350, 290]

    crop, adj_box = extract_plate_crop(dummy_frame, plate_bbox)
    assert crop is not None
    assert crop.shape[1] >= 150


def test_vehicle_body_roi_isolation():
    dummy_vehicle = np.ones((200, 300, 3), dtype=np.uint8) * 100
    body_roi = extract_vehicle_body_roi(dummy_vehicle)
    assert body_roi is not None
    assert body_roi.shape[0] < dummy_vehicle.shape[0]
    assert body_roi.shape[1] < dummy_vehicle.shape[1]


def test_image_quality_assessment():
    dummy_sharp = np.random.randint(0, 255, (120, 240, 3), dtype=np.uint8)
    q = compute_image_quality(dummy_sharp)
    assert q.score > 0.0
    assert q.is_valid is True

    # Blank flat image should have low sharpness
    dummy_flat = np.ones((80, 80, 3), dtype=np.uint8) * 50
    q_flat = compute_image_quality(dummy_flat)
    assert q_flat.sharpness == 0.0
