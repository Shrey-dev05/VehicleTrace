"""
Tests for Vehicle Colour Classifier and CIE L*a*b* Paint Panel Analysis
"""
import numpy as np
import pytest
from backend.app.attributes.colour_classifier import VehicleColourClassifier


def test_canonical_colour_classes():
    clf = VehicleColourClassifier()
    assert "white" in clf.classes
    assert "black" in clf.classes
    assert "silver_grey" in clf.classes
    assert "red" in clf.classes
    assert "blue" in clf.classes
    assert "green" in clf.classes
    assert "yellow" in clf.classes
    assert "orange" in clf.classes


def test_colour_analyzer_synthetic_all_classes():
    clf = VehicleColourClassifier()
    test_cases = [
        ("red", [0, 0, 220]),
        ("maroon", [20, 20, 140]),
        ("royal_blue", [220, 80, 20]),
        ("navy_blue", [130, 35, 15]),
        ("pure_white", [245, 245, 245]),
        ("deep_black", [15, 15, 15]),
        ("metallic_silver", [185, 185, 185]),
        ("dark_grey", [75, 75, 75]),
        ("taxi_yellow", [0, 215, 240]),
        ("vibrant_orange", [0, 130, 245]),
        ("forest_green", [25, 140, 35])
    ]

    expected_map = {
        "red": "red", "maroon": "red",
        "royal_blue": "blue", "navy_blue": "blue",
        "pure_white": "white", "deep_black": "black",
        "metallic_silver": "silver_grey", "dark_grey": "silver_grey",
        "taxi_yellow": "yellow", "vibrant_orange": "orange",
        "forest_green": "green"
    }

    for name, bgr in test_cases:
        crop = np.zeros((120, 160, 3), dtype=np.uint8)
        crop[:, :] = bgr
        crop[-15:, :] = [20, 20, 20]  # simulated tire
        res = clf.predict(crop)
        assert res["value"] == expected_map[name], f"Failed for {name}: expected {expected_map[name]}, got {res['value']}"
        assert res["score"] > 0.5


def test_colour_analyzer_abstention_on_corrupt():
    clf = VehicleColourClassifier()
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    res = clf.predict(empty_crop)
    assert res["value"] == "unknown"
    assert res["status"] == "unknown"


def test_normalize_colour_name():
    from backend.app.attributes.colour_classifier import normalize_colour_name
    assert normalize_colour_name("Silver") == "silver_grey"
    assert normalize_colour_name("grey") == "silver_grey"
    assert normalize_colour_name("dark_blue") == "blue"
    assert normalize_colour_name("maroon") == "red"
    assert normalize_colour_name("white") == "white"
    assert normalize_colour_name("golden") == "yellow"
