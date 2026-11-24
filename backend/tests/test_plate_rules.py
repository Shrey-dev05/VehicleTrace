"""
Tests for Indian Number Plate Syntax and Validation Rules
"""
import pytest
from backend.app.core.plate_rules import (
    validate_plate, correct_plate_text, normalize_plate_text, is_indian_plate_format
)


def test_standard_plate_validation():
    v = validate_plate("DL01AB1234")
    assert v.format_ok is True
    assert v.state_ok is True
    assert v.rto_ok is True
    assert v.plate_type == "standard"
    assert v.state_code == "DL"


def test_bharat_series_validation():
    v = validate_plate("22BH1234AA")
    assert v.format_ok is True
    assert v.plate_type == "bh"
    assert v.number == "1234"


def test_partial_wildcard_plate():
    v = validate_plate("CG07A?1234")
    assert v.plate_type == "partial"
    assert v.format_ok is False


def test_plate_text_correction():
    # O in place of 0 in numeric section
    corrected = correct_plate_text("DL01AB123O")
    assert corrected.endswith("0")

    # I in place of 1 in numeric section
    corrected_i = correct_plate_text("DL01ABI234")
    assert "1" in corrected_i


def test_is_indian_plate_format_standard():
    assert is_indian_plate_format("DL12CC3495") is True
    assert is_indian_plate_format("MH02CB1234") is True
    assert is_indian_plate_format("CH01AA0001") is True
    assert is_indian_plate_format("HR26DQ5555") is True


def test_is_indian_plate_format_partial_wildcard():
    # Number digit occluded with ?
    assert is_indian_plate_format("DL12CC?495") is True
    assert is_indian_plate_format("MH02CB1?34") is True
    assert is_indian_plate_format("CG07A?1234") is True


def test_is_indian_plate_format_bharat_series():
    assert is_indian_plate_format("22BH1234AA") is True
    assert is_indian_plate_format("21BH?234AB") is True


def test_is_indian_plate_format_invalid_and_foreign_rejection():
    assert is_indian_plate_format("OBDB4668") is False   # Malaysian car
    assert is_indian_plate_format("MW51VSU") is False    # UK plate
    assert is_indian_plate_format("TEST1234") is False   # Fake state
    assert is_indian_plate_format("112233") is False     # Pure digits
    assert is_indian_plate_format("???") is False        # Too short
    assert is_indian_plate_format("??????????") is False # No known characters
