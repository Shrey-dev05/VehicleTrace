"""
Tests for Vehicle Registry Candidate Matching
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database.models import Base, VehicleRegistry
from backend.app.matching.registry_matcher import RegistryMatcher, score_plate_compatibility


@pytest.fixture
def mock_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Seed test registry
    records = [
        VehicleRegistry(registration_number="DL01AB1234", vehicle_type="sedan", colour="white", make="Honda", model="City"),
        VehicleRegistry(registration_number="MH02CD5678", vehicle_type="SUV", colour="silver_grey", make="Tata", model="Nexon"),
        VehicleRegistry(registration_number="KA03EF9012", vehicle_type="hatchback", colour="red", make="Maruti_Suzuki", model="Swift")
    ]
    db.add_all(records)
    db.commit()
    yield db
    db.close()


def test_exact_plate_score():
    score, desc = score_plate_compatibility("DL01AB1234", "DL01AB1234")
    assert score == 1.0


def test_wildcard_plate_score():
    score, desc = score_plate_compatibility("DL01AB??34", "DL01AB1234")
    assert score > 0.70
    assert "wildcard" in desc.lower()


def test_mismatch_plate_penalty():
    score, desc = score_plate_compatibility("HR26DQ1111", "DL01AB1234")
    assert score < 0.40


def test_registry_matching_top_candidate(mock_db):
    matcher = RegistryMatcher(acceptance_threshold=0.45)
    res = matcher.match_candidates(
        db=mock_db,
        observed_plate="MH02CD5678",
        observed_colour="silver_grey",
        observed_make="Tata",
        observed_model="Nexon",
        observed_type="SUV"
    )

    assert res["status"] == "CANDIDATES_FOUND"
    top = res["candidates"][0]
    assert top["registration_number"] == "MH02CD5678"
    assert top["ranking_score"] > 0.90


def test_registry_matching_abstention(mock_db):
    matcher = RegistryMatcher(acceptance_threshold=0.45)
    # Completely unfamiliar unregistered vehicle
    res = matcher.match_candidates(
        db=mock_db,
        observed_plate="ZZ99ZZ9999",
        observed_colour="orange",
        observed_make="Unknown",
        observed_model="Unknown",
        observed_type="bus"
    )
    assert res["status"] == "NO_CONFIDENT_MATCH"


def test_find_close_matches_with_wildcards(mock_db):
    matcher = RegistryMatcher()
    matches = matcher.find_close_matches(
        db=mock_db,
        observed_plate="MH02CD?678",
        observed_colour="silver_grey",
        observed_make="Tata",
        observed_model="Nexon",
        observed_type="SUV",
        limit=5
    )
    assert len(matches) > 0
    top = matches[0]
    assert top["registration_number"] == "MH02CD5678"
    assert top["match_percentage"] >= 90
    assert top["manufacturer"] == "Tata"
    assert top["vehicle_model"] == "Nexon"


def test_find_close_matches_from_json_registry(mock_db):
    import json
    from backend.app.core.config import settings
    registry_file = settings.base_dir / "backend" / "data" / "vehicle_registry.json"
    if registry_file.exists():
        with open(registry_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records[:10]:
            mock_db.add(VehicleRegistry(
                registration_number=r["registration_number"],
                vehicle_type=r.get("category_type", "car"),
                colour=r.get("body_colour", "unknown"),
                make=r.get("manufacturer", "Unknown"),
                model=r.get("vehicle_model", "Unknown"),
                state=r.get("state", "DL")
            ))
        mock_db.commit()

        matcher = RegistryMatcher()
        # Observed plate with missing digits '?'
        matches = matcher.find_close_matches(
            db=mock_db,
            observed_plate="CG04KL?821",
            observed_colour="White",
            observed_make="Maruti Suzuki",
            observed_model="Swift",
            observed_type="car",
            limit=5
        )
        assert len(matches) > 0
        top = matches[0]
        assert top["registration_number"] == "CG04KL4821"
        assert top["match_percentage"] >= 90
        assert top["manufacturer"] == "Maruti Suzuki"
        assert top["vehicle_model"] == "Swift"
        assert top["body_colour"] == "White"


def test_find_close_matches_abstains_when_no_missing_digits(mock_db):
    matcher = RegistryMatcher()
    # Plate is completely known without missing digits
    matches = matcher.find_close_matches(
        db=mock_db,
        observed_plate="MH02CD5678",
        observed_colour="silver_grey",
        observed_make="Tata",
        observed_model="Nexon",
        observed_type="SUV"
    )
    # Must only find close matches if the number extracted has missing digits
    assert matches == []
