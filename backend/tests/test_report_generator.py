"""
Unit Tests for Forensic Investigation Evidence Report Generator
Verifies:
- Outcome Mode 1: Complete Number Extracted
- Outcome Mode 2: Partial / Missing Number – Vehicle Dataset Matching (wildcard '?' never guessed)
- Outcome Mode 3: Number Plate Not Readable – Dataset-based Visual Attribute Matching
- The three core questions answered upfront in Executive Summary
- Candidate comparison view & evidence-based explanations
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database.models import Base, CaseRecord, VehicleFusion, VehicleObservation, VehicleRegistry
from backend.app.reports.report_generator import generate_case_report_html


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Seed Registry Vehicles
    registry_vehicles = [
        VehicleRegistry(registration_number="CG07AB1234", vehicle_type="SUV", colour="White", make="Tata", model="Nexon", state="CG", rto="Bhilai"),
        VehicleRegistry(registration_number="CG07AC1234", vehicle_type="SUV", colour="White", make="Tata", model="Nexon", state="CG", rto="Bhilai"),
        VehicleRegistry(registration_number="CG07AD2234", vehicle_type="SUV", colour="White", make="Tata", model="Punch", state="CG", rto="Bhilai"),
        VehicleRegistry(registration_number="CG07AF1234", vehicle_type="SUV", colour="White", make="Mahindra", model="XUV300", state="CG", rto="Bhilai"),
        VehicleRegistry(registration_number="CG07AB9234", vehicle_type="SUV", colour="Silver", make="Tata", model="Nexon", state="CG", rto="Bhilai"),
        VehicleRegistry(registration_number="MH12RT7305", vehicle_type="Sedan", colour="Black", make="Hyundai", model="Elantra", state="MH", rto="Pune")
    ]
    db.add_all(registry_vehicles)
    db.commit()

    yield db
    db.close()


def test_report_mode_1_complete_number_extracted(test_db):
    """Outcome 1: Complete number plate extracted without unreadable characters."""
    case = CaseRecord(
        id="VT-CASE-001",
        title="Incident at Crossing A",
        source_type="video",
        source_path="/path/to/cctv_traffic_01.mp4"
    )
    fusion = VehicleFusion(
        case_id="VT-CASE-001",
        track_id="trk_01",
        fused_plate="CG07AB1234",
        plate_conf=0.95,
        plate_status="clean",
        fused_colour="white",
        fused_colour_score=0.96,
        fused_make="Tata",
        fused_make_score=0.90,
        fused_model="Nexon",
        fused_model_score=0.85,
        fused_type="SUV",
        fused_type_score=0.92,
        frames_used=24
    )
    test_db.add_all([case, fusion])
    test_db.commit()

    html = generate_case_report_html("VT-CASE-001", test_db)

    # Verifications for Outcome Mode 1
    assert "MODE 1: COMPLETE NUMBER EXTRACTED" in html
    assert "CG07AB1234" in html
    assert "Executive Evidence Summary" in html
    assert "What did AI detect from CCTV?" in html
    assert "How much plate could it reliably extract?" in html
    assert "Most relevant dataset matches" in html
    assert "SECTION 1" in html
    assert "SECTION 2" in html
    assert "SECTION 3" in html
    assert "SECTION 4" in html
    assert "SECTION 5" in html
    assert "Tata" in html
    assert "Nexon" in html


def test_report_mode_2_partial_missing_number_dataset_matching(test_db):
    """Outcome 2: Partial plate with wildcard '?' matching top 5 candidates."""
    case = CaseRecord(
        id="VT-CASE-002",
        title="High Speed Corridor Breach",
        source_type="video",
        source_path="/path/to/corridor_cam.mp4"
    )
    fusion = VehicleFusion(
        case_id="VT-CASE-002",
        track_id="trk_02",
        fused_plate="CG07A??234",
        plate_conf=0.88,
        plate_status="partial",
        fused_colour="white",
        fused_colour_score=0.94,
        fused_make="Tata",
        fused_make_score=0.87,
        fused_model="Unknown",  # Model is unknown
        fused_model_score=0.40,
        fused_type="SUV",
        fused_type_score=0.91,
        frames_used=18
    )
    test_db.add_all([case, fusion])
    test_db.commit()

    html = generate_case_report_html("VT-CASE-002", test_db)

    # Verifications for Outcome Mode 2
    assert "MODE 2: PARTIAL / MISSING NUMBER" in html
    assert "CG07A" in html
    assert "occluded-char" in html  # '?' highlighted, never guessed
    assert "Tata" in html
    assert "Top candidate identified as" in html
    assert "CG07AB1234" in html
    assert "Evidence-Based Weighted Ranking Algorithm" in html
    assert "Side-by-Side Candidate Comparison Matrix" in html
    assert "Investigative Evidence Explanation" in html


def test_report_mode_3_number_plate_not_readable(test_db):
    """Outcome 3: Plate unreadable; identification executed using visual similarity."""
    case = CaseRecord(
        id="VT-CASE-003",
        title="Night Patrol Surveillance",
        source_type="image",
        source_path="/path/to/vehicle_still.jpg"
    )
    fusion = VehicleFusion(
        case_id="VT-CASE-003",
        track_id="trk_03",
        fused_plate="NOT_READABLE",
        plate_conf=0.0,
        plate_status="unknown",
        fused_colour="white",
        fused_colour_score=0.92,
        fused_make="Tata",
        fused_make_score=0.85,
        fused_model="Unknown",
        fused_model_score=0.30,
        fused_type="SUV",
        fused_type_score=0.89,
        frames_used=1
    )
    test_db.add_all([case, fusion])
    test_db.commit()

    html = generate_case_report_html("VT-CASE-003", test_db)

    # Verifications for Outcome Mode 3
    assert "MODE 3: NUMBER PLATE NOT READABLE" in html
    assert "UNREADABLE / NOT DETECTED" in html
    assert "Dynamic Redistribution Mode" in html
    assert "S = 0.35M + 0.25C + 0.25B + 0.15V" in html
    assert "Visual ROI" in html
    assert "Forensic Occlusion Integrity Principle" in html


def test_individual_track_report_generation(test_db):
    """Verify that an individual report can be generated for a specific track ID."""
    case = CaseRecord(
        id="VT-CASE-MULTI",
        title="Multi Vehicle Scene",
        source_type="video",
        source_path="/path/to/multi.mp4"
    )
    fusion_1 = VehicleFusion(
        case_id="VT-CASE-MULTI",
        track_id="trk_alpha",
        fused_plate="CG07AB1234",
        plate_conf=0.96,
        plate_status="clean",
        fused_colour="white",
        fused_colour_score=0.95,
        fused_make="Tata",
        fused_make_score=0.92,
        fused_model="Nexon",
        fused_model_score=0.88,
        fused_type="SUV",
        fused_type_score=0.94,
        frames_used=20
    )
    fusion_2 = VehicleFusion(
        case_id="VT-CASE-MULTI",
        track_id="trk_beta",
        fused_plate="MH12RT7305",
        plate_conf=0.93,
        plate_status="clean",
        fused_colour="black",
        fused_colour_score=0.91,
        fused_make="Hyundai",
        fused_make_score=0.89,
        fused_model="Elantra",
        fused_model_score=0.84,
        fused_type="Sedan",
        fused_type_score=0.90,
        frames_used=15
    )
    test_db.add_all([case, fusion_1, fusion_2])
    test_db.commit()

    # Request individual report for trk_alpha only
    html_alpha = generate_case_report_html("VT-CASE-MULTI", test_db, track_id="trk_alpha")
    assert "INDIVIDUAL VEHICLE FORENSIC REPORT" in html_alpha
    assert "TRACK #trk_alpha" in html_alpha
    assert "CG07AB1234" in html_alpha
    # trk_beta data must NOT be in the individual dossier for trk_alpha
    assert "MH12RT7305" not in html_alpha

    # Request invalid track ID returns friendly not found page
    html_404 = generate_case_report_html("VT-CASE-MULTI", test_db, track_id="non_existent")
    assert "Individual Vehicle Track Not Found" in html_404


def test_complete_plate_cross_verification_anomaly(test_db):
    """When all digits are visible but detected visual attributes contradict registry, flag anomaly."""
    case = CaseRecord(
        id="VT-CASE-ANOMALY",
        title="Suspect Cloned Vehicle",
        source_type="video",
        source_path="/path/to/suspect.mp4"
    )
    # CG07AB1234 in test_db registry is a White Tata Nexon SUV.
    # Here CCTV detects the plate on a Red Mahindra vehicle.
    fusion = VehicleFusion(
        case_id="VT-CASE-ANOMALY",
        track_id="trk_anomaly",
        fused_plate="CG07AB1234",
        plate_conf=0.97,
        plate_status="clean",
        fused_colour="red",
        fused_colour_score=0.96,
        fused_make="Mahindra",
        fused_make_score=0.90,
        fused_model="Thar",
        fused_model_score=0.82,
        fused_type="SUV",
        fused_type_score=0.92,
        frames_used=30
    )
    test_db.add_all([case, fusion])
    test_db.commit()

    html = generate_case_report_html("VT-CASE-ANOMALY", test_db, track_id="trk_anomaly")
    assert "CENTRALISED REGISTRY CROSS-VERIFICATION" in html
    assert "ANOMALY DETECTED" in html
    assert "SUSPECT CLONED / REPAINTED VEHICLE" in html
    assert "Centralised Dashboard Status" in html


def test_partial_plate_5_compatible_vehicles_section(test_db):
    """When some digits are occluded (?), Section 4 displays 5 Most Compatible Vehicles."""
    case = CaseRecord(
        id="VT-CASE-PARTIAL",
        title="Corridor Cam Night",
        source_type="video",
        source_path="/path/to/partial.mp4"
    )
    fusion = VehicleFusion(
        case_id="VT-CASE-PARTIAL",
        track_id="trk_partial",
        fused_plate="CG07A??234",
        plate_conf=0.85,
        plate_status="partial",
        fused_colour="white",
        fused_colour_score=0.95,
        fused_make="Tata",
        fused_make_score=0.91,
        fused_model="Nexon",
        fused_model_score=0.86,
        fused_type="SUV",
        fused_type_score=0.94,
        frames_used=22
    )
    test_db.add_all([case, fusion])
    test_db.commit()

    html = generate_case_report_html("VT-CASE-PARTIAL", test_db, track_id="trk_partial")
    assert "SECTION 4: 5 MOST COMPATIBLE VEHICLES" in html
    assert "Evidence-Based Weighted Ranking Algorithm" in html
    assert "Side-by-Side Candidate Comparison Matrix" in html
    assert "CG07AB1234" in html

