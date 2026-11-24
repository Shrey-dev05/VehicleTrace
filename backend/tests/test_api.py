"""
End-to-End API Integration Tests
"""
import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import init_db

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_registry_list_endpoint():
    response = client.get("/api/matching/registry")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "registration_number" in data[0]


def test_cases_list_endpoint():
    response = client.get("/api/cases")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_image_analysis_pipeline_integration():
    # Create synthetic test car image (white car with simulated license plate area)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 230  # Light white vehicle body
    # Draw simulated dark bumper and plate region
    cv2.rectangle(img, (240, 340), (400, 390), (255, 255, 255), -1)
    cv2.putText(img, "DL01AB1234", (250, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    _, encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/analysis/image",
        files={"file": ("test_vehicle.jpg", img_bytes, "image/jpeg")},
        data={"title": "Integration Test Image"}
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "case_id" in res_data
    assert res_data["status"] == "completed"
    case_id = res_data["case_id"]

    # Test candidate matching for this case
    match_resp = client.post(
        f"/api/matching/cases/{case_id}/run",
        json={"use_investigator_corrections": True}
    )
    assert match_resp.status_code == 200
    match_data = match_resp.json()
    assert "tracks_matched" in match_data

    # Test investigator manual review correction
    track_id = list(match_data["tracks_matched"].keys())[0]
    corr_resp = client.post(
        f"/api/cases/{case_id}/confirm",
        json={
            "track_id": track_id,
            "attribute_name": "make",
            "corrected_value": "Tata",
            "investigator_id": "INV-TEST",
            "reason": "Verified front grille in CCTV frame"
        }
    )
    assert corr_resp.status_code == 200
    assert corr_resp.json()["status"] == "SUCCESS"

    # Test HTML report generation
    report_resp = client.get(f"/api/reports/{case_id}/html")
    assert report_resp.status_code == 200
    assert "VEHICLE TRACE" in report_resp.text
