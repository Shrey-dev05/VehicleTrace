# VEHICLE TRACE — CCTV ANPR & Vehicle Attribute Recognition

An AI-assisted surveillance vehicle identification and traceability platform.

VEHICLE TRACE processes CCTV camera images and video feeds to detect vehicles, read number plates via OCR, and identify vehicle body colour, manufacturer (make), model, and vehicle category.

---

## Key Features

1. **Dual-Path Architecture**: Strictly isolates the vehicle body crop (for body colour, make, model) from the number plate crop (for OCR).
2. **Indian License Plate Normalisation**: Supports standard Indian plates (`DL01AB1234`), 2-line stacked plates, and Bharat (BH) series (`22BH1234AA`).
3. **Multi-Frame Temporal Evidence Fusion**: Combines multi-frame observations per tracked vehicle using quality-weighted probability voting.
4. **Registry Candidate Ranking**: Compares observed evidence against vehicle registry databases with wildcard preservation and explainable scoring.
5. **Investigator Audit Trail**: Allows investigators to correct or confirm attributes without overwriting original AI model predictions.
6. **Web Dashboard & Reports**: Modern web UI with live CCTV video progress, evidence frame inspection, and forensic HTML report generation.

---

## Directory Structure

```text
CORE_V1/
├── frontend/
│   ├── index.html                  # Investigator Web Portal
│   └── static/
│       ├── css/styles.css          # UI styles
│       └── js/app.js               # Frontend application controller
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application assembly
│   │   ├── api/                    # REST endpoints (analysis, cases, matching)
│   │   ├── core/                   # Configuration, device management, plate rules
│   │   ├── database/               # SQLAlchemy models and session initialization
│   │   ├── detection/              # Vehicle and plate detectors
│   │   ├── attributes/             # Colour, make, model classifiers & fusion
│   │   ├── recognition/            # Plate OCR engine & temporal plate fusion
│   │   ├── tracking/               # Spatial vehicle tracker & box association
│   │   ├── matching/               # Vehicle registry candidate ranking engine
│   │   └── reports/                # Forensic HTML case report generator
│   └── tests/                      # Pytest unit and integration test suite
├── training/
│   ├── configs/                    # Canonical class taxonomies (JSON)
│   ├── prepare_crops.py            # Crop preparation from raw annotations
│   ├── validate_annotations.py     # Split disjointness and class support check
│   ├── train_colour.py             # MobileNetV3 colour training script
│   ├── train_make.py               # Make classifier training script
│   ├── train_model.py              # Hierarchical model classifier training script
│   └── evaluate_attributes.py      # Evaluation and abstention reporting
├── storage/                        # SQLite DB, uploads, crops, outputs
├── docs/                           # Architecture documentation
└── README.md
```

---

## Getting Started

### 1. Run Unit & Integration Tests
```bash
python -m pytest backend/tests -v
```

### 2. Start the Application Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open your browser at `http://localhost:8000/` to access the investigator portal.
API documentation is available at `http://localhost:8000/docs`.

### 3. Training Custom Models
```bash
# 1. Prepare vehicle crops from metadata
python training/prepare_crops.py --input-csv path/to/annotations.csv --images-dir path/to/images

# 2. Validate annotation consistency
python training/validate_annotations.py --meta-csv training/datasets/vehicle_crops.csv

# 3. Train Colour Classifier
python training/train_colour.py --epochs 15

# 4. Train Make Classifier
python training/train_make.py --epochs 20

# 5. Evaluate on challenge set
python training/evaluate_attributes.py
```

---

## API Summary

- `POST /api/analysis/image`: Single image vehicle detection and attribute analysis.
- `POST /api/analysis/video`: Asynchronous CCTV video processing job.
- `GET /api/analysis/jobs/{job_id}`: Poll video analysis status and progress.
- `GET /api/analysis/jobs/{job_id}/results`: Retrieve final fused vehicle records.
- `GET /api/analysis/jobs/{job_id}/frames`: Retrieve frame-level evidence observations.
- `POST /api/matching/cases/{case_id}/run`: Run registry candidate ranking.
- `POST /api/cases/{case_id}/confirm`: Record investigator attribute correction.
- `GET /api/reports/{case_id}/html`: Generate printable forensic evidence report.
