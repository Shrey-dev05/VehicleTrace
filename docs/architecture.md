# VEHICLE TRACE — Architecture & Pipeline Design

## 1. System Overview

VEHICLE TRACE is an AI-assisted CCTV vehicle identification and traceability platform. The system operates on single surveillance images or multi-frame video footage to extract:
- Number plate registration text via optical character recognition (OCR)
- Dominant vehicle body colour
- Manufacturer / make
- Hierarchical model
- Vehicle category / type

Results are fused across video frames and matched against an authorised vehicle registry to provide investigative candidate ranking with full explainability and immutable human audit trails.

---

## 2. Core Architectural Decision: Dual Image Paths

A primary design requirement is that **the number plate crop is never used to predict the vehicle's body colour, make, or model**. 

Two distinct visual pipelines are maintained:

```text
               Input Image / Video Stream
                           |
                           v
              Vehicle Detector & Tracker
                           |
            +--------------+--------------+
            |                             |
            v                             v
    Full Vehicle Crop             Plate Detector
            |                             |
            v                             v
    Body ROI Isolation                Plate Crop
            |                             |
            v                             v
Attribute Classifiers (Color/Make/Model) Deskew / CLAHE / Upscale
            |                             |
            |                             v
            |                         OCR Engine
            |                             |
            +--------------+--------------+
                           |
                           v
            Frame Observation Record
                           |
                           v
             Multi-Frame Temporal Fusion
                           |
                           v
             Registry Candidate Ranking
                           |
                           v
        Investigator Review & Forensic Report
```

If the input is a close-up crop of a number plate alone (no vehicle silhouette or body panels visible), the pipeline returns plate text and marks colour, make, and model as `unknown`.

---

## 3. Module Breakdown

### 3.1 Detection & Crop Extraction
- **Vehicle Detector (`backend/app/detection/vehicle_detector.py`)**: Pretrained YOLOv8 for spatial vehicle localization (`car`, `motorcycle_scooter`, `bus`, `truck`).
- **Plate Detector (`backend/app/detection/plate_detector.py`)**: License plate detector with rectangular edge contour morphological fallback.
- **Crop Extractor (`backend/app/attributes/crop_extractor.py`)**: Performs boundary expansion with strict frame bounds checking, quality grading (sharpness, contrast, brightness, aspect ratio), and central vehicle body panel extraction to exclude tires, glass, and road reflections.

### 3.2 Vehicle Attribute Classification
- **Colour Classifier (`backend/app/attributes/colour_classifier.py`)**: 11 canonical classes (`white`, `black`, `silver_grey`, `red`, `blue`, `green`, `yellow`, `brown`, `orange`, `other`, `unknown`). MobileNetV3 PyTorch checkpoint with calibrated OpenCV HSV dominant body color analyzer baseline.
- **Make Classifier (`backend/app/attributes/make_classifier.py`)**: 17 canonical Indian market manufacturers (`Maruti_Suzuki`, `Tata`, `Hyundai`, `Mahindra`, `Honda`, `Toyota`, etc.).
- **Model Classifier (`backend/app/attributes/model_classifier.py`)**: Hierarchical model classification constrained by the predicted make. Returns `unknown` or `low_confidence` when visual evidence is insufficient.

### 3.3 License Plate OCR & Syntax Rules
- **Syntax Rules (`backend/app/core/plate_rules.py`)**: Full Indian plate syntax validation supporting Standard series (`DL01AB1234`) and Bharat (BH) series (`22BH1234AA`). Character confusion correction (`O` <-> `0`, `I` <-> `1`, `Z` <-> `2`, `B` <-> `8`).
- **OCR Engine (`backend/app/recognition/ocr_engine.py`)**: TrOCR model runner with contrast enhancement (CLAHE) and contour heuristic reader. Preserves `?` wildcards for partial or obscured characters.

### 3.4 Temporal Tracking & Multi-Frame Fusion
- **Spatial Tracker (`backend/app/tracking/tracker.py`)**: IoU-based vehicle tracker generating local video track IDs (`TRK-001`, `TRK-002`).
- **Attribute Fusion (`backend/app/attributes/attribute_fusion.py`)**: Quality-weighted probability averaging per track, conflict detection, and best crop selection.
- **Plate Fusion (`backend/app/recognition/plate_fusion.py`)**: Character slot voting resolving character ambiguities across frames.

### 3.5 Registry Candidate Matching & Explainability
- **Registry Matcher (`backend/app/matching/registry_matcher.py`)**: Ranks database candidates:
  - Plate compatibility (50% weight, `?` wildcard non-penalizing, mismatch heavily penalized)
  - Make/model compatibility (25% weight)
  - Colour compatibility (15% weight)
  - Vehicle type compatibility (10% weight)
  - Returns `NO_CONFIDENT_MATCH` if no candidate meets the acceptance threshold.

### 3.6 Storage & Investigator Audit
- **Database (`backend/app/database/models.py`)**:
  - `cases`: investigation session metadata.
  - `vehicle_attribute_observations`: frame-level observations.
  - `vehicle_attribute_fusion`: fused track-level consensus.
  - `investigator_attribute_corrections`: human confirmations and audit trail without overwriting raw AI predictions.
  - `vehicle_registry`: sample authorised vehicle registry.
