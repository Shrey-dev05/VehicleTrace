"""
End-to-End Analysis Service — High-Performance Streaming & Keyframe Fusion
Optimized for low-latency CCTV video decoding, adaptive keyframe sampling, and in-memory track state.
"""
import os
import cv2
import uuid
import json
import time
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.plate_rules import is_indian_plate_format
from backend.app.database.session import SessionLocal
from backend.app.database.models import CaseRecord, VehicleObservation, VehicleFusion, VehicleRegistry
from backend.app.attributes.crop_extractor import (
    extract_vehicle_crop, extract_plate_crop, compute_image_quality
)
from backend.app.detection.vehicle_detector import VehicleDetector
from backend.app.detection.plate_detector import PlateDetector
from backend.app.attributes.colour_classifier import VehicleColourClassifier
from backend.app.attributes.make_classifier import VehicleMakeClassifier
from backend.app.attributes.model_classifier import VehicleModelClassifier
from backend.app.attributes.attribute_fusion import AttributeFusionEngine
from backend.app.recognition.ocr_engine import OCREngine
from backend.app.recognition.plate_fusion import PlateFusionEngine
from backend.app.tracking.tracker import SpatialVehicleTracker
from backend.app.services.gemini_vision_service import GeminiVisionService
from backend.app.matching.registry_matcher import RegistryMatcher

log = logging.getLogger(__name__)


@dataclass
class VideoTrackState:
    track_id: str
    best_v_crop: Optional[Any] = None
    best_v_box: Optional[List[int]] = None
    best_v_quality: float = 0.0
    best_v_time: float = 0.0
    best_p_crop: Optional[Any] = None
    best_p_box: Optional[List[int]] = None
    best_p_area: int = 0
    best_p_time: float = 0.0
    last_ocr_frame: int = -999
    settled: bool = False
    vtype: str = "car"
    vtype_conf: float = 0.8
    frame_count: int = 0
    last_p_search_area: int = 0
    last_p_search_frame: int = -999
    plate_reads: List[Dict[str, Any]] = None
    key_frames: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.plate_reads is None:
            self.plate_reads = []
        if self.key_frames is None:
            self.key_frames = []


class AnalysisPipeline:
    def __init__(self):
        log.info("Initializing Analysis Pipeline components...")
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.colour_classifier = VehicleColourClassifier()
        self.make_classifier = VehicleMakeClassifier()
        self.model_classifier = VehicleModelClassifier()
        self.ocr_engine = OCREngine()
        self.attribute_fusion = AttributeFusionEngine()
        self.plate_fusion = PlateFusionEngine()
        self.gemini_vision = GeminiVisionService()
        self.registry_matcher = RegistryMatcher()

    def process_image(self, image_path: str, case_id: str, db: Session) -> Dict[str, Any]:
        """
        Process single image:
        1. Vehicle detection
        2. Vehicle crop extraction
        3. Attribute inference (strictly on full vehicle crop)
        4. Plate detection and OCR (strictly on plate crop)
        5. Persist observations and fusion records
        """
        frame = cv2.imread(image_path)
        if frame is None or frame.size == 0:
            raise ValueError(f"Cannot load image at {image_path}")

        h, w = frame.shape[:2]
        img_quality = compute_image_quality(frame)

        # Check if input is a tight plate-only crop
        is_plate_only = img_quality.is_plate_only

        if is_plate_only:
            ocr_res = self.ocr_engine.read_plate(frame)
            tid = "TRK-001"
            plate_crop_rel = f"crops/{case_id}_{tid}_plate.jpg"
            plate_crop_abs = settings.storage_dir / plate_crop_rel
            cv2.imwrite(str(plate_crop_abs), frame)

            obs = VehicleObservation(
                case_id=case_id,
                track_id=tid,
                frame_index=0,
                frame_timestamp_ms=0.0,
                vehicle_bbox=json.dumps([0, 0, w, h]),
                vehicle_crop_path=plate_crop_rel,
                plate_bbox=json.dumps([0, 0, w, h]),
                plate_crop_path=plate_crop_rel,
                vehicle_type_value="unknown",
                vehicle_type_score=0.0,
                colour_value="unknown",
                colour_score=0.0,
                colour_alternatives=json.dumps([]),
                make_value="Unknown",
                make_score=0.0,
                make_alternatives=json.dumps([]),
                model_value="Unknown",
                model_score=0.0,
                model_candidates=json.dumps([]),
                plate_text=ocr_res.text,
                plate_score=ocr_res.confidence,
                quality_score=img_quality.score,
                model_version="v1.0"
            )
            db.add(obs)

            fusion = VehicleFusion(
                case_id=case_id,
                track_id=tid,
                fused_type="unknown",
                fused_type_score=0.0,
                fused_colour="unknown",
                fused_colour_score=0.0,
                colour_alternatives=json.dumps([]),
                fused_make="Unknown",
                fused_make_score=0.0,
                make_alternatives=json.dumps([]),
                fused_model="Unknown",
                fused_model_score=0.0,
                model_candidates=json.dumps([]),
                fused_plate=ocr_res.text,
                plate_conf=ocr_res.confidence,
                plate_status=ocr_res.status,
                unknown_positions=json.dumps(ocr_res.unknown_positions),
                fusion_status="predicted" if ocr_res.status == "clean" else "low_confidence",
                frames_used=1,
                timestamps_ms=json.dumps([0.0]),
                best_vehicle_crop_path=plate_crop_rel,
                best_plate_crop_path=plate_crop_rel,
                model_versions="v1.0"
            )
            db.add(fusion)
            db.commit()

            return {
                "case_id": case_id,
                "is_plate_only": True,
                "vehicles": [{
                    "track_id": tid,
                    "vehicle": {
                        "type": {"value": "unknown", "score": 0.0, "status": "unknown"},
                        "colour": {"value": "unknown", "score": 0.0, "status": "unknown", "alternatives": []},
                        "make": {"value": "Unknown", "score": 0.0, "status": "unknown", "alternatives": []},
                        "model": {"value": "Unknown", "score": 0.0, "status": "unknown", "candidates": []}
                    },
                    "plate": {
                        "text": ocr_res.text,
                        "confidence": ocr_res.confidence,
                        "status": ocr_res.status,
                        "unknown_positions": ocr_res.unknown_positions
                    },
                    "crops": {
                        "vehicle_crop": f"/storage/{plate_crop_rel}",
                        "plate_crop": f"/storage/{plate_crop_rel}"
                    }
                }]
            }

        # Normal scene image: allow fallback if no vehicle detected in single query image
        v_dets = self.vehicle_detector.detect(frame, allow_fullscreen_fallback=True)
        plate_dets = self.plate_detector.detect(frame)

        results = []
        for idx, vd in enumerate(v_dets):
            tid = f"VEH-{idx+1:02d}"
            v_box = vd["bbox"]
            vtype = vd["vehicle_type"]

            v_crop, _ = extract_vehicle_crop(frame, v_box)
            if v_crop is None:
                continue

            v_rel_path = f"crops/{case_id}_{tid}_v.jpg"
            v_abs_path = settings.storage_dir / v_rel_path
            cv2.imwrite(str(v_abs_path), v_crop)

            # Predict Body Attributes strictly from vehicle crop
            colour_res = self.colour_classifier.predict(v_crop)
            make_res = self.make_classifier.predict(v_crop, vehicle_type=vtype)
            model_res = self.model_classifier.predict(v_crop, make=make_res["value"], vehicle_type=vtype)

            # Associate plate
            associated_plate = None
            for pd in plate_dets:
                if (v_box[0] <= pd["bbox"][0] <= v_box[2]) and (v_box[1] <= pd["bbox"][1] <= v_box[3]):
                    associated_plate = pd
                    break

            if associated_plate is None:
                local_plates = self.plate_detector.detect(v_crop, conf_override=0.18)
                if local_plates:
                    lp = local_plates[0]
                    associated_plate = {
                        "bbox": [
                            v_box[0] + lp["bbox"][0],
                            v_box[1] + lp["bbox"][1],
                            v_box[0] + lp["bbox"][2],
                            v_box[1] + lp["bbox"][3]
                        ],
                        "confidence": lp["confidence"]
                    }

            plate_text = ""
            plate_conf = 0.0
            plate_status = "unknown"
            unknown_positions = []
            p_rel_path = None
            p_crop = None

            if associated_plate is not None:
                p_crop, _ = extract_plate_crop(frame, associated_plate["bbox"])
                if p_crop is not None and p_crop.size > 0:
                    p_rel_path = f"crops/{case_id}_{tid}_p.jpg"
                    p_abs_path = settings.storage_dir / p_rel_path
                    cv2.imwrite(str(p_abs_path), p_crop)

            # High-accuracy Gemini Multimodal Vision analysis if enabled
            if settings.use_gemini_vision:
                try:
                    g_res = self.gemini_vision.analyze_vehicle_and_plate(v_crop, p_crop)
                    if g_res.get("success"):
                        # If Gemini extracted plate
                        if g_res.get("plate_text") and g_res.get("plate_confidence", 0.0) >= 0.70:
                            plate_text = g_res["plate_text"]
                            plate_conf = g_res["plate_confidence"]
                            plate_status = "clean"
                        # Upgrade Color, Make, Model, Type with Gemini Multimodal intelligence
                        if g_res.get("colour") and g_res["colour"] not in ("unknown", "other", ""):
                            colour_res["value"] = g_res["colour"]
                            colour_res["score"] = max(colour_res["score"], g_res["colour_confidence"])
                            colour_res["status"] = "predicted"
                        if g_res.get("make") and g_res["make"] != "Unknown":
                            make_res["value"] = g_res["make"]
                            make_res["score"] = max(make_res["score"], g_res["make_confidence"])
                            make_res["status"] = "predicted"
                        if g_res.get("model") and g_res["model"] != "Unknown":
                            model_res["value"] = g_res["model"]
                            model_res["score"] = max(model_res["score"], g_res["model_confidence"])
                            model_res["status"] = "predicted"
                        if g_res.get("vehicle_type") and g_res["vehicle_type"] not in ("unknown", "other", ""):
                            vtype = g_res["vehicle_type"]
                except Exception as ex:
                    log.warning(f"Gemini analysis error on image: {ex}")

            # Local TrOCR fallback if plate not recognized or Gemini disabled
            if p_crop is not None and p_crop.size > 0 and (not plate_text or plate_status == "unknown"):
                ocr_res = self.ocr_engine.read_plate(p_crop)
                if ocr_res.text:
                    plate_text = ocr_res.text
                    plate_conf = ocr_res.confidence
                    plate_status = ocr_res.status
                    unknown_positions = ocr_res.unknown_positions

            # Strictly validate Indian vehicle registration format
            if plate_text:
                if is_indian_plate_format(plate_text):
                    if "?" in plate_text:
                        plate_status = "partial"
                        unknown_positions = [i for i, c in enumerate(plate_text) if c == "?"]
                    else:
                        plate_status = "clean"
                else:
                    log.info(f"Rejecting plate '{plate_text}' in image analysis: not valid Indian registration format.")
                    plate_text = ""
                    plate_conf = 0.0
                    plate_status = "unreadable"
                    unknown_positions = []

            # If make/model still unknown but plate is known, query registry
            if (make_res["value"] in ("Unknown", "Other", "") or model_res["value"] in ("Unknown", "Other", "")) and plate_text:
                clean_p = plate_text.replace(" ", "").replace("-", "")
                if "?" not in clean_p:
                    reg_exact = db.query(VehicleRegistry).filter(VehicleRegistry.registration_number == clean_p).first()
                    if reg_exact:
                        make_res["value"] = reg_exact.make
                        make_res["score"] = 0.95
                        make_res["status"] = "predicted"
                        model_res["value"] = reg_exact.model
                        model_res["score"] = 0.95
                        model_res["status"] = "predicted"
                        if vtype in ("car", "unknown"):
                            vtype = reg_exact.vehicle_type
                else:
                    reg_candidates = self.registry_matcher.find_close_matches(
                        db=db,
                        observed_plate=plate_text,
                        observed_colour=colour_res["value"],
                        limit=1
                    )
                    if reg_candidates and reg_candidates[0]["ranking_score"] >= 0.85:
                        top_c = reg_candidates[0]
                        if make_res["value"] in ("Unknown", "Other", ""):
                            make_res["value"] = top_c["manufacturer"]
                            make_res["score"] = round(top_c["ranking_score"], 2)
                            make_res["status"] = "predicted"
                        if model_res["value"] in ("Unknown", "Other", ""):
                            model_res["value"] = top_c["vehicle_model"]
                            model_res["score"] = round(top_c["ranking_score"], 2)
                            model_res["status"] = "predicted"
                        if vtype in ("car", "unknown"):
                            vtype = top_c["category_type"]

            f_status = "predicted" if (plate_status in ("clean", "partial") or colour_res["status"] == "predicted") else "low_confidence"

            obs = VehicleObservation(
                case_id=case_id,
                track_id=tid,
                frame_index=0,
                frame_timestamp_ms=0.0,
                vehicle_bbox=json.dumps(v_box),
                vehicle_crop_path=v_rel_path,
                plate_bbox=json.dumps(associated_plate["bbox"]) if associated_plate else None,
                plate_crop_path=p_rel_path,
                vehicle_type_value=vtype,
                vehicle_type_score=vd["confidence"],
                colour_value=colour_res["value"],
                colour_score=colour_res["score"],
                colour_alternatives=json.dumps(colour_res.get("alternatives", [])),
                make_value=make_res["value"],
                make_score=make_res["score"],
                make_alternatives=json.dumps(make_res.get("alternatives", [])),
                model_value=model_res["value"],
                model_score=model_res["score"],
                model_candidates=json.dumps(model_res.get("candidates", [])),
                plate_text=plate_text,
                plate_score=plate_conf,
                quality_score=img_quality.score,
                model_version="v1.0"
            )
            db.add(obs)

            fusion = VehicleFusion(
                case_id=case_id,
                track_id=tid,
                fused_type=vtype,
                fused_type_score=vd["confidence"],
                fused_colour=colour_res["value"],
                fused_colour_score=colour_res["score"],
                colour_alternatives=json.dumps(colour_res.get("alternatives", [])),
                fused_make=make_res["value"],
                fused_make_score=make_res["score"],
                make_alternatives=json.dumps(make_res.get("alternatives", [])),
                fused_model=model_res["value"],
                fused_model_score=model_res["score"],
                model_candidates=json.dumps(model_res.get("candidates", [])),
                fused_plate=plate_text,
                plate_conf=plate_conf,
                plate_status=plate_status,
                unknown_positions=json.dumps(unknown_positions),
                fusion_status=f_status,
                frames_used=1,
                timestamps_ms=json.dumps([0.0]),
                best_vehicle_crop_path=v_rel_path,
                best_plate_crop_path=p_rel_path,
                model_versions="v1.0"
            )
            db.add(fusion)

            has_complete_plate = bool(
                plate_text
                and plate_text not in ("NOT_READABLE", "UNKNOWN")
                and "?" not in plate_text
                and len(plate_text.replace(" ", "")) >= 4
                and plate_status not in ("unreadable", "unknown")
                and p_rel_path
            )

            close_matches = []
            if plate_text and "?" in plate_text:
                close_matches = self.registry_matcher.find_close_matches(
                    db=db,
                    observed_plate=plate_text,
                    observed_colour=colour_res["value"],
                    observed_make=make_res["value"],
                    observed_model=model_res["value"],
                    observed_type=vtype,
                    limit=5
                )

            results.append({
                "track_id": tid,
                "bbox": v_box,
                "has_complete_plate": has_complete_plate,
                "close_matches": close_matches,
                "vehicle": {
                    "type": {"value": vtype, "score": vd["confidence"], "status": "predicted"},
                    "colour": colour_res,
                    "make": make_res,
                    "model": model_res
                },
                "plate": {
                    "text": plate_text,
                    "confidence": plate_conf,
                    "status": plate_status,
                    "unknown_positions": unknown_positions
                },
                "crops": {
                    "vehicle_crop": f"/storage/{v_rel_path}",
                    "plate_crop": f"/storage/{p_rel_path}" if p_rel_path else None
                }
            })

        db.commit()
        return {
            "case_id": case_id,
            "is_plate_only": False,
            "vehicles": results
        }

    def process_video_async(self, video_path: str, case_id: str):
        """
        High-Performance Video Pipeline:
        - Downsampled video decoding at ~2.0 FPS (sample_stride = fps / 2).
        - In-memory tracking & best-crop candidate retention (no disk spam during frame iteration).
        - Single-pass attribute extraction & OCR on the single best crop per tracked vehicle.
        - Persist only best vehicle & plate crop to disk.
        """
        db = SessionLocal()
        case = db.query(CaseRecord).filter(CaseRecord.id == case_id).first()
        if not case:
            db.close()
            return

        case.status = "processing"
        case.progress = 5.0
        db.commit()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            duration_sec = total_frames / fps
            case.fps = fps
            case.total_frames = total_frames
            case.duration_sec = duration_sec
            db.commit()

            tracker = SpatialVehicleTracker(iou_threshold=0.20, max_lost_frames=15)
            # Track profile store: track_id -> VideoTrackState
            tracks: Dict[str, VideoTrackState] = {}

            # Adaptive sampling stride: ~1.5 frames per second (high temporal resolution without CPU bottleneck)
            sample_stride = max(1, int(round(fps / 1.5)))

            frame_idx = 0
            while True:
                if frame_idx % sample_stride == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    t_ms = (frame_idx / fps) * 1000.0

                    # Vehicle detection strictly on real vehicles (no fullscreen fallback)
                    v_dets = self.vehicle_detector.detect(frame, allow_fullscreen_fallback=False)
                    v_boxes = [d["bbox"] for d in v_dets]
                    det_map = {tuple(d["bbox"]): d for d in v_dets}

                    tracked_vehicles = tracker.update(v_boxes)

                    for tid, v_box in tracked_vehicles:
                        vw = v_box[2] - v_box[0]
                        vh = v_box[3] - v_box[1]
                        v_area = vw * vh
                        # Filter out distant background noise (< 75x35)
                        if vw < 75 or vh < 35:
                            continue

                        if tid not in tracks:
                            v_info = det_map.get(tuple(v_box), {})
                            tracks[tid] = VideoTrackState(
                                track_id=tid,
                                vtype=v_info.get("vehicle_type", "car"),
                                vtype_conf=v_info.get("confidence", 0.8)
                            )
                        ts = tracks[tid]
                        ts.frame_count += 1

                        # Extract vehicle crop
                        v_crop, _ = extract_vehicle_crop(frame, v_box)
                        if v_crop is None or v_crop.size == 0:
                            continue

                        q = compute_image_quality(v_crop)
                        if q.score > ts.best_v_quality:
                            ts.best_v_quality = q.score
                            ts.best_v_crop = v_crop.copy()
                            ts.best_v_box = v_box
                            ts.best_v_time = t_ms

                        # Smart plate search criteria:
                        # 1. Skip if vehicle is too small for a readable plate (< 90x45)
                        # 2. Search if track has no plate crop yet
                        # 3. Re-search if vehicle area grew by >= 15% (getting closer & sharper)
                        # 4. Or periodically every 4 sampled frames
                        should_search_plate = False
                        if vw >= 90 and vh >= 45:
                            if ts.best_p_crop is None:
                                should_search_plate = True
                            elif v_area > ts.last_p_search_area * 1.15:
                                should_search_plate = True
                            elif (frame_idx - ts.last_p_search_frame) >= (sample_stride * 4):
                                should_search_plate = True

                        if should_search_plate:
                            ts.last_p_search_frame = frame_idx
                            ts.last_p_search_area = max(ts.last_p_search_area, v_area)
                            local_plates = self.plate_detector.detect(v_crop, conf_override=0.18)
                            if local_plates:
                                lp = local_plates[0]
                                pw = lp["bbox"][2] - lp["bbox"][0]
                                ph = lp["bbox"][3] - lp["bbox"][1]
                                p_area = pw * ph

                                p_crop, _ = extract_plate_crop(v_crop, lp["bbox"])
                                if p_crop is not None and p_crop.size > 0:
                                    if p_area > ts.best_p_area:
                                        ts.best_p_area = p_area
                                        ts.best_p_crop = p_crop.copy()
                                        ts.best_p_time = t_ms
                                        ts.best_p_box = [
                                            v_box[0] + lp["bbox"][0],
                                            v_box[1] + lp["bbox"][1],
                                            v_box[0] + lp["bbox"][2],
                                            v_box[1] + lp["bbox"][3]
                                        ]

                    # Update progress every ~8 sampled frames
                    if frame_idx % (sample_stride * 8) == 0:
                        pct = 5.0 + (frame_idx / total_frames) * 75.0
                        case.progress = round(min(80.0, pct), 1)
                        db.commit()

                else:
                    # Fast-forward without CPU decompression for skipped frames
                    ret = cap.grab()
                    if not ret:
                        break

                frame_idx += 1

            cap.release()

            # Finalization Phase: Keyframe Attribute Recognition & Disk Persistence
            case.progress = 85.0
            db.commit()

            # Filter spurious 1-frame transient tracks in multi-frame video
            valid_tracks: List[VideoTrackState] = []
            for tid, ts in tracks.items():
                if ts.best_v_crop is None:
                    continue
                if total_frames > 15 and ts.frame_count < 2 and ts.best_p_crop is None and ts.best_v_quality < 35.0:
                    continue
                valid_tracks.append(ts)

            if not valid_tracks:
                valid_tracks = [ts for ts in tracks.values() if ts.best_v_crop is not None]

            # Rate-controlled Gemini Multimodal Vision API calls across valid tracks
            gemini_results = {}
            if settings.use_gemini_vision and valid_tracks:
                for idx_ts, ts in enumerate(valid_tracks):
                    try:
                        if idx_ts > 0:
                            time.sleep(0.15)
                        res = self.gemini_vision.analyze_vehicle_and_plate(ts.best_v_crop, ts.best_p_crop)
                        gemini_results[ts.track_id] = res
                    except Exception as ex:
                        log.warning(f"Gemini analysis error on video track {ts.track_id}: {ex}")
                        gemini_results[ts.track_id] = {"success": False, "error": str(ex)}

            for ts in valid_tracks:
                tid = ts.track_id

                # Run Colour, Make, Model on the track's single best vehicle crop
                c_res = self.colour_classifier.predict(ts.best_v_crop)
                m_res = self.make_classifier.predict(ts.best_v_crop, vehicle_type=ts.vtype)
                mod_res = self.model_classifier.predict(ts.best_v_crop, make=m_res["value"], vehicle_type=ts.vtype)

                g_res = gemini_results.get(tid, {})

                plate_text = ""
                plate_conf = 0.0
                plate_status = "unreadable"
                unknown_positions = []

                # If Gemini Vision successfully extracted plate with high confidence, use directly (saves 1.2s CPU TrOCR)
                if g_res.get("success") and g_res.get("plate_text") and g_res.get("plate_confidence", 0.0) >= 0.70:
                    plate_text = g_res["plate_text"]
                    plate_conf = g_res["plate_confidence"]
                    plate_status = "clean"
                elif ts.best_p_crop is not None and ts.best_p_crop.size > 0:
                    # Fallback to local TrOCR
                    ocr_res = self.ocr_engine.read_plate(ts.best_p_crop)
                    plate_text = ocr_res.text
                    plate_conf = ocr_res.confidence
                    plate_status = ocr_res.status
                    unknown_positions = ocr_res.unknown_positions

                # Strictly validate Indian vehicle registration format
                if plate_text:
                    if is_indian_plate_format(plate_text):
                        if "?" in plate_text:
                            plate_status = "partial"
                            unknown_positions = [i for i, c in enumerate(plate_text) if c == "?"]
                        else:
                            plate_status = "clean"
                    else:
                        log.info(f"Rejecting plate '{plate_text}' for track {tid}: does not conform to Indian format.")
                        plate_text = ""
                        plate_conf = 0.0
                        plate_status = "unreadable"
                        unknown_positions = []

                # Apply Gemini multimodal enhancements for make, model, colour, and vehicle type
                if g_res.get("success"):
                    if g_res.get("colour") and g_res["colour"] not in ("unknown", "other", ""):
                        c_res["value"] = g_res["colour"]
                        c_res["score"] = max(c_res["score"], g_res["colour_confidence"])
                        c_res["status"] = "predicted"
                    if g_res.get("make") and g_res["make"] != "Unknown":
                        m_res["value"] = g_res["make"]
                        m_res["score"] = max(m_res["score"], g_res["make_confidence"])
                        m_res["status"] = "predicted"
                    if g_res.get("model") and g_res["model"] != "Unknown":
                        mod_res["value"] = g_res["model"]
                        mod_res["score"] = max(mod_res["score"], g_res["model_confidence"])
                        mod_res["status"] = "predicted"
                    if g_res.get("vehicle_type") and g_res["vehicle_type"] not in ("unknown", "other", ""):
                        ts.vtype = g_res["vehicle_type"]

                # If make/model still unknown but plate is known, query registry
                if (m_res["value"] in ("Unknown", "Other", "") or mod_res["value"] in ("Unknown", "Other", "")) and plate_text:
                    clean_p = plate_text.replace(" ", "").replace("-", "")
                    if "?" not in clean_p:
                        reg_exact = db.query(VehicleRegistry).filter(VehicleRegistry.registration_number == clean_p).first()
                        if reg_exact:
                            m_res["value"] = reg_exact.make
                            m_res["score"] = 0.95
                            m_res["status"] = "predicted"
                            mod_res["value"] = reg_exact.model
                            mod_res["score"] = 0.95
                            mod_res["status"] = "predicted"
                            if ts.vtype in ("car", "unknown"):
                                ts.vtype = reg_exact.vehicle_type
                    else:
                        reg_candidates = self.registry_matcher.find_close_matches(
                            db=db,
                            observed_plate=plate_text,
                            observed_colour=c_res["value"],
                            limit=1
                        )
                        if reg_candidates and reg_candidates[0]["ranking_score"] >= 0.85:
                            top_c = reg_candidates[0]
                            if m_res["value"] in ("Unknown", "Other", ""):
                                m_res["value"] = top_c["manufacturer"]
                                m_res["score"] = round(top_c["ranking_score"], 2)
                                m_res["status"] = "predicted"
                            if mod_res["value"] in ("Unknown", "Other", ""):
                                mod_res["value"] = top_c["vehicle_model"]
                                mod_res["score"] = round(top_c["ranking_score"], 2)
                                mod_res["status"] = "predicted"
                            if ts.vtype in ("car", "unknown"):
                                ts.vtype = top_c["category_type"]

                # Save ONLY best representative vehicle and plate crops to disk
                v_rel = f"crops/{case_id}_{tid}_best_v.jpg"
                v_abs = settings.storage_dir / v_rel
                cv2.imwrite(str(v_abs), ts.best_v_crop)

                p_rel = None
                if ts.best_p_crop is not None:
                    p_rel = f"crops/{case_id}_{tid}_best_p.jpg"
                    p_abs = settings.storage_dir / p_rel
                    cv2.imwrite(str(p_abs), ts.best_p_crop)

                # Persist observation record
                obs_rec = VehicleObservation(
                    case_id=case_id,
                    track_id=tid,
                    frame_index=0,
                    frame_timestamp_ms=ts.best_v_time,
                    vehicle_bbox=json.dumps(ts.best_v_box or []),
                    vehicle_crop_path=v_rel,
                    plate_bbox=json.dumps(ts.best_p_box or []) if ts.best_p_box else None,
                    plate_crop_path=p_rel,
                    vehicle_type_value=ts.vtype,
                    vehicle_type_score=ts.vtype_conf,
                    colour_value=c_res["value"],
                    colour_score=c_res["score"],
                    colour_alternatives=json.dumps(c_res.get("alternatives", [])),
                    make_value=m_res["value"],
                    make_score=m_res["score"],
                    make_alternatives=json.dumps(m_res.get("alternatives", [])),
                    model_value=mod_res["value"],
                    model_score=mod_res["score"],
                    model_candidates=json.dumps(mod_res.get("candidates", [])),
                    plate_text=plate_text,
                    plate_score=plate_conf,
                    quality_score=ts.best_v_quality,
                    model_version="v1.0"
                )
                db.add(obs_rec)

                # Fusion record
                f_status = "predicted" if (plate_status == "clean" or c_res["status"] == "predicted") else "low_confidence"
                fusion = VehicleFusion(
                    case_id=case_id,
                    track_id=tid,
                    fused_type=ts.vtype,
                    fused_type_score=ts.vtype_conf,
                    fused_colour=c_res["value"],
                    fused_colour_score=c_res["score"],
                    colour_alternatives=json.dumps(c_res.get("alternatives", [])),
                    fused_make=m_res["value"],
                    fused_make_score=m_res["score"],
                    make_alternatives=json.dumps(m_res.get("alternatives", [])),
                    fused_model=mod_res["value"],
                    fused_model_score=mod_res["score"],
                    model_candidates=json.dumps(mod_res.get("candidates", [])),
                    fused_plate=plate_text,
                    plate_conf=plate_conf,
                    plate_status=plate_status,
                    unknown_positions=json.dumps(unknown_positions),
                    fusion_status=f_status,
                    frames_used=1 if plate_text else 0,
                    timestamps_ms=json.dumps([ts.best_v_time]),
                    best_vehicle_crop_path=v_rel,
                    best_plate_crop_path=p_rel,
                    model_versions="v1.0"
                )
                db.add(fusion)

            case.status = "completed"
            case.progress = 100.0
            db.commit()
            log.info(f"Video job for case {case_id} completed successfully.")

        except Exception as e:
            log.error(f"Video processing failed for case {case_id}: {e}", exc_info=True)
            case.status = "failed"
            case.error_message = str(e)
            db.commit()
        finally:
            db.close()


# Global pipeline singleton
pipeline_instance = AnalysisPipeline()
