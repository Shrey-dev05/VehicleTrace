"""
Analysis API Endpoints
"""
import os
import uuid
import shutil
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.plate_rules import is_indian_plate_format
from backend.app.database.session import get_db, SessionLocal
from backend.app.database.models import CaseRecord, VehicleObservation, VehicleFusion
from backend.app.services.analysis_service import pipeline_instance
from backend.app.matching.registry_matcher import RegistryMatcher

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
registry_matcher = RegistryMatcher()


@router.post("/image")
async def analyze_image(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Synchronous analysis of a single vehicle image or plate crop.
    Preserves two separate paths: vehicle crop for colour/make/model and plate crop for OCR.
    """
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    safe_filename = f"{case_id}{ext}"
    dest_path = settings.upload_dir / safe_filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    case_title = title or f"Image Analysis {case_id}"
    case = CaseRecord(
        id=case_id,
        title=case_title,
        source_type="image",
        source_path=str(dest_path),
        status="processing",
        progress=20.0
    )
    db.add(case)
    db.commit()

    try:
        results = pipeline_instance.process_image(str(dest_path), case_id, db)
        case.status = "completed"
        case.progress = 100.0
        db.commit()
        return {
            "case_id": case_id,
            "title": case_title,
            "status": "completed",
            "results": results
        }
    except Exception as e:
        case.status = "failed"
        case.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")


@router.post("/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Asynchronous video analysis job endpoint. Returns job_id immediately.
    """
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    safe_filename = f"{case_id}{ext}"
    dest_path = settings.upload_dir / safe_filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    case_title = title or f"CCTV Video Analysis {case_id}"
    case = CaseRecord(
        id=case_id,
        title=case_title,
        source_type="video",
        source_path=str(dest_path),
        status="pending",
        progress=0.0
    )
    db.add(case)
    db.commit()

    # Launch asynchronous background task
    background_tasks.add_task(pipeline_instance.process_video_async, str(dest_path), case_id)

    return {
        "job_id": case_id,
        "case_id": case_id,
        "title": case_title,
        "status": "pending",
        "message": "Video analysis queued successfully. Poll status with /api/analysis/jobs/{job_id}"
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    case = db.query(CaseRecord).filter(CaseRecord.id == job_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Job/Case not found")

    return {
        "job_id": case.id,
        "status": case.status,
        "progress": case.progress,
        "error_message": case.error_message,
        "fps": case.fps,
        "total_frames": case.total_frames,
        "duration_sec": case.duration_sec,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None
    }


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str, db: Session = Depends(get_db)):
    case = db.query(CaseRecord).filter(CaseRecord.id == job_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Job/Case not found")

    fusions = db.query(VehicleFusion).filter(VehicleFusion.case_id == job_id).all()
    results = []
    for f in fusions:
        import json
        plate_str = (f.fused_plate or "").strip()
        is_indian = is_indian_plate_format(plate_str)
        has_valid_plate = bool(
            plate_str
            and is_indian
            and f.plate_status not in ("unreadable", "unknown")
            and f.best_plate_crop_path
        )
        close_matches = []
        if plate_str and "?" in plate_str:
            close_matches = registry_matcher.find_close_matches(
                db=db,
                observed_plate=plate_str,
                observed_colour=f.fused_colour or "unknown",
                observed_make=f.fused_make or "Unknown",
                observed_model=f.fused_model or "Unknown",
                observed_type=f.fused_type or "car",
                limit=5
            )

        results.append({
            "track_id": f.track_id,
            "has_complete_plate": has_valid_plate,
            "is_indian_format": is_indian,
            "close_matches": close_matches,
            "vehicle": {
                "vehicle_type": {
                    "value": f.fused_type,
                    "score": f.fused_type_score,
                    "status": "predicted"
                },
                "colour": {
                    "value": f.fused_colour,
                    "score": f.fused_colour_score,
                    "status": f.fusion_status,
                    "alternatives": json.loads(f.colour_alternatives or "[]")
                },
                "make": {
                    "value": f.fused_make,
                    "score": f.fused_make_score,
                    "status": f.fusion_status,
                    "alternatives": json.loads(f.make_alternatives or "[]")
                },
                "model": {
                    "value": f.fused_model,
                    "score": f.fused_model_score,
                    "status": f.fusion_status,
                    "candidates": json.loads(f.model_candidates or "[]")
                }
            },
            "plate": {
                "text": f.fused_plate,
                "confidence": f.plate_conf,
                "status": f.plate_status,
                "unknown_positions": json.loads(f.unknown_positions or "[]")
            },
            "evidence": {
                "fusion_status": f.fusion_status,
                "frames_used": f.frames_used,
                "timestamps_ms": json.loads(f.timestamps_ms or "[]"),
                "vehicle_crop_path": f"/storage/{f.best_vehicle_crop_path}" if f.best_vehicle_crop_path else None,
                "plate_crop_path": f"/storage/{f.best_plate_crop_path}" if f.best_plate_crop_path else None
            }
        })

    return {
        "job_id": job_id,
        "status": case.status,
        "vehicles": results
    }


@router.get("/jobs/{job_id}/frames")
async def get_job_frame_evidence(job_id: str, db: Session = Depends(get_db)):
    obs_list = db.query(VehicleObservation).filter(VehicleObservation.case_id == job_id).all()
    frames = []
    import json
    for o in obs_list:
        frames.append({
            "id": o.id,
            "track_id": o.track_id,
            "frame_index": o.frame_index,
            "timestamp_ms": o.frame_timestamp_ms,
            "vehicle_bbox": json.loads(o.vehicle_bbox or "[]"),
            "plate_bbox": json.loads(o.plate_bbox or "[]") if o.plate_bbox else None,
            "vehicle_crop": f"/storage/{o.vehicle_crop_path}" if o.vehicle_crop_path else None,
            "plate_crop": f"/storage/{o.plate_crop_path}" if o.plate_crop_path else None,
            "plate_text": o.plate_text,
            "plate_score": o.plate_score,
            "colour_value": o.colour_value,
            "colour_score": o.colour_score,
            "make_value": o.make_value,
            "make_score": o.make_score,
            "quality_score": o.quality_score
        })
    return {"job_id": job_id, "frames": frames}
