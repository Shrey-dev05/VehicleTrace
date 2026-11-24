"""
Case Management and Manual Review API Endpoints
Enables investigator attribute correction without overwriting original AI model predictions.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import CaseRecord, VehicleFusion, InvestigatorCorrection, VehicleObservation

router = APIRouter(prefix="/api/cases", tags=["cases"])


class AttributeCorrectionRequest(BaseModel):
    track_id: str
    attribute_name: str  # colour, vehicle_type, make, model, plate
    corrected_value: str
    investigator_id: Optional[str] = "INV-001"
    reason: Optional[str] = "Visual inspection by investigator"


@router.get("")
async def list_cases(db: Session = Depends(get_db)):
    cases = db.query(CaseRecord).order_by(CaseRecord.created_at.desc()).all()
    return [{
        "id": c.id,
        "title": c.title,
        "source_type": c.source_type,
        "status": c.status,
        "progress": c.progress,
        "created_at": c.created_at.isoformat() if c.created_at else None
    } for c in cases]


@router.get("/{case_id}")
async def get_case_details(case_id: str, db: Session = Depends(get_db)):
    case = db.query(CaseRecord).filter(CaseRecord.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    fusions = db.query(VehicleFusion).filter(VehicleFusion.case_id == case_id).all()
    corrections = db.query(InvestigatorCorrection).filter(InvestigatorCorrection.case_id == case_id).all()

    corr_list = [{
        "id": c.id,
        "track_id": c.track_id,
        "attribute_name": c.attribute_name,
        "original_prediction": c.original_prediction,
        "corrected_value": c.corrected_value,
        "investigator_id": c.investigator_id,
        "reason": c.reason,
        "created_at": c.created_at.isoformat() if c.created_at else None
    } for c in corrections]

    return {
        "case": {
            "id": case.id,
            "title": case.title,
            "source_type": case.source_type,
            "status": case.status,
            "progress": case.progress,
            "duration_sec": case.duration_sec,
            "created_at": case.created_at.isoformat() if case.created_at else None
        },
        "vehicles_count": len(fusions),
        "corrections": corr_list
    }


@router.post("/{case_id}/confirm")
async def confirm_investigator_correction(
    case_id: str,
    req: AttributeCorrectionRequest,
    db: Session = Depends(get_db)
):
    """
    Save investigator-confirmed attribute/plate edit separately without overwriting original AI prediction.
    """
    case = db.query(CaseRecord).filter(CaseRecord.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    fusion = db.query(VehicleFusion).filter(
        VehicleFusion.case_id == case_id,
        VehicleFusion.track_id == req.track_id
    ).first()

    if not fusion:
        raise HTTPException(status_code=404, detail=f"Track {req.track_id} not found in case {case_id}")

    # Determine original prediction value
    orig_val = ""
    if req.attribute_name == "colour":
        orig_val = fusion.fused_colour
    elif req.attribute_name == "make":
        orig_val = fusion.fused_make
    elif req.attribute_name == "model":
        orig_val = fusion.fused_model
    elif req.attribute_name == "vehicle_type":
        orig_val = fusion.fused_type
    elif req.attribute_name == "plate":
        orig_val = fusion.fused_plate

    # Record correction in immutable audit table
    corr = InvestigatorCorrection(
        case_id=case_id,
        track_id=req.track_id,
        attribute_name=req.attribute_name,
        original_prediction=orig_val,
        corrected_value=req.corrected_value,
        investigator_id=req.investigator_id,
        reason=req.reason
    )
    db.add(corr)
    fusion.fusion_status = "human_confirmed"
    db.commit()

    return {
        "status": "SUCCESS",
        "message": f"Attribute '{req.attribute_name}' correction saved for track {req.track_id}.",
        "correction": {
            "track_id": req.track_id,
            "attribute_name": req.attribute_name,
            "original_prediction": orig_val,
            "corrected_value": req.corrected_value,
            "investigator_id": req.investigator_id
        }
    }
