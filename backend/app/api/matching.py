"""
Registry Matching API Endpoints
Matches observed/confirmed vehicle attributes to authorized registry database.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import CaseRecord, VehicleFusion, InvestigatorCorrection, VehicleRegistry
from backend.app.matching.registry_matcher import RegistryMatcher

router = APIRouter(prefix="/api/matching", tags=["matching"])
matcher = RegistryMatcher(acceptance_threshold=0.45)


class MatchRequest(BaseModel):
    track_id: Optional[str] = None
    use_investigator_corrections: bool = True


@router.post("/cases/{case_id}/run")
async def run_registry_matching(
    case_id: str,
    req: MatchRequest,
    db: Session = Depends(get_db)
):
    """
    Run candidate ranking against vehicle registry for a given track in a case.
    """
    case = db.query(CaseRecord).filter(CaseRecord.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    query = db.query(VehicleFusion).filter(VehicleFusion.case_id == case_id)
    if req.track_id:
        query = query.filter(VehicleFusion.track_id == req.track_id)
    fusions = query.all()

    if not fusions:
        raise HTTPException(status_code=404, detail="No vehicle records found for this case")

    results_by_track = {}
    for f in fusions:
        plate = f.fused_plate
        colour = f.fused_colour
        make = f.fused_make
        model = f.fused_model
        vtype = f.fused_type

        # Override with investigator corrections if requested
        if req.use_investigator_corrections:
            corrs = db.query(InvestigatorCorrection).filter(
                InvestigatorCorrection.case_id == case_id,
                InvestigatorCorrection.track_id == f.track_id
            ).all()
            for c in corrs:
                if c.attribute_name == "plate":
                    plate = c.corrected_value
                elif c.attribute_name == "colour":
                    colour = c.corrected_value
                elif c.attribute_name == "make":
                    make = c.corrected_value
                elif c.attribute_name == "model":
                    model = c.corrected_value
                elif c.attribute_name == "vehicle_type":
                    vtype = c.corrected_value

        match_output = matcher.match_candidates(
            db=db,
            observed_plate=plate,
            observed_colour=colour,
            observed_make=make,
            observed_model=model,
            observed_type=vtype,
            limit=5
        )

        results_by_track[f.track_id] = {
            "query_attributes": {
                "plate": plate,
                "colour": colour,
                "make": make,
                "model": model,
                "vehicle_type": vtype,
                "has_investigator_overrides": req.use_investigator_corrections
            },
            "ranking_result": match_output
        }

    return {
        "case_id": case_id,
        "tracks_matched": results_by_track
    }


class DirectMatchQuery(BaseModel):
    plate: Optional[str] = ""
    colour: Optional[str] = "unknown"
    make: Optional[str] = "Unknown"
    model: Optional[str] = "Unknown"
    vehicle_type: Optional[str] = "car"
    limit: Optional[int] = 5


@router.post("/find-matches")
async def find_registry_matches(
    query: DirectMatchQuery,
    db: Session = Depends(get_db)
):
    """
    Directly query vehicle registry for candidates closely matching an observed plate
    and attributes (especially partial plates with '?').
    """
    matches = matcher.find_close_matches(
        db=db,
        observed_plate=query.plate or "",
        observed_colour=query.colour or "unknown",
        observed_make=query.make or "Unknown",
        observed_model=query.model or "Unknown",
        observed_type=query.vehicle_type or "car",
        limit=query.limit or 5
    )
    return {"matches": matches}


@router.get("/registry")
async def list_registry_vehicles(limit: int = 50, db: Session = Depends(get_db)):
    records = db.query(VehicleRegistry).limit(limit).all()
    return [{
        "registration_number": r.registration_number,
        "make": r.make,
        "model": r.model,
        "colour": r.colour,
        "vehicle_type": r.vehicle_type,
        "year": r.year,
        "state": r.state,
        "rto": r.rto
    } for r in records]
