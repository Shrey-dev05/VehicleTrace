"""
VEHICLE TRACE - FastAPI Backend Application
"""
import os
import logging
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, Depends, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import init_db, get_db
from backend.app.api.analysis import router as analysis_router
from backend.app.api.cases import router as cases_router
from backend.app.api.matching import router as matching_router
from backend.app.reports.report_generator import generate_case_report_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("vehicle_trace")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-assisted CCTV vehicle identification, ANPR and attribute recognition system."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database schema and seed sample registry
@app.on_event("startup")
def on_startup():
    log.info("Initializing database...")
    init_db()
    log.info("Vehicle Trace application ready.")

# Mount storage for evidence images and crops
app.mount("/storage", StaticFiles(directory=str(settings.storage_dir)), name="storage")

# Mount frontend static files
frontend_dir = settings.base_dir / "frontend"
static_dir = frontend_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routers
app.include_router(analysis_router)
app.include_router(cases_router)
app.include_router(matching_router)


@app.get("/api/reports/{case_id}/html", response_class=HTMLResponse)
async def get_report_html(case_id: str, track_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Generate and return forensic HTML investigation report for case or specific vehicle track."""
    return generate_case_report_html(case_id, db, track_id=track_id)


@app.get("/api/reports/{case_id}/tracks/{track_id}/html", response_class=HTMLResponse)
async def get_track_report_html(case_id: str, track_id: str, db: Session = Depends(get_db)):
    """Generate and return dedicated individual forensic report for a specific vehicle track / plate."""
    return generate_case_report_html(case_id, db, track_id=track_id)


@app.get("/")
async def root():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "VEHICLE TRACE API is operational", "docs": "/docs"}
