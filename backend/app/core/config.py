"""
Application Configuration and Settings
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CROP_DIR = STORAGE_DIR / "crops"
OUTPUT_DIR = STORAGE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "model_artifacts"
CONFIG_DIR = BASE_DIR / "training" / "configs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AppSettings(BaseModel):
    # App info
    app_name: str = "VEHICLE TRACE - ANPR & Attribute Recognition"
    version: str = "1.0.0"
    debug: bool = False
    
    # Storage
    base_dir: Path = BASE_DIR
    storage_dir: Path = STORAGE_DIR
    upload_dir: Path = UPLOAD_DIR
    crop_dir: Path = CROP_DIR
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR
    config_dir: Path = CONFIG_DIR

    # Database
    database_url: str = f"sqlite:///{STORAGE_DIR / 'vehicle_trace.db'}"

    # Hardware & Inference
    device: str = "cpu"  # "cuda" or "cpu", auto-detected
    fp16: bool = False
    
    # Gemini Multimodal Vision API
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6I9yrkzU1U6HgVkD5mHUn3widwxttAqcL7-z3jXN5S2jA")
    gemini_model: str = "gemini-3.1-flash-lite"
    use_gemini_vision: bool = True
    
    # Thresholds
    vehicle_det_conf: float = 0.35
    plate_det_conf: float = 0.25
    ocr_min_conf: float = 0.30
    min_quality_score: float = 15.0
    colour_conf_threshold: float = 0.50
    make_conf_threshold: float = 0.50
    model_conf_threshold: float = 0.60
    
    # Video tracking & sampling
    video_sample_fps: float = 5.0
    frame_skip: int = 2
    tracker_type: str = "bytetrack"
    
    # Model Artifact Paths
    vehicle_yolo_path: Path = MODEL_DIR / "vehicle_detector" / "yolov8n.pt"
    plate_yolo_path: Path = MODEL_DIR / "plate_detector" / "yolo_plate.pt"
    colour_model_path: Path = MODEL_DIR / "colour_classifier" / "colour_best.pth"
    make_model_path: Path = MODEL_DIR / "make_classifier" / "make_best.pth"
    model_classifier_path: Path = MODEL_DIR / "model_classifier" / "model_best.pth"

settings = AppSettings()
