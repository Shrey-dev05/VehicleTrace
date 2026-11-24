"""
Vehicle Detector using Pretrained YOLO
Detects vehicles (cars, motorcycles, buses, trucks) and generates full-vehicle bounding boxes.
"""
import logging
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from backend.app.core.config import settings
from backend.app.core.device import get_optimal_device

log = logging.getLogger(__name__)

# COCO vehicle classes: 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle_scooter",
    5: "bus",
    7: "truck"
}


class VehicleDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        self.device_info = get_optimal_device()
        self.conf = conf_threshold or settings.vehicle_det_conf
        
        # Load YOLO model
        if model_path and os.path.isfile(model_path):
            self.model = YOLO(model_path)
        else:
            # Ultralytics will load standard pretrained yolov8n.pt
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception as e:
                log.warning(f"Could not load yolov8n.pt ({e}), using fallback detector.")
                self.model = None

    def detect(self, frame: np.ndarray, allow_fullscreen_fallback: bool = False) -> List[Dict[str, Any]]:
        """
        Detect vehicles in frame.
        Returns list of dicts:
        [{
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "vehicle_type": str,
            "class_id": int
        }]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections = []

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=frame,
                    classes=list(COCO_VEHICLE_CLASSES.keys()),
                    conf=self.conf,
                    device=0 if self.device_info.device == "cuda" else "cpu",
                    verbose=False
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.int().cpu().tolist()
                    confs = boxes.conf.float().cpu().tolist()
                    cls_ids = boxes.cls.int().cpu().tolist()

                    for b, score, cid in zip(xyxy, confs, cls_ids):
                        vtype = COCO_VEHICLE_CLASSES.get(cid, "car")
                        detections.append({
                            "bbox": b,
                            "confidence": round(score, 3),
                            "vehicle_type": vtype,
                            "class_id": cid
                        })
            except Exception as e:
                log.error(f"YOLO vehicle detection failed: {e}")

        # Fallback if no vehicle was detected and allow_fullscreen_fallback is True (for static images)
        if not detections and allow_fullscreen_fallback:
            detections.append({
                "bbox": [0, 0, w, h],
                "confidence": 0.50,
                "vehicle_type": "car",
                "class_id": 2
            })

        return detections
