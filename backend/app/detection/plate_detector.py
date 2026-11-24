"""
License Plate Detector
Locates license plates within vehicle crops or scene images using fine-tuned YOLO.
"""
import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from backend.app.core.config import settings
from backend.app.core.device import get_optimal_device

log = logging.getLogger(__name__)


class PlateDetector:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = None):
        self.device_info = get_optimal_device()
        self.conf = conf_threshold or settings.plate_det_conf
        self.model = None

        candidate_paths = [
            model_path,
            str(settings.plate_yolo_path),
            str(settings.model_dir / "plate_detector" / "yolo_plate.pt"),
            r"C:\Users\shrey\.cache\huggingface\hub\models--subash1652007--indian-plate-ocr\snapshots\82232c6220919c8c60d148ad74eb13dbea37e260\yolo_best.pt",
            r"C:\Users\shrey\OneDrive\Desktop\Core\yolo_best.pt"
        ]

        for p in candidate_paths:
            if p and os.path.isfile(p):
                try:
                    self.model = YOLO(p)
                    log.info(f"Loaded plate detection model from {p}")
                    break
                except Exception as e:
                    log.warning(f"Failed to load plate weights from {p}: {e}")

    def detect(self, image: np.ndarray, conf_override: Optional[float] = None, imgsz: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Detect license plates in an image.
        Returns: list of dicts: [{"bbox": [x1, y1, x2, y2], "confidence": float}]
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        plates = []
        conf_to_use = conf_override or self.conf

        if self.model is not None:
            try:
                # Dynamic imgsz: 320 for vehicle crops (e.g. <= 480px) is ~2x faster with identical accuracy
                if imgsz is None:
                    max_dim = max(h, w)
                    chosen_imgsz = 320 if max_dim <= 480 else 640
                else:
                    chosen_imgsz = imgsz

                res = self.model.predict(
                    source=image,
                    conf=conf_to_use,
                    imgsz=chosen_imgsz,
                    device=0 if self.device_info.device == "cuda" else "cpu",
                    verbose=False
                )[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.int().cpu().tolist()
                    confs = res.boxes.conf.float().cpu().tolist()
                    for box, c in zip(xyxy, confs):
                        plates.append({
                            "bbox": box,
                            "confidence": round(c, 3)
                        })
            except Exception as e:
                log.warning(f"Plate detector YOLO inference error: {e}")

        # If no plate detected via YOLO, run morphological/contour heuristic fallback
        if not plates:
            plates = self._detect_heuristic(image)

        return plates

    def _detect_heuristic(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Morphological plate candidate detector based on high-contrast rectangular bands.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        
        # Sobel vertical edges
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        
        # Morphological close with horizontal kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(abs_grad_x, cv2.MORPH_CLOSE, kernel)
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = float(cw) / max(ch, 1)
            # Standard plate aspect ratio ~ 1.8 to 5.8
            if 1.8 <= aspect <= 5.8 and cw >= 35 and ch >= 10 and (cw * ch) < (w * h * 0.4):
                candidates.append({
                    "bbox": [x, y, x + cw, y + ch],
                    "confidence": 0.40
                })

        candidates.sort(key=lambda c: (c["bbox"][1] / max(h, 1)), reverse=True)
        return candidates[:2]
