"""
Vehicle and License Plate Crop Extractor with Paint Panel Isolation
Preserves coordinate traceability and isolates true painted body panels from glass, tires, and background.
"""
import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class CropQuality:
    score: float
    sharpness: float
    brightness: float
    contrast: float
    width: int
    height: int
    aspect_ratio: float
    is_valid: bool
    is_plate_only: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


def compute_image_quality(crop: np.ndarray, min_quality: float = 15.0) -> CropQuality:
    """Compute sharpness, contrast, brightness, and overall quality score."""
    if crop is None or crop.size == 0:
        return CropQuality(0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, False, False)

    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    aspect = float(w) / max(h, 1)

    sharp_norm = min(sharpness / 120.0, 1.0)
    res_norm = min(w / 160.0, 1.0)
    contrast_norm = min(contrast / 50.0, 1.0)
    bright_norm = 1.0 - min(abs(brightness - 128.0) / 100.0, 1.0)

    score = 100.0 * (0.35 * sharp_norm + 0.25 * res_norm + 0.20 * contrast_norm + 0.20 * bright_norm)
    is_valid = (score >= min_quality) and (w >= 32) and (h >= 32)
    
    # Check if this crop is likely a tight license plate rather than full vehicle
    is_plate_only = (2.0 <= aspect <= 5.8) and (w < 450 and h < 160)

    return CropQuality(
        score=round(score, 1),
        sharpness=round(sharpness, 1),
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        width=w,
        height=h,
        aspect_ratio=round(aspect, 2),
        is_valid=is_valid,
        is_plate_only=is_plate_only,
        details={
            "sharp_norm": round(sharp_norm, 2),
            "contrast_norm": round(contrast_norm, 2),
            "bright_norm": round(bright_norm, 2)
        }
    )


def extract_vehicle_crop(
    frame: np.ndarray,
    bbox: List[int],
    expand_ratio: float = 0.06
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Extract vehicle crop with slight boundary expansion while ensuring bounds checking.
    """
    if frame is None or frame.size == 0:
        return None, bbox

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(bw * expand_ratio)
    pad_y = int(bh * expand_ratio)

    ax1 = max(0, x1 - pad_x)
    ay1 = max(0, y1 - pad_y)
    ax2 = min(w, x2 + pad_x)
    ay2 = min(h, y2 + pad_y)

    crop = frame[ay1:ay2, ax1:ax2]
    if crop is None or crop.size == 0:
        return None, [ax1, ay1, ax2, ay2]

    return crop, [ax1, ay1, ax2, ay2]


def extract_plate_crop(
    frame: np.ndarray,
    bbox: List[int],
    expand_ratio_x: float = 0.06,
    expand_ratio_y: float = 0.10
) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Extract plate crop with precise padding to avoid clipping characters
    while preventing background bumper intrusion.
    """
    if frame is None or frame.size == 0:
        return None, bbox

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = max(3, int(round(bw * expand_ratio_x)))
    pad_y = max(3, int(round(bh * expand_ratio_y)))

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    crop = frame[py1:py2, px1:px2]
    if crop is None or crop.size == 0:
        return None, [px1, py1, px2, py2]

    return crop, [px1, py1, px2, py2]


def extract_vehicle_body_roi(vehicle_crop: np.ndarray) -> np.ndarray:
    """
    Isolate pure painted body panels:
    - Excludes upper 22% (windshield, roofline, sky reflection)
    - Excludes lower 22% (tires, wheels, road asphalt, shadow)
    - In the central 56% vertical band, excludes specular glares and deep shadows.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return vehicle_crop

    vh, vw = vehicle_crop.shape[:2]
    if vh < 20 or vw < 20:
        return vehicle_crop

    # Focus on the true body band: doors, hood, fenders
    y1 = int(vh * 0.22)
    y2 = int(vh * 0.78)
    x1 = int(vw * 0.15)
    x2 = int(vw * 0.85)

    body_slice = vehicle_crop[y1:y2, x1:x2]
    if body_slice.size == 0:
        return vehicle_crop

    # Mask out extreme reflections / shadows to isolate solid paint
    hsv = cv2.cvtColor(body_slice, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Paint mask: exclude deep black shadow (< 25) and blinding white specular glare (> 248 with low sat)
    paint_mask = (v >= 25) & ~((v > 248) & (s < 20))
    
    # If valid paint pixels exist, keep them
    if np.count_nonzero(paint_mask) > (body_slice.shape[0] * body_slice.shape[1] * 0.2):
        return body_slice
    return vehicle_crop
