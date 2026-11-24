"""
Gemini Multimodal Vision Service
Analyzes vehicle and license plate crops using Google Gemini Vision (gemini-3.5-flash-lite)
for state-of-the-art OCR, manufacturer/logo detection, vehicle model identification, and color classification.
"""
import logging
import base64
import json
import re
import cv2
import requests
import numpy as np
from typing import Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.core.plate_rules import is_indian_plate_format
from backend.app.attributes.colour_classifier import normalize_colour_name

log = logging.getLogger(__name__)


class GeminiVisionService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def analyze_vehicle_and_plate(self, vehicle_crop: np.ndarray, plate_crop: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze vehicle and optional license plate cutout using Gemini Vision.
        Returns:
        {
            "success": bool,
            "plate_text": str,
            "plate_confidence": float,
            "colour": str,
            "colour_confidence": float,
            "make": str,
            "make_confidence": float,
            "model": str,
            "model_confidence": float,
            "vehicle_type": str
        }
        """
        if not self.api_key:
            return {"success": False, "error": "No Gemini API key configured"}

        if vehicle_crop is None or vehicle_crop.size == 0:
            return {"success": False, "error": "Invalid vehicle crop"}

        try:
            parts = []

            # Optimize vehicle crop resolution for badge/emblem and body geometry recognition
            vh, vw = vehicle_crop.shape[:2]
            if max(vh, vw) < 420:
                scale_up = 420.0 / float(max(1, max(vh, vw)))
                v_resized = cv2.resize(vehicle_crop, (int(vw * scale_up), int(vh * scale_up)), interpolation=cv2.INTER_CUBIC)
            elif max(vh, vw) > 800:
                scale_down = 800.0 / float(max(vh, vw))
                v_resized = cv2.resize(vehicle_crop, (int(vw * scale_down), int(vh * scale_down)), interpolation=cv2.INTER_AREA)
            else:
                v_resized = vehicle_crop

            _, v_buf = cv2.imencode(".jpg", v_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            v_b64 = base64.b64encode(v_buf).decode("utf-8")

            prompt = (
                "You are an expert CCTV vehicle forensics investigator specializing in vehicle identification and Indian ANPR.\n"
                "Analyze the provided vehicle imagery thoroughly and accurately:\n\n"
                "1. Vehicle Make / Automaker / Logo: Inspect the front grille emblem, rear badge, wheel hubcaps, and design styling. "
                "Identify the exact automaker brand (e.g. Maruti Suzuki, Hyundai, Tata, Mahindra, Toyota, Honda, Kia, "
                "Volkswagen, Skoda, MG, Nissan, Ford, Renault, BMW, Mercedes-Benz, Audi, Royal Enfield, Hero, Bajaj, TVS, etc.).\n"
                "2. Vehicle Model: Identify the exact commercial vehicle model (e.g. Swift, Baleno, Creta, Nexon, Thar, City, "
                "Innova Crysta, Fortuner, Seltos, Dzire, Brezza, Scorpio, XUV700, Tiago, Safari, Harrier, Venue, i20, Splendor, Pulsar, etc.).\n"
                "3. Body Paint Color: Identify the vehicle body paint color. Choose from: "
                "['white', 'black', 'silver_grey', 'red', 'blue', 'green', 'yellow', 'orange', 'brown']. "
                "(If metallic silver or grey, return 'silver_grey'; if dark blue or navy, return 'blue'; if maroon or dark red, return 'red'). "
                "Pay special attention to day/night CCTV lighting: do not confuse streetlamp amber or brake glare with vehicle body paint.\n"
                "4. Registration Plate (Indian Standard & Bharat BH Series):\n"
                "  * Standard Indian Format: 2-letter State/UT code (e.g. DL, MH, KA, HR, UP, GJ, TN, CG, RJ, PB, WB, AP, TS, etc.) + 1-2 digit RTO code + optional 1-3 letters + 1-4 digit number (e.g. CG04KL4821, MH12RT7305, DL08CA1189, KA03MN6642).\n"
                "  * Bharat (BH) Series: 2-digit year (e.g. 22) + 'BH' + 4-digit number + 1-2 letters (e.g. 22BH1234AA).\n"
                "  * STRICT NO-GUESSING RULE: If any character or digit is occluded, partially obscured, blurry, or missing, DO NOT GUESS OR INVENT IT. You MUST put a question mark '?' for that specific digit/letter (e.g. 'CG04KL?821', 'MH12??7305').\n"
                "  * If the plate does NOT conform to Indian registration plate formats, or is foreign, fake, or completely missing, return an empty string \"\".\n"
                "5. Category / Body Style: Identify the exact vehicle category from: "
                "['Hatchback', 'Sedan', 'SUV', 'Compact SUV', 'MPV', 'Van', 'Motorcycle', 'Scooter', 'Truck', 'Bus', 'Auto Rickshaw'].\n\n"
                "Return a strict JSON object with these exact keys:\n"
                "{\n"
                '  "plate_text": "...",\n'
                '  "plate_confidence": 0.95,\n'
                '  "vehicle_color": "white",\n'
                '  "color_confidence": 0.95,\n'
                '  "make": "...",\n'
                '  "make_confidence": 0.90,\n'
                '  "model": "...",\n'
                '  "model_confidence": 0.90,\n'
                '  "vehicle_type": "Hatchback"\n'
                "}"
            )

            parts.append({"text": prompt})
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": v_b64
                }
            })

            # If a separate plate cutout is provided, enhance contrast and append
            if plate_crop is not None and plate_crop.size > 0:
                ph, pw = plate_crop.shape[:2]
                if pw < 180 or ph < 45:
                    scale_p = max(2.0, 220.0 / max(1, pw))
                    p_up = cv2.resize(plate_crop, (int(pw * scale_p), int(ph * scale_p)), interpolation=cv2.INTER_CUBIC)
                else:
                    p_up = plate_crop.copy()

                try:
                    lab = cv2.cvtColor(p_up, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l_clahe = clahe.apply(l)
                    p_enhanced = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
                except Exception:
                    p_enhanced = p_up

                _, p_buf = cv2.imencode(".jpg", p_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
                p_b64 = base64.b64encode(p_buf).decode("utf-8")
                parts.append({"text": "Here is also an enlarged, high-contrast cutout of the vehicle license plate:"})
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": p_b64
                    }
                })

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.0
                }
            }

            # Candidate model fallback chain to survive 429/503 rate limits
            candidate_models = []
            for m in [self.model, "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]:
                if m and m not in candidate_models:
                    candidate_models.append(m)

            last_err = "No response"
            data = None

            for m in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
                try:
                    resp = requests.post(url, json=payload, timeout=25)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        candidates = res_json.get("candidates", [])
                        if candidates:
                            out_text = candidates[0]["content"]["parts"][0]["text"].strip()
                            data = json.loads(out_text)
                            break
                    elif resp.status_code in (429, 503):
                        log.warning(f"Gemini model {m} hit status {resp.status_code}. Falling back to next model...")
                        import time
                        time.sleep(0.4)
                        last_err = f"API {resp.status_code}"
                    else:
                        log.warning(f"Gemini model {m} returned {resp.status_code}: {resp.text[:100]}")
                        last_err = f"API {resp.status_code}"
                except Exception as ex:
                    log.warning(f"Gemini request exception on model {m}: {ex}")
                    last_err = str(ex)

            if not data:
                return {"success": False, "error": last_err}

            # Clean and normalize fields
            raw_plate = str(data.get("plate_text", "")).strip().upper().replace(" ", "").replace("-", "")
            plate = re.sub(r"[^A-Z0-9?]", "", raw_plate)

            # Reject non-Indian or invalid plate format
            if plate and not is_indian_plate_format(plate):
                log.info(f"Gemini plate '{plate}' rejected: does not match Indian registration format.")
                plate = ""

            raw_color = str(data.get("vehicle_color", "")).strip()
            color = normalize_colour_name(raw_color)

            make = str(data.get("make", "")).strip()
            model_name = str(data.get("model", "")).strip()
            vtype = str(data.get("vehicle_type", "car")).strip()

            return {
                "success": True,
                "plate_text": plate,
                "plate_confidence": float(data.get("plate_confidence", 0.90)),
                "colour": color,
                "colour_confidence": float(data.get("color_confidence", 0.90)),
                "make": make if make.lower() not in ("unknown", "other", "") else "Unknown",
                "make_confidence": float(data.get("make_confidence", 0.85)),
                "model": model_name if model_name.lower() not in ("unknown", "other", "") else "Unknown",
                "model_confidence": float(data.get("model_confidence", 0.80)),
                "vehicle_type": vtype if vtype.lower() not in ("unknown", "") else "Car"
            }

        except Exception as e:
            log.warning(f"Gemini Vision inference exception: {e}")
            return {"success": False, "error": str(e)}
