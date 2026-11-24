"""
License Plate OCR Recognizer Engine — High-Precision Multi-Variant TrOCR & Indian Syntax Fitting
Applies deskewing, ViT patch upscaling, CLAHE, bilateral filtering, stacked plate unrolling,
multi-beam generation, and position-aware syntax correction.
"""
import os
import re
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import torch

from backend.app.core.config import settings
from backend.app.core.device import get_optimal_device
from backend.app.core.plate_rules import (
    validate_plate, correct_plate_text, normalize_plate_text, PlateValidation,
    STATE_CODES, LAYOUTS, DIGIT_TO_LETTER, LETTER_TO_DIGIT, fix_state_code
)
from backend.app.attributes.crop_extractor import compute_image_quality

log = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    clean_text: str
    confidence: float
    validation: PlateValidation
    unknown_positions: List[int]
    status: str  # "clean", "partial", "unreadable"
    quality_score: float
    variant_used: str = "original"


# =====================================================================
# Image Preprocessing & Variant Generation
# =====================================================================

def estimate_skew(gray: np.ndarray) -> float:
    """Median angle of near-horizontal edges."""
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=max(20, int(0.4 * w)), maxLineGap=6)
    if lines is None:
        return 0.0
    angs = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 30:
            angs.append(a)
    return float(np.median(angs)) if angs else 0.0


def deskew_crop(crop: np.ndarray, skew: float) -> np.ndarray:
    if not (1.5 < abs(skew) <= 22):
        return crop
    h, w = crop.shape[:2]
    pad = int(max(h, w) * 0.1)
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    hp, wp = padded.shape[:2]
    M = cv2.getRotationMatrix2D((wp / 2, hp / 2), skew, 1.0)
    rotated = cv2.warpAffine(padded, M, (wp, hp), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated[pad:pad + h, pad:pad + w]


def upscale_plate(crop: np.ndarray, min_w: int = 320, min_h: int = 64) -> np.ndarray:
    """Ensure sufficient pixel resolution for ViT patch tokens."""
    h, w = crop.shape[:2]
    if w <= 0 or h <= 0:
        return crop
    scale = max(min_w / w if w < min_w else 1.0, min_h / h if h < min_h else 1.0)
    if scale <= 1.0:
        return crop
    return cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)


def split_and_stitch_stacked_plate(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    For 2-line Indian plates (aspect < 2.5):
    Split top and bottom halves and stitch horizontally side-by-side with a separator.
    """
    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return None
    mid_h = int(round(h * 0.55))
    top = crop[0:mid_h, :]
    bottom = crop[h - mid_h:, :]
    th, tw = top.shape[:2]
    bh, bw = bottom.shape[:2]
    target_h = max(th, bh)
    if th != target_h:
        top = cv2.resize(top, (int(round(tw * target_h / max(th, 1))), target_h), interpolation=cv2.INTER_CUBIC)
    if bh != target_h:
        bottom = cv2.resize(bottom, (int(round(bw * target_h / max(bh, 1))), target_h), interpolation=cv2.INTER_CUBIC)
    gap = np.full((target_h, max(12, int(target_h * 0.25)), 3), 255, dtype=crop.dtype)
    return np.hstack([top, gap, bottom])


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_unsharp(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, amount, blurred, 1.0 - amount, 0)


def generate_ocr_variants(base: np.ndarray, aspect: float) -> List[Tuple[str, np.ndarray]]:
    variants = [
        ("original", base),
        ("unsharp", apply_unsharp(base, 1.4)),
        ("clahe", apply_clahe(base)),
        ("bi_clahe", apply_clahe(cv2.bilateralFilter(base, 5, 45, 45)))
    ]
    if aspect < 2.4:
        stitched = split_and_stitch_stacked_plate(base)
        if stitched is not None:
            variants.append(("stacked_stitched", stitched))
            variants.append(("stacked_clahe", apply_clahe(stitched)))
    return variants


# =====================================================================
# OCR Recognizer Engine
# =====================================================================

class OCREngine:
    def __init__(self):
        self.device_info = get_optimal_device()
        self.device = self.device_info.device
        self.dtype = torch.float16 if self.device_info.fp16 else torch.float32
        self.proc = None
        self.model = None
        self._init_model()

    def _init_model(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        candidate_dirs = [
            str(settings.model_dir / "trocr_plate"),
            r"C:\Users\shrey\.cache\huggingface\hub\models--subash1652007--indian-plate-ocr\snapshots\82232c6220919c8c60d148ad74eb13dbea37e260\trocr_plate",
            r"C:\Users\shrey\OneDrive\Desktop\Core\trocr_plate"
        ]

        for d in candidate_dirs:
            if d and os.path.isdir(d):
                try:
                    self.proc = TrOCRProcessor.from_pretrained(d)
                    self.model = VisionEncoderDecoderModel.from_pretrained(d, torch_dtype=self.dtype).to(self.device).eval()
                    log.info(f"Loaded fine-tuned Indian plate TrOCR model from {d}")
                    return
                except Exception as e:
                    log.warning(f"Failed to load TrOCR from {d}: {e}")

        log.warning("Could not load TrOCR weights; using syntactic fallback.")

    @torch.inference_mode()
    def _run_trocr_batch(self, imgs: List[np.ndarray], beams: int = 2, n_ret: int = 2) -> List[List[Tuple[str, float]]]:
        if self.model is None or not imgs:
            return [[("", 0.0)] for _ in imgs]

        pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in imgs]
        pv = self.proc(images=pils, return_tensors="pt").pixel_values
        pv = pv.to(self.device, dtype=self.dtype, non_blocking=True)

        out = self.model.generate(
            pv, max_length=13, num_beams=beams, num_return_sequences=n_ret,
            output_scores=True, return_dict_in_generate=True, early_stopping=(beams > 1)
        )
        dec = self.proc.batch_decode(out.sequences, skip_special_tokens=True)

        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            probs = torch.exp(out.sequences_scores.float()).clamp(0.01, 1.0).cpu().tolist()
        elif out.scores:
            stacked = torch.stack(out.scores, dim=1).softmax(dim=-1)
            tokens = out.sequences[:, 1:1 + stacked.shape[1]].unsqueeze(-1)
            token_probs = stacked.gather(dim=-1, index=tokens).squeeze(-1)
            probs = token_probs.mean(dim=-1).clamp(0.01, 1.0).cpu().tolist()
        else:
            probs = [0.85] * len(out.sequences)

        results = []
        for i in range(len(imgs)):
            sub = []
            for j in range(n_ret):
                idx = i * n_ret + j
                raw_s = dec[idx] if idx < len(dec) else ""
                txt = normalize_plate_text(raw_s)
                sub.append((txt, float(probs[idx]) if idx < len(probs) else 0.5))
            results.append(sub)
        return results

    def _rank_candidates(self, candidates: List[Tuple[str, float]]) -> Optional[Tuple[str, float, PlateValidation, float]]:
        """Rank candidates using probability combined with Indian syntax layout score."""
        best = None
        for raw, prob in candidates:
            if not raw or len(raw) < 3:
                continue
            corrected = correct_plate_text(raw)
            if not corrected or len(corrected) < 3:
                continue
            v = validate_plate(corrected)
            valid_bonus = 0.40 if (v.format_ok and v.state_ok) else (0.20 if v.format_ok else 0.0)
            score = prob * (0.25 + 0.75 * v.score) + valid_bonus
            if best is None or score > best[3]:
                best = (corrected, prob, v, score)
        return best

    def read_plate(self, plate_crop: np.ndarray) -> OCRResult:
        """
        Process plate crop through multi-variant TrOCR beam generation
        and position-aware Indian syntactic correction.
        """
        if plate_crop is None or plate_crop.size == 0:
            return OCRResult("", "", 0.0, validate_plate(""), [], "unreadable", 0.0)

        # Apply gentle neutral padding to prevent character boundary clipping
        h, w = plate_crop.shape[:2]
        pad_x = max(4, int(round(w * 0.05)))
        pad_y = max(3, int(round(h * 0.08)))
        padded_crop = cv2.copyMakeBorder(plate_crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        q = compute_image_quality(padded_crop)
        gray = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2GRAY) if padded_crop.ndim == 3 else padded_crop
        skew = estimate_skew(gray)

        deskewed = deskew_crop(padded_crop, skew)
        base = upscale_plate(deskewed, min_w=320, min_h=64)

        variants = generate_ocr_variants(base, q.aspect_ratio)

        # 1. TrOCR Multi-Variant Inference with beams=2 and n_ret=2
        if self.model is not None:
            imgs = [img for _, img in variants]
            batch_outs = self._run_trocr_batch(imgs, beams=2, n_ret=2)
            
            all_cands = []
            best_rank = None
            best_variant_name = "original"
            for (v_name, _), cands in zip(variants, batch_outs):
                for c_txt, c_prob in cands:
                    if c_txt and len(c_txt) >= 3:
                        all_cands.append((c_txt, c_prob))
                r = self._rank_candidates(cands)
                if r and (best_rank is None or r[3] > best_rank[3]):
                    best_rank = r
                    best_variant_name = v_name

            # Also rank over the full pool of candidates across all variants
            pool_rank = self._rank_candidates(all_cands)
            if pool_rank and (best_rank is None or pool_rank[3] > best_rank[3]):
                best_rank = pool_rank

            if best_rank is not None:
                text, prob, val, _ = best_rank
                unknown_pos = [i for i, c in enumerate(text) if c == "?"]
                status = "clean" if (val.format_ok and val.state_ok and "?" not in text) else "partial"

                return OCRResult(
                    text=text,
                    clean_text=text.replace("?", ""),
                    confidence=round(prob, 3),
                    validation=val,
                    unknown_positions=unknown_pos,
                    status=status,
                    quality_score=q.score,
                    variant_used=best_variant_name
                )


        # 2. If no candidate met the threshold
        return OCRResult(
            text="",
            clean_text="",
            confidence=0.0,
            validation=validate_plate(""),
            unknown_positions=[],
            status="unreadable",
            quality_score=q.score,
            variant_used="none"
        )
