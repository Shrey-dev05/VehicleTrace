"""
Vehicle Registry Matching Engine
Compares visual evidence (plate, colour, make, model, type) against registered vehicle database.
Ranks candidates with explainability and abstains when evidence is insufficient.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from backend.app.database.models import VehicleRegistry

log = logging.getLogger(__name__)


class Tuple_PlateMatch:
    def __init__(self, score: float, notes: str):
        self.score = score
        self.notes = notes

    def __iter__(self):
        yield self.score
        yield self.notes


def score_plate_compatibility(observed_plate: str, candidate_plate: str) -> Tuple_PlateMatch:
    """
    Score plate similarity preserving wildcard '?' character slots.
    Known characters matching = +1.0
    Wildcard '?' = neutral (0.60)
    Explicit mismatch in known position = -1.5 (heavy penalty)
    """
    obs = (observed_plate or "").upper().replace(" ", "").replace("-", "").replace(".", "")
    cand = (candidate_plate or "").upper().replace(" ", "").replace("-", "").replace(".", "")

    if not obs:
        return Tuple_PlateMatch(0.3, "No observed plate text available.")

    if obs == cand:
        return Tuple_PlateMatch(1.0, "Exact 100% character match.")

    # Normalization for single digit RTO (e.g. DL8CA1189 -> DL08CA1189)
    if len(obs) == 9 and len(cand) == 10 and obs[:2].isalpha() and obs[2].isdigit() and obs[3].isalpha():
        obs = obs[:2] + "0" + obs[2:]
    elif len(cand) == 9 and len(obs) == 10 and cand[:2].isalpha() and cand[2].isdigit() and cand[3].isalpha():
        cand = cand[:2] + "0" + cand[2:]

    # Compare position-wise if lengths match
    if len(obs) == len(cand):
        matches = 0
        wildcards = 0
        mismatches = 0
        for o_char, c_char in zip(obs, cand):
            if o_char == "?":
                wildcards += 1
            elif o_char == c_char:
                matches += 1
            else:
                mismatches += 1

        total_chars = len(obs)
        if mismatches == 0:
            score = 0.6 + 0.4 * (matches / total_chars)
            desc = f"Pattern compatible: {matches} exact matches, {wildcards} wildcard positions."
        else:
            penalty = mismatches * 0.35
            score = max(0.0, (matches / total_chars) - penalty)
            desc = f"Mismatch: {mismatches} conflicting characters, {matches} matching."
        return Tuple_PlateMatch(round(score, 3), desc)

    # Substring or Levenshtein similarity for slight length variances
    common = sum(1 for c in set(obs) if c != "?" and c in set(cand))
    approx_score = round(min(0.7, common / max(len(cand), 1)), 3)
    return Tuple_PlateMatch(approx_score, f"Length divergence ({len(obs)} vs {len(cand)}), approx common characters.")


def score_make_compatibility(observed_make: str, candidate_make: str) -> Tuple[float, str]:
    if not observed_make or observed_make.lower() in ("unknown", "other", ""):
        return 0.5, "Make unconfirmed from camera view."

    o = observed_make.lower().replace("_", " ").replace("-", " ").strip()
    c = (candidate_make or "").lower().replace("_", " ").replace("-", " ").strip()

    if not c:
        return 0.5, "Registry make unavailable."

    if o == c:
        return 1.0, f"Make matched: {candidate_make}."

    # Substring / brand alias matching (e.g. Maruti vs Maruti Suzuki, Mercedes vs Mercedes-Benz)
    if o in c or c in o:
        return 0.95, f"Make highly consistent: {candidate_make}."

    return 0.1, f"Make conflict: observed {observed_make} vs registry {candidate_make}."


def score_model_compatibility(observed_model: str, candidate_model: str) -> Tuple[float, str]:
    if not observed_model or observed_model.lower() in ("unknown", "other", ""):
        return 0.5, "Model unconfirmed from camera view."

    o = observed_model.lower().replace("_", " ").replace("-", " ").strip()
    c = (candidate_model or "").lower().replace("_", " ").replace("-", " ").strip()

    if not c:
        return 0.5, "Registry model unavailable."

    if o == c:
        return 1.0, f"Model matched: {candidate_model}."

    # Word overlap or substring (e.g. "Innova" vs "Innova Crysta", "Swift" vs "Swift Dzire")
    o_words = set(o.split())
    c_words = set(c.split())
    if o_words & c_words or o in c or c in o:
        return 0.90, f"Model consistent: {candidate_model}."

    return 0.1, f"Model conflict: observed {observed_model} vs registry {candidate_model}."


def score_colour_compatibility(observed_colour: str, candidate_colour: str) -> Tuple[float, str]:
    if not observed_colour or observed_colour.lower() in ("unknown", "other", ""):
        return 0.5, "Colour unconfirmed from camera view."

    o = observed_colour.lower().replace("_", " ").strip()
    c = (candidate_colour or "").lower().replace("_", " ").strip()

    if not c:
        return 0.5, "Registry colour unavailable."

    if o == c:
        return 1.0, f"Colour matched: {candidate_colour}."

    silver_grey_group = {"silver", "grey", "gray", "silver_grey", "silver grey"}
    if o in silver_grey_group and c in silver_grey_group:
        return 0.90, f"Colour consistent: {candidate_colour} (silver/grey tone)."

    blue_group = {"blue", "dark blue", "navy", "navy blue", "cyan"}
    if o in blue_group and c in blue_group:
        return 0.90, f"Colour consistent: {candidate_colour} (blue tone)."

    red_group = {"red", "maroon", "burgundy"}
    if o in red_group and c in red_group:
        return 0.85, f"Colour consistent: {candidate_colour} (red tone)."

    return 0.15, f"Colour discrepancy: observed {observed_colour} vs registry {candidate_colour}."


def score_type_compatibility(observed_type: str, candidate_type: str) -> Tuple[float, str]:
    if not observed_type or observed_type.lower() in ("unknown", "other", ""):
        return 0.5, "Vehicle type unconfirmed."

    o = observed_type.lower().replace("_", " ").strip()
    c = (candidate_type or "").lower().replace("_", " ").strip()

    if not c:
        return 0.5, "Registry type unavailable."

    if o == c:
        return 1.0, f"Type matched: {candidate_type}."

    car_categories = {
        "car", "sedan", "hatchback", "suv", "compact suv", "micro suv",
        "mpv", "coupe suv", "off-road suv", "van", "coupe"
    }
    if o in car_categories and c in car_categories:
        return 0.90, f"Category compatible: {candidate_type}."

    two_wheeler = {"motorcycle", "scooter", "motorcycle_scooter", "electric scooter", "bike"}
    if o in two_wheeler and c in two_wheeler:
        return 0.95, f"Category compatible: {candidate_type} (two-wheeler)."

    heavy = {"truck", "tipper truck", "heavy commercial vehicle", "light commercial vehicle", "commercial vehicle"}
    if o in heavy and c in heavy:
        return 0.95, f"Category compatible: {candidate_type} (commercial)."

    bus = {"bus", "mini bus"}
    if o in bus and c in bus:
        return 0.95, f"Category compatible: {candidate_type} (bus)."

    return 0.20, f"Type divergence: observed {observed_type} vs registry {candidate_type}."



def generate_evidence_explanation(
    p_score: float,
    make_score: float,
    model_score: float,
    colour_score: float,
    type_score: float,
    has_wildcard: bool = False,
    is_plate_missing: bool = False
) -> str:
    """
    Generate clear, human-readable forensic rationale explaining why candidate was shortlisted.
    """
    if is_plate_missing:
        parts = []
        if make_score >= 0.8:
            parts.append("Make")
        if model_score >= 0.8:
            parts.append("Model")
        if colour_score >= 0.8:
            parts.append("Colour")
        if type_score >= 0.8:
            parts.append("Body Type")
        if len(parts) >= 3:
            return f"Visual attribute match ({', '.join(parts)}); plate unavailable"
        elif len(parts) >= 1:
            return f"Partial visual match ({', '.join(parts)}); plate unreadable"
        return "Visual similarity on available attributes; plate unconfirmed"

    # Plate available or partial
    if p_score >= 0.85:
        if make_score >= 0.8 and colour_score >= 0.8 and model_score >= 0.8:
            return "Strong plate and complete attribute match"
        elif make_score >= 0.8 and model_score >= 0.8:
            return "Strong colour, model and plate compatibility" if colour_score >= 0.8 else "Plate and model match; colour differs"
        elif colour_score >= 0.8 and make_score < 0.7:
            return "Plate and colour match; manufacturer differs"
        elif make_score >= 0.8:
            return "Strong partial plate and attribute match"
        return "Strong plate compatibility; visual attributes aligned"
    elif p_score >= 0.60:
        if make_score >= 0.8 and colour_score >= 0.8:
            return "Strong partial plate and attribute match"
        elif make_score >= 0.8:
            return "Partial plate and manufacturer match"
        elif colour_score >= 0.8:
            return "Partial plate and colour match"
        return "Pattern compatible on known characters"
    else:
        if colour_score >= 0.8 and make_score >= 0.8:
            return "Colour and manufacturer match; plate divergent"
        return "Low character compatibility; secondary candidate"


class RegistryMatcher:
    def __init__(self, acceptance_threshold: float = 0.45):
        self.acceptance_threshold = acceptance_threshold

    def match_candidates(
        self,
        db: Session,
        observed_plate: str,
        observed_colour: str = "unknown",
        observed_make: str = "Unknown",
        observed_model: str = "Unknown",
        observed_type: str = "car",
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Rank registry candidates according to weighted evidence:
        - When plate is present/partial:
          S = 0.40P + 0.20M + 0.15C + 0.15B + 0.10V
        - When plate is missing/unreadable:
          Dynamically redistribute weights across available attributes:
          S = 0.35M + 0.25C + 0.25B + 0.15V
        """
        all_records = db.query(VehicleRegistry).all()
        if not all_records:
            return {
                "status": "NO_RECORDS",
                "message": "Vehicle registry database has no records.",
                "candidates": []
            }

        clean_plate = (observed_plate or "").strip().upper()
        is_plate_available = bool(clean_plate and clean_plate not in ("NOT_READABLE", "UNKNOWN", "NONE", ""))
        has_wildcard = "?" in clean_plate

        candidate_scores = []

        for rec in all_records:
            # 1. Plate Compatibility
            if is_plate_available:
                p_score, p_desc = score_plate_compatibility(clean_plate, rec.registration_number)
            else:
                p_score = 0.0
                p_desc = "Number plate not readable / unavailable in evidence."

            # 2. Make and Model Compatibility
            make_score, make_notes = score_make_compatibility(observed_make, rec.make)
            model_score, model_notes = score_model_compatibility(observed_model, rec.model)

            # 3. Colour Compatibility
            colour_score, colour_notes = score_colour_compatibility(observed_colour, rec.colour)

            # 4. Vehicle Type Compatibility
            type_score, type_notes = score_type_compatibility(observed_type, rec.vehicle_type)

            # Evidence-based weighted scoring formula
            if is_plate_available:
                # S = 0.40P + 0.20M + 0.15C + 0.15B + 0.10V
                total_ranking_score = round(
                    (0.40 * p_score) +
                    (0.20 * make_score) +
                    (0.15 * colour_score) +
                    (0.15 * type_score) +
                    (0.10 * model_score),
                    3
                )
            else:
                # Dynamically redistribute across visual attributes:
                # S = 0.35M + 0.25C + 0.25B + 0.15V
                total_ranking_score = round(
                    (0.35 * make_score) +
                    (0.25 * colour_score) +
                    (0.25 * type_score) +
                    (0.15 * model_score),
                    3
                )

            evidence_explanation = generate_evidence_explanation(
                p_score=p_score,
                make_score=make_score,
                model_score=model_score,
                colour_score=colour_score,
                type_score=type_score,
                has_wildcard=has_wildcard,
                is_plate_missing=(not is_plate_available)
            )

            candidate_scores.append({
                "registration_number": rec.registration_number,
                "registry_details": {
                    "make": rec.make,
                    "model": rec.model,
                    "colour": rec.colour,
                    "vehicle_type": rec.vehicle_type,
                    "year": rec.year,
                    "owner_masked": rec.owner_masked,
                    "state": rec.state,
                    "rto": rec.rto
                },
                "ranking_score": total_ranking_score,
                "evidence_explanation": evidence_explanation,
                "compatibility": {
                    "plate_score": p_score,
                    "plate_notes": p_desc,
                    "colour_score": colour_score,
                    "colour_notes": colour_notes,
                    "make_score": make_score,
                    "make_notes": make_notes,
                    "model_score": model_score,
                    "model_notes": model_notes,
                    "type_score": type_score,
                    "type_notes": type_notes
                }
            })

        # Sort descending by ranking score
        candidate_scores.sort(key=lambda c: c["ranking_score"], reverse=True)
        top_candidates = candidate_scores[:limit]

        # Check if top candidate satisfies acceptance threshold
        if not top_candidates or top_candidates[0]["ranking_score"] < self.acceptance_threshold:
            return {
                "status": "NO_CONFIDENT_MATCH",
                "acceptance_threshold": self.acceptance_threshold,
                "message": "No registry vehicle met the confidence threshold. Visual evidence is insufficient or unregistered.",
                "candidates": top_candidates
            }

        return {
            "status": "CANDIDATES_FOUND",
            "acceptance_threshold": self.acceptance_threshold,
            "message": f"Identified {len(top_candidates)} matching registry candidates.",
            "candidates": top_candidates
        }

    def find_close_matches(
        self,
        db: Session,
        observed_plate: str,
        observed_colour: str = "unknown",
        observed_make: str = "Unknown",
        observed_model: str = "Unknown",
        observed_type: str = "car",
        limit: int = 5,
        min_score: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Find vehicle registry candidates closely matching the observed plate.
        Strictly restricted: ONLY executed if the extracted plate text has missing digits ('?').
        """
        clean_obs_plate = (observed_plate or "").strip()
        # Strictly require missing digits ('?') in the observed plate
        if "?" not in clean_obs_plate:
            return []

        all_records = db.query(VehicleRegistry).all()
        if not all_records:
            return []

        has_wildcard = True
        matches = []

        for rec in all_records:
            p_score, p_desc = score_plate_compatibility(clean_obs_plate, rec.registration_number)

            # When wildcard exists, heavily filter out any plate with conflicting characters
            if has_wildcard and p_score < 0.50:
                continue

            make_score, make_notes = score_make_compatibility(observed_make, rec.make)
            model_score, model_notes = score_model_compatibility(observed_model, rec.model)
            colour_score, colour_notes = score_colour_compatibility(observed_colour, rec.colour)
            type_score, type_notes = score_type_compatibility(observed_type, rec.vehicle_type)

            # S = 0.40P + 0.20M + 0.15C + 0.15B + 0.10V
            total_ranking_score = round(
                (0.40 * p_score) +
                (0.20 * make_score) +
                (0.15 * colour_score) +
                (0.15 * type_score) +
                (0.10 * model_score),
                3
            )

            if total_ranking_score < min_score:
                continue

            evidence_explanation = generate_evidence_explanation(
                p_score=p_score,
                make_score=make_score,
                model_score=model_score,
                colour_score=colour_score,
                type_score=type_score,
                has_wildcard=True,
                is_plate_missing=False
            )

            # Build readable attributes matched tags
            attr_matched = []
            if p_score >= 0.80:
                attr_matched.append(f"Plate: {int(round(p_score*100))}%")
            if make_score >= 0.80:
                attr_matched.append(f"Make: {rec.make}")
            if model_score >= 0.80:
                attr_matched.append(f"Model: {rec.model}")
            if colour_score >= 0.80:
                attr_matched.append(f"Colour: {rec.colour}")
            if type_score >= 0.80:
                attr_matched.append(f"Type: {rec.vehicle_type}")

            matches.append({
                "registration_number": rec.registration_number,
                "manufacturer": rec.make,
                "vehicle_model": rec.model,
                "body_colour": rec.colour,
                "category_type": rec.vehicle_type,
                "state": rec.state,
                "ranking_score": total_ranking_score,
                "match_percentage": int(round(total_ranking_score * 100)),
                "evidence_explanation": evidence_explanation,
                "plate_match_score": p_score,
                "plate_notes": p_desc,
                "attributes_matched": attr_matched,
                "compatibility": {
                    "plate_score": p_score,
                    "plate_notes": p_desc,
                    "colour_score": colour_score,
                    "colour_notes": colour_notes,
                    "make_score": make_score,
                    "make_notes": make_notes,
                    "model_score": model_score,
                    "model_notes": model_notes,
                    "type_score": type_score,
                    "type_notes": type_notes
                }
            })

        matches.sort(key=lambda m: m["ranking_score"], reverse=True)
        return matches[:limit]
