"""
Temporal Plate Fusion
Fuses license plate characters across video track frames using character slot voting.
"""
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from backend.app.core.plate_rules import validate_plate, correct_plate_text


class PlateFusionEngine:
    def fuse_plate_reads(self, plate_reads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        plate_reads: list of dicts with:
        - text: str
        - confidence: float
        - quality_score: float
        - validation_score: float
        """
        if not plate_reads:
            return {
                "fused_plate": "",
                "plate_conf": 0.0,
                "plate_status": "unknown",
                "unknown_positions": [],
                "votes": 0
            }

        # Filter out empty reads
        valid_reads = [r for r in plate_reads if r.get("text")]
        if not valid_reads:
            return {
                "fused_plate": "",
                "plate_conf": 0.0,
                "plate_status": "unknown",
                "unknown_positions": [],
                "votes": 0
            }

        # Find most common string length
        lengths = [len(r["text"]) for r in valid_reads]
        target_len = Counter(lengths).most_common(1)[0][0]

        # Filter reads matching dominant length
        matching_reads = [r for r in valid_reads if len(r["text"]) == target_len]
        if not matching_reads:
            matching_reads = valid_reads

        # Character slot voting
        fused_chars = []
        conf_scores = []
        unknown_positions = []

        for pos in range(target_len):
            char_votes = defaultdict(float)
            for r in matching_reads:
                text = r["text"]
                if pos < len(text):
                    c = text[pos]
                    weight = max(0.2, r.get("confidence", 0.5)) * max(0.2, r.get("quality_score", 20.0) / 100.0)
                    char_votes[c] += weight

            if char_votes:
                best_char, best_weight = max(char_votes.items(), key=lambda x: x[1])
                total_w = sum(char_votes.values())
                slot_conf = best_weight / total_w
                if best_char == "?" or slot_conf < 0.40:
                    fused_chars.append("?")
                    unknown_positions.append(pos)
                else:
                    fused_chars.append(best_char)
                conf_scores.append(slot_conf)
            else:
                fused_chars.append("?")
                unknown_positions.append(pos)
                conf_scores.append(0.0)

        fused_str = "".join(fused_chars)
        corrected_str = correct_plate_text(fused_str)
        val = validate_plate(corrected_str)

        avg_conf = round(sum(conf_scores) / max(len(conf_scores), 1), 3)
        status = "clean" if (val.format_ok and val.state_ok and "?" not in corrected_str) else "partial"

        return {
            "fused_plate": corrected_str,
            "plate_conf": avg_conf,
            "plate_status": status,
            "unknown_positions": unknown_positions,
            "votes": len(matching_reads)
        }
