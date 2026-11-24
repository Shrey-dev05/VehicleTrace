"""
Indian Number Plate Syntax, Normalisation, Validation and Correction
Supports Standard Indian Plates (e.g. DL01AB1234) and Bharat (BH) Series (e.g. 22BH1234AA)
"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

STATE_CODES = frozenset(
    "AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP "
    "MZ NL OD OR PB PY RJ SK TN TR TS TG UK UA UP WB".split()
)

# Standard format: State(2) + RTO(1-2) + Series(0-3) + Number(1-4)
PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$")
# Bharat Series: Year(2) + BH + Number(4) + Series(1-2)
BH_RE = re.compile(r"^(\d{2})(BH)(\d{4})([A-Z]{1,2})$")

PREFIX_PATTERNS = ("IND", "1ND", "TND", "LND", "ND", "IN", "ID", "IO")

DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "T": "7", "J": "1", "Z": "2", "E": "3", "A": "4",
    "S": "5", "G": "6", "B": "8",
}

LAYOUTS = (
    (2, 2, 4), (1, 3, 4), (2, 1, 4), (1, 2, 4), (2, 3, 4), (1, 1, 4), (2, 0, 4),
    (2, 1, 3), (2, 0, 3), (2, 0, 2), (1, 0, 4), (1, 0, 3)
)


@dataclass
class PlateValidation:
    format_ok: bool
    state_ok: bool
    rto_ok: bool
    score: float
    plate_type: str = "standard"  # "standard" | "bh" | "partial" | "unknown"
    state_code: Optional[str] = None
    rto_code: Optional[str] = None
    series: Optional[str] = None
    number: Optional[str] = None


def normalize_plate_text(text: Optional[str]) -> str:
    """Normalize input string to uppercase alphanumeric with '?' preserved for unknown characters."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9?]", "", text.upper())


def validate_plate(text: str) -> PlateValidation:
    t = normalize_plate_text(text)
    if not t:
        return PlateValidation(False, False, False, 0.0, "unknown")
    
    # Check if string contains wildcards '?'
    if "?" in t:
        return PlateValidation(False, False, False, 0.25, "partial")

    # Bharat (BH) Series
    m_bh = BH_RE.match(t)
    if m_bh:
        yr = int(m_bh.group(1))
        yr_ok = 20 <= yr <= 35
        num_ok = 1 <= int(m_bh.group(3)) <= 9999
        score = 0.5 + (0.3 if yr_ok else 0.0) + (0.2 if num_ok else 0.0)
        return PlateValidation(
            format_ok=True,
            state_ok=True,
            rto_ok=True,
            score=round(score, 3),
            plate_type="bh",
            series=m_bh.group(4),
            number=m_bh.group(3)
        )

    # Standard Series
    m = PLATE_RE.match(t)
    if not m:
        return PlateValidation(False, False, False, 0.0, "unknown")

    state = m.group(1)
    rto = m.group(2)
    series = m.group(3)
    num = m.group(4)

    state_ok = state in STATE_CODES
    rto_ok = 1 <= int(rto) <= 99
    num_len = len(num)
    score = 0.4 + (0.3 if state_ok else 0.0) + (0.15 if rto_ok else 0.0) + (0.15 if num_len == 4 else 0.05)
    
    return PlateValidation(
        format_ok=True,
        state_ok=state_ok,
        rto_ok=rto_ok,
        score=round(score, 3),
        plate_type="standard",
        state_code=state,
        rto_code=rto,
        series=series,
        number=num
    )


def is_indian_plate_format(text: Optional[str]) -> bool:
    """
    Check if a text conforms to official Indian registration plate formats:
    1. Standard State/UT Series: State(2) + RTO(1-2) + Optional Series(0-3) + Number(1-4)
       e.g. DL12CC3495, MH02CB1234, CH01AA0001, HR26DQ5555, DL1C1234
    2. Bharat (BH) Series: Year(2) + 'BH' + Number(4) + Series(1-2)
       e.g. 22BH1234AA
    Allows '?' for partially occluded/unreadable characters, provided the syntactic
    structure adheres to Indian registration rules.
    Non-matching formats (foreign, arbitrary words, invalid state codes) return False.
    """
    if not text:
        return False
    t = re.sub(r"[^A-Z0-9?]", "", text.upper())
    if len(t) < 5 or len(t) > 11:
        return False

    # Must contain at least 3 known alphanumeric characters to avoid pure noise
    known_chars = re.sub(r"[^A-Z0-9]", "", t)
    if len(known_chars) < 3:
        return False

    # Check Bharat (BH) Series: e.g. 22BH1234AA, 22BH?234AA
    m_bh = re.match(r"^([\d?]{2})(BH|B\?|\?H)([\d?]{4})([A-Z?]{1,2})$", t)
    if m_bh:
        return True

    # Standard format: State(2) + RTO(1-2) + Series(0-3) + Number(1-4)
    m_std = re.match(r"^([A-Z?]{2})([\d?]{1,2})([A-Z?]{0,3})([\d?]{1,4})$", t)
    if not m_std:
        return False

    st, rto, series, num = m_std.groups()

    # State validation: if both characters of state are known, must match valid Indian State/UT code
    if "?" not in st:
        if st not in STATE_CODES:
            return False
    else:
        # If one character is ?, verify at least one state code matches
        matching = [s for s in STATE_CODES if (s[0] == st[0] or s[1] == st[1])]
        if not matching:
            return False

    if len(num) == 0:
        return False

    return True


def fix_state_code(st: str) -> Tuple[str, int]:
    if st in STATE_CODES:
        return st, 0
    alt_map = {
        "0": ["O", "D"], "O": ["D"], "D": ["O"],
        "1": ["I", "T", "L"], "I": ["1", "T", "L"], "L": ["I"],
        "4": ["A"], "A": ["4"],
        "6": ["G"], "G": ["6", "C"], "C": ["G"],
        "7": ["T", "Z"], "T": ["7"],
        "8": ["B"], "B": ["8", "P"], "P": ["B"],
        "5": ["S"], "S": ["5"], "2": ["Z"], "Z": ["2"],
        "N": ["M"], "M": ["N"],
    }
    for i in range(len(st)):
        for alt in alt_map.get(st[i], []):
            c = st[:i] + alt + st[i+1:]
            if c in STATE_CODES:
                return c, 1
    for sc in STATE_CODES:
        if sum(1 for a, b in zip(st, sc) if a != b) == 1:
            return sc, 2
    return st, 0


def correct_plate_text(text: str) -> str:
    """Correct character confusion according to expected Indian layout syntax."""
    t = normalize_plate_text(text)
    if not t or len(t) < 4:
        return t

    # Strip prefixes like IND
    for pfx in PREFIX_PATTERNS:
        if t.startswith(pfx) and len(t) > len(pfx) + 3:
            t = t[len(pfx):]
            break

    # If valid as is, return
    v = validate_plate(t)
    if v.format_ok and v.state_ok:
        return t

    best_candidate = t
    best_score = -1.0

    # Layout fitting
    for (r, s, d) in LAYOUTS:
        if 2 + r + s + d != len(t):
            continue
        pattern = "L" * 2 + "D" * r + "L" * s + "D" * d
        out, subs, ok = [], 0, True
        for kind, c in zip(pattern, t):
            if kind == "L":
                if c.isalpha():
                    out.append(c)
                elif c in DIGIT_TO_LETTER:
                    out.append(DIGIT_TO_LETTER[c]); subs += 1
                else:
                    ok = False; break
            else:
                if c.isdigit():
                    out.append(c)
                elif c in LETTER_TO_DIGIT:
                    out.append(LETTER_TO_DIGIT[c]); subs += 1
                else:
                    ok = False; break
        if not ok:
            continue
        
        st_cand = "".join(out[:2])
        if st_cand not in STATE_CODES:
            fixed_st, st_subs = fix_state_code(st_cand)
            if fixed_st in STATE_CODES:
                out[:2] = list(fixed_st)
                subs += st_subs

        cand_str = "".join(out)
        val = validate_plate(cand_str)
        if val.format_ok:
            sc = val.score - (subs * 0.08)
            if sc > best_score:
                best_score = sc
                best_candidate = cand_str

    return best_candidate
