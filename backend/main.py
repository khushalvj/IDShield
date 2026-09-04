from io import BytesIO
import re
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps


# ---------------------------------------------------------
# TESSERACT
# ---------------------------------------------------------

import os
import shutil

_tesseract = shutil.which("tesseract")
if _tesseract:
    pytesseract.pytesseract.tesseract_cmd = _tesseract
elif os.name == "nt":
    _windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_windows_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _windows_tesseract


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = FastAPI(
    title="IDShield API",
    description="AI-assisted identity and document screening backend",
    version="1.1.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://idshield-1.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "IDShield",
        "status": "online",
        "version": app.version,
    }


# ---------------------------------------------------------
# IMAGE QUALITY
# ---------------------------------------------------------

def calculate_image_quality(image: Image.Image) -> dict:
    gray = np.array(image.convert("L"))
    height, width = gray.shape

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    score = 100
    issues = []

    if width < 1000 or height < 700:
        score -= 20
        issues.append("Image resolution is low.")

    if blur_score < 80:
        score -= 30
        issues.append("Image appears blurred.")

    if brightness < 55:
        score -= 15
        issues.append("Image is dark.")

    if brightness > 235:
        score -= 15
        issues.append("Image is overexposed.")

    score = max(0, min(score, 100))

    if score >= 80:
        status = "good"
    elif score >= 60:
        status = "fair"
    else:
        status = "poor"

    return {
        "score": score,
        "status": status,
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "issues": issues,
    }


# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

def _ocr_pass(image: Image.Image, psm: int) -> dict:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    config = f"--oem 3 --psm {psm}"

    text = pytesseract.image_to_string(
        gray,
        config=config,
    ).strip()

    data = pytesseract.image_to_data(
        gray,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    confidences = []
    words = []

    for i, raw_conf in enumerate(data["conf"]):
        try:
            conf = float(raw_conf)
        except (ValueError, TypeError):
            continue

        value = str(data["text"][i]).strip()

        if value and conf >= 25:
            words.append({
                "text": value,
                "confidence": round(conf, 2),
            })

        if conf >= 0:
            confidences.append(conf)

    confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    return {
        "text": text,
        "confidence": round(confidence, 2),
        "words": words,
        "method": f"psm_{psm}",
    }


def run_ocr(image: Image.Image) -> dict:
    """Run several OCR views and keep the strongest evidence from each."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = max(2.0, min(3.0, 1800 / max(gray.shape[1], 1)))
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
    sharp = cv2.addWeighted(clahe, 1.35, cv2.GaussianBlur(clahe, (0, 0), 2), -0.35, 0)
    binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)

    samples = [(up, "full"), (clahe, "clahe"), (sharp, "sharp"), (binary, "adaptive")]
    # Add a bottom-third view because passport MRZ text is dense and small.
    bottom = up[int(up.shape[0] * 0.60):, :]
    samples.append((bottom, "bottom"))

    passes = []
    for sample, label in samples:
        for psm in (6, 11):
            config = f"--oem 3 --psm {psm}"
            text = pytesseract.image_to_string(sample, config=config).strip()
            data = pytesseract.image_to_data(sample, config=config, output_type=pytesseract.Output.DICT)
            confs = []
            for raw in data.get("conf", []):
                try:
                    value = float(raw)
                    if value >= 0:
                        confs.append(value)
                except (ValueError, TypeError):
                    pass
            confidence = round(sum(confs) / len(confs), 2) if confs else 0.0
            passes.append({"text": text, "confidence": confidence, "method": f"{label}_psm_{psm}"})

    best = max(passes, key=lambda item: item["confidence"])
    # Combine unique OCR lines from the best passes. This improves recovery of
    # fields that one preprocessing variant misses without inventing values.
    lines = []
    seen = set()
    for item in sorted(passes, key=lambda x: x["confidence"], reverse=True):
        for line in item["text"].splitlines():
            clean = line.strip()
            key = re.sub(r"\s+", " ", clean).upper()
            if clean and key not in seen:
                seen.add(key)
                lines.append(clean)
    return {
        "text": "\n".join(lines),
        "confidence": best["confidence"],
        "method": f"ensemble_{best['method']}",
        "passes": [{"method": x["method"], "confidence": x["confidence"]} for x in passes],
    }


def run_mrz_ocr(image: Image.Image) -> dict:
    """Focused OCR for the bottom machine-readable zone of a passport."""
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    y0 = int(height * 0.62)
    crop = rgb[y0:height, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    scale = max(2.5, 1800 / max(width, 1))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    variants = [
        (gray, 6),
        (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 6),
        (gray, 7),
    ]

    results = []
    for sample, psm in variants:
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        text = pytesseract.image_to_string(sample, config=config).strip()
        data = pytesseract.image_to_data(sample, config=config, output_type=pytesseract.Output.DICT)
        confs = []
        for raw in data.get("conf", []):
            try:
                value = float(raw)
                if value >= 0:
                    confs.append(value)
            except (ValueError, TypeError):
                pass
        results.append({
            "text": text,
            "confidence": round(sum(confs) / len(confs), 2) if confs else 0.0,
            "method": f"mrz_psm_{psm}",
        })

    best = max(results, key=lambda item: item["confidence"])
    return best


# ---------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------

def normalize_text(value: str) -> str:
    value = value.upper()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^A-Z0-9 /().,&'-]", "", value)
    return value.strip()


FIELD_LABELS = [
    "FULL NAME",
    "NAME",
    "HOLDER NAME",
    "SURNAME",
    "GIVEN NAMES",
    "DATE OF BIRTH",
    "DOB",
    "DATE OF ISSUE",
    "ISSUE DATE",
    "DATE OF EXPIRY",
    "EXPIRY DATE",
    "VALID UNTIL",
    "NATIONALITY",
    "DOCUMENT NUMBER",
    "ID NUMBER",
    "IDENTITY NUMBER",
    "CARD NUMBER",
    "LICENCE NUMBER",
    "LICENSE NUMBER",
    "PASSPORT NUMBER",
    "PASSPORT NUMBER / TYPE",
    "VISA NUMBER",
    "VISA TYPE",
    "ENTRY PERMIT",
    "ENTRIES",
    "DURATION OF STAY",
    "PLACE OF ISSUE",
    "PLACE OF BIRTH",
    "ISSUING AUTHORITY",
    "DRIVING LICENCE",
    "DRIVING LICENSE",
    "IDENTITY CARD",
    "ID CARD",
    "NATIONAL ID",
    "IDENTIFICATION CARD",
]


def _clean_field_value(value: str) -> str:
    value = normalize_text(value)

    # Remove accidental label glued to the value.
    for label in sorted(FIELD_LABELS, key=len, reverse=True):
        if value.startswith(label):
            value = value[len(label):].strip(" :-")

    return value


def _valid_name(value: str | None) -> bool:
    if not value:
        return False

    value = normalize_text(value)
    tokens = value.split()

    if len(value) < 5 or len(tokens) < 2:
        return False

    bad = {
        "NAME",
        "NAMES",
        "SURNAME",
        "GIVEN",
        "PASSPORT",
        "NATIONALITY",
        "NUMBER",
    }

    return not all(token in bad for token in tokens)


def _valid_number(value: str | None) -> bool:
    if not value:
        return False

    value = normalize_text(value).replace(" ", "")

    if not 5 <= len(value) <= 20:
        return False

    if not re.fullmatch(r"[A-Z0-9/-]+", value):
        return False

    return bool(re.search(r"[A-Z]", value)) and bool(
        re.search(r"\d", value)
    )


def _valid_nationality(value: str | None) -> bool:
    if not value:
        return False

    value = normalize_text(value)

    known = {
        "INDIAN",
        "IND",
        "INDIA",
        "AMERICAN",
        "USA",
        "BRITISH",
        "CANADIAN",
        "AUSTRALIAN",
        "NEPALESE",
        "BHUTANESE",
        "BANGLADESHI",
    }

    return value in known


def _extract_date(value: str | None) -> str | None:
    if not value:
        return None

    value = normalize_text(value)

    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s+[A-Z]{3,9}\s+\d{4}\b",
        r"\b[A-Z]{3,9}\s+\d{1,2},?\s+\d{4}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(0)

    return None


def extract_visible_fields(text: str, document_type: str = "passport") -> dict:
    """Extract visible fields conservatively, with MRZ-safe fallbacks later."""
    fields = {}

    def find_value(label: str) -> str | None:
        lines = text.splitlines()
        label_upper = label.upper()

        for index, raw_line in enumerate(lines):
            line = normalize_text(raw_line)
            if label_upper not in line:
                continue

            remainder = re.sub(
                rf".*?{re.escape(label_upper)}\s*[:/\-]?\s*",
                "",
                line,
                count=1,
                flags=re.IGNORECASE,
            ).strip()

            if remainder and remainder.upper() != label_upper:
                return _clean_field_value(remainder)

            if index + 1 < len(lines):
                nxt = normalize_text(lines[index + 1])
                if nxt and nxt.upper() not in {x.upper() for x in FIELD_LABELS}:
                    return _clean_field_value(nxt)

        return None

    if document_type == "passport":
        surname = find_value("SURNAME")
        given_names = find_value("GIVEN NAMES")
        nationality = find_value("NATIONALITY")
        dob = find_value("DATE OF BIRTH")
        sex = find_value("SEX")
        expiry = find_value("DATE OF EXPIRY")
        issue = find_value("DATE OF ISSUE")
        pob = find_value("PLACE OF BIRTH")
        authority = find_value("ISSUING AUTHORITY")

        if _valid_name(surname):
            fields["surname"] = surname
        if _valid_name(given_names):
            fields["given_names"] = given_names
        if _valid_nationality(nationality):
            fields["nationality"] = nationality

        parsed = _extract_date(dob)
        if parsed:
            fields["date_of_birth"] = parsed
        parsed = _extract_date(issue)
        if parsed:
            fields["date_of_issue"] = parsed
        parsed = _extract_date(expiry)
        if parsed:
            fields["date_of_expiry"] = parsed

        if sex:
            sex = normalize_text(sex)
            if sex in {"M", "F", "MALE", "FEMALE"}:
                fields["sex"] = sex

        if pob and len(pob) >= 3:
            fields["place_of_birth"] = pob
        if authority and len(authority) >= 3:
            fields["issuing_authority"] = authority

        patterns = [
            r"PASSPORT\s+NUMBER\s*/?\s*TYPE\s*[:\-]?\s*([A-Z0-9]{6,15})",
            r"PASSPORT\s+NUMBER\s*[:\-]?\s*([A-Z0-9]{6,15})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.upper(), flags=re.IGNORECASE)
            if match and _valid_number(match.group(1)):
                fields["passport_number"] = match.group(1).upper()
                break

    return fields


# ---------------------------------------------------------
# DOCUMENT CLASSIFICATION
# ---------------------------------------------------------

def detect_document_type(text: str) -> dict:
    normalized = " ".join(text.upper().split())

    scores = {
        "passport": 0,
        "visa": 0,
        "id_card": 0,
        "driving_licence": 0,
    }

    evidence = {
        "passport": [],
        "visa": [],
        "id_card": [],
        "driving_licence": [],
    }

    if "REPUBLIC OF INDIA" in normalized:
        scores["passport"] += 45
        evidence["passport"].append("REPUBLIC OF INDIA")

    if "PASSPORT" in normalized:
        scores["passport"] += 35
        evidence["passport"].append("PASSPORT")

    if "PASSPORT NUMBER" in normalized:
        scores["passport"] += 45
        evidence["passport"].append("PASSPORT NUMBER")

    if "MACHINE READABLE" in normalized:
        scores["passport"] += 35
        evidence["passport"].append("MACHINE READABLE")

    if "P<IND" in normalized:
        scores["passport"] += 60
        evidence["passport"].append("INDIAN PASSPORT MRZ")

    if "P IND" in normalized:
        scores["passport"] += 45
        evidence["passport"].append("P IND MRZ PATTERN")

    if (
        "REPUBLIC" in normalized
        and "INDIA" in normalized
        and (
            "PASSPORT" in normalized
            or "PASSPART" in normalized
            or "P<IND" in normalized
        )
    ):
        scores["passport"] = max(scores["passport"], 90)
        evidence["passport"].append("INDIAN PASSPORT SIGNATURE")

    if "VISA" in normalized:
        scores["visa"] += 45
        evidence["visa"].append("VISA")

    if "VISA NUMBER" in normalized:
        scores["visa"] += 25
        evidence["visa"].append("VISA NUMBER")

    if "ENTRY PERMIT" in normalized:
        scores["visa"] += 30
        evidence["visa"].append("ENTRY PERMIT")

    if "DURATION OF STAY" in normalized:
        scores["visa"] += 25
        evidence["visa"].append("DURATION OF STAY")

    # Generic identity-card signals. These are intentionally combined with
    # stronger document-specific signals below so ordinary documents do not
    # get classified as IDs merely because they contain the word "ID".
    if "IDENTITY CARD" in normalized:
        scores["id_card"] += 45
        evidence["id_card"].append("IDENTITY CARD")

    if "ID CARD" in normalized:
        scores["id_card"] += 40
        evidence["id_card"].append("ID CARD")

    if "NATIONAL ID" in normalized:
        scores["id_card"] += 45
        evidence["id_card"].append("NATIONAL ID")

    if "IDENTIFICATION CARD" in normalized:
        scores["id_card"] += 45
        evidence["id_card"].append("IDENTIFICATION CARD")

    # Common Indian identity-document signatures.
    indian_id_signals = [
        ("AADHAAR", 60),
        ("AADHAR", 60),
        ("UNIQUE IDENTIFICATION", 50),
        ("PERMANENT ACCOUNT NUMBER", 55),
        ("PAN CARD", 55),
        ("ELECTION COMMISSION", 50),
        ("ELECTOR PHOTO ID", 55),
        ("EPIC NO", 55),
        ("VOTER ID", 55),
    ]
    for signal, points in indian_id_signals:
        if signal in normalized:
            scores["id_card"] += points
            evidence["id_card"].append(signal)

    if "DRIVING LICENCE" in normalized or "DRIVING LICENSE" in normalized:
        scores["driving_licence"] += 60
        evidence["driving_licence"].append("DRIVING LICENCE")

    if "LICENCE NUMBER" in normalized or "LICENSE NUMBER" in normalized:
        scores["driving_licence"] += 25
        evidence["driving_licence"].append("LICENCE NUMBER")

    scores = {
        key: min(value, 100)
        for key, value in scores.items()
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Conservative acceptance thresholds reduce false positives from ordinary
    # documents that happen to contain words such as "visa" or "ID".
    minimum_score = {
        "passport": 45,
        "visa": 55,
        "id_card": 45,
        "driving_licence": 60,
    }.get(best_type, 45)

    if best_score < minimum_score:
        return {
            "type": "unknown",
            "confidence": 0,
            "evidence": [],
            "candidates": scores,
        }

    return {
        "type": best_type,
        "confidence": best_score,
        "evidence": evidence[best_type],
        "candidates": scores,
    }


# ---------------------------------------------------------
# DOCUMENT ORIENTATION
# ---------------------------------------------------------

def _rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """Rotate using PIL's positive angle convention for counter-clockwise turns."""
    if angle == 0:
        return image.copy()
    return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def auto_orient_document(image: Image.Image) -> tuple[Image.Image, int, str]:
    """
    Normalize EXIF orientation and only rotate when OCR/classification gives
    evidence that another right-angle orientation is a supported ID.

    This avoids Tesseract OSD incorrectly rotating an already-readable
    document and accidentally turning a valid passport into an unsupported
    document.
    """
    base = ImageOps.exif_transpose(image).convert("RGB")

    # First, test the supplied orientation. Do not rotate a document that is
    # already recognized confidently enough as a supported identity document.
    try:
        original_ocr = run_ocr(base)
        original_classification = detect_document_type(original_ocr["text"])
        if original_classification.get("type") != "unknown":
            return base, 0, "original_orientation"
    except Exception:
        pass

    # Only search rotations when the original orientation was not recognized.
    # A rotation wins only when it produces a supported classification.
    candidates = []
    for correction in (90, 180, 270):
        candidate = _rotate_image(base, correction)
        try:
            ocr_probe = run_ocr(candidate)
            classification = detect_document_type(ocr_probe["text"])
            cls_conf = float(classification.get("confidence", 0) or 0)
            ocr_conf = float(ocr_probe.get("confidence", 0) or 0)
            supported = classification.get("type") != "unknown"
            evidence_count = len(classification.get("evidence", []) or [])
            score = (10000 if supported else 0) + cls_conf * 100 + evidence_count * 10 + ocr_conf
            candidates.append((score, correction, candidate, classification))
        except Exception:
            continue

    if candidates:
        _, correction, oriented, classification = max(candidates, key=lambda item: item[0])
        if classification.get("type") != "unknown":
            return oriented, correction, "ocr_classification_rotation"

    return base, 0, "original_orientation"


# ---------------------------------------------------------
# MRZ
# ---------------------------------------------------------

def clean_mrz_line(line: str) -> str:
    line = line.upper().strip().replace(" ", "")

    replacements = {
        "«": "<",
        "‹": "<",
        "|": "<",
        "—": "<",
        "_": "<",
    }

    for old, new in replacements.items():
        line = line.replace(old, new)

    return re.sub(r"[^A-Z0-9<]", "", line)


def find_mrz(text: str):
    candidates = []

    for raw_line in text.splitlines():
        cleaned = clean_mrz_line(raw_line)

        if 30 <= len(cleaned) <= 55:
            candidates.append(cleaned)

    for index, first in enumerate(candidates):
        if not first.startswith("P<"):
            continue

        for second in candidates[index + 1:]:
            if len(second) >= 30:
                return [first, second]

    # Fallback: inspect long chunks containing P<
    compact = re.sub(r"\s+", "", text.upper())
    match = re.search(
        r"(P<[A-Z0-9<]{25,})([A-Z0-9<]{25,})",
        compact,
    )

    if match:
        return [
            clean_mrz_line(match.group(1)),
            clean_mrz_line(match.group(2)),
        ]

    return None


def mrz_check_digit(value: str) -> int:
    weights = [7, 3, 1]
    total = 0

    for index, char in enumerate(value):
        if char.isdigit():
            number = int(char)
        elif "A" <= char <= "Z":
            number = ord(char) - ord("A") + 10
        else:
            number = 0

        total += number * weights[index % 3]

    return total % 10


def validate_checksum(value: str, check_digit: str) -> bool:
    return (
        check_digit.isdigit()
        and mrz_check_digit(value) == int(check_digit)
    )


def _clean_mrz_name(value: str) -> str:
    value = value.replace("<", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_passport_mrz(mrz_lines):
    if not mrz_lines or len(mrz_lines) < 2:
        return None

    line1 = clean_mrz_line(mrz_lines[0]).ljust(44, "<")[:44]
    line2 = clean_mrz_line(mrz_lines[1]).ljust(44, "<")[:44]

    document_type = line1[0:2]
    issuing_country = line1[2:5]

    names = line1[5:44]
    parts = names.split("<<", 1)

    surname = _clean_mrz_name(parts[0])

    given_names = ""
    if len(parts) > 1:
        given_names = _clean_mrz_name(parts[1])

    # Remove trailing filler / OCR noise from given names.
    given_names = re.split(
        r"\s+[A-Z]\s*$",
        given_names,
    )[0].strip()

    passport_number = line2[0:9].replace("<", "")
    nationality = line2[10:13]
    date_of_birth = line2[13:19]
    sex = line2[20]
    date_of_expiry = line2[21:27]
    personal_number = line2[28:42]

    checksums = {
        "passport_number": validate_checksum(
            line2[0:9],
            line2[9],
        ),
        "date_of_birth": validate_checksum(
            line2[13:19],
            line2[19],
        ),
        "date_of_expiry": validate_checksum(
            line2[21:27],
            line2[27],
        ),
        "personal_number": validate_checksum(
            line2[28:42],
            line2[42],
        ),
    }

    composite_value = (
        line2[0:10]
        + line2[13:20]
        + line2[21:28]
        + line2[28:43]
    )

    checksums["composite"] = validate_checksum(
        composite_value,
        line2[43],
    )

    checksums["all_valid"] = all(
        checksums.values()
    )

    return {
        "document_type": document_type,
        "issuing_country": issuing_country,
        "surname": surname,
        "given_names": given_names,
        "passport_number": passport_number,
        "nationality": nationality,
        "date_of_birth": date_of_birth,
        "sex": sex,
        "date_of_expiry": date_of_expiry,
        "personal_number": personal_number.replace("<", ""),
        "checksums": checksums,
    }


def enrich_fields_from_mrz(fields: dict, mrz: dict | None) -> dict:
    """Produce clean dashboard fields, using validated MRZ data to repair OCR noise."""
    result = dict(fields)
    if not mrz:
        return result

    # For passports, MRZ is structurally constrained. Prefer it over obvious
    # OCR fragments such as two-letter surname noise or raw YYMMDD strings.
    for key in ("surname", "given_names", "nationality", "passport_number", "sex"):
        value = mrz.get(key)
        if value:
            result[key] = value

    dob_display = mrz_date_to_iso(mrz.get("date_of_birth"), "dob")
    expiry_display = mrz_date_to_iso(mrz.get("date_of_expiry"), "expiry")
    if dob_display:
        result["date_of_birth"] = dob_display
        result["date_of_birth_display"] = dob_display
    if expiry_display:
        result["date_of_expiry"] = expiry_display
        result["date_of_expiry_display"] = expiry_display

    return result


def mrz_date_to_iso(value: str | None, kind: str) -> str | None:
    """Convert YYMMDD MRZ dates to ISO dates for the dashboard."""
    if not value or not re.fullmatch(r"\d{6}", str(value)):
        return value

    yy = int(value[0:2])
    mm = int(value[2:4])
    dd = int(value[4:6])

    if kind == "expiry":
        year = 2000 + yy
    else:
        current_year = datetime.now().year
        current_yy = current_year % 100
        year = 2000 + yy if yy <= current_yy else 1900 + yy

    try:
        return datetime(year, mm, dd).strftime("%Y-%m-%d")
    except ValueError:
        return value


# ---------------------------------------------------------
# CONSISTENCY
# ---------------------------------------------------------

KNOWN_NATIONALITIES = {
    "IND",
    "USA",
    "GBR",
    "CAN",
    "AUS",
    "NPL",
    "BTN",
    "BGD",
}


def _normalize_compare(value: str | None) -> str:
    if not value:
        return ""

    value = normalize_text(value)

    aliases = {
        "INDIAN": "IND",
        "INDIA": "IND",
    }

    value = aliases.get(value, value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%y%m%d")
        except ValueError:
            pass
    return value


def compare_fields(
    visible: dict,
    mrz: dict | None,
) -> list[dict]:
    if not mrz:
        return []

    comparisons = [
        ("passport_number", "passport_number"),
        ("nationality", "nationality"),
        ("date_of_birth", "date_of_birth"),
        ("date_of_expiry", "date_of_expiry"),
        ("surname", "surname"),
        ("given_names", "given_names"),
        ("sex", "sex"),
    ]

    results = []

    for field, mrz_field in comparisons:
        visible_value = visible.get(field)
        mrz_value = mrz.get(mrz_field)

        if field == "nationality" and mrz_value:
            mrz_value = (
                mrz_value
                if mrz_value in KNOWN_NATIONALITIES
                else mrz_value
            )

        left = _normalize_compare(visible_value)
        right = _normalize_compare(mrz_value)

        if not left or not right:
            status = "not_comparable"
            message = "Comparable value was not reliably extracted."

        elif left == right:
            status = "match"
            message = "Visible value matches the MRZ."

        else:
            status = "mismatch"
            message = (
                f"Visible value '{visible_value}' differs "
                f"from MRZ value '{mrz_value}'."
            )

        results.append({
            "field": field,
            "visible": visible_value,
            "mrz": mrz_value,
            "status": status,
            "message": message,
        })

    return results


# ---------------------------------------------------------
# TAMPERING SCREENING
# ---------------------------------------------------------

def analyze_tampering(image: Image.Image) -> dict:
    """Run lightweight, explainable visual-forensics screening."""
    gray = np.array(image.convert("L"))
    h, w = gray.shape
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Compare local texture statistics across a grid. A forged pasted region
    # can have noticeably different noise/texture from surrounding regions.
    tiles = []
    for gy in range(3):
        for gx in range(3):
            y0, y1 = gy*h//3, (gy+1)*h//3
            x0, x1 = gx*w//3, (gx+1)*w//3
            tile = gray[y0:y1, x0:x1]
            tiles.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()))
    median_noise = float(np.median(tiles)) if tiles else 0.0
    deviations = [abs(x - median_noise) / max(median_noise, 1.0) for x in tiles]
    texture_outliers = sum(d > 1.8 for d in deviations)

    score = 0
    indicators = []
    if texture_outliers >= 2:
        score += 20
        indicators.append("Localized texture/noise inconsistency detected.")
    if lap_var > 3500 and edge_density > 0.20:
        score += 10
        indicators.append("Strong high-frequency detail detected; manual inspection is recommended.")
    if edge_density > 0.32:
        score += 5
        indicators.append("Unusually dense edge structure detected.")

    score = min(score, 45)
    status = "high" if score >= 30 else "medium" if score >= 15 else "low"
    if not indicators:
        indicators.append("No obvious visual tampering indicators detected.")

    return {
        "checked": True,
        "score": score,
        "status": status,
        "indicators": indicators,
        "method": "multi-region texture and edge analysis",
        "metrics": {
            "edge_density": round(edge_density, 4),
            "laplacian_variance": round(lap_var, 2),
            "texture_outlier_regions": texture_outliers,
        },
        "note": "Image-forensics screening is an indicator and does not establish authenticity by itself.",
    }


# ---------------------------------------------------------
# FACE PRESENCE
# ---------------------------------------------------------

def detect_face_presence(image: Image.Image) -> dict:
    """Detect portrait regions using OpenCV Haar cascades when available.

    Some OpenCV builds used in lightweight environments may not expose
    CascadeClassifier or Haar-cascade data. Face detection must never crash
    the complete document-screening request, so such cases are returned as
    an explicit manual-review state.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        return {
            "status": "not_available",
            "faces_detected": 0,
            "boxes": [],
            "message": (
                "Face detection is unavailable in the current OpenCV build. "
                "Continue document screening and flag for manual review."
            ),
        }

    if not hasattr(cv2, "data") or not hasattr(cv2.data, "haarcascades"):
        return {
            "status": "not_available",
            "faces_detected": 0,
            "boxes": [],
            "message": (
                "OpenCV Haar-cascade data is unavailable. "
                "Continue document screening and flag for manual review."
            ),
        }

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    alt = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )

    if cascade.empty() and alt.empty():
        return {
            "status": "not_available",
            "faces_detected": 0,
            "boxes": [],
            "message": (
                "OpenCV face-detection models could not be loaded. "
                "Continue document screening and flag for manual review."
            ),
        }

    h, w = gray.shape
    rois = [
        (0, 0, w, h),
        (int(w * 0.45), 0, int(w * 0.55), int(h * 0.70)),
        (0, 0, int(w * 0.55), int(h * 0.70)),
        (int(w * 0.20), int(h * 0.05), int(w * 0.80), int(h * 0.85)),
    ]

    boxes = []
    for x, y, rw, rh in rois:
        crop = gray[y:y + rh, x:x + rw]
        if crop.size == 0:
            continue

        scale = max(1.0, min(3.5, 1600 / max(crop.shape[1], 1)))
        sample = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        if hasattr(cv2, "createCLAHE"):
            sample = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(sample)

        for detector in (cascade, alt):
            if detector.empty():
                continue
            for neighbors in (5, 3):
                faces = detector.detectMultiScale(
                    sample,
                    1.08,
                    neighbors,
                    minSize=(28, 28),
                )
                if len(faces):
                    for fx, fy, fw, fh in faces:
                        boxes.append((
                            int(x + fx / scale),
                            int(y + fy / scale),
                            int(fw / scale),
                            int(fh / scale),
                        ))
                    break
            if boxes:
                break
        if boxes:
            break

    # De-duplicate overlapping detections.
    unique = []
    for box in boxes:
        x, y, bw, bh = box
        duplicate = False
        for ux, uy, uw, uh in unique:
            ix = max(0, min(x + bw, ux + uw) - max(x, ux))
            iy = max(0, min(y + bh, uy + uh) - max(y, uy))
            inter = ix * iy
            union = bw * bh + uw * uh - inter
            if union and inter / union > 0.35:
                duplicate = True
                break
        if not duplicate:
            unique.append(box)

    if not unique:
        return {
            "status": "not_detected",
            "faces_detected": 0,
            "boxes": [],
            "message": "No clear portrait region was detected.",
        }

    return {
        "status": "detected",
        "faces_detected": len(unique),
        "boxes": unique,
        "message": (
            "Portrait face region detected. Reference-face comparison can "
            "be performed when a second image is supplied."
        ),
    }


# ---------------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------------

def calculate_risk(
    document_type: dict,
    fields: dict,
    mrz: dict | None,
    consistency: list[dict],
    image_quality: dict,
    ocr: dict,
    tampering: dict,
    face: dict,
) -> dict:

    score = 0
    reasons = []
    checks = []

    # Document classification
    if document_type["type"] == "unknown":
        score += 40
        checks.append({
            "check": "document_type",
            "status": "warning",
            "message": (
                "Document type could not be confidently identified."
            ),
        })
        reasons.append(
            "Document type could not be confidently identified."
        )
    else:
        checks.append({
            "check": "document_type",
            "status": "passed",
            "message": (
                f"Document identified as "
                f"{document_type['type']}."
            ),
        })

    # Image quality is reported, not treated as fraud by itself.
    quality_status = (
        "passed"
        if image_quality["status"] == "good"
        else "warning"
    )

    checks.append({
        "check": "image_quality",
        "status": quality_status,
        "score": image_quality["score"],
        "issues": image_quality["issues"],
        "message": (
            "Image quality is sufficient for screening."
            if quality_status == "passed"
            else (
                "Image quality affects extraction reliability; "
                "it is not by itself evidence of fraud."
            )
        ),
    })

    # OCR confidence is extraction quality.
    ocr_status = "passed" if ocr["confidence"] >= 70 else "warning"

    checks.append({
        "check": "ocr_quality",
        "status": ocr_status,
        "confidence": ocr["confidence"],
        "message": (
            "OCR extraction quality is acceptable."
            if ocr_status == "passed"
            else (
                "OCR confidence is limited; extracted values "
                "should be reviewed."
            )
        ),
    })

    # Required fields.
    important_fields = [
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "date_of_expiry",
        "passport_number",
    ]

    missing = []

    for field in important_fields:
        visible_value = fields.get(field)
        mrz_value = mrz.get(field) if mrz else None

        if not visible_value and not mrz_value:
            missing.append(field)

    if missing:
        score += min(len(missing) * 4, 20)
        checks.append({
            "check": "required_fields",
            "status": "warning",
            "missing": missing,
            "message": (
                "Some expected fields could not be reliably extracted."
            ),
        })
        reasons.append(
            "Some expected fields could not be reliably extracted."
        )
    else:
        checks.append({
            "check": "required_fields",
            "status": "passed",
            "missing": [],
            "message": "All primary passport fields were extracted.",
        })

    # MRZ.
    if mrz is None:
        checks.append({
            "check": "mrz",
            "status": "not_detected",
            "message": "No machine-readable zone was detected.",
        })
        if document_type["type"] == "passport":
            reasons.append(
                "MRZ was not detected; additional verification may be required."
            )
    else:
        checks.append({
            "check": "mrz",
            "status": "detected",
            "message": "Machine-readable zone detected.",
        })

        checksums = mrz.get("checksums", {})
        invalid = [
            name
            for name, valid in checksums.items()
            if valid is False
        ]

        if invalid:
            critical = {
                "passport_number",
                "date_of_birth",
                "date_of_expiry",
                "composite",
            }

            critical_invalid = [
                item for item in invalid
                if item in critical
            ]

            if critical_invalid:
                score += 15
                checksum_message = (
                    "One or more critical MRZ checksums failed."
                )
                reasons.append(
                    "One or more critical MRZ checksums failed."
                )
            elif invalid == ["personal_number"]:
                score += 3
                checksum_message = (
                    "MRZ personal-number checksum could not be "
                    "validated; treated as a minor screening warning."
                )
                reasons.append(
                    "MRZ personal-number checksum could not be "
                    "validated; treated as a minor screening warning."
                )
            else:
                score += 5
                checksum_message = (
                    "One or more non-critical MRZ checksums "
                    "could not be validated."
                )
                reasons.append(
                    "One or more non-critical MRZ checksums "
                    "could not be validated."
                )

            checks.append({
                "check": "mrz_checksums",
                "status": "failed",
                "failed_fields": invalid,
                "message": checksum_message,
            })
        else:
            checks.append({
                "check": "mrz_checksums",
                "status": "passed",
                "message": "MRZ checksum validation passed.",
            })

    # Consistency.
    mismatches = [
        item for item in consistency
        if item["status"] == "mismatch"
    ]

    comparable_matches = [
        item for item in consistency
        if item["status"] == "match"
    ]

    if mismatches:
        score += min(len(mismatches) * 12, 36)
        checks.append({
            "check": "visible_mrz_consistency",
            "status": "failed",
            "mismatches": [item["field"] for item in mismatches],
            "message": "One or more visible fields differ from the MRZ.",
        })
        reasons.append(
            "Visible document fields do not fully agree with the MRZ."
        )
    elif comparable_matches:
        checks.append({
            "check": "visible_mrz_consistency",
            "status": "passed",
            "matches": len(comparable_matches),
            "message": (
                "Comparable visible fields are consistent with the MRZ."
            ),
        })
    else:
        checks.append({
            "check": "visible_mrz_consistency",
            "status": "not_comparable",
            "message": (
                "There was not enough reliable visible data "
                "for a full consistency comparison."
            ),
        })

    # Tampering screening.
    if tampering["status"] == "high":
        score += 25
        checks.append({
            "check": "tampering_screening",
            "status": "warning",
            "score": tampering["score"],
            "indicators": tampering["indicators"],
            "message": (
                "Multiple visual-forensics indicators require review."
            ),
        })
        reasons.append(
            "Visual-forensics screening found indicators requiring review."
        )
    elif tampering["status"] == "medium":
        score += 10
        checks.append({
            "check": "tampering_screening",
            "status": "warning",
            "score": tampering["score"],
            "indicators": tampering["indicators"],
            "message": (
                "Some visual-forensics indicators were detected."
            ),
        })
        reasons.append(
            "Some visual-forensics indicators were detected."
        )
    else:
        checks.append({
            "check": "tampering_screening",
            "status": "passed",
            "score": tampering["score"],
            "indicators": tampering["indicators"],
            "message": (
                "No obvious visual tampering indicators were detected."
            ),
        })

    # Expiry screening.
    expiry_value = (
        mrz.get("date_of_expiry")
        if mrz else fields.get("date_of_expiry")
    )
    expiry_iso = mrz_date_to_iso(expiry_value, "expiry")
    if expiry_iso and re.fullmatch(r"\d{4}-\d{2}-\d{2}", expiry_iso):
        try:
            expiry_date = datetime.strptime(expiry_iso, "%Y-%m-%d").date()
            if expiry_date < datetime.now().date():
                score += 25
                checks.append({
                    "check": "expiry",
                    "status": "failed",
                    "message": f"Document expiry date has passed ({expiry_iso}).",
                })
                reasons.append("Document appears to be expired based on the extracted expiry date.")
            else:
                checks.append({
                    "check": "expiry",
                    "status": "passed",
                    "message": f"Document expiry date is valid ({expiry_iso}).",
                })
        except ValueError:
            pass

    # Face presence.
    if face["status"] == "not_detected":
        checks.append({
            "check": "face_verification",
            "status": "not_available",
            "message": face["message"],
        })
    else:
        checks.append({
            "check": "face_verification",
            "status": "detected",
            "message": face["message"],
        })

    score = min(max(score, 0), 100)

    if score >= 60:
        level = "high"
    elif score >= 30:
        level = "medium"
    else:
        level = "low"

    if level == "low":
        summary = (
            "No major screening issues were detected. "
            "Further identity verification may still be required."
        )
    elif level == "medium":
        summary = (
            "Some screening indicators require additional "
            "verification before a final decision."
        )
    else:
        summary = (
            "Significant screening indicators were detected. "
            "Manual review is strongly recommended."
        )

    return {
        "score": score,
        "level": level,
        "summary": summary,
        "reasons": reasons,
        "checks": checks,
    }


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
):
    """Run the complete IDShield screening pipeline on one document image."""
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Upload JPG, PNG, WEBP or BMP.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 12 MB.")

    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.") from exc

    width, height = image.size
    if width < 250 or height < 180:
        raise HTTPException(
            status_code=400,
            detail="Image is too small for reliable document screening. Use a clearer scan or photo.",
        )
    if width > 7000 or height > 7000:
        image.thumbnail((7000, 7000), Image.Resampling.LANCZOS)

    image = image.convert("RGB")
    image, orientation_correction, orientation_method = auto_orient_document(image)
    image_quality = calculate_image_quality(image)
    ocr = run_ocr(image)

    document_type = detect_document_type(ocr["text"])

    # Hard gate: IDShield must never turn an unrecognized/random document
    # into a normal "low risk" identity result. Classification happens before
    # the expensive screening stages, and uncertain documents are returned as
    # unsupported for officer handling.
    if document_type["type"] == "unknown":
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "analysis": {
                "document": {
                    **document_type,
                    "supported": False,
                    "status": "unsupported",
                    "message": (
                        "The uploaded file could not be confidently identified "
                        "as a supported identity document. Analysis stopped."
                    ),
                },
                "image_quality": image_quality,
                "orientation": {
                    "correction_degrees": orientation_correction,
                    "method": orientation_method,
                },
                "ocr": ocr,
                "visible_fields": {},
                "mrz": {
                    "detected": False,
                    "lines": None,
                    "parsed": None,
                    "fallback_ocr_used": False,
                    "fallback_confidence": None,
                },
                "consistency": [],
                "tampering": {"status": "not_run", "notes": ["Not run because document type is unsupported."]},
                "face_verification": {"status": "not_run", "message": "Not run because document type is unsupported."},
                "risk": {
                    "score": None,
                    "level": "unsupported",
                    "reasons": [
                        "Document type could not be confidently identified."
                    ],
                    "checks": [
                        {
                            "check": "document_type",
                            "status": "blocked",
                            "message": "Unsupported or unrecognized document.",
                        }
                    ],
                },
                "verification": {
                    "decision": "unsupported_document",
                    "note": "Analysis stopped before identity-document screening.",
                },
            },
        }

    visible_fields = extract_visible_fields(ocr["text"], document_type["type"])

    mrz_lines = find_mrz(ocr["text"]) if document_type["type"] == "passport" else None
    mrz_ocr = None

    if document_type["type"] == "passport" and mrz_lines is None:
        mrz_ocr = run_mrz_ocr(image)
        mrz_lines = find_mrz(mrz_ocr["text"])

    mrz_data = parse_passport_mrz(mrz_lines) if document_type["type"] == "passport" else None
    visible_fields = enrich_fields_from_mrz(visible_fields, mrz_data)

    if mrz_data:
        mrz_data = dict(mrz_data)
        mrz_data["date_of_birth_display"] = mrz_date_to_iso(
            mrz_data.get("date_of_birth"), "dob"
        )
        mrz_data["date_of_expiry_display"] = mrz_date_to_iso(
            mrz_data.get("date_of_expiry"), "expiry"
        )

    consistency = compare_fields(visible_fields, mrz_data)
    tampering = analyze_tampering(image)
    face = detect_face_presence(image)

    risk = calculate_risk(
        document_type=document_type,
        fields=visible_fields,
        mrz=mrz_data,
        consistency=consistency,
        image_quality=image_quality,
        ocr=ocr,
        tampering=tampering,
        face=face,
    )

    decision = "manual_review" if risk["level"] in {"medium", "high"} else "screened_low_risk"

    document_type = {**document_type, "supported": True, "status": "supported"}

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "analysis": {
            "document": document_type,
            "image_quality": image_quality,
            "orientation": {
                "correction_degrees": orientation_correction,
                "method": orientation_method,
            },
            "ocr": ocr,
            "visible_fields": visible_fields,
            "mrz": {
                "detected": mrz_lines is not None,
                "lines": mrz_lines,
                "parsed": mrz_data,
                "fallback_ocr_used": mrz_ocr is not None,
                "fallback_confidence": mrz_ocr["confidence"] if mrz_ocr else None,
            },
            "consistency": consistency,
            "tampering": tampering,
            "face_verification": face,
            "risk": risk,
            "verification": {
                "decision": decision,
                "note": (
                    "Screening indicators support officer review; they do not establish "
                    "legal document authenticity by themselves."
                ),
            },
        },
    }


