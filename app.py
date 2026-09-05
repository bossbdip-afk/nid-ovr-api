from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

APP_VERSION = "14.6.14-admin-login"
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(15 * 1024 * 1024)))
OCR_LANG = os.getenv("OCR_LANG", "ben+eng")
OCR_SCALE = float(os.getenv("OCR_SCALE", "2.2"))
OCR_MODE = os.getenv("OCR_MODE", "auto").strip().lower()  # auto | always | off
EXTRACT_TIMEOUT_SECONDS = float(os.getenv("EXTRACT_TIMEOUT_SECONDS", "120"))
MAX_CONCURRENT_EXTRACTS = max(1, int(os.getenv("MAX_CONCURRENT_EXTRACTS", "1")))
EXTRACT_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)

TEMPLATE_DATA_DIR = Path(os.getenv("TEMPLATE_DATA_DIR", "./data")).resolve()
TEMPLATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_PDF_PATH = TEMPLATE_DATA_DIR / "card-template.pdf"
TEMPLATE_MAPPING_PATH = TEMPLATE_DATA_DIR / "card-mapping.json"
ADMIN_TEMPLATE_KEY = os.getenv("ADMIN_TEMPLATE_KEY", "").strip()
MAX_TEMPLATE_BYTES = int(os.getenv("MAX_TEMPLATE_BYTES", str(10 * 1024 * 1024)))

def require_template_admin(request: Request) -> None:
    if not ADMIN_TEMPLATE_KEY:
        raise HTTPException(status_code=503, detail="Admin login Render-এ configure করা হয়নি।")
    if request.headers.get("x-admin-key", "") != ADMIN_TEMPLATE_KEY:
        raise HTTPException(status_code=401, detail="Admin key সঠিক নয়।")


DEFAULT_ORIGINS = [
    "https://sbtechinfo.liveblog365.com",
    "http://sbtechinfo.liveblog365.com",
    "https://singtonid.liveblog365.com",
    "http://singtonid.liveblog365.com",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

# Always keep the known frontend origins enabled, even when Render has an
# older ALLOWED_ORIGINS environment variable configured. Extra origins can
# still be added through ALLOWED_ORIGINS as a comma-separated list.
env_origins = [
    x.strip()
    for x in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if x.strip()
]
origins = list(dict.fromkeys(DEFAULT_ORIGINS + env_origins))

app = FastAPI(title="NID Bengali Field OCR API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ------------------------- Unicode / Bengali helpers -------------------------
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
BENGALI_MARKS = set("ঁংঃ়ািীুূৃৄেৈোৌ্ৗ")
TRIVIAL_PUNCT_RE = re.compile(r"^[\s,.;:|/\\\-–—_()\[\]{}]+$")


def _canon_bengali_forms(s: str) -> str:
    # Tesseract often emits decomposed nukta forms. Convert them to the single
    # Bengali code points used by the source PDFs and by the existing UI.
    return (
        s.replace("য়", "য়")
        .replace("ড়", "ড়")
        .replace("ঢ়", "ঢ়")
    )


def clean_text(value: Any) -> str:
    s = unicodedata.normalize("NFC", str(value or ""))
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = _canon_bengali_forms(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()


def clean_bengali(value: Any, *, allow_ascii: bool = True) -> str:
    s = clean_text(value)

    # PDF text maps in this NID family sometimes duplicate the visarga in the
    # honorific or expose a split pre-base vowel. These are structural glyph
    # repairs, not name dictionary replacements.
    honorific_rules = [
        (r"^ম(?:ে|ো)া?ঃ{2,}", "মোঃ"),
        (r"^মোঃ{2,}", "মোঃ"),
        (r"^ম(?:ে|ো)া?ছাঃ", "মোছাঃ"),
        (r"^মো\s+ছাঃ", "মোছাঃ"),
        (r"^মা\s*ছাঃ", "মোছাঃ"),
        # Some source PDFs split the female honorific across a text-map
        # boundary even though the same cell visually reads "মোসাঃ".
        # This repairs spacing only; it does not replace any person-name word.
        (r"^মো\s+সাঃ(?=\s|$)", "মোসাঃ"),
    ]
    for pat, repl in honorific_rules:
        s = re.sub(pat, repl, s)

    # A combining mark must stay attached to the preceding Bengali base.
    s = re.sub(r"\s+([ঁংঃ়ািীুূৃৄেৈোৌ্ৗ])", r"\1", s)

    # Remove obvious text-map garbage while preserving Bengali, Latin letters,
    # digits, and the punctuation used in NID values/addresses.
    if BENGALI_RE.search(s):
        if allow_ascii:
            s = re.sub(r"[^\u0980-\u09FFA-Za-z0-9\s()\-–—,./:+]", " ", s)
        else:
            s = re.sub(r"[^\u0980-\u09FF0-9\s()\-–—,./:+]", " ", s)

    s = re.sub(r"\s+", " ", s).strip(" ,;|/")
    s = re.sub(r"ঃ{2,}", "ঃ", s)

    # Re-attach a lone Bengali glyph fragment to the preceding Bengali token.
    # Broken PDF text maps commonly detach the final consonant (e.g. "আছফু ল").
    # Keep the real standalone conjunction "ও" untouched.
    s = re.sub(r"([\u0980-\u09FF]{2,})\s+([\u0980-\u09FF])(?![\u0980-\u09FF])",
               lambda m: m.group(0) if m.group(2) == "ও" else m.group(1) + m.group(2), s)
    return s.strip()


def meaningful(value: Any) -> bool:
    s = clean_text(value)
    return bool(s and not TRIVIAL_PUNCT_RE.match(s))


def to_bengali_digits(value: Any) -> str:
    """Convert ASCII digits to Bengali digits for display only.

    Source extraction/debug values remain unchanged so field provenance and
    original PDF data can still be inspected exactly as parsed.
    """
    return clean_text(value).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def compact_for_compare(s: str) -> str:
    s = clean_bengali(s).lower()
    return re.sub(r"[\s,.;:()\-–—_/]+", "", s)


def obvious_damage(s: str) -> int:
    s0 = clean_text(s)
    score = 0
    if re.search(r"ঃ{2,}", s0):
        score += 5
    # Malformed honorific produced by glyph-order extraction (e.g. মোঃাঃ).
    if re.search(r"মোঃ[ািীুূৃেৈোৌ]+ঃ", s0):
        score += 15
    if re.search(r"\s+[ঁংঃ়ািীুূৃৄেৈোৌ্ৗ]", s0):
        score += 5
    # Same-cell visual extraction can occasionally duplicate a Bengali mark
    # (for example, াা / ীী / ংং). Treat that candidate as damaged.
    if re.search(r"([ঁংঃ়ািীুূৃৄেৈোৌ্ৗ])\1", s0):
        score += 5
    if re.search(r"[^\u0980-\u09FFA-Za-z0-9\s()\-–—,./:+]", s0):
        score += 4
    # One-character Bengali fragments between otherwise Bengali tokens are a
    # strong sign of a broken visual/text ordering.
    toks = s0.split()
    for t in toks:
        if len(t) == 1 and BENGALI_RE.search(t) and t not in {"ও"}:
            score += 2
    return score


def exact_cell_text(page: fitz.Page, cell) -> str:
    """Rebuild text from the exact PDF glyphs inside one table cell.

    PyMuPDF's higher-level text/table extraction can invent spaces between
    Bengali glyph runs. Here we preserve only spaces that physically exist as
    space glyphs in the source PDF, and use one space only between visual lines.
    This is generic and field/PDF independent.
    """
    if not cell:
        return ""
    try:
        rawdict = page.get_text("rawdict", clip=fitz.Rect(cell))
    except Exception:
        return ""
    lines: list[str] = []
    for block in rawdict.get("blocks", []):
        for line in block.get("lines", []):
            chars: list[str] = []
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    chars.append(str(ch.get("c", "")))
            line_text = "".join(chars).strip()
            if line_text:
                lines.append(line_text)
    text = clean_text(" ".join(lines))
    # Some Bengali PDF fonts expose the same combining mark twice at the same
    # visual position. Collapse only adjacent duplicate combining marks; this
    # does not merge separate words or alter base letters.
    text = re.sub(r"([ঁংঃ়ািীুূৃৄেৈোৌ্ৗ])\1+", r"\1", text)
    return text


def choose_visual_cell_text(raw: str, visual: str) -> tuple[str, str] | None:
    """Safely repair Bengali address spacing/order from the same PDF cell.

    PyMuPDF table extraction can split or reorder Bengali glyphs while the
    page text in the exact same cell remains correct. No dictionary or
    PDF-specific replacement is used here.
    """
    raw_clean = clean_bengali(raw)
    visual_clean = clean_bengali(visual)
    if not raw_clean or not visual_clean:
        return None

    rc = compact_for_compare(raw_clean)
    vc = compact_for_compare(visual_clean)
    if not rc or not vc:
        return None

    # If both candidates contain the same characters and differ only by
    # whitespace, prefer the exact-cell glyph reconstruction. Table extraction
    # can invent spaces inside Bengali words (e.g. "মৌ জা"), whereas the raw
    # glyph stream preserves only real source spaces.
    if rc == vc and raw_clean != visual_clean:
        return visual_clean, "pdf_cell_explicit_spacing"

    raw_damage = obvious_damage(raw)
    visual_damage = obvious_damage(visual)
    sim = similarity(raw_clean, visual_clean)

    # Repair detached marks / broken fragments only if the visual text is
    # clearly cleaner and still represents the same source value.
    if raw_damage > visual_damage and sim >= 0.55:
        return visual_clean, "pdf_cell_visual_repair"

    # Repair pure character-order swaps only when both same-cell candidates
    # contain exactly the same characters and the visual candidate is clean.
    if (
        visual_damage == 0
        and sim >= 0.72
        and len(rc) == len(vc)
        and sorted(rc) == sorted(vc)
    ):
        return visual_clean, "pdf_cell_visual_order"

    return None

def only_extra_marks(raw: str, ocr: str) -> bool:
    """True when OCR merely dropped one Bengali mark that the PDF text kept."""
    r = compact_for_compare(raw)
    o = compact_for_compare(ocr)
    if len(r) != len(o) + 1:
        return False
    for i in range(len(r)):
        candidate = r[:i] + r[i + 1 :]
        if candidate == o and r[i] in BENGALI_MARKS:
            return True
    return False


def similarity(a: str, b: str) -> float:
    a = compact_for_compare(a)
    b = compact_for_compare(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class OcrCandidate:
    text: str = ""
    confidence: float = 0.0


_TESSERACT_AVAILABLE = bool(pytesseract is not None and shutil.which("tesseract"))

def tesseract_available() -> bool:
    return _TESSERACT_AVAILABLE

def should_run_ocr(raw: Any) -> bool:
    """Fast-path policy for OCR.

    Tesseract startup is expensive on Render's small instances. The old code
    launched it twice for every Bengali cell even when the PDF text layer was
    already clean, which could turn one request into 20+ OCR subprocesses.
    In auto mode we trust clean embedded PDF text and OCR only missing or
    visibly damaged Bengali values. Set OCR_MODE=always to restore v14.0
    behavior, or OCR_MODE=off to disable OCR entirely.
    """
    if OCR_MODE == "off":
        return False
    if OCR_MODE == "always":
        return True
    raw_s = clean_text(raw)
    if not meaningful(raw_s):
        return True
    if not BENGALI_RE.search(raw_s):
        return False
    return obvious_damage(raw_s) > 0


def _ocr_image(img: Image.Image, psm: int) -> OcrCandidate:
    if not tesseract_available():
        return OcrCandidate()
    try:
        data = pytesseract.image_to_data(
            img,
            lang=OCR_LANG,
            config=f"--oem 1 --psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return OcrCandidate()

    words: list[str] = []
    confs: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = clean_text(text)
        try:
            c = float(conf)
        except Exception:
            c = -1.0
        if text and c >= 0:
            words.append(text)
            confs.append(c)
    if not words:
        return OcrCandidate()
    return OcrCandidate(clean_bengali(" ".join(words)), sum(confs) / len(confs))


def ocr_cell(page: fitz.Page, bbox: Any, *, allow_ascii: bool = True) -> OcrCandidate:
    if not bbox or not tesseract_available():
        return OcrCandidate()
    rect = fitz.Rect(bbox)
    # Avoid table borders; a tiny inset gives Tesseract a cleaner line.
    inset_x = min(1.5, rect.width * 0.02)
    inset_y = min(1.0, rect.height * 0.05)
    rect = fitz.Rect(rect.x0 + inset_x, rect.y0 + inset_y, rect.x1 - inset_x, rect.y1 - inset_y)
    if rect.width <= 2 or rect.height <= 2:
        return OcrCandidate()

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(OCR_SCALE, OCR_SCALE), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception:
        return OcrCandidate()

    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    # Add white padding so glyphs touching a cell edge are not clipped.
    gray = ImageOps.expand(gray, border=max(5, int(gray.height * 0.12)), fill=255)

    # PSM 7 is the normal one-line cell path. A second Tesseract subprocess
    # is only worth paying for when the first pass is weak.
    first = _ocr_image(gray, 7)
    candidates = [first]
    if not first.text or first.confidence < 52:
        candidates.append(_ocr_image(gray, 6))
    best = max(candidates, key=lambda c: (c.confidence, len(c.text)))
    best.text = clean_bengali(best.text, allow_ascii=allow_ascii)
    return best


def choose_text(raw: str, ocr: OcrCandidate) -> tuple[str, str, float]:
    raw_clean = clean_bengali(raw)
    ocr_clean = clean_bengali(ocr.text)
    if not raw_clean:
        return ocr_clean, "ocr", ocr.confidence
    if not ocr_clean or ocr.confidence < 42:
        return raw_clean, "pdf_text", 100.0
    if compact_for_compare(raw_clean) == compact_for_compare(ocr_clean):
        return raw_clean, "pdf_text+ocr_agree", max(ocr.confidence, 90.0)

    # Protect chandrabindu / vowel marks when OCR simply drops one mark from an
    # otherwise clean source value (e.g. খাঁ -> খা, সাজেদা -> সাজেদ).
    if only_extra_marks(raw_clean, ocr_clean) and obvious_damage(raw_clean) == 0:
        return raw_clean, "pdf_text_preserve_mark", 90.0

    sim = similarity(raw_clean, ocr_clean)
    raw_damage = obvious_damage(raw)

    # If the text layer is visibly broken, visual OCR wins when it remains
    # recognizably close to the same field.
    if raw_damage > 0 and ocr.confidence >= 45 and sim >= 0.55:
        return ocr_clean, "ocr_repaired_damaged_text", ocr.confidence

    # The most common remaining corruption is character order inside a word
    # (খাতনু/খাতুন, আমিরলু/আমিরুল, মেলানহ্দ/মেলান্দহ). Prefer OCR only when
    # both candidates contain the same characters in a different order. This
    # avoids changing a perfectly valid source word merely because OCR confused
    # one glyph (e.g. টংগের -> উংগের).
    rc = compact_for_compare(raw_clean)
    oc = compact_for_compare(ocr_clean)
    if ocr.confidence >= 58 and sim >= 0.72 and len(rc) == len(oc) and sorted(rc) == sorted(oc):
        return ocr_clean, "ocr_visual_order", ocr.confidence

    return raw_clean, "pdf_text", 100.0




# Central field-name registry. Extraction is driven by source labels rather than
# PDF filename/tag/layout version. Aliases here represent label spelling variants
# only; values are never borrowed from another semantic field.
FIELD_LABELS: dict[str, list[str]] = {
    "nid": ["National ID"],
    "pin": ["Pin"],
    "siNo": ["Sl No", "SI No", "SL No", "Serial No", "Serial Number", "S/L No"],
    "voterNo": ["Voter No"],
    "nameBn": ["Name(Bangla)", "Name (Bangla)"],
    "nameEn": ["Name(English)", "Name (English)"],
    "dob": ["Date of Birth"],
    "birthPlace": ["Birth Place"],
    "bloodGroup": ["Blood Group", "Blood group"],
    "father": ["Father Name"],
    "mother": ["Mother Name"],
    "spouse": ["Spouse Name"],
    "gender": ["Gender"],
    "religion": ["Religion"],
    "voterArea": ["Voter Area"],
}

ADDRESS_FIELD_LABELS: dict[str, list[str]] = {
    "rmo": ["RMO"],
    "municipality": ["City Corporation Or Municipality", "City Corporation/Or Municipality", "City Corporation / Municipality", "Municipality"],
    "unionWard": ["Union/Ward", "Union Ward"],
    "mouzaMoholla": ["Mouza/Moholla", "Mouza Moholla"],
    "additionalMouzaMoholla": ["Additional Mouza/Moholla", "Additional Mouza Moholla"],
    "additionalVillageRoad": ["Additional Village/Road", "Additional Village Road"],
    "villageRoad": ["Village/Road", "Village Road"],
    "holding": ["Home/Holding No", "Home/Holding", "Home Holding No"],
    "postOffice": ["Post Office", "Post Ofcfie"],
    "postalCode": ["Postal Code", "Post Code"],
    "upazila": ["Upozila", "Upazila"],
    "district": ["District"],
    "region": ["Region"],
}

# ------------------------------ Table helpers ------------------------------

def norm_label(value: Any) -> str:
    s = clean_text(value).lower()
    s = s.replace("\n", " ")
    s = re.sub(r"[^a-z0-9/()]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def label_match(actual: Any, wanted: str, threshold: float = 0.78) -> bool:
    a = norm_label(actual)
    w = norm_label(wanted)
    if not a or not w:
        return False
    if a == w:
        return True
    return SequenceMatcher(None, a, w).ratio() >= threshold


def get_table(page: fitz.Page):
    try:
        tables = page.find_tables().tables
    except Exception:
        tables = []
    if not tables:
        return None
    # NID PDFs in this family have one dominant form table per page.
    return max(tables, key=lambda t: (t.row_count * max(1, t.col_count), (fitz.Rect(t.bbox).width * fitz.Rect(t.bbox).height)))


def get_tables(page: fitz.Page) -> list[Any]:
    """Return every detected table on a page, largest first.

    Field extraction must not assume that the wanted label lives in one
    dominant table. Keeping all tables lets the same source field be found in
    different PDF structures without document-specific branches.
    """
    try:
        tables = list(page.find_tables().tables)
    except Exception:
        tables = []
    return sorted(
        tables,
        key=lambda t: (t.row_count * max(1, t.col_count), fitz.Rect(t.bbox).width * fitz.Rect(t.bbox).height),
        reverse=True,
    )


def collect_document_tables(doc: fitz.Document) -> list[tuple[fitz.Page, Any, list[list[Any]]]]:
    out: list[tuple[fitz.Page, Any, list[list[Any]]]] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        for table in get_tables(page):
            try:
                data = table.extract()
            except Exception:
                continue
            out.append((page, table, data))
    return out


def simple_value_any_table(
    tables: list[tuple[fitz.Page, Any, list[list[Any]]]],
    labels: Iterable[str],
    *,
    ocr_bengali: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Find a source field by label across all pages/tables."""
    for page, table, data in tables:
        found = find_simple_row(table, labels, data=data)
        if not found:
            continue
        value, meta = simple_value(page, table, labels, ocr_bengali=ocr_bengali, data=data)
        if meaningful(value):
            meta = dict(meta)
            meta["page"] = page.number + 1
            return value, meta
    return "", {"source": "missing"}


def find_address_table(
    tables: list[tuple[fitz.Page, Any, list[list[Any]]]], section: str
) -> tuple[fitz.Page, Any, list[list[Any]]] | None:
    """Locate the table that actually contains an address section label."""
    for page, table, data in tables:
        for row in data:
            if _row_label_col(row, section, 0.84) is not None:
                return page, table, data
    return None


def find_simple_row(table, labels: Iterable[str], data=None):
    """Find a label in any table column and return (row_index, row, label_col).

    Older PDFs put labels in column 0; the supplied ``old`` PDF has a leading
    blank column and puts the same labels in column 1.  Searching every cell
    keeps both layouts working without a document-specific offset.
    """
    if table is None:
        return None
    if data is None:
        data = table.extract()
    wanted = [norm_label(x) for x in labels]
    for ri, row in enumerate(data):
        if not row:
            continue
        for ci, cell in enumerate(row):
            actual = norm_label(cell)
            if actual and actual in wanted:
                return ri, row, ci
    for ri, row in enumerate(data):
        if not row:
            continue
        for ci, cell in enumerate(row):
            if not norm_label(cell):
                continue
            for label in labels:
                if label_match(cell, label, 0.90):
                    return ri, row, ci
    return None

def simple_value(page: fitz.Page, table, labels: Iterable[str], *, ocr_bengali: bool = False, data=None) -> tuple[str, dict[str, Any]]:
    found = find_simple_row(table, labels, data=data)
    if not found:
        return "", {"source": "missing"}
    ri, row, li = found
    vi = li + 1
    raw = row[vi] if vi < len(row) and row[vi] is not None else ""
    raw_clean = clean_bengali(raw) if BENGALI_RE.search(str(raw or "")) else clean_text(raw)
    if not ocr_bengali or not BENGALI_RE.search(str(raw or "")):
        return raw_clean, {"source": "pdf_table"}

    # Bengali identity fields can suffer the same table-text spacing/order
    # damage as address cells. Prefer the native text from the exact same PDF
    # cell only when the conservative visual comparator proves it is cleaner.
    # This is field-agnostic and does not use name/place dictionaries.
    cell = table.rows[ri].cells[vi] if vi < len(table.rows[ri].cells) else None
    visual_text = ""
    if cell:
        try:
            visual_text = page.get_text("text", clip=fitz.Rect(cell)).strip()
        except Exception:
            visual_text = ""
    visual_choice = choose_visual_cell_text(str(raw or ""), visual_text)
    if visual_choice is not None:
        value, source = visual_choice
        return value, {
            "source": source,
            "raw": clean_text(raw),
            "visual": clean_text(visual_text),
        }

    if not should_run_ocr(raw):
        return raw_clean, {"source": "pdf_table_fast"}
    ocr = ocr_cell(page, cell)
    value, source, conf = choose_text(str(raw or ""), ocr)
    return value, {"source": source, "ocr_confidence": round(conf, 1), "raw": clean_text(raw), "ocr": ocr.text}

def _row_label_col(row, wanted: str, threshold: float = 0.84):
    if not row:
        return None
    for ci, cell in enumerate(row):
        if label_match(cell, wanted, threshold):
            return ci
    return None


def address_rows(table, section: str, next_section: str | None, data=None):
    if table is None:
        return []
    if data is None:
        data = table.extract()
    start = None
    end = len(data)
    for i, row in enumerate(data):
        if _row_label_col(row, section, 0.84) is not None:
            start = i
            break
    if start is None:
        return []
    if next_section:
        for i in range(start + 1, len(data)):
            if _row_label_col(data[i], next_section, 0.84) is not None:
                end = i
                break
    return list(range(start, end))


def address_field(page: fitz.Page, table, row_indices: list[int], labels: list[str], *, ocr_bengali: bool = True, data=None) -> tuple[str, dict[str, Any]]:
    if data is None:
        data = table.extract() if table else []
    wanted = [norm_label(x) for x in labels]
    candidates = []
    for ri in row_indices:
        row = data[ri]
        # Search every cell for a label and use the immediately following cell
        # as its value. This supports both 5-column and leading-blank 7-column forms.
        for li, actual in enumerate(row):
            an = norm_label(actual)
            if not an:
                continue
            exact = an in wanted
            fuzzy = any(label_match(actual, label, 0.90) for label in labels)
            if exact or fuzzy:
                vi = li + 1
                candidates.append((0 if exact else 1, ri, li, vi, row))
    candidates.sort(key=lambda x: x[0])
    for _, ri, li, vi, row in candidates:
        raw = row[vi] if vi < len(row) and row[vi] is not None else ""
        if not meaningful(raw):
            continue
        raw_clean = clean_bengali(raw)
        if not ocr_bengali or not BENGALI_RE.search(str(raw or "")):
            return raw_clean, {"source": "pdf_table"}

        cell = table.rows[ri].cells[vi] if vi < len(table.rows[ri].cells) else None
        visual_text = ""
        if cell:
            try:
                visual_text = exact_cell_text(page, cell)
            except Exception:
                visual_text = ""

        visual_choice = choose_visual_cell_text(str(raw or ""), visual_text)
        if visual_choice is not None:
            value, source = visual_choice
            return value, {
                "source": source,
                "raw": clean_text(raw),
                "visual": clean_text(visual_text),
            }

        if not should_run_ocr(raw):
            return raw_clean, {"source": "pdf_table_fast"}
        ocr = ocr_cell(page, cell)
        value, source, conf = choose_text(str(raw or ""), ocr)
        return value, {"source": source, "ocr_confidence": round(conf, 1), "raw": clean_text(raw), "ocr": ocr.text}
    return "", {"source": "missing"}

def build_address(page: fitz.Page, table, section: str, next_section: str | None, data=None, *, bengali_postal_digits: bool = False) -> tuple[str, dict[str, Any]]:
    """Build the compact card address without dropping source values.

    Card output follows the compact reference-card format:
    Village/Road -> Post Office + Postal Code -> Upazila -> District.
    Other source address pieces are preserved in debug metadata only.
    Blank values are never invented.
    """
    rows = address_rows(table, section, next_section, data=data)
    if not rows:
        return "", {"source": "missing"}

    rmo, m_rmo = address_field(page, table, rows, ADDRESS_FIELD_LABELS["rmo"], data=data)
    municipality, m_muni = address_field(page, table, rows, ADDRESS_FIELD_LABELS["municipality"], data=data)
    union_ward, m_union = address_field(page, table, rows, ADDRESS_FIELD_LABELS["unionWard"], data=data)
    mouza, m_mouza = address_field(page, table, rows, ADDRESS_FIELD_LABELS["mouzaMoholla"], data=data)
    additional_mouza, m_add_mouza = address_field(page, table, rows, ADDRESS_FIELD_LABELS["additionalMouzaMoholla"], data=data)
    additional_village, m_add = address_field(page, table, rows, ADDRESS_FIELD_LABELS["additionalVillageRoad"], data=data)
    village, m_vil = address_field(page, table, rows, ADDRESS_FIELD_LABELS["villageRoad"], data=data)
    holding, m_hold = address_field(page, table, rows, ADDRESS_FIELD_LABELS["holding"], ocr_bengali=False, data=data)
    post, m_post = address_field(page, table, rows, ADDRESS_FIELD_LABELS["postOffice"], data=data)
    postal, m_postal = address_field(page, table, rows, ADDRESS_FIELD_LABELS["postalCode"], ocr_bengali=False, data=data)
    upazila, m_up = address_field(page, table, rows, ADDRESS_FIELD_LABELS["upazila"], data=data)
    district, m_dist = address_field(page, table, rows, ADDRESS_FIELD_LABELS["district"], data=data)
    region, m_region = address_field(page, table, rows, ADDRESS_FIELD_LABELS["region"], data=data)

    # Locality fallback is used only when the primary Village/Road field is blank.
    # Keep every source field distinct: a Mouza/Moholla value must never be
    # reported or stored as if it came from Village/Road.
    locality_value = ""
    locality_source = ""
    locality_label = ""
    locality_fallback = False
    locality_candidates = [
        ("villageRoad", "গ্রাম/রাস্তা", village, False),
        ("additionalVillageRoad", "অতিরিক্ত গ্রাম/রাস্তা", additional_village, True),
        ("mouzaMoholla", "মৌজা/মহল্লা", mouza, True),
        ("additionalMouzaMoholla", "অতিরিক্ত মৌজা/মহল্লা", additional_mouza, True),
        ("municipality", "পৌরসভা/সিটি", municipality, True),
    ]
    for source_name, display_label, candidate, is_fallback in locality_candidates:
        if meaningful(candidate):
            locality_value = candidate
            locality_source = source_name
            locality_label = display_label
            locality_fallback = is_fallback
            break

    parts: list[str] = []
    postal_display = to_bengali_digits(postal) if bengali_postal_digits else clean_text(postal)

    if meaningful(locality_value):
        # Card display keeps the legacy compact address label regardless of
        # which source field supplied the fallback value. Source provenance
        # remains available below in localitySelection/sourceFields.
        parts.append(f"গ্রাম/রাস্তা: {clean_bengali(locality_value)}")

    if meaningful(post):
        post_part = f"ডাকঘর: {clean_bengali(post)}"
        if meaningful(postal):
            post_part += f" - {postal_display}"
        parts.append(post_part)
    elif meaningful(postal):
        # Do not mistake Postal Code for Post Office when the source value is blank.
        parts.append(f"পোস্ট কোড: {postal_display}")

    if meaningful(upazila):
        parts.append(clean_bengali(upazila))
    if meaningful(district):
        parts.append(clean_bengali(district))

    meta = {
        "source": "pdf_table_compact",
        "rmo": m_rmo,
        "municipality": m_muni,
        "unionWard": m_union,
        # Keep legacy debug keys source-true instead of collapsing fallback fields.
        "mouza": m_mouza,
        "village": m_vil,
        "holding": m_hold,
        "post": m_post,
        "postal": m_postal,
        "upazila": m_up,
        "district": m_dist,
        "region": m_region,
        # Preserve exact field-to-field extraction for diagnostics.
        "sourceFields": {
            "villageRoad": clean_bengali(village) if meaningful(village) else "",
            "additionalVillageRoad": clean_bengali(additional_village) if meaningful(additional_village) else "",
            "mouzaMoholla": clean_bengali(mouza) if meaningful(mouza) else "",
            "additionalMouzaMoholla": clean_bengali(additional_mouza) if meaningful(additional_mouza) else "",
            "postOffice": clean_bengali(post) if meaningful(post) else "",
            "postalCode": clean_text(postal) if meaningful(postal) else "",
            "upazila": clean_bengali(upazila) if meaningful(upazila) else "",
            "district": clean_bengali(district) if meaningful(district) else "",
            "holding": clean_bengali(holding) if meaningful(holding) else "",
            "unionWard": clean_bengali(union_ward) if meaningful(union_ward) else "",
            "municipality": clean_bengali(municipality) if meaningful(municipality) else "",
            "region": clean_bengali(region) if meaningful(region) else "",
            "rmo": clean_bengali(rmo) if meaningful(rmo) else "",
        },
        "localitySelection": {
            "sourceField": locality_source,
            "value": clean_bengali(locality_value) if meaningful(locality_value) else "",
            "displayLabel": "গ্রাম/রাস্তা" if meaningful(locality_value) else "",
            "sourceLabel": locality_label,
            "fallbackUsed": locality_fallback,
        },
        "components": {
            "villageRoad": clean_bengali(village) if meaningful(village) else "",
            "additionalVillageRoad": clean_bengali(additional_village) if meaningful(additional_village) else "",
            "mouzaMoholla": clean_bengali(mouza) if meaningful(mouza) else "",
            "additionalMouzaMoholla": clean_bengali(additional_mouza) if meaningful(additional_mouza) else "",
            "postOffice": clean_bengali(post) if meaningful(post) else "",
            "postalCode": clean_text(postal) if meaningful(postal) else "",
            "upazila": clean_bengali(upazila) if meaningful(upazila) else "",
            "district": clean_bengali(district) if meaningful(district) else "",
            "holding": clean_bengali(holding) if meaningful(holding) else "",
            "unionWard": clean_bengali(union_ward) if meaningful(union_ward) else "",
            "municipality": clean_bengali(municipality) if meaningful(municipality) else "",
            "region": clean_bengali(region) if meaningful(region) else "",
            "rmo": clean_bengali(rmo) if meaningful(rmo) else "",
        },
    }
    return ", ".join(parts), meta


def _pixmap_png_data_url(doc: fitz.Document, xref: int, smask: int = 0) -> str:
    """Return a PNG data URL for a PDF image, rebuilding its soft mask when present."""
    try:
        pix = fitz.Pixmap(doc, xref)
        if smask:
            mask = fitz.Pixmap(doc, smask)
            pix = fitz.Pixmap(pix, mask)
        # Keep alpha, but normalize non-RGB color spaces so browsers render it reliably.
        if pix.colorspace is not None and pix.colorspace.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        png = pix.tobytes("png")
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    except Exception:
        return ""


def extract_cardholder_signature(doc: fitz.Document, page: fitz.Page) -> tuple[str, dict[str, Any]]:
    """Extract the card-holder signature from page 1 without fixed page crops.

    The supplied PDF family places a portrait image and a separate wide signature
    image on the same side of the page. The signature may use an /SMask, so it is
    reconstructed from the PDF XObject + mask. Detection is position-based relative
    to the actual portrait rectangle, not hard-coded left/right page coordinates.
    """
    items: list[dict[str, Any]] = []
    for raw in page.get_images(full=True):
        xref, smask, w, h = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
        if w <= 0 or h <= 0:
            continue
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            if rect.is_empty or rect.is_infinite or rect.width <= 0 or rect.height <= 0:
                continue
            items.append({
                "xref": xref,
                "smask": smask,
                "w": w,
                "h": h,
                "rect": rect,
                "pixel_ratio": w / max(h, 1),
                "placed_ratio": rect.width / max(rect.height, 0.01),
            })

    # Portrait: tall image, reasonably sized, and in the upper part of page.
    # Do not assume it is on the left: the supplied NIDFN PDFs place it on the right.
    portraits = [
        it for it in items
        if it["h"] > it["w"] * 1.08
        and it["rect"].height >= 45
        and it["rect"].width >= 30
        and it["rect"].y0 < page.rect.height * 0.55
        and it["rect"].get_area() < page.rect.get_area() * 0.20
    ]
    portraits.sort(key=lambda it: it["rect"].get_area(), reverse=True)
    portrait = portraits[0] if portraits else None

    def overlap_x(a: fitz.Rect, b: fitz.Rect) -> float:
        inter = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        return inter / max(1.0, min(a.width, b.width))

    candidates: list[tuple[float, dict[str, Any]]] = []
    for it in items:
        r = it["rect"]
        # Signature must be wide/short both in intrinsic pixels and on-page placement.
        if it["pixel_ratio"] < 2.0 or it["placed_ratio"] < 2.0:
            continue
        if r.height > 55 or r.width < 35:
            continue
        if r.get_area() > page.rect.get_area() * 0.05:
            continue

        score = 0.0
        if portrait is not None:
            pr = portrait["rect"]
            gap = r.y0 - pr.y1
            # Signature is directly below portrait in this PDF family.
            if gap < -4 or gap > max(95.0, pr.height * 0.75):
                continue
            ov = overlap_x(r, pr)
            if ov < 0.55:
                continue
            # Prefer same width / x-position as portrait and a small vertical gap.
            center_delta = abs((r.x0 + r.x1) / 2 - (pr.x0 + pr.x1) / 2)
            width_delta = abs(r.width - pr.width)
            score += 160.0
            score -= abs(gap) * 1.6
            score += ov * 45.0
            score -= center_delta * 0.8
            score -= width_delta * 0.25
        else:
            # Conservative no-portrait fallback: wide small XObject in upper half.
            if r.y0 > page.rect.height * 0.55:
                continue
            score += 10.0 - r.y0 * 0.005

        if it["smask"]:
            score += 30.0
        if 2.5 <= it["pixel_ratio"] <= 6.5:
            score += 12.0
        candidates.append((score, it))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if candidates:
        best = candidates[0][1]
        data_url = _pixmap_png_data_url(doc, best["xref"], best["smask"])
        if data_url:
            r = best["rect"]
            debug = {
                "source": "pdf_image_xobject",
                "xref": best["xref"],
                "smask": best["smask"],
                "pixelSize": [best["w"], best["h"]],
                "rect": [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)],
            }
            if portrait is not None:
                pr = portrait["rect"]
                debug["portraitRect"] = [round(pr.x0, 2), round(pr.y0, 2), round(pr.x1, 2), round(pr.y1, 2)]
            return data_url, debug

    # Deliberately do not use a fixed page-coordinate crop. A wrong crop can show
    # table labels as a fake signature. If no trustworthy signature XObject is found,
    # return missing so the UI can clearly report that instead of displaying bad data.
    debug: dict[str, Any] = {"source": "missing"}
    if portrait is not None:
        pr = portrait["rect"]
        debug["portraitRect"] = [round(pr.x0, 2), round(pr.y0, 2), round(pr.x1, 2), round(pr.y1, 2)]
    return "", debug



def extract_cardholder_fingerprint(doc: fitz.Document, page: fitz.Page) -> tuple[str, dict[str, Any]]:
    """Extract a fingerprint image placed below the portrait on page 1.

    Signature detection remains separate. Fingerprints in the supplied old-format
    PDF are near-square / portrait-ish image XObjects, while signatures are wide.
    """
    items: list[dict[str, Any]] = []
    for raw in page.get_images(full=True):
        xref, smask, w, h = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
        if w <= 0 or h <= 0:
            continue
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            if rect.is_empty or rect.is_infinite or rect.width <= 0 or rect.height <= 0:
                continue
            items.append({"xref":xref,"smask":smask,"w":w,"h":h,"rect":rect,
                          "pixel_ratio":w/max(h,1),"placed_ratio":rect.width/max(rect.height,0.01)})

    portraits = [it for it in items if it["h"] > it["w"]*1.08 and it["rect"].height >= 45
                 and it["rect"].width >= 30 and it["rect"].y0 < page.rect.height*0.55
                 and it["rect"].get_area() < page.rect.get_area()*0.20]
    portraits.sort(key=lambda it: it["rect"].get_area(), reverse=True)
    portrait = portraits[0] if portraits else None
    if portrait is None:
        return "", {"source":"missing","reason":"portrait_not_found"}

    pr=portrait["rect"]
    def overlap_x(a,b):
        inter=max(0.0,min(a.x1,b.x1)-max(a.x0,b.x0))
        return inter/max(1.0,min(a.width,b.width))

    candidates=[]
    for it in items:
        if it["xref"] == portrait["xref"]:
            continue
        r=it["rect"]
        # Fingerprint is not a wide signature: accept near-square / portrait-ish.
        if not (0.55 <= it["pixel_ratio"] <= 1.45 and 0.50 <= it["placed_ratio"] <= 1.45):
            continue
        gap=r.y0-pr.y1
        if gap < -4 or gap > max(125.0,pr.height*1.05):
            continue
        ov=overlap_x(r,pr)
        if ov < 0.55:
            continue
        if r.get_area() > page.rect.get_area()*0.08:
            continue
        center_delta=abs((r.x0+r.x1)/2-(pr.x0+pr.x1)/2)
        width_delta=abs(r.width-pr.width)
        score=150.0-abs(gap)*1.1+ov*45.0-center_delta*0.8-width_delta*0.18
        # Fingerprints tend to have substantial height and no soft mask requirement.
        if r.height >= pr.height*0.55: score += 15.0
        candidates.append((score,it))
    candidates.sort(key=lambda x:x[0], reverse=True)
    if not candidates:
        return "", {"source":"missing","portraitRect":[round(pr.x0,2),round(pr.y0,2),round(pr.x1,2),round(pr.y1,2)]}
    best=candidates[0][1]
    data_url=_pixmap_png_data_url(doc,best["xref"],best["smask"])
    if not data_url:
        return "", {"source":"missing","reason":"image_decode_failed"}
    r=best["rect"]
    return data_url,{"source":"pdf_image_xobject","xref":best["xref"],"smask":best["smask"],
                     "pixelSize":[best["w"],best["h"]],
                     "rect":[round(r.x0,2),round(r.y0,2),round(r.x1,2),round(r.y1,2)],
                     "portraitRect":[round(pr.x0,2),round(pr.y0,2),round(pr.x1,2),round(pr.y1,2)]}

def extract_document(pdf_bytes: bytes) -> dict[str, Any]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"PDF খোলা যায়নি: {e}") from e
    if doc.page_count < 1:
        raise ValueError("PDF-এ কোনো page নেই।")

    p1 = doc[0]
    signature_data_url, signature_debug = extract_cardholder_signature(doc, p1)
    fingerprint_data_url, fingerprint_debug = extract_cardholder_fingerprint(doc, p1)

    # Field discovery is document-wide: every detected table on every page is
    # eligible. PDF tag/name/old-new format is intentionally ignored.
    document_tables = collect_document_tables(doc)
    if not document_tables:
        raise ValueError("PDF-এ কোনো field table শনাক্ত করা যায়নি।")

    fields: dict[str, str] = {}
    debug: dict[str, Any] = {}

    ocr_fields = {"nameBn", "birthPlace", "father", "mother", "spouse", "voterArea"}
    for key in (
        "nid", "pin", "siNo", "voterNo", "nameBn", "nameEn", "dob",
        "birthPlace", "bloodGroup", "father", "mother", "spouse", "gender",
        "religion", "voterArea",
    ):
        fields[key], debug[key] = simple_value_any_table(
            document_tables, FIELD_LABELS[key], ocr_bengali=key in ocr_fields
        )

    # Some NID PDFs expose the serial label outside the detected table or split
    # the label/value into separate text spans. If the normal table lookup did
    # not find it, use the embedded text layer as a cheap fallback (no OCR).
    if not meaningful(fields.get("siNo")):
        page_text = clean_text(p1.get_text("text"))
        serial_patterns = [
            r"(?:S\s*/\s*L|SL|Sl|SI|Serial)\s*\.?\s*(?:No|Number)\s*[:#.-]?\s*([A-Za-z0-9/-]{2,})",
            r"(?:ক্রমিক|সিরিয়াল)\s*(?:নং|নম্বর)?\s*[:#.-]?\s*([A-Za-z0-9০-৯/-]{2,})",
        ]
        for pat in serial_patterns:
            m = re.search(pat, page_text, flags=re.IGNORECASE)
            if m:
                fields["siNo"] = clean_text(m.group(1))
                debug["siNo"] = {"source": "pdf_text_fallback"}
                break

    present_source = find_address_table(document_tables, "Present Address")
    if present_source:
        page, table, data = present_source
        fields["presentAddress"], debug["presentAddress"] = build_address(page, table, "Present Address", "Permanent Address", data=data, bengali_postal_digits=True)
        debug["presentAddress"]["page"] = page.number + 1
    else:
        fields["presentAddress"] = ""
        debug["presentAddress"] = {"source": "missing"}

    permanent_source = find_address_table(document_tables, "Permanent Address")
    if permanent_source:
        page, table, data = permanent_source
        fields["permanentAddress"], debug["permanentAddress"] = build_address(page, table, "Permanent Address", "Foreign Address", data=data)
        debug["permanentAddress"]["page"] = page.number + 1
    else:
        fields["permanentAddress"] = ""
        debug["permanentAddress"] = {"source": "missing"}

    # Final generic normalization pass. Do not invent missing values.
    for k, v in list(fields.items()):
        if k in {"nameBn", "father", "mother", "spouse", "birthPlace", "voterArea", "presentAddress", "permanentAddress"}:
            fields[k] = clean_bengali(v)
        else:
            fields[k] = clean_text(v)

    return {
        "ok": True,
        "engine": "python-pymupdf-visual-ocr-v14.3.2",
        "version": APP_VERSION,
        "ocr_available": tesseract_available(),
        "pages": doc.page_count,
        "fields": fields,
        "signatureDataUrl": signature_data_url,
        "fingerprintDataUrl": fingerprint_data_url,
        "debug": {**debug, "signature": signature_debug, "fingerprint": fingerprint_debug},
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "engine": "python-pymupdf-visual-ocr-v14.3.2",
        "version": APP_VERSION,
        "message": "NID Bengali OCR API is running. Use /health or POST /extract.",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "engine": "python-pymupdf-visual-ocr-v14.3.2",
        "version": APP_VERSION,
        "pymupdf": fitz.VersionBind,
        "tesseract": tesseract_available(),
        "ocr_lang": OCR_LANG,
        "ocr_mode": OCR_MODE,
        "ocr_scale": OCR_SCALE,
    }



@app.post("/admin/login")
async def admin_login(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not ADMIN_TEMPLATE_KEY:
        raise HTTPException(status_code=503, detail="Admin login Render-এ configure করা হয়নি।")
    key = str(payload.get("key", "")).strip()
    if not key or key != ADMIN_TEMPLATE_KEY:
        raise HTTPException(status_code=401, detail="Admin key সঠিক নয়।")
    return {"ok": True}


@app.get("/template/config")
async def template_config() -> dict[str, Any]:
    mapping = None
    if TEMPLATE_MAPPING_PATH.exists():
        try:
            mapping = json.loads(TEMPLATE_MAPPING_PATH.read_text(encoding="utf-8"))
        except Exception:
            mapping = None
    return {
        "ok": True,
        "template_exists": TEMPLATE_PDF_PATH.exists(),
        "mapping": mapping,
        "persistent_note": "Render local filesystem is ephemeral unless a persistent disk is configured.",
    }


@app.get("/template/pdf")
async def template_pdf():
    if not TEMPLATE_PDF_PATH.exists():
        raise HTTPException(status_code=404, detail="Template PDF পাওয়া যায়নি।")
    return FileResponse(TEMPLATE_PDF_PATH, media_type="application/pdf", filename="card-template.pdf")


@app.post("/template/pdf")
async def upload_template_pdf(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    require_template_admin(request)
    name = (file.filename or "").lower()
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not name.endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Template হিসেবে শুধু PDF দিন।")
    data = await file.read(MAX_TEMPLATE_BYTES + 1)
    if len(data) > MAX_TEMPLATE_BYTES:
        raise HTTPException(status_code=413, detail="Template PDF size limit-এর চেয়ে বড়।")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Template ফাইলটি valid PDF নয়।")
    try:
        check_doc = fitz.open(stream=data, filetype="pdf")
        if check_doc.page_count < 1:
            raise ValueError("empty PDF")
        check_doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Template PDF খোলা যাচ্ছে না বা corrupt।") from e
    tmp = TEMPLATE_PDF_PATH.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(TEMPLATE_PDF_PATH)
    return {"ok": True, "bytes": len(data), "filename": file.filename or "card-template.pdf"}


@app.post("/template/mapping")
async def save_template_mapping(request: Request, mapping: dict[str, Any] = Body(...)) -> dict[str, Any]:
    require_template_admin(request)
    if not isinstance(mapping, dict) or not mapping:
        raise HTTPException(status_code=400, detail="Mapping খালি হতে পারবে না।")
    allowed = {"photo","signature","fingerprint","biometric","nameBn","nameEn","father","mother","dob","nid","presentAddress","birthPlace","bloodGroup","issueDate","barcode"}
    clean: dict[str, Any] = {}
    for field, cfg in mapping.items():
        if field not in allowed or not isinstance(cfg, dict):
            continue
        try:
            x, y, w, h = (float(cfg[k]) for k in ("x","y","w","h"))
        except Exception:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.001 and y + h <= 1.001):
            continue
        clean[field] = {
            "x": x, "y": y, "w": w, "h": h,
            "fontSizePct": float(cfg.get("fontSizePct", 0.018)),
            "color": str(cfg.get("color", "#111111"))[:20],
            "align": str(cfg.get("align", "left")) if str(cfg.get("align", "left")) in {"left","center","right"} else "left",
            "weight": str(cfg.get("weight", "400")) if str(cfg.get("weight", "400")) in {"400","600","700"} else "400",
            "fit": str(cfg.get("fit", "contain")) if str(cfg.get("fit", "contain")) in {"contain","cover","fill"} else "contain",
        }
    if not clean:
        raise HTTPException(status_code=400, detail="Valid mapping পাওয়া যায়নি।")
    tmp = TEMPLATE_MAPPING_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(TEMPLATE_MAPPING_PATH)
    return {"ok": True, "fields": list(clean.keys())}


@app.delete("/template/config")
async def clear_template_config(request: Request) -> dict[str, Any]:
    require_template_admin(request)
    for path in (TEMPLATE_PDF_PATH, TEMPLATE_MAPPING_PATH):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    return {"ok": True}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    name = (file.filename or "").lower()
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not name.endswith(".pdf"):
        raise HTTPException(status_code=415, detail="শুধু PDF ফাইল দিন।")
    pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF ফাইলটি নির্ধারিত size limit-এর চেয়ে বড়।")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="ফাইলটি valid PDF নয়।")
    try:
        # IMPORTANT: PyMuPDF table analysis + Tesseract are synchronous CPU /
        # subprocess work. Running them directly inside this async route blocks
        # Uvicorn's event loop, which makes Render's /health check time out and
        # can restart the free instance mid-request. Keep the event loop free by
        # doing extraction in a worker thread. One heavy job at a time also
        # prevents the small free instance from being overloaded.
        async with EXTRACT_SEMAPHORE:
            return await asyncio.wait_for(
                asyncio.to_thread(extract_document, pdf_bytes),
                timeout=EXTRACT_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError as e:
        raise HTTPException(status_code=504, detail="PDF processing timeout হয়েছে। আবার চেষ্টা করুন।") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction ব্যর্থ: {e}") from e
