from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

APP_VERSION = "14.0.0"
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(15 * 1024 * 1024)))
OCR_LANG = os.getenv("OCR_LANG", "ben+eng")
OCR_SCALE = float(os.getenv("OCR_SCALE", "3.2"))
EXTRACT_TIMEOUT_SECONDS = float(os.getenv("EXTRACT_TIMEOUT_SECONDS", "120"))
MAX_CONCURRENT_EXTRACTS = max(1, int(os.getenv("MAX_CONCURRENT_EXTRACTS", "1")))
EXTRACT_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_EXTRACTS)

DEFAULT_ORIGINS = [
    "https://sbtechinfo.liveblog365.com",
    "http://sbtechinfo.liveblog365.com",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",") if x.strip()]

app = FastAPI(title="NID Bengali Field OCR API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
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
    return s.strip()


def meaningful(value: Any) -> bool:
    s = clean_text(value)
    return bool(s and not TRIVIAL_PUNCT_RE.match(s))


def compact_for_compare(s: str) -> str:
    s = clean_bengali(s).lower()
    return re.sub(r"[\s,.;:()\-–—_/]+", "", s)


def obvious_damage(s: str) -> int:
    s0 = clean_text(s)
    score = 0
    if re.search(r"ঃ{2,}", s0):
        score += 5
    if re.search(r"\s+[ঁংঃ়ািীুূৃৄেৈোৌ্ৗ]", s0):
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


def tesseract_available() -> bool:
    return bool(pytesseract is not None and shutil.which("tesseract"))


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

    candidates = [_ocr_image(gray, 7), _ocr_image(gray, 6)]
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


def find_simple_row(table, labels: Iterable[str]):
    if table is None:
        return None
    data = table.extract()
    wanted = [norm_label(x) for x in labels]
    # Exact normalized labels first. This prevents Father Name from ever being
    # accepted for Mother Name just because the strings are similar.
    for ri, row in enumerate(data):
        if not row:
            continue
        actual = norm_label(row[0] if len(row) > 0 else "")
        if actual and actual in wanted:
            return ri, row
    # Fuzzy fallback is only for genuine extraction typos (e.g. Post Ofcfie).
    for ri, row in enumerate(data):
        if not row:
            continue
        actual = row[0] if len(row) > 0 else ""
        for label in labels:
            if label_match(actual, label, 0.90):
                return ri, row
    return None


def simple_value(page: fitz.Page, table, labels: Iterable[str], *, ocr_bengali: bool = False) -> tuple[str, dict[str, Any]]:
    found = find_simple_row(table, labels)
    if not found:
        return "", {"source": "missing"}
    ri, row = found
    raw = row[1] if len(row) > 1 and row[1] is not None else ""
    raw_clean = clean_bengali(raw) if BENGALI_RE.search(str(raw or "")) else clean_text(raw)
    if not ocr_bengali or not BENGALI_RE.search(str(raw or "")):
        return raw_clean, {"source": "pdf_table"}
    cell = table.rows[ri].cells[1] if len(table.rows[ri].cells) > 1 else None
    ocr = ocr_cell(page, cell)
    value, source, conf = choose_text(str(raw or ""), ocr)
    return value, {"source": source, "ocr_confidence": round(conf, 1), "raw": clean_text(raw), "ocr": ocr.text}


def address_rows(table, section: str, next_section: str | None):
    if table is None:
        return []
    data = table.extract()
    start = None
    end = len(data)
    for i, row in enumerate(data):
        if row and len(row) > 0 and label_match(row[0], section, 0.84):
            start = i
            break
    if start is None:
        return []
    if next_section:
        for i in range(start + 1, len(data)):
            row = data[i]
            if row and len(row) > 0 and label_match(row[0], next_section, 0.84):
                end = i
                break
    return list(range(start, end))


def address_field(page: fitz.Page, table, row_indices: list[int], labels: list[str], *, ocr_bengali: bool = True) -> tuple[str, dict[str, Any]]:
    data = table.extract() if table else []
    # Address rows have two label/value pairs: columns 1/2 and 3/4.
    wanted = [norm_label(x) for x in labels]
    candidates = []
    for ri in row_indices:
        row = data[ri]
        for li, vi in ((1, 2), (3, 4)):
            if li >= len(row):
                continue
            actual = row[li]
            an = norm_label(actual)
            if not an:
                continue
            exact = an in wanted
            fuzzy = any(label_match(actual, label, 0.90) for label in labels)
            if exact or fuzzy:
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
        ocr = ocr_cell(page, cell)
        value, source, conf = choose_text(str(raw or ""), ocr)
        return value, {"source": source, "ocr_confidence": round(conf, 1), "raw": clean_text(raw), "ocr": ocr.text}
    return "", {"source": "missing"}


def build_address(page: fitz.Page, table, section: str, next_section: str | None) -> tuple[str, dict[str, Any]]:
    rows = address_rows(table, section, next_section)
    if not rows:
        return "", {"source": "missing"}

    additional_village, m_add = address_field(page, table, rows, ["Additional Village/Road"])
    village, m_vil = address_field(page, table, rows, ["Village/Road"])
    holding, m_hold = address_field(page, table, rows, ["Home/Holding No", "Home/Holding"], ocr_bengali=False)
    post, m_post = address_field(page, table, rows, ["Post Office", "Post Ofcfie"])
    postal, m_postal = address_field(page, table, rows, ["Postal Code", "Post Code"], ocr_bengali=False)
    upazila, m_up = address_field(page, table, rows, ["Upozila", "Upazila"])
    district, m_dist = address_field(page, table, rows, ["District"])

    v = additional_village or village
    if not meaningful(holding):
        holding = ""

    parts: list[str] = []
    if v:
        parts.append(f"গ্রাম/রাস্তা: {v}")
    if holding:
        parts.append(f"হোল্ডিং নং: {holding}")
    if post:
        parts.append("ডাকঘর: " + post + ((" - " + postal) if postal else ""))
    elif postal:
        parts.append("পোস্ট কোড: " + postal)
    if upazila:
        parts.append(upazila)
    if district:
        parts.append(district)

    meta = {
        "village": m_add if additional_village else m_vil,
        "holding": m_hold,
        "post": m_post,
        "postal": m_postal,
        "upazila": m_up,
        "district": m_dist,
    }
    return clean_bengali(", ".join(parts)), meta


def extract_document(pdf_bytes: bytes) -> dict[str, Any]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"PDF খোলা যায়নি: {e}") from e
    if doc.page_count < 1:
        raise ValueError("PDF-এ কোনো page নেই।")

    p1 = doc[0]
    t1 = get_table(p1)
    if t1 is None:
        raise ValueError("প্রথম page-এর NID table শনাক্ত করা যায়নি।")

    # IMPORTANT: process page 1 before calling find_tables() on another page.
    # PyMuPDF table finder objects are page-bound and some versions reuse
    # internal finder state across calls.
    p2 = doc[1] if doc.page_count > 1 else None

    fields: dict[str, str] = {}
    debug: dict[str, Any] = {}

    simple_specs = {
        "nid": (["National ID"], False),
        "pin": (["Pin"], False),
        "siNo": (["Sl No", "SI No"], False),
        "voterNo": (["Voter No"], False),
        "nameBn": (["Name(Bangla)"], True),
        "nameEn": (["Name(English)"], False),
        "dob": (["Date of Birth"], False),
        "birthPlace": (["Birth Place"], True),
        "father": (["Father Name"], True),
        "mother": (["Mother Name"], True),
        "spouse": (["Spouse Name"], True),
        "gender": (["Gender"], False),
    }
    for key, (labels, use_ocr) in simple_specs.items():
        fields[key], debug[key] = simple_value(p1, t1, labels, ocr_bengali=use_ocr)

    fields["presentAddress"], debug["presentAddress"] = build_address(p1, t1, "Present Address", "Permanent Address")
    fields["permanentAddress"], debug["permanentAddress"] = build_address(p1, t1, "Permanent Address", "Foreign Address")

    t2 = get_table(p2) if p2 is not None else None
    if p2 is not None and t2 is not None:
        fields["religion"], debug["religion"] = simple_value(p2, t2, ["Religion"], ocr_bengali=False)
        fields["voterArea"], debug["voterArea"] = simple_value(p2, t2, ["Voter Area"], ocr_bengali=True)
    else:
        fields["religion"] = ""
        fields["voterArea"] = ""
        debug["religion"] = {"source": "missing"}
        debug["voterArea"] = {"source": "missing"}

    # Final generic normalization pass. Do not invent missing values.
    for k, v in list(fields.items()):
        if k in {"nameBn", "father", "mother", "spouse", "birthPlace", "voterArea", "presentAddress", "permanentAddress"}:
            fields[k] = clean_bengali(v)
        else:
            fields[k] = clean_text(v)

    return {
        "ok": True,
        "engine": "python-pymupdf-visual-ocr-v14",
        "version": APP_VERSION,
        "ocr_available": tesseract_available(),
        "pages": doc.page_count,
        "fields": fields,
        "debug": debug,
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "ok": True,
        "engine": "python-pymupdf-visual-ocr-v14",
        "version": APP_VERSION,
        "message": "NID Bengali OCR API is running. Use /health or POST /extract.",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "engine": "python-pymupdf-visual-ocr-v14",
        "version": APP_VERSION,
        "pymupdf": fitz.VersionBind,
        "tesseract": tesseract_available(),
        "ocr_lang": OCR_LANG,
    }


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
