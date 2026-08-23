# SBTechInfo NID Bengali OCR API

FastAPI + PyMuPDF + Tesseract Bengali OCR service.

Endpoints:
- `GET /` status
- `GET /health` health check
- `POST /extract` multipart PDF field extraction

## v14.3 performance + serial fix
- Added a positioned-text fast path that runs before PyMuPDF `find_tables()`.
- Normal text-layer PDFs avoid expensive table detection and OCR.
- `find_tables()` + Tesseract remain as compatibility fallback for unusual/scanned PDFs.
- Serial value `0` is treated as a placeholder/missing value.
- Serial is recovered from `Sl No`, `SI No`, `SL No`, `S/L No`, `Serial No`, or `Serial Number` text.
- Existing CORS support for `singtonid.liveblog365.com` is preserved.

The real PDF processing time depends on PDF structure and Render load/cold-start. On the fast path, backend parsing itself is designed to complete far below the old 15-second table-detection path.
