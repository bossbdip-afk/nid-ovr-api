# SBTechInfo NID Bengali OCR API

FastAPI + PyMuPDF + Tesseract Bengali OCR service.

Endpoints:
- `GET /` status
- `GET /health` health check
- `POST /extract` multipart PDF field extraction

This repo includes `render.yaml` and a Dockerfile for Render deployment.

## v14.1 performance update
- Default `OCR_MODE=auto`: clean Bengali text from the PDF layer skips Tesseract.
- OCR runs only for visibly damaged Bengali text.
- OCR scale reduced from 3.2 to 2.2.
- PSM 6 fallback runs only when the first OCR pass is weak.
- Tesseract availability detection is cached.

For legacy behavior set `OCR_MODE=always`.
