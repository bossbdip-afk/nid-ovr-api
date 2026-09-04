# SB Tech Info OCR Backend - Checked Build 14.3.1

FastAPI backend for PDF field extraction and the template-mapping workflow.

## Main endpoints

- `GET /health` - service health
- `POST /extract` - extract fields from the uploaded source PDF
- `GET /template/config` - get saved template status + mapping
- `GET /template/pdf` - download the currently saved blank template
- `POST /template/pdf` - upload/replace blank PDF template
- `POST /template/mapping` - save visual field mapping
- `DELETE /template/config` - clear saved template + mapping

## Mapping fields

`photo`, `signature`, `nameBn`, `nameEn`, `father`, `mother`, `dob`, `nid`, `presentAddress`, `birthPlace`, `bloodGroup`, `issueDate`, `barcode`.

The 2D barcode itself is rendered by the frontend from extracted `nameEn`, `dob`, and `pin` data. The backend returns the extracted `pin` field through `/extract`.

## Render / Docker

The repository root should contain exactly the normal deploy files, including a file named **`Dockerfile`** (no `.txt` extension).

Template files are stored under `TEMPLATE_DATA_DIR` (default `./data`). Render's normal local filesystem is ephemeral, so uploaded templates can disappear after restart/redeploy unless a persistent disk is configured. The frontend also has its browser-side fallback.

Optional environment variables:

- `ADMIN_TEMPLATE_KEY` - protects template upload/mapping/delete endpoints via `X-Admin-Key`
- `TEMPLATE_DATA_DIR` - directory used to store template + mapping
- `MAX_TEMPLATE_BYTES` - maximum template PDF size (default 10 MB)
- Existing OCR variables in `render.yaml` remain supported.

## Signature extraction fix (v14.3.2)
- `/extract` now returns `signatureDataUrl` when the card-holder signature is stored as a PDF image XObject.
- Transparent signatures that use an `/SMask` are reconstructed as PNG instead of being missed by JPEG scanning.
- The candidate is selected from the wide image placed directly below the portrait on page 1.



Signature fix v14.3.3: single active frontend processPDF flow, backend /SMask extraction, and rendered-crop fallback below the portrait when an embedded signature image is not directly decodable.


## Signature extraction fix (v14.3.4)
- Portrait detection no longer assumes the image is on the left side of the page.
- Signature is selected by its actual PDF XObject placement directly below the detected portrait.
- Fixed page-coordinate signature cropping was removed to prevent table text from being shown as a signature.


## Fingerprint extraction fix (v14.3.5)
- `/extract` returns `fingerprintDataUrl` when a near-square fingerprint image XObject is placed below the portrait.
- Signature extraction remains unchanged and separate.
- Table parsing now supports PDFs with a leading blank column, so old-format forms still return identity/address fields.
