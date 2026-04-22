# CollectorVision — Example Server

A minimal FastAPI server that exposes card identification as a REST API.
Drop-in compatible with the `07_web_scanner` client.

## Install

```bash
pip install collectorvision fastapi "uvicorn[standard]"
```

## Run

```bash
# Local catalog file
python server.py --catalog ./milo1-scryfall-mtg-2026-04.npz

# Auto-download from HuggingFace (cached in ~/.cache/collectorvision/)
python server.py --hfd HanClinto/milo scryfall-mtg

# Pre-cropped images (skip corner detection)
python server.py --catalog ./catalog.npz --detector-none

# Filter blurry / no-card frames in a video pipeline
python server.py --catalog ./catalog.npz --min-sharpness 0.02

# HTTPS (required for camera access from phones / other LAN devices)
python server.py --catalog ./catalog.npz --ssl
```

Browse to **http://localhost:8000** → Swagger UI.

## Endpoints

### `POST /identify` — base64 JSON (07_web_scanner compatible)

```json
{
  "records": [
    { "_base64": "<base64-encoded JPEG or PNG>" }
  ]
}
```

Response:
```json
{
  "records": [
    {
      "_status":    { "code": 200, "text": "OK" },
      "card_present": true,
      "sharpness":  0.042,
      "card_id":    "abc123...",
      "confidence": 0.94,
      "alternatives": [
        { "card_id": "def456...", "confidence": 0.61 }
      ],
      "crop_jpeg":  "<base64 preview of the dewarped card>"
    }
  ]
}
```

### `POST /identify/upload` — multipart file upload

Simpler for testing:

```bash
# Single image
curl -X POST http://localhost:8000/identify/upload \
     -F "files=@card.jpg"

# Multiple frames of the same card (votes are summed)
curl -X POST http://localhost:8000/identify/upload \
     -F "files=@frame1.jpg" -F "files=@frame2.jpg"
```

### `GET /health`

```json
{ "status": "ok", "version": "0.1.0.dev0" }
```

## Notes

- The catalog and detector are lazy-loaded on the first request and reused for
  all subsequent calls — startup is fast, first request takes ~1–2 s.
- `sharpness` in the response is the SimCC mean-peak score; low values (< 0.02)
  indicate no card is visible in the frame.  Set `--min-sharpness 0.02` to
  automatically skip these frames.
- For multi-card identification in one request, send separate records in the
  `POST /identify` body (one record per card, not per frame).
- For iOS/Android examples, see `examples/ios/` and `examples/android/` (coming soon).
