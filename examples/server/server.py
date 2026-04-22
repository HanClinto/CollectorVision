#!/usr/bin/env python3
"""CollectorVision — plug-and-play card identification server.

A minimal FastAPI server that exposes card identification as a REST API.
Uses a single Identifier instance (lazy-loaded on first request).

Compatible with the 07_web_scanner client — accepts the same
``{"records": [{"_base64": "..."}]}`` request format and returns the
same ``{"records": [...], "_status": {...}}`` envelope.

Usage
-----
    # Install deps
    pip install collectorvision fastapi uvicorn[standard]

    # Run (downloads gallery automatically on first start)
    python server.py --gallery ./magic-scryfall-milo1-2026-04.npz

    # Or with HuggingFace auto-download (requires network on first run)
    python server.py --hfd HanClinto/milo scryfall-mtg

    # HTTPS (required for camera access from other devices on the LAN)
    python server.py --gallery ./gallery.npz --ssl

Endpoints
---------
    POST /identify
        Body: {"records": [{"_base64": "<base64-encoded image>", ...}]}
        Returns per-record identification results.

    POST /identify/upload
        Body: multipart form with one or more image files.
        Simpler for curl / browser testing.

    GET  /health
        Returns {"status": "ok"} — useful for readiness probes.

    GET  /
        Redirects to /docs (Swagger UI).
"""
from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

import collector_vision as cvg

# ---------------------------------------------------------------------------
# CLI args (parsed once so uvicorn can also pick them up)
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="CollectorVision identification server")
_parser.add_argument("--gallery",  type=Path,
                     help="Path to a local .npz gallery file")
_parser.add_argument("--hfd",      nargs=2, metavar=("REPO", "NAME"),
                     help="Auto-download gallery from HuggingFace: --hfd REPO NAME")
_parser.add_argument("--host",     default="127.0.0.1")
_parser.add_argument("--port",     type=int, default=8000)
_parser.add_argument("--top-k",    type=int, default=5,
                     help="Number of alternatives to return per image")
_parser.add_argument("--min-sharpness", type=float, default=0.0,
                     help="SimCC sharpness gate [0-1]; 0=disabled (default). "
                          "~0.02 skips frames with no visible card.")
_parser.add_argument("--detector-none", action="store_true",
                     help="Skip corner detection — treat inputs as pre-cropped card images.")
_parser.add_argument("--ssl",      action="store_true",
                     help="Serve over HTTPS using a self-signed certificate.")
_args, _ = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CollectorVision",
    description="Card identification API — feed it an image, get back a card identity.",
    version=cvg.__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local tool — all origins OK
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def _root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "version": cvg.__version__}


# ---------------------------------------------------------------------------
# Lazy Identifier singleton
# ---------------------------------------------------------------------------

_identifier: cvg.Identifier | None = None


def _get_identifier() -> cvg.Identifier:
    global _identifier
    if _identifier is not None:
        return _identifier

    # Gallery source
    if _args.gallery:
        gallery_src = _args.gallery
    elif _args.hfd:
        repo, name = _args.hfd
        gallery_src = cvg.HFD(repo, name)
    else:
        raise HTTPException(
            status_code=503,
            detail="No gallery configured. Start the server with --gallery or --hfd.",
        )

    # Detector
    if _args.detector_none:
        detector = None
    else:
        from collector_vision.detectors.neural import NeuralCornerDetector
        detector = NeuralCornerDetector(min_sharpness=_args.min_sharpness)

    _identifier = cvg.Identifier(gallery_src, detector=detector)
    return _identifier


# ---------------------------------------------------------------------------
# Image decoding helpers
# ---------------------------------------------------------------------------

def _bgr_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (unsupported format or corrupt data)")
    return bgr


def _bgr_from_b64(b64: str) -> np.ndarray:
    try:
        return _bgr_from_bytes(base64.b64decode(b64))
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}") from exc


def _crop_jpeg(bgr: np.ndarray, max_dim: int = 300) -> str:
    """Return a small JPEG preview of the crop as a base64 string."""
    h, w = bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Core identification logic
# ---------------------------------------------------------------------------

def _identify_bgr(bgr: np.ndarray, *, top_k: int) -> dict:
    """Run the full pipeline on one BGR image. Returns a record dict."""
    t0 = time.perf_counter()
    identifier = _get_identifier()

    try:
        result = identifier.identify(bgr, top_k=top_k)
    except Exception as exc:
        return {
            "_status":     {"code": 500, "text": f"Identification error: {exc}"},
            "card_present": False,
        }

    t1 = time.perf_counter()

    # Build the dewarped crop preview using the detector directly so we can
    # include it in the response without running inference twice.
    # (If detector=None, the full image is the crop.)
    detector = identifier._get_detector()
    sharpness = None
    crop_jpeg = None
    card_present = True

    if detector is not None:
        from collector_vision.identifier import _dewarp
        det_result = detector.detect(bgr)
        sharpness = det_result.extra.get("sharpness")
        card_present = det_result.card_present

        if det_result.card_present and det_result.corners is not None:
            crop_bgr = _dewarp(bgr, det_result.corners)
            crop_jpeg = _crop_jpeg(crop_bgr)
        else:
            # No card detected — still return identification (low confidence)
            crop_jpeg = _crop_jpeg(bgr)
    else:
        crop_jpeg = _crop_jpeg(bgr)

    out: dict = {
        "_status":     {"code": 200, "text": "OK"},
        "card_present": card_present,
        "_timing":     {"total_ms": round((t1 - t0) * 1000, 1)},
    }

    if sharpness is not None:
        out["sharpness"] = round(sharpness, 5)

    if crop_jpeg:
        out["crop_jpeg"] = crop_jpeg

    if result.ids:
        out["ids"]        = result.ids
        out["confidence"] = round(result.confidence, 4)
        out["alternatives"] = [
            {"ids": alt.ids, "confidence": round(alt.confidence, 4)}
            for alt in result.alternatives
        ]

    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/identify")
async def identify(request: Request):
    """Identify cards from base64-encoded images.

    Accepts the same JSON envelope as the 07_web_scanner API::

        {
          "records": [
            {"_base64": "<base64-encoded JPEG or PNG>"},
            ...
          ]
        }

    Per-record optional fields:

    - ``"top_k"`` — override the number of alternatives (default: server ``--top-k``)

    Returns::

        {
          "records": [
            {
              "_status":    {"code": 200, "text": "OK"},
              "card_present": true,
              "sharpness":  0.042,
              "crop_jpeg":  "<base64 preview>",
              "ids":        {"scryfall_id": "..."},
              "confidence": 0.94,
              "alternatives": [...]
            }
          ]
        }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    records = body.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=400, detail="'records' must be a non-empty list")

    results = []
    for rec in records:
        b64 = rec.get("_base64") or rec.get("base64")
        if not b64:
            results.append({
                "_status":     {"code": 400, "text": "Missing '_base64' field"},
                "card_present": False,
            })
            continue

        top_k = int(rec.get("top_k", _args.top_k))

        try:
            bgr = _bgr_from_b64(b64)
        except ValueError as exc:
            results.append({
                "_status":     {"code": 400, "text": str(exc)},
                "card_present": False,
            })
            continue

        results.append(_identify_bgr(bgr, top_k=top_k))

    return JSONResponse({"records": results})


@app.post("/identify/upload")
async def identify_upload(
    files: list[UploadFile] = File(...),
    top_k: int = None,
):
    """Identify cards from uploaded image files (multipart form).

    Simpler than the base64 endpoint — useful for curl, browser forms,
    and quick testing::

        curl -X POST http://localhost:8000/identify/upload \\
             -F "files=@card.jpg"

    Multiple files are treated as frames of the **same physical card** and
    votes are summed before ranking (same as ``Identifier.identify(*frames)``).
    To identify multiple *different* cards in one request, use ``POST /identify``
    with separate records.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    resolved_top_k = top_k if top_k is not None else _args.top_k

    if len(files) == 1:
        # Single frame — use _identify_bgr for the full record response
        data = await files[0].read()
        try:
            bgr = _bgr_from_bytes(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse(_identify_bgr(bgr, top_k=resolved_top_k))

    # Multiple frames — run multi-frame voting via Identifier.identify()
    bgrs = []
    for f in files:
        data = await f.read()
        try:
            bgrs.append(_bgr_from_bytes(data))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {exc}")

    t0 = time.perf_counter()
    identifier = _get_identifier()
    result = identifier.identify(*bgrs, top_k=resolved_top_k)
    t1 = time.perf_counter()

    return JSONResponse({
        "_status":     {"code": 200, "text": "OK"},
        "card_present": True,
        "ids":          result.ids,
        "confidence":   round(result.confidence, 4),
        "alternatives": [
            {"ids": alt.ids, "confidence": round(alt.confidence, 4)}
            for alt in result.alternatives
        ],
        "frame_results": [
            {"ids": fr.ids, "confidence": round(fr.confidence, 4)}
            for fr in result.frame_results
        ],
        "_timing": {"total_ms": round((t1 - t0) * 1000, 1)},
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if _args.ssl:
        # Generate a throwaway self-signed cert for LAN access
        # (browser will warn; acceptable for local dev)
        import ssl, tempfile, subprocess  # noqa: E401
        with tempfile.TemporaryDirectory() as tmp:
            cert = f"{tmp}/cert.pem"
            key  = f"{tmp}/key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key, "-out", cert, "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
            ], check=True, capture_output=True)
            uvicorn.run(
                "server:app", host=_args.host, port=_args.port,
                ssl_certfile=cert, ssl_keyfile=key,
            )
    else:
        uvicorn.run("server:app", host=_args.host, port=_args.port)
