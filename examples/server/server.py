#!/usr/bin/env python3
"""CollectorVision — plug-and-play card identification server.

A minimal FastAPI server that exposes card identification as a REST API.
Catalog and detector are loaded lazily on first request and reused.

Usage
-----
    # Install deps
    pip install "collectorvision[server]"

    # Run with local catalog file
    python server.py --catalog ./milo1-scryfall-mtg-2026-04.npz

    # Run with HuggingFace auto-download (downloads on first request)
    python server.py --hfd HanClinto/milo scryfall-mtg

    # HTTPS (required for camera access from other devices on the LAN)
    python server.py --catalog ./catalog.npz --ssl

Endpoints
---------
    POST /identify
        Body: {"records": [{"_base64": "<base64-encoded image>", ...}]}
        Returns per-record identification results.

    POST /identify/upload
        Body: multipart form with one or more image files.
        Simpler for curl / browser testing.

    GET  /health
        Returns {"status": "ok"}.

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
# CLI args
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="CollectorVision identification server")
_parser.add_argument("--catalog",  type=Path,
                     help="Path to a local .npz catalog file")
_parser.add_argument("--hfd",      nargs=2, metavar=("REPO", "KEY"),
                     help="Auto-download catalog from HuggingFace: --hfd REPO KEY")
_parser.add_argument("--host",     default="127.0.0.1")
_parser.add_argument("--port",     type=int, default=8000)
_parser.add_argument("--top-k",    type=int, default=5)
_parser.add_argument("--min-sharpness", type=float, default=0.0,
                     help="SimCC sharpness gate; 0=disabled. ~0.02 skips blank frames.")
_parser.add_argument("--detector-none", action="store_true",
                     help="Skip corner detection — inputs are pre-cropped card images.")
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
    allow_origins=["*"],
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
# Lazy singletons
# ---------------------------------------------------------------------------

_catalog: cvg.Catalog | None = None
_detector: cvg.NeuralCornerDetector | None = None
_detector_loaded = False


def _get_catalog() -> cvg.Catalog:
    global _catalog
    if _catalog is not None:
        return _catalog
    if _args.catalog:
        _catalog = cvg.Catalog.load(_args.catalog)
    elif _args.hfd:
        repo, key = _args.hfd
        _catalog = cvg.Catalog.load(f"hf://{repo}/{key}")
    else:
        raise HTTPException(
            status_code=503,
            detail="No catalog configured. Start the server with --catalog or --hfd.",
        )
    return _catalog


def _get_detector() -> cvg.NeuralCornerDetector | None:
    global _detector, _detector_loaded
    if not _detector_loaded:
        if not _args.detector_none:
            _detector = cvg.NeuralCornerDetector()
        _detector_loaded = True
    return _detector


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
    t0 = time.perf_counter()
    catalog  = _get_catalog()
    detector = _get_detector()

    sharpness  = None
    card_present = True
    crop_jpeg  = None

    from PIL import Image

    # Step 1: detect + dewarp
    if detector is not None:
        detection = detector.detect(bgr, min_sharpness=_args.min_sharpness)
        sharpness = detection.extra.get("sharpness", 0.0)

        if not detection.card_present:
            return {
                "_status":      {"code": 200, "text": "OK"},
                "card_present": False,
                "sharpness":    round(float(sharpness), 5),
                "_timing":      {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
            }

        crop_pil  = detection.dewarp(bgr)
        crop_bgr  = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
        crop_jpeg = _crop_jpeg(crop_bgr)
    else:
        sharpness = None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop_pil  = Image.fromarray(rgb)
        crop_jpeg = _crop_jpeg(bgr)

    # Step 2: embed + search
    try:
        emb  = catalog.embedder.embed(crop_pil)
        hits = catalog.search(emb, top_k=top_k)
    except Exception as exc:
        return {
            "_status":      {"code": 500, "text": f"Identification error: {exc}"},
            "card_present": False,
        }

    t1 = time.perf_counter()

    out: dict = {
        "_status":      {"code": 200, "text": "OK"},
        "card_present": card_present,
        "_timing":      {"total_ms": round((t1 - t0) * 1000, 1)},
    }
    if sharpness is not None:
        out["sharpness"] = round(float(sharpness), 5)
    if crop_jpeg:
        out["crop_jpeg"] = crop_jpeg
    if hits:
        best_score, best_id = hits[0]
        out["card_id"]      = best_id
        out["confidence"]   = round(best_score, 4)
        out["alternatives"] = [
            {"card_id": cid, "confidence": round(score, 4)}
            for score, cid in hits[1:]
        ]
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/identify")
async def identify(request: Request):
    """Identify cards from base64-encoded images.

    Body::

        {
          "records": [
            {"_base64": "<base64-encoded JPEG or PNG>"},
            ...
          ]
        }

    Returns::

        {
          "records": [
            {
              "_status":    {"code": 200, "text": "OK"},
              "card_present": true,
              "sharpness":  0.042,
              "crop_jpeg":  "<base64 preview>",
              "card_id":    "abc123...",
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
            results.append({"_status": {"code": 400, "text": "Missing '_base64' field"},
                            "card_present": False})
            continue

        top_k = int(rec.get("top_k", _args.top_k))
        try:
            bgr = _bgr_from_b64(b64)
        except ValueError as exc:
            results.append({"_status": {"code": 400, "text": str(exc)},
                            "card_present": False})
            continue

        results.append(_identify_bgr(bgr, top_k=top_k))

    return JSONResponse({"records": results})


@app.post("/identify/upload")
async def identify_upload(
    files: list[UploadFile] = File(...),
    top_k: int = None,
):
    """Identify cards from uploaded image files (multipart form).

    Multiple files are treated as frames of the **same physical card** and
    scores are summed before ranking.  To identify multiple *different* cards,
    use ``POST /identify`` with separate records.

    Example::

        curl -X POST http://localhost:8000/identify/upload -F "files=@card.jpg"
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    resolved_top_k = top_k if top_k is not None else _args.top_k

    if len(files) == 1:
        data = await files[0].read()
        try:
            bgr = _bgr_from_bytes(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse(_identify_bgr(bgr, top_k=resolved_top_k))

    # Multiple frames — aggregate scores
    catalog  = _get_catalog()
    detector = _get_detector()

    from collections import defaultdict
    from PIL import Image

    t0 = time.perf_counter()
    score_map: dict[str, float] = defaultdict(float)

    for f in files:
        data = await f.read()
        try:
            bgr = _bgr_from_bytes(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {exc}")

        if detector is not None:
            detection = detector.detect(bgr, min_sharpness=_args.min_sharpness)
            if not detection.card_present:
                continue  # skip blurry / no-card frames
            crop_pil = detection.dewarp(bgr)
        else:
            crop_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        emb = catalog.embedder.embed(crop_pil)
        for score, card_id in catalog.search(emb, top_k=resolved_top_k):
            score_map[card_id] += score

    hits = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:resolved_top_k]
    t1 = time.perf_counter()

    best_id, best_score = hits[0]
    return JSONResponse({
        "_status":      {"code": 200, "text": "OK"},
        "card_present": True,
        "card_id":      best_id,
        "confidence":   round(best_score, 4),
        "alternatives": [{"card_id": cid, "confidence": round(s, 4)} for cid, s in hits[1:]],
        "_timing":      {"total_ms": round((t1 - t0) * 1000, 1)},
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if _args.ssl:
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmp:
            cert, key = f"{tmp}/cert.pem", f"{tmp}/key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key, "-out", cert, "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
            ], check=True, capture_output=True)
            uvicorn.run("server:app", host=_args.host, port=_args.port,
                        ssl_certfile=cert, ssl_keyfile=key)
    else:
        uvicorn.run("server:app", host=_args.host, port=_args.port)
