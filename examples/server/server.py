#!/usr/bin/env python3
"""CollectorVision — plug-and-play card identification server.

A minimal FastAPI server that exposes card identification as a REST API.
Catalog and detector load once at startup and are reused across requests.

Usage
-----
    pip install "collector-vision[server]"

    python server.py --catalog ./milo1-scryfall-mtg-2026-04.npz
    python server.py --hfd HanClinto/milo scryfall-mtg
    python server.py --catalog ./catalog.npz --ssl

Endpoints
---------
    POST /identify          JSON body: {"records": [{"_base64": "<image>"}]}
    POST /identify/upload   Multipart form with one or more image files.
    GET  /health            {"status": "ok"}
    GET  /                  Redirects to /docs (Swagger UI).
"""
from __future__ import annotations

import base64
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image

import collector_vision as cvg

# ---------------------------------------------------------------------------
# Startup configuration — set before creating the TestClient or calling
# uvicorn.run().  The lifespan handler reads these module-level variables.
# ---------------------------------------------------------------------------

catalog_source: str | Path | None = None   # path, hf:// URI, or None
top_k_default:  int   = 5
min_sharpness:  float = 0.0
detector_none:  bool  = False              # True → skip detection (pre-cropped inputs)


def configure(
    catalog: "str | Path | None" = None,
    top_k: int = 5,
    min_sharpness_val: float = 0.0,
    no_detector: bool = False,
) -> None:
    """Configure the server before startup (used by tests and scripts)."""
    global catalog_source, top_k_default, min_sharpness, detector_none
    catalog_source = catalog
    top_k_default  = top_k
    min_sharpness  = min_sharpness_val
    detector_none  = no_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not catalog_source:
        raise RuntimeError("No catalog configured. Call configure() or use --catalog / --hfd.")
    app.state.catalog  = cvg.Catalog.load(catalog_source)
    app.state.detector = None if detector_none else cvg.NeuralCornerDetector()
    yield


app = FastAPI(
    title="CollectorVision",
    description="Card identification API — feed it an image, get back a card identity.",
    version=cvg.__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Core pipeline — detect, dewarp, embed, search
# ---------------------------------------------------------------------------

def _decode_bgr(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (unsupported format or corrupt data)")
    return bgr


def _identify(bgr: np.ndarray, catalog: cvg.Catalog,
              detector: "cvg.NeuralCornerDetector | None", top_k: int) -> dict:
    t0 = time.perf_counter()

    # Detect + dewarp (or pass straight through if detection is disabled)
    sharpness = None
    if detector is not None:
        det = detector.detect(bgr, min_sharpness=min_sharpness)
        sharpness = det.sharpness
        if not det.card_present:
            return {"card_present": False, "sharpness": sharpness,
                    "_timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)}}
        crop = det.dewarp(bgr)
    else:
        crop = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Thumbnail of the crop for visual confirmation in the response
    crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    h, w = crop_bgr.shape[:2]
    scale = min(1.0, 300 / max(h, w))
    if scale < 1.0:
        crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    crop_jpeg = base64.b64encode(buf.tobytes()).decode()

    # Embed + search
    emb  = catalog.embedder.embed(crop)
    hits = catalog.search(emb, top_k=top_k)

    best_score, best_id = hits[0]
    result = {
        "card_present": True,
        "card_id":      best_id,
        "confidence":   round(best_score, 4),
        "alternatives": [{"card_id": cid, "confidence": round(s, 4)} for s, cid in hits[1:]],
        "crop_jpeg":    crop_jpeg,
        "_timing":      {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
    }
    if sharpness is not None:
        result["sharpness"] = round(float(sharpness), 5)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def _root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "version": cvg.__version__}


@app.post("/identify")
async def identify(request: Request):
    """Identify cards from base64-encoded images.

    Body: ``{"records": [{"_base64": "<JPEG or PNG as base64>"}, ...]}``

    Each record may include ``"top_k"`` to override the server default.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    records = body.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=400, detail="'records' must be a non-empty list")

    catalog  = request.app.state.catalog
    detector = request.app.state.detector
    results  = []

    for rec in records:
        b64 = rec.get("_base64") or rec.get("base64")
        if not b64:
            results.append({"_status": {"code": 400, "text": "Missing '_base64' field"},
                            "card_present": False})
            continue
        try:
            bgr = _decode_bgr(base64.b64decode(b64))
        except Exception as exc:
            results.append({"_status": {"code": 400, "text": str(exc)},
                            "card_present": False})
            continue
        results.append(_identify(bgr, catalog, detector,
                                  top_k=int(rec.get("top_k", top_k_default))))

    return JSONResponse({"records": results})


@app.post("/identify/upload")
async def identify_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    top_k: int | None = None,
):
    """Identify cards from uploaded image files (multipart form).

    Multiple files are treated as frames of the **same physical card** —
    scores are summed before ranking.

    Example: ``curl -X POST http://localhost:8000/identify/upload -F "files=@card.jpg"``
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    catalog  = request.app.state.catalog
    detector = request.app.state.detector
    k        = top_k if top_k is not None else top_k_default

    if len(files) == 1:
        data = await files[0].read()
        return JSONResponse(_identify(_decode_bgr(data), catalog, detector, top_k=k))

    # Multiple frames — aggregate scores across frames of the same card
    t0        = time.perf_counter()
    score_map: dict[str, float] = defaultdict(float)

    for f in files:
        data = await f.read()
        try:
            bgr = _decode_bgr(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {exc}")

        if detector is not None:
            det = detector.detect(bgr, min_sharpness=min_sharpness)
            if not det.card_present:
                continue
            crop = det.dewarp(bgr)
        else:
            crop = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        for score, card_id in catalog.search(catalog.embedder.embed(crop), top_k=k):
            score_map[card_id] += score

    hits = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    if not hits:
        return JSONResponse({"card_present": False,
                             "_timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)}})

    best_id, best_score = hits[0]
    return JSONResponse({
        "card_present": True,
        "card_id":      best_id,
        "confidence":   round(best_score, 4),
        "alternatives": [{"card_id": cid, "confidence": round(s, 4)} for cid, s in hits[1:]],
        "_timing":      {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="CollectorVision identification server")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--catalog", type=Path, help="Path to a local .npz catalog file")
    g.add_argument("--hfd",     nargs=2, metavar=("REPO", "KEY"),
                   help="Auto-download from HuggingFace: --hfd REPO KEY")
    p.add_argument("--host",          default="127.0.0.1")
    p.add_argument("--port",          type=int, default=8000)
    p.add_argument("--top-k",         type=int, default=5)
    p.add_argument("--min-sharpness", type=float, default=0.0,
                   help="SimCC sharpness gate; 0=disabled. ~0.02 skips blank frames.")
    p.add_argument("--detector-none", action="store_true",
                   help="Skip corner detection — inputs are pre-cropped card images.")
    p.add_argument("--ssl",           action="store_true",
                   help="Serve over HTTPS using a self-signed certificate.")
    args = p.parse_args()

    configure(
        catalog=f"hf://{args.hfd[0]}/{args.hfd[1]}" if args.hfd else args.catalog,
        top_k=args.top_k,
        min_sharpness_val=args.min_sharpness,
        no_detector=args.detector_none,
    )

    if args.ssl:
        import subprocess, tempfile
        tmp  = tempfile.mkdtemp()
        cert, key = f"{tmp}/cert.pem", f"{tmp}/key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048",
                        "-keyout", key, "-out", cert, "-days", "365", "-nodes",
                        "-subj", "/CN=localhost"],
                       check=True, capture_output=True)
        uvicorn.run("server:app", host=args.host, port=args.port,
                    ssl_certfile=cert, ssl_keyfile=key)
    else:
        uvicorn.run("server:app", host=args.host, port=args.port)
