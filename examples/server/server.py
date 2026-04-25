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
    POST /identify          JSON body (supports rolling embedding buffer — see below).
    POST /identify/upload   Multipart form, single image. Simpler for curl / testing.
    GET  /health            {"status": "ok"}
    GET  /                  Redirects to /docs (Swagger UI).

Live-camera rolling buffer
--------------------------
Every response includes the embedding for that frame.  For a live feed, maintain
a client-side deque of the last N embeddings and send them back with the next
request.  The server averages the buffer with the current frame before searching,
giving a consensus identification without re-uploading any image data::

    from collections import deque
    buffer = deque(maxlen=5)

    while capturing:
        frame = grab_frame()
        result = requests.post("/identify", json={
            "_base64": to_b64(frame),
            "prior_embeddings": list(buffer),
        }).json()

        if result["card_present"]:
            buffer.append(result["embedding"])
            print(result["card_id"], result["confidence"])
"""

from __future__ import annotations

import base64
import time
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
# Configuration — set before the lifespan starts (TestClient or uvicorn.run)
# ---------------------------------------------------------------------------

catalog_source: str | Path | None = None
top_k_default: int = 5
min_sharpness: float = 0.0
detector_none: bool = False
min_prior_similarity: float = 0.7  # drop prior embeddings with cosine sim < this


def configure(
    catalog: str | Path | None = None,
    top_k: int = 5,
    min_sharpness_val: float = 0.0,
    no_detector: bool = False,
    min_prior_sim: float = 0.7,
) -> None:
    """Configure the server before startup (used by tests and scripts)."""
    global catalog_source, top_k_default, min_sharpness, detector_none, min_prior_similarity
    catalog_source = catalog
    top_k_default = top_k
    min_sharpness = min_sharpness_val
    detector_none = no_detector
    min_prior_similarity = min_prior_sim


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not catalog_source:
        raise RuntimeError("No catalog configured. Call configure() or use --catalog / --hfd.")
    app.state.catalog = cvg.Catalog.load(catalog_source)
    app.state.detector = None if detector_none else cvg.NeuralCornerDetector()
    yield


app = FastAPI(
    title="CollectorVision",
    description="Card identification API — feed it an image, get back a card identity.",
    version=cvg.__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _decode_bgr(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (unsupported format or corrupt data)")
    return bgr


def _identify(
    bgr: np.ndarray,
    catalog: cvg.Catalog,
    detector: cvg.NeuralCornerDetector | None,
    top_k: int,
    prior_embeddings: list[list[float]] | None = None,
) -> dict:
    t0 = time.perf_counter()

    # Detect + dewarp (or pass straight through if detection is disabled)
    sharpness = None
    if detector is not None:
        det = detector.detect(bgr, min_sharpness=min_sharpness)
        sharpness = det.sharpness
        if not det.card_present:
            return {
                "card_present": False,
                "sharpness": sharpness,
                "_timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
            }
        crop = det.dewarp(bgr)
    else:
        crop = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # Thumbnail of the dewarped crop for visual confirmation
    crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    h, w = crop_bgr.shape[:2]
    scale = min(1.0, 300 / max(h, w))
    if scale < 1.0:
        crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    crop_jpeg = base64.b64encode(buf.tobytes()).decode()

    # Embed the current frame
    current_emb = catalog.embedder.embed(crop)

    # Sum prior embeddings from the client's rolling buffer with the current frame.
    # Priors below min_prior_similarity (cosine sim, via dot product on unit vectors)
    # are discarded — bad corner grabs produce distant vectors that would dilute the sum.
    # Renormalization is skipped: scaling a query doesn't affect cosine-similarity
    # rankings against a normalized gallery.
    if prior_embeddings:
        kept = [current_emb]
        for e in prior_embeddings:
            e_arr = np.array(e, dtype=np.float32)
            if float(np.dot(current_emb, e_arr)) >= min_prior_similarity:
                kept.append(e_arr)
        search_emb = np.stack(kept).sum(axis=0)
    else:
        search_emb = current_emb

    hits = catalog.search(search_emb, top_k=top_k)
    best_score, best_id = hits[0]

    result = {
        "card_present": True,
        "card_id": best_id,
        "confidence": round(float(best_score), 4),
        "alternatives": [{"card_id": cid, "confidence": round(float(s), 4)} for s, cid in hits[1:]],
        "embedding": current_emb.tolist(),  # client stores this in its rolling buffer
        "crop_jpeg": crop_jpeg,
        "_timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
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
    """Identify a card from a base64-encoded image.

    Body::

        {
          "_base64": "<JPEG or PNG as base64>",
          "top_k": 5,
          "prior_embeddings": [[...128 floats...], ...]
        }

    ``prior_embeddings`` is optional.  Populate it from the ``"embedding"``
    fields of recent responses to improve identification accuracy across a
    live camera feed without re-uploading image data.  Priors whose cosine
    similarity with the current frame falls below the server's
    ``min_prior_similarity`` threshold are silently dropped before the sum.

    Response includes ``"embedding"`` — the 128-d vector for this frame.
    Add it to your client-side rolling buffer for the next request.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    b64 = body.get("_base64") or body.get("base64")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing '_base64' field")

    try:
        bgr = _decode_bgr(base64.b64decode(b64))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    prior = body.get("prior_embeddings") or []
    top_k = int(body.get("top_k", top_k_default))

    return JSONResponse(
        _identify(bgr, request.app.state.catalog, request.app.state.detector, top_k, prior)
    )


@app.post("/identify/upload")
async def identify_upload(
    request: Request,
    file: UploadFile = File(...),
    top_k: int | None = None,
):
    """Identify a card from an uploaded image file.

    Simpler than ``/identify`` for curl / browser testing.  Does not support
    the rolling embedding buffer — use ``/identify`` for live-camera clients.

    Example::

        curl -X POST http://localhost:8000/identify/upload -F "file=@card.jpg"
    """
    data = await file.read()
    try:
        bgr = _decode_bgr(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    k = top_k if top_k is not None else top_k_default
    return JSONResponse(_identify(bgr, request.app.state.catalog, request.app.state.detector, k))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="CollectorVision identification server")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--catalog", type=Path, help="Path to a local .npz catalog file")
    g.add_argument(
        "--hfd",
        nargs=2,
        metavar=("REPO", "KEY"),
        help="Auto-download from HuggingFace: --hfd REPO KEY",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="SimCC sharpness gate; 0=disabled. ~0.02 skips blank frames.",
    )
    p.add_argument(
        "--min-prior-sim",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for rolling-buffer priors (0–1). "
        "Priors below this value are discarded before averaging.",
    )
    p.add_argument(
        "--detector-none",
        action="store_true",
        help="Skip corner detection — inputs are pre-cropped card images.",
    )
    p.add_argument(
        "--ssl", action="store_true", help="Serve over HTTPS using a self-signed certificate."
    )
    args = p.parse_args()

    configure(
        catalog=f"hf://{args.hfd[0]}/{args.hfd[1]}" if args.hfd else args.catalog,
        top_k=args.top_k,
        min_sharpness_val=args.min_sharpness,
        min_prior_sim=args.min_prior_sim,
        no_detector=args.detector_none,
    )

    if args.ssl:
        import subprocess
        import tempfile

        tmp = tempfile.mkdtemp()
        cert, key = f"{tmp}/cert.pem", f"{tmp}/key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                key,
                "-out",
                cert,
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        uvicorn.run(
            "server:app", host=args.host, port=args.port, ssl_certfile=cert, ssl_keyfile=key
        )
    else:
        uvicorn.run("server:app", host=args.host, port=args.port)
