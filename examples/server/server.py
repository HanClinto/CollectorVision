#!/usr/bin/env python3
"""CollectorVision — plug-and-play card identification server.

A minimal FastAPI server that exposes card identification as a REST API.
Catalog and detector load once at startup and are reused across requests.

Usage
-----
    pip install "collector-vision[server]"

    python server.py --catalog mtg
    python server.py --catalog mtg --catalog pokemon
    python server.py --catalog mtg:tcgplayer
    python server.py --catalog ./milo1-scryfall-mtg-2026-04.npz
    python server.py --hfd HanClinto/milo scryfall-mtg
    python server.py --catalog ./catalog.npz --ssl

Multiple ``--catalog`` flags load several games into one server; every
catalog must share the same embedding model. Requests may pin a search to one
loaded game via a ``"game"`` field/param, or omit it to search all of them.
Some games have more than one Catalog v2 source with different metadata
fields (e.g. MTG has both ``scryfall`` and ``tcgplayer``) — pick one with
``--catalog game:source``, e.g. ``--catalog mtg:tcgplayer``. Responses include
which ``source`` a match came from, since metadata field names differ by
source.

Endpoints
---------
    POST /identify            JSON body (supports rolling embedding buffer — see below).
    POST /identify/upload     Multipart form, single image. Simpler for curl / testing.
    POST /identify/embeddings Precomputed embedding vector(s) — no image, no detection.
    GET  /health              {"status": "ok"}
    GET  /                    Redirects to /docs (Swagger UI).

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
import json
import logging
import time
from collections.abc import Sequence
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
# Scan logging — persists to a file so events survive past a screen/terminal
# scrollback. One line per /identify* call, independent of uvicorn's access log.
# ---------------------------------------------------------------------------

scan_logger = logging.getLogger("collectorvision.scans")
scan_logger.setLevel(logging.INFO)
scan_logger.propagate = False
if not scan_logger.handlers:
    _log_dir = Path(__file__).resolve().parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    _handler = logging.FileHandler(_log_dir / "scans.log")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    scan_logger.addHandler(_handler)
    scan_logger.addHandler(logging.StreamHandler())


def _log_scan(endpoint: str, result: dict) -> None:
    if result.get("card_present"):
        scan_logger.info(
            "%s card_present=True card_id=%s game=%s source=%s name=%r confidence=%s ms=%s",
            endpoint,
            result.get("card_id"),
            result.get("game"),
            result.get("source"),
            result.get("name"),
            result.get("confidence"),
            (result.get("_timing") or {}).get("total_ms"),
        )
    else:
        scan_logger.info(
            "%s card_present=False sharpness=%s",
            endpoint,
            result.get("sharpness"),
        )

# ---------------------------------------------------------------------------
# Configuration — set before the lifespan starts (TestClient or uvicorn.run)
# ---------------------------------------------------------------------------

# One or more games/catalogs to load. A bare game name (e.g. "mtg") resolves
# to the recommended Catalog v2 snapshot for that game; paths and hf:// URIs
# still load Catalog v1 files, per collector_vision.Catalog.load()'s dispatch.
catalog_sources: list[str | Path] = []
top_k_default: int = 5
min_sharpness: float = 0.0
detector_none: bool = False
embeddings_only: bool = False
min_prior_similarity: float = 0.7  # drop prior embeddings with cosine sim < this


def configure(
    catalog: str | Path | Sequence[str | Path] | None = None,
    top_k: int = 5,
    min_sharpness_val: float = 0.0,
    no_detector: bool = False,
    embeddings_only_mode: bool = False,
    min_prior_sim: float = 0.7,
) -> None:
    """Configure the server before startup (used by tests and scripts).

    ``catalog`` may be a single game name/path, or a sequence of them to load
    multiple games into one server (e.g. ``["mtg", "pokemon"]``).
    """
    global catalog_sources, top_k_default, min_sharpness, detector_none
    global embeddings_only, min_prior_similarity
    if catalog is None:
        catalog_sources = []
    elif isinstance(catalog, (str, Path)):
        catalog_sources = [catalog]
    else:
        catalog_sources = list(catalog)
    top_k_default = top_k
    min_sharpness = min_sharpness_val
    detector_none = no_detector
    embeddings_only = embeddings_only_mode
    min_prior_similarity = min_prior_sim


def _catalog_embedding_identity(catalog: cvg.CatalogLike) -> str:
    """Stable key identifying which embedder a catalog requires."""
    embedding = getattr(catalog, "embedding", None)  # CatalogV2
    if embedding is not None:
        return embedding.model
    return json.dumps(getattr(catalog, "embedder_spec", {}), sort_keys=True)  # CatalogV1


def _parse_catalog_spec(raw: str | Path) -> tuple[str | Path, str | None]:
    """Parse an optional ``game:source`` spec, e.g. ``"mtg:tcgplayer"``.

    Only applies to bare Catalog v2 game identifiers — games can have more
    than one source (e.g. MTG has both ``scryfall`` and ``tcgplayer``
    snapshots, with different metadata fields). Paths and ``hf://`` URIs
    (Catalog v1) pass through unchanged.
    """
    if not isinstance(raw, str):
        return raw, None
    if raw.startswith("hf://") or raw.lower().endswith(".npz") or "/" in raw or "\\" in raw:
        return raw, None
    if ":" in raw:
        game, source = raw.split(":", 1)
        return game, (source or None)
    return raw, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not catalog_sources:
        raise RuntimeError("No catalog configured. Call configure() or use --catalog / --hfd.")
    catalogs = {}
    for raw_source in catalog_sources:
        game, source = _parse_catalog_spec(raw_source)
        catalog = cvg.Catalog.load(game, source=source) if source else cvg.Catalog.load(game)
        catalogs[str(game)] = catalog
    identities = {_catalog_embedding_identity(c) for c in catalogs.values()}
    if len(identities) > 1:
        raise RuntimeError(
            f"Loaded catalogs use different embedding models ({sorted(identities)}); "
            "every catalog on one server must share the same embedder."
        )
    app.state.catalogs = catalogs
    app.state.detector = None if detector_none or embeddings_only else cvg.NeuralCornerDetector()
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


def _hits_for_catalog(catalog: cvg.CatalogLike, game: str, query_emb: np.ndarray, top_k: int) -> list[dict]:
    """Search one catalog, attaching record metadata (name/identifiers) when available.

    Catalog v2 exposes ``search_records`` with the full card record; Catalog v1
    only exposes bare ``(score, card_id)`` pairs, so those fields are ``None``.
    Metadata field names differ by source (e.g. MTG's ``scryfall`` source uses
    ``set``/``set_name``; its ``tcgplayer`` source uses ``set`` for the full
    set name instead) — callers should branch on ``source`` when mapping.
    """
    source = getattr(catalog, "source", None)
    if hasattr(catalog, "search_records"):
        return [
            {
                "score": record["score"],
                "id": record["id"],
                "game": game,
                "source": source,
                "name": record.get("name"),
                "identifiers": record.get("identifiers") or {},
                "metadata": record.get("metadata"),
            }
            for record in catalog.search_records(query_emb, top_k=top_k)
        ]
    return [
        {
            "score": score,
            "id": cid,
            "game": game,
            "source": source,
            "name": None,
            "identifiers": {},
            "metadata": None,
        }
        for score, cid in catalog.search(query_emb, top_k=top_k)
    ]


def _search_multi(
    catalogs: dict[str, cvg.CatalogLike],
    query_emb: np.ndarray,
    top_k: int,
    game: str | None = None,
) -> list[dict]:
    """Search one game's catalog, or merge top_k hits across every loaded game."""
    if game is not None:
        if game not in catalogs:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown game {game!r}; loaded games: {sorted(catalogs)}",
            )
        hits = _hits_for_catalog(catalogs[game], game, query_emb, top_k)
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits[:top_k]

    merged = [
        hit for g, cat in catalogs.items() for hit in _hits_for_catalog(cat, g, query_emb, top_k)
    ]
    merged.sort(key=lambda hit: hit["score"], reverse=True)
    return merged[:top_k]


def _alternative(hit: dict) -> dict:
    return {
        "card_id": hit["id"],
        "game": hit["game"],
        "name": hit["name"],
        "confidence": round(float(hit["score"]), 4),
    }


def _combine_embeddings(embeddings: list[list[float]]) -> np.ndarray:
    """Average multiple frame embeddings of the same physical card into one query vector.

    Averaging (rather than summing) keeps the reported cosine-similarity
    ``confidence`` on the same ~[-1, 1] scale as a single-frame query; ranking
    is identical either way since it's just a positive scalar multiple.
    """
    if not embeddings:
        raise ValueError("embeddings must contain at least one vector")
    arrays = [np.array(e, dtype=np.float32) for e in embeddings]
    dim = arrays[0].shape
    if any(a.shape != dim for a in arrays):
        raise ValueError("all embeddings must be the same length")
    return np.stack(arrays).mean(axis=0)


def _identify(
    bgr: np.ndarray,
    catalogs: dict[str, cvg.CatalogLike],
    detector: cvg.NeuralCornerDetector | None,
    top_k: int,
    prior_embeddings: list[list[float]] | None = None,
    game: str | None = None,
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

    # Embed the current frame — any loaded catalog's embedder works, they all share one model
    current_emb = next(iter(catalogs.values())).embedder.embed(crop)

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

    hits = _search_multi(catalogs, search_emb, top_k, game=game)
    best = hits[0]

    result = {
        "card_present": True,
        "card_id": best["id"],
        "game": best["game"],
        "source": best["source"],
        "name": best["name"],
        "identifiers": best["identifiers"],
        "metadata": best["metadata"],
        "confidence": round(float(best["score"]), 4),
        "alternatives": [_alternative(hit) for hit in hits[1:]],
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
    if embeddings_only:
        raise HTTPException(status_code=404, detail="Image endpoints are disabled in embeddings-only mode")
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
    game = body.get("game")

    result = _identify(
        bgr, request.app.state.catalogs, request.app.state.detector, top_k, prior, game
    )
    _log_scan("/identify", result)
    return JSONResponse(result)


@app.post("/identify/upload")
async def identify_upload(
    request: Request,
    file: UploadFile = File(...),
    top_k: int | None = None,
    game: str | None = None,
):
    """Identify a card from an uploaded image file.

    Simpler than ``/identify`` for curl / browser testing.  Does not support
    the rolling embedding buffer — use ``/identify`` for live-camera clients.

    Example::

        curl -X POST http://localhost:8000/identify/upload -F "file=@card.jpg"
    """
    if embeddings_only:
        raise HTTPException(status_code=404, detail="Image endpoints are disabled in embeddings-only mode")
    data = await file.read()
    try:
        bgr = _decode_bgr(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    k = top_k if top_k is not None else top_k_default
    result = _identify(bgr, request.app.state.catalogs, request.app.state.detector, k, game=game)
    _log_scan("/identify/upload", result)
    return JSONResponse(result)


@app.post("/identify/embeddings")
async def identify_embeddings(request: Request):
    """Identify a card from one or more precomputed embedding vectors.

    Skips detection/embedding entirely — for clients (e.g. an on-device driver)
    that already ran the corner detector and embedder locally and only need
    the catalog lookup.

    Body::

        {
          "embeddings": [[...128 floats...], [...], ...],
          "game": "mtg",
          "top_k": 5
        }

    ``embeddings`` holds one or more 128-d vectors for the *same physical
    card* — for example one per lighting frame (dome/left/right), or more in
    the future for multi-frame consensus. They are averaged into a single
    query vector before searching, keeping ``confidence`` on the same scale
    as a single-frame query.

    ``game`` is optional; when omitted, every loaded catalog is searched and
    the best overall match is returned along with which game it came from.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    raw_embeddings = body.get("embeddings")
    if not raw_embeddings or not isinstance(raw_embeddings, list):
        raise HTTPException(status_code=400, detail="Missing or empty 'embeddings' field")

    try:
        query_emb = _combine_embeddings(raw_embeddings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    game = body.get("game")
    top_k = int(body.get("top_k", top_k_default))

    t0 = time.perf_counter()
    hits = _search_multi(request.app.state.catalogs, query_emb, top_k, game=game)
    best = hits[0]

    result = {
        "card_present": True,
        "card_id": best["id"],
        "game": best["game"],
        "source": best["source"],
        "name": best["name"],
        "identifiers": best["identifiers"],
        "metadata": best["metadata"],
        "confidence": round(float(best["score"]), 4),
        "alternatives": [_alternative(hit) for hit in hits[1:]],
        "embedding_count": len(raw_embeddings),
        "_timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
    }
    _log_scan("/identify/embeddings", result)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="CollectorVision identification server")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--catalog",
        action="append",
        help=(
            "Game name (for v2), optionally 'game:source' (e.g. 'mtg:tcgplayer'), "
            "or a local .npz catalog path. Repeat to load multiple games."
        ),
    )
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
        "--embeddings-only",
        action="store_true",
        help="Load only catalogs; disable image endpoints and all ONNX vision models.",
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
        embeddings_only_mode=args.embeddings_only,
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
        uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=cert, ssl_keyfile=key)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
