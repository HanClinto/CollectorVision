#!/usr/bin/env python3
"""Nifty Scanner overlay served by CollectorVision + FastAPI.

This ports the small Node/Socket.IO server from
https://github.com/paul-lrr/nifty-recognizer to Python while using
CollectorVision for card identification. The original overlay/controller pages
are served from ``public/`` and talk to a tiny Socket.IO-compatible browser shim
that forwards ``card_image`` messages over a native WebSocket.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import collector_vision as cvg

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
CARDS = PUBLIC / "cards"
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

catalog_source: str | Path | None = None
top_k_default = 5
min_sharpness = 0.0
detector_none = False
min_prior_similarity = 0.7
log = logging.getLogger("nifty_cv_scanner")


def _format_elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def configure(
    catalog: str | Path | None = None,
    top_k: int = 5,
    min_sharpness_val: float = 0.0,
    no_detector: bool = False,
    min_prior_sim: float = 0.7,
) -> None:
    """Configure the app before startup."""
    global catalog_source, top_k_default, min_sharpness, detector_none, min_prior_similarity
    catalog_source = catalog
    top_k_default = top_k
    min_sharpness = min_sharpness_val
    detector_none = no_detector
    min_prior_similarity = min_prior_sim


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, event: str, data: dict[str, Any], exclude: WebSocket | None = None) -> None:
        message = {"event": event, "data": data}
        stale: list[WebSocket] = []
        for websocket in self._connections:
            if websocket is exclude:
                continue
            try:
                await websocket.send_json(message)
            except RuntimeError:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not catalog_source:
        raise RuntimeError("No catalog configured. Use --catalog or --hfd.")

    startup = time.perf_counter()
    log.info("Starting CollectorVision Nifty Scanner")

    catalog_start = time.perf_counter()
    log.info("Loading catalog from %s", catalog_source)
    app.state.catalog = cvg.Catalog.load(catalog_source)
    log.info(
        "Catalog ready in %s (%s cards, source=%s)",
        _format_elapsed(catalog_start),
        len(app.state.catalog.card_ids),
        app.state.catalog.source,
    )

    if detector_none:
        log.info("Corner detector disabled; incoming frames are treated as pre-cropped cards")
        app.state.detector = None
    else:
        detector_start = time.perf_counter()
        log.info("Loading Cornelius corner detector ONNX model")
        app.state.detector = cvg.NeuralCornerDetector()
        log.info("Corner detector ready in %s", _format_elapsed(detector_start))

    embedder_start = time.perf_counter()
    log.info("Loading Milo embedder ONNX model")
    _ = app.state.catalog.embedder
    log.info("Embedder ready in %s", _format_elapsed(embedder_start))
    log.info("Nifty Scanner ready in %s", _format_elapsed(startup))
    yield


app = FastAPI(
    title="CollectorVision Nifty Scanner",
    description="Python/CollectorVision port of paul-lrr/nifty-recognizer.",
    version=cvg.__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _decode_bgr(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (unsupported format or corrupt data)")
    return bgr


def _card_image_src(card_id: str) -> str:
    if card_id.isdigit():
        return f"http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid={card_id}&type=card"
    if UUID_RE.match(card_id):
        return f"/card-image/{card_id}"
    return f"/cards/{card_id}.jpg"


def _oracle_id_for(catalog: cvg.Catalog, card_id: str) -> str | None:
    return catalog.card_to_oracle.get(card_id) or None


def _identify(
    bgr: np.ndarray,
    catalog: cvg.Catalog,
    detector: cvg.NeuralCornerDetector | None,
    top_k: int,
    prior_embeddings: list[list[float]] | None = None,
    min_sharpness_override: float | None = None,
    min_prior_similarity_override: float | None = None,
    rotation_invariant: bool = True,
) -> dict[str, Any]:
    start = time.perf_counter()
    sharpness = None
    detector_presence = None
    corners = None

    if detector is not None:
        detection = detector.detect(
            bgr,
            min_sharpness=min_sharpness if min_sharpness_override is None else min_sharpness_override,
        )
        sharpness = detection.sharpness
        detector_presence = detection.extra.get("presence")
        if detection.corners is not None:
            corners = detection.corners.tolist()
        if not detection.card_present:
            return {
                "card_present": False,
                "sharpness": sharpness,
                "detector_presence": detector_presence,
                "corners": corners,
                "_timing": {"total_ms": round((time.perf_counter() - start) * 1000, 1)},
            }
        crop = detection.dewarp(bgr)
    else:
        crop = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    height, width = crop_bgr.shape[:2]
    scale = min(1.0, 300 / max(height, width))
    if scale < 1.0:
        crop_bgr = cv2.resize(crop_bgr, (int(width * scale), int(height * scale)))
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    crop_jpeg = base64.b64encode(buf.tobytes()).decode()

    candidates: list[tuple[str, Image.Image]] = [("upright", crop)]
    if rotation_invariant:
        candidates.append(("rotated_180", cvg.rotate_card_180(crop)))

    best_result: tuple[float, str, str, np.ndarray, list[tuple[float, str]], dict[str, Any]] | None = None
    for orientation, candidate_crop in candidates:
        current_emb = catalog.embedder.embed(candidate_crop)
        search_emb = current_emb
        prior_stats: dict[str, Any] = {
            "provided": len(prior_embeddings or []),
            "kept": 0,
            "best_similarity": None,
        }
        if prior_embeddings:
            kept = [current_emb]
            prior_threshold = (
                min_prior_similarity
                if min_prior_similarity_override is None
                else min_prior_similarity_override
            )
            for embedding in prior_embeddings:
                embedding_arr = np.array(embedding, dtype=np.float32)
                similarity = float(np.dot(current_emb, embedding_arr))
                prior_stats["best_similarity"] = (
                    similarity
                    if prior_stats["best_similarity"] is None
                    else max(float(prior_stats["best_similarity"]), similarity)
                )
                if similarity >= prior_threshold:
                    kept.append(embedding_arr)
            prior_stats["kept"] = len(kept) - 1
            search_emb = np.stack(kept).sum(axis=0)

        search_norm = float(np.linalg.norm(search_emb))
        if search_norm > 0:
            search_emb = search_emb / search_norm

        hits = catalog.search(search_emb, top_k=top_k)
        score, card_id = hits[0]
        if best_result is None or score > best_result[0]:
            best_result = (score, card_id, orientation, current_emb, hits, prior_stats)

    if best_result is None:
        raise RuntimeError("No search candidates produced a result")

    best_score, best_id, best_orientation, current_emb, hits, prior_stats = best_result
    best_oracle_id = _oracle_id_for(catalog, best_id)
    result: dict[str, Any] = {
        "card_present": True,
        "card_id": best_id,
        "oracle_id": best_oracle_id,
        "secondaryId": best_oracle_id,
        "secondaryIdField": "oracleId" if best_oracle_id else None,
        "oracleId": best_oracle_id,
        "confidence": round(float(best_score), 4),
        "alternatives": [
            {
                "card_id": card_id,
                "oracle_id": _oracle_id_for(catalog, card_id),
                "confidence": round(float(score), 4),
            }
            for score, card_id in hits[1:]
        ],
        "embedding": current_emb.tolist(),
        "crop_jpeg": crop_jpeg,
        "image_src": _card_image_src(best_id),
        "orientation": best_orientation,
        "rotation_invariant": rotation_invariant,
        "prior_similarity": prior_stats["best_similarity"],
        "prior_embeddings_kept": prior_stats["kept"],
        "prior_embeddings_provided": prior_stats["provided"],
        "detector_presence": detector_presence,
        "corners": corners,
        "_timing": {"total_ms": round((time.perf_counter() - start) * 1000, 1)},
    }
    if sharpness is not None:
        result["sharpness"] = round(float(sharpness), 5)
    return result


def _records_response(result: dict[str, Any]) -> dict[str, Any]:
    record = {"_status": {"code": 200, "text": "OK"}, **result}
    return {"records": [record]}


def _optional_float(
    primary: dict[str, Any], fallback: dict[str, Any], *names: str
) -> float | None:
    for name in names:
        value = primary.get(name, fallback.get(name))
        if value is None or value == "":
            continue
        return float(value)
    return None


def _optional_bool(primary: dict[str, Any], fallback: dict[str, Any], default: bool, *names: str) -> bool:
    for name in names:
        value = primary.get(name, fallback.get(name))
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).lower() not in {"0", "false", "off", "no"}
    return default


async def _recognize_and_optionally_broadcast(
    request: Request,
    bgr: np.ndarray,
    top_k: int,
    prior_embeddings: list[list[float]] | None = None,
    broadcast: bool = False,
    min_sharpness_override: float | None = None,
    min_prior_similarity_override: float | None = None,
    rotation_invariant: bool = True,
) -> dict[str, Any]:
    result = _identify(
        bgr,
        request.app.state.catalog,
        request.app.state.detector,
        top_k,
        prior_embeddings,
        min_sharpness_override,
        min_prior_similarity_override,
        rotation_invariant,
    )
    if broadcast and result.get("card_present"):
        await manager.broadcast("card_image", {"auto": True, "src": result["image_src"]})
    return result


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(PUBLIC / "index.html", media_type="text/html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": cvg.__version__}


@app.get("/socket.io/socket.io.js", include_in_schema=False)
async def socket_shim() -> FileResponse:
    return FileResponse(PUBLIC / "socket.io.js", media_type="application/javascript")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")
            data = message.get("data") or {}
            if event == "card_image":
                await manager.broadcast("card_image", data, exclude=websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/cardmatch/{card_id}", include_in_schema=False)
async def cardmatch(card_id: str) -> PlainTextResponse:
    await manager.broadcast("card_image", {"auto": True, "src": _card_image_src(card_id)})
    return PlainTextResponse(card_id)


@app.get("/card-image/{card_id}", include_in_schema=False)
async def card_image(card_id: str) -> RedirectResponse:
    if card_id.isdigit():
        return RedirectResponse(
            url=f"http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid={card_id}&type=card"
        )
    if UUID_RE.match(card_id):
        return RedirectResponse(
            url=f"https://api.scryfall.com/cards/{card_id}?format=image&version=normal"
        )
    raise HTTPException(status_code=404, detail="Only Multiverse IDs and Scryfall UUIDs are supported")


@app.post("/identify")
async def identify(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    record_mode = False
    record = body
    if isinstance(body.get("records"), list) and body["records"]:
        record = body["records"][0]
        record_mode = True

    b64 = record.get("_base64") or record.get("base64")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing '_base64' field")

    try:
        bgr = _decode_bgr(base64.b64decode(b64))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await _recognize_and_optionally_broadcast(
        request,
        bgr,
        int(record.get("top_k", body.get("top_k", top_k_default))),
        record.get("prior_embeddings") or body.get("prior_embeddings") or [],
        bool(record.get("broadcast", body.get("broadcast", False))),
        _optional_float(record, body, "min_sharpness"),
        _optional_float(record, body, "min_prior_similarity"),
        _optional_bool(record, body, True, "rotation_invariant", "scan_upside_down_cards"),
    )
    return JSONResponse(_records_response(result) if record_mode else result)


@app.post("/recognize/upload")
async def recognize_upload(
    request: Request,
    file: UploadFile = File(...),
    top_k: int | None = None,
    broadcast: bool = True,
    min_sharpness: float | None = None,
    min_prior_similarity: float | None = None,
    rotation_invariant: bool = True,
) -> JSONResponse:
    data = await file.read()
    try:
        bgr = _decode_bgr(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await _recognize_and_optionally_broadcast(
        request,
        bgr,
        top_k if top_k is not None else top_k_default,
        broadcast=broadcast,
        min_sharpness_override=min_sharpness,
        min_prior_similarity_override=min_prior_similarity,
        rotation_invariant=rotation_invariant,
    )
    return JSONResponse(result)


@app.post("/identify/upload")
async def identify_upload(
    request: Request,
    file: UploadFile = File(...),
    top_k: int | None = None,
    min_sharpness: float | None = None,
    min_prior_similarity: float | None = None,
    rotation_invariant: bool = True,
) -> JSONResponse:
    return await recognize_upload(
        request,
        file,
        top_k=top_k,
        broadcast=False,
        min_sharpness=min_sharpness,
        min_prior_similarity=min_prior_similarity,
        rotation_invariant=rotation_invariant,
    )


app.mount("/cards", StaticFiles(directory=CARDS, check_dir=False), name="cards")
app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="public")


if __name__ == "__main__":
    import argparse

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="CollectorVision-backed Nifty Scanner overlay")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--catalog", type=Path, help="Path to a local CollectorVision .npz catalog")
    group.add_argument(
        "--hfd",
        nargs=2,
        metavar=("REPO", "KEY"),
        help="Auto-download from HuggingFace, e.g. --hfd HanClinto/milo scryfall-mtg",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="SimCC sharpness gate; 0 disables filtering, ~0.02 skips blank frames.",
    )
    parser.add_argument(
        "--min-prior-sim",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for rolling-buffer priors.",
    )
    parser.add_argument(
        "--detector-none",
        action="store_true",
        help="Skip corner detection for already-cropped card images.",
    )
    args = parser.parse_args()

    configure(
        catalog=f"hf://{args.hfd[0]}/{args.hfd[1]}" if args.hfd else args.catalog,
        top_k=args.top_k,
        min_sharpness_val=args.min_sharpness,
        no_detector=args.detector_none,
        min_prior_sim=args.min_prior_sim,
    )
    base_url = f"http://{args.host}:{args.port}"
    log.info("Starting web server on %s", base_url)
    log.info("Scanner:    %s/scanner.html", base_url)
    log.info("Overlay:    %s/card-view.html", base_url)
    log.info("Controller: %s/card-controller.html", base_url)
    uvicorn.run(app, host=args.host, port=args.port)
