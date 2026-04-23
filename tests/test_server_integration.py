"""Integration tests for examples/server/server.py.

Uses FastAPI's TestClient — no real HTTP server needed.  Exercises the full
pipeline (detect → dewarp → embed → search) with the real catalog and sample
image, keeping the example and its integration test synonymous.
"""
import base64
import sys
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"

sys.path.insert(0, str(ROOT / "examples" / "server"))

from server import app, configure  # noqa: E402
from fastapi.testclient import TestClient

SCRYING_GLASS_ID = "7286819f-6c57-4503-898c-528786ad86e9"


@pytest.fixture(scope="module")
def client():
    configure(catalog="hf://HanClinto/milo/scryfall-mtg")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def client_no_detector():
    configure(catalog="hf://HanClinto/milo/scryfall-mtg", no_detector=True)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_bytes():
    if not SAMPLE.exists():
        pytest.skip(f"Sample image not found: {SAMPLE}")
    return SAMPLE.read_bytes()


# ---------------------------------------------------------------------------
# /health  /
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_redirects(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"] == "/docs"


# ---------------------------------------------------------------------------
# /identify  (JSON endpoint)
# ---------------------------------------------------------------------------

def test_identify_missing_base64(client):
    r = client.post("/identify", json={"not_base64": "x"})
    assert r.status_code == 400


def test_identify_bad_base64(client):
    r = client.post("/identify", json={"_base64": "!!!notbase64!!!"})
    assert r.status_code == 400


def test_identify_scrying_glass(client, sample_bytes):
    b64 = base64.b64encode(sample_bytes).decode()
    r   = client.post("/identify", json={"_base64": b64})
    assert r.status_code == 200

    body = r.json()
    assert body["card_present"] is True
    assert body["card_id"] == SCRYING_GLASS_ID
    assert body["confidence"] > 0.8
    assert "alternatives" in body
    assert "crop_jpeg" in body
    assert "_timing" in body

    # Response must include the embedding so clients can build a rolling buffer
    emb = body["embedding"]
    assert isinstance(emb, list)
    assert len(emb) == 128


def test_identify_rolling_buffer(client, sample_bytes):
    """Prior embeddings from previous frames improve the consensus search."""
    b64 = base64.b64encode(sample_bytes).decode()

    # First call — get the embedding for this frame
    r1  = client.post("/identify", json={"_base64": b64})
    emb = r1.json()["embedding"]

    # Second call — send that embedding back as a prior; result should still be correct
    r2   = client.post("/identify", json={"_base64": b64, "prior_embeddings": [emb]})
    body = r2.json()
    assert body["card_present"] is True
    assert body["card_id"] == SCRYING_GLASS_ID
    assert body["confidence"] > 0.8


# ---------------------------------------------------------------------------
# /identify/upload  (single-image form endpoint)
# ---------------------------------------------------------------------------

def test_identify_upload_scrying_glass(client, sample_bytes):
    r = client.post(
        "/identify/upload",
        files={"file": ("card.jpg", sample_bytes, "image/jpeg")},
    )
    assert r.status_code == 200

    body = r.json()
    assert body["card_present"] is True
    assert body["card_id"] == SCRYING_GLASS_ID
    assert body["confidence"] > 0.8
    assert "embedding" in body   # upload endpoint also returns embedding
    assert "crop_jpeg" in body


# ---------------------------------------------------------------------------
# detector_none mode
# ---------------------------------------------------------------------------

def test_identify_detector_none(client_no_detector, sample_bytes):
    b64 = base64.b64encode(sample_bytes).decode()
    r   = client_no_detector.post("/identify", json={"_base64": b64})
    assert r.status_code == 200
    body = r.json()
    assert body["card_present"] is True
    assert "card_id" in body
    assert "embedding" in body
