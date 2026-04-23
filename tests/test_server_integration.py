"""Integration tests for examples/server/server.py.

Uses FastAPI's TestClient so no real HTTP server is needed.  The test loads
the real catalog and detector (same as production) to exercise the full pipeline
end-to-end, keeping examples and integration tests synonymous.

The catalog is loaded from the HuggingFace cache if present, or downloaded on
first run.  Tests are skipped if the sample image is missing.
"""
import base64
import sys
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"

# Add examples/server to path so `import server` resolves correctly when
# uvicorn reloads the module by name ("server:app").
sys.path.insert(0, str(ROOT / "examples" / "server"))

from server import app, configure  # noqa: E402  (path manipulation above)
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Shared TestClient with the full pipeline (detect + embed + search)."""
    configure(catalog="hf://HanClinto/milo/scryfall-mtg")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def client_no_detector():
    """TestClient with detection skipped — inputs are pre-cropped."""
    configure(catalog="hf://HanClinto/milo/scryfall-mtg", no_detector=True)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_bytes():
    if not SAMPLE.exists():
        pytest.skip(f"Sample image not found: {SAMPLE}")
    return SAMPLE.read_bytes()


# ---------------------------------------------------------------------------
# /health
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
# /identify  (base64 JSON endpoint)
# ---------------------------------------------------------------------------

def test_identify_missing_records(client):
    r = client.post("/identify", json={})
    assert r.status_code == 400


def test_identify_missing_base64(client):
    r = client.post("/identify", json={"records": [{"not_base64": "x"}]})
    assert r.status_code == 200
    rec = r.json()["records"][0]
    assert rec["_status"]["code"] == 400


def test_identify_bad_base64(client):
    r = client.post("/identify", json={"records": [{"_base64": "!!!notbase64!!!"}]})
    assert r.status_code == 200
    rec = r.json()["records"][0]
    assert rec["_status"]["code"] == 400


def test_identify_scrying_glass_base64(client, sample_bytes):
    b64 = base64.b64encode(sample_bytes).decode()
    r = client.post("/identify", json={"records": [{"_base64": b64}]})
    assert r.status_code == 200

    records = r.json()["records"]
    assert len(records) == 1
    rec = records[0]

    assert rec["card_present"] is True
    assert rec["card_id"] == "7286819f-6c57-4503-898c-528786ad86e9"
    assert rec["confidence"] > 0.8
    assert "alternatives" in rec
    assert "crop_jpeg" in rec
    assert "_timing" in rec


def test_identify_multiple_records(client, sample_bytes):
    b64 = base64.b64encode(sample_bytes).decode()
    r = client.post("/identify", json={"records": [{"_base64": b64}, {"_base64": b64}]})
    assert r.status_code == 200
    assert len(r.json()["records"]) == 2


# ---------------------------------------------------------------------------
# /identify/upload  (multipart form endpoint)
# ---------------------------------------------------------------------------

def test_identify_upload_no_files(client):
    r = client.post("/identify/upload")
    assert r.status_code == 422  # FastAPI validation: missing required field


def test_identify_upload_scrying_glass(client, sample_bytes):
    r = client.post(
        "/identify/upload",
        files={"files": ("card.jpg", sample_bytes, "image/jpeg")},
    )
    assert r.status_code == 200

    body = r.json()
    assert body["card_present"] is True
    assert body["card_id"] == "7286819f-6c57-4503-898c-528786ad86e9"
    assert body["confidence"] > 0.8
    assert "crop_jpeg" in body


def test_identify_upload_multi_frame_aggregation(client, sample_bytes):
    """Multiple uploads of the same image should still find the card."""
    r = client.post(
        "/identify/upload",
        files=[
            ("files", ("frame1.jpg", sample_bytes, "image/jpeg")),
            ("files", ("frame2.jpg", sample_bytes, "image/jpeg")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["card_present"] is True
    assert body["card_id"] == "7286819f-6c57-4503-898c-528786ad86e9"


# ---------------------------------------------------------------------------
# detector_none mode (pre-cropped inputs)
# ---------------------------------------------------------------------------

def test_identify_upload_detector_none(client_no_detector, sample_bytes):
    """With detector_none=True the raw photo is passed directly to the embedder."""
    r = client_no_detector.post(
        "/identify/upload",
        files={"files": ("card.jpg", sample_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    # Without dewarping the raw photo may not match cleanly, but the server
    # should always return a valid response shape.
    assert body["card_present"] is True
    assert "card_id" in body
    assert "confidence" in body
