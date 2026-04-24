"""Regression tests for captured browser frames.

Captures are permanent historical records — they should never need to be
deleted to keep the suite green.  A capture documents a specific device,
browser, and card at a point in time.

The primary assertion is end-to-end: given a frame from the camera,
does the full Python pipeline return the expected card ID?

Drop ``.json.gz`` bundles (produced by the **⬇ Download Debug Bundle**
button in the web scanner) into ``tests/fixtures/captures/``, then run::

    python scripts/ingest_bug_reports.py <issue-number>

to annotate each bundle with ``expectedCardId`` and generate per-pipeline
manifests.  The only field required for a test to assert anything meaningful
is ``expectedCardId``.

What is asserted
----------------
* Frame decodes and dimensions match ``processCanvas``.
* Full Python pipeline (detect → dewarp → embed → search) returns
  ``expectedCardId`` as the top-1 hit.  Skipped (with a warning) if
  ``expectedCardId`` is not set in the bundle.

Cross-pipeline comparison (corners, embeddings) is handled by
``test_pipeline_consistency.py``, which reads pre-generated manifests.
"""
from __future__ import annotations

import base64
import gzip
import json
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = ROOT / "tests" / "fixtures" / "captures"


def _find_captures() -> list[Path]:
    if not CAPTURES_DIR.exists():
        return []
    return sorted(CAPTURES_DIR.glob("*.json.gz"))


class CaptureRegressionTests(unittest.TestCase):
    """One test per .json.gz in tests/fixtures/captures/."""

    detector = None
    catalog = None

    @classmethod
    def setUpClass(cls) -> None:
        if not _find_captures():
            return
        import collector_vision as cvg  # noqa: PLC0415
        cls.detector = cvg.NeuralCornerDetector()
        cls.catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

    def _run_capture(self, capture_path: Path) -> None:
        if self.detector is None:
            self.skipTest("no captures found at class setup time")

        with gzip.open(capture_path, "rb") as fh:
            bundle: dict = json.load(fh)

        expected_card_id: str | None = bundle.get("expectedCardId") or None

        # Decode the frame.
        png_bytes = base64.b64decode(bundle["framePng"])
        bgr = cv2.imdecode(
            np.frombuffer(png_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(bgr, f"Could not decode framePng in {capture_path.name}")

        # Dimension sanity-check.
        h, w = bgr.shape[:2]
        if "processCanvas" in bundle:
            self.assertEqual(w, bundle["processCanvas"]["width"],
                f"{capture_path.name}: decoded width mismatch")
            self.assertEqual(h, bundle["processCanvas"]["height"],
                f"{capture_path.name}: decoded height mismatch")

        # --- End-to-end identity: picture → card ID ---
        if not expected_card_id:
            warnings.warn(
                f"{capture_path.name}: no expectedCardId set — "
                f"run 'python scripts/ingest_bug_reports.py' to annotate. "
                f"Skipping identity assertion.",
                stacklevel=2,
            )
            return

        detection = self.detector.detect(bgr)
        self.assertTrue(
            detection.card_present,
            f"{capture_path.name}: no card detected in frame "
            f"(sharpness={detection.sharpness:.3f}).",
        )

        crop = detection.dewarp(bgr)
        emb  = self.catalog.embedder.embed(crop)
        hits = self.catalog.search(emb, top_k=1)
        _, top_id = hits[0]

        self.assertEqual(
            top_id,
            expected_card_id,
            f"{capture_path.name}: expected {expected_card_id!r}, got {top_id!r}.",
        )


def _make_test(path: Path):
    def test_method(self: CaptureRegressionTests) -> None:
        self._run_capture(path)

    test_method.__name__ = f"test_capture_{path.stem.replace('.', '_')}"
    test_method.__doc__ = f"Detect card in captured frame: {path.name}"
    return test_method


for _path in _find_captures():
    _test = _make_test(_path)
    setattr(CaptureRegressionTests, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()

