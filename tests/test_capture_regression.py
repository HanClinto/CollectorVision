"""Regression tests for captured browser frames.

Drop capture bundles from the web-scanner debug dock into::

    tests/fixtures/captures/

Each capture is a **single gzip-compressed JSON file** (``*.json.gz``) produced
by the **Capture** button in the debug dock.  The bundle contains:

* ``framePng``       — base64-encoded lossless PNG of the ``processCanvas``
  pixels, ready to decode with ``cv2.imdecode()``.
* ``processCanvas``  — recorded pixel dimensions for cross-checking.
* ``orderedCorners`` / ``sharpness`` — browser-reported detection results.
* ``expectedCardId`` — Scryfall UUID identifying the correct card (``null``
  until a developer fills it in via ``ingest_bug_reports.py``).
* ``consoleLog``     — full runtime log at the time of capture.

Tests are generated dynamically: one test method per ``.json.gz`` file so
pytest reports pass/fail per capture independently.

What is asserted
----------------
All captures:
  * frame decodes correctly and dimensions match ``processCanvas``
  * Python detector finds ``card_present=True`` with ``sharpness >= 0.02``
  * Browser-vs-Python corner agreement within 0.15 normalised units
    (skipped when ``expectedCardId`` is set, because the capture may exist
    precisely because the browser was reporting wrong corners)

Captures with ``expectedCardId`` set:
  * the full Python pipeline (detect → dewarp → embed → search) returns
    the expected Scryfall ID as the top-1 hit

Usage
-----
1. On the device, open the debug dock and tap **Capture**.
2. One file downloads: ``cv_…_capture.json.gz``.
3. Copy it into ``tests/fixtures/captures/``.
4. Run::

       python -m pytest tests/test_capture_regression.py -v
"""
from __future__ import annotations

import base64
import gzip
import json
import unittest
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
    """One test per .json.gz in tests/fixtures/captures/.

    The class has no fixed test_ methods; they are attached dynamically below
    so that pytest counts and reports each capture file individually.
    """

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
            self.skipTest("no detector (no captures found at class setup time)")

        with gzip.open(capture_path, "rb") as fh:
            bundle: dict = json.load(fh)

        expected_card_id: str | None = bundle.get("expectedCardId") or None

        # Decode the inline base64 PNG back to a BGR numpy array.
        png_bytes = base64.b64decode(bundle["framePng"])
        bgr = cv2.imdecode(
            np.frombuffer(png_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(bgr, f"Could not decode framePng in {capture_path.name}")

        h, w = bgr.shape[:2]
        if "processCanvas" in bundle:
            self.assertEqual(
                w,
                bundle["processCanvas"]["width"],
                f"{capture_path.name}: decoded width {w} != processCanvas.width "
                f"{bundle['processCanvas']['width']}",
            )
            self.assertEqual(
                h,
                bundle["processCanvas"]["height"],
                f"{capture_path.name}: decoded height {h} != processCanvas.height "
                f"{bundle['processCanvas']['height']}",
            )

        detection = self.detector.detect(bgr)

        self.assertTrue(
            detection.card_present,
            f"{capture_path.name}: card not detected "
            f"(python sharpness={detection.sharpness:.3f}, "
            f"browser sharpness={bundle.get('sharpness')})\n"
            f"Last console log entries:\n"
            + "\n".join(
                f"  [{e['level']}] {e['message']}"
                for e in bundle.get("consoleLog", [])[-10:]
            ),
        )
        self.assertGreater(
            detection.sharpness,
            0.02,
            f"{capture_path.name}: sharpness {detection.sharpness:.3f} below threshold",
        )

        # Browser-vs-Python corner agreement.
        # Skipped when expectedCardId is set: the capture may exist precisely
        # because the browser was reporting wrong corners (the Android bug).
        if (
            bundle.get("orderedCorners")
            and detection.corners is not None
            and not expected_card_id
        ):
            browser_corners = sorted(
                [c["x"], c["y"]] for c in bundle["orderedCorners"]
            )
            python_corners = sorted(detection.corners.tolist())
            max_delta = float(
                np.abs(np.array(browser_corners) - np.array(python_corners)).max()
            )
            self.assertLess(
                max_delta,
                0.15,
                f"{capture_path.name}: corners differ by {max_delta:.3f} "
                f"(browser={browser_corners}, python={python_corners})",
            )

        # Full-pipeline identity check when expectedCardId is populated.
        if expected_card_id:
            self.assertIsNotNone(
                self.catalog,
                "catalog not loaded — cannot run identity check",
            )
            crop = detection.dewarp(bgr)
            emb = self.catalog.embedder.embed(crop)
            hits = self.catalog.search(emb, top_k=1)
            _, top_id = hits[0]
            self.assertEqual(
                top_id,
                expected_card_id,
                f"{capture_path.name}: expected {expected_card_id!r}, got {top_id!r}",
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
