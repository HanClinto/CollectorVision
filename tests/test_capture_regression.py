"""Regression tests for captured browser frames.

Captures are **permanent historical records** — they should never need to be
deleted to keep the test suite green.  Each capture exercises a specific device
/ browser combination and may include evidence of a known bug.

Drop capture bundles from the web-scanner debug dock into::

    tests/fixtures/captures/

Each capture is a **single gzip-compressed JSON file** (``*.json.gz``) produced
by the **Capture** button in the debug dock.  Key fields:

* ``framePng``       — base64-encoded lossless PNG of the ``processCanvas``.
* ``processCanvas``  — recorded pixel dimensions.
* ``orderedCorners`` — browser-reported corners **at the time of capture**
  (may be wrong if taken during a browser bug).
* ``expectedCardId`` — Scryfall UUID of the correct card (set by the ingest
  script from the GitHub issue).
* ``pythonCorners``  — Python CPU reference corners (set by ingest script).
* ``knownIssue``     — GitHub issue URL if this capture documents a live bug,
  or ``null`` if it is a clean healthy capture.
* ``consoleLog``     — full runtime log at capture time.

What is asserted
----------------
**All captures:**
  * Frame decodes and dimensions match ``processCanvas``.
  * Python detector finds ``card_present=True`` with ``sharpness >= 0.02``.
  * Python corners have not drifted from the stored ``pythonCorners`` reference
    (catches regressions in the Python library/model).
  * Full-pipeline identity (detect → dewarp → embed → search) returns
    ``expectedCardId`` as the top-1 hit when that field is set.

**Clean captures** (``knownIssue`` is null/absent):
  * Browser corners agree with Python within 0.15 normalised units.
    Failure here means a regression was introduced in the browser pipeline.

**Bug-report captures** (``knownIssue`` is set):
  * Browser corners are asserted to **disagree** with Python (canary).
    If they suddenly agree, the bug has been fixed; the test fails with a
    "bug appears fixed" message — update the bundle and ingest a clean capture.

This design means:
  * Old bug captures never cause test failures just by existing.
  * If the underlying bug is silently fixed the canary fires, so we know.
  * Adding a clean post-fix capture to the suite is the way to confirm a fix.

Usage
-----
1. On the device, open Settings → tap **⬇ Download Debug Bundle**.
2. Share the ``.json.gz`` file and file a GitHub issue via **↗ Open GitHub Issue**.
3. Run ``python scripts/ingest_bug_reports.py <issue-number>`` to download and
   annotate locally.
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

        # Python-corner determinism: if the bundle was annotated with known-good
        # Python corners (by ingest_bug_reports.py), assert that re-running the
        # detector now produces the same result.  This catches model or
        # preprocessing regressions in the Python library itself.
        if bundle.get("pythonCorners") and detection.corners is not None:
            stored = sorted([c["x"], c["y"]] for c in bundle["pythonCorners"])
            fresh  = sorted(detection.corners.tolist())
            drift  = float(
                np.abs(np.array(stored) - np.array(fresh)).max()
            )
            self.assertLess(
                drift,
                0.05,
                f"{capture_path.name}: Python corner drift {drift:.3f} — "
                f"model output changed since capture was annotated "
                f"(stored={stored}, now={fresh})",
            )

        # Browser-vs-Python corner agreement.
        #
        # The logic depends on whether this capture documents a known bug:
        #
        #   knownIssue = None  → clean capture; assert browser ≈ python.
        #     Failure means a regression was introduced in the browser pipeline.
        #
        #   knownIssue = str   → bug-report capture; assert browser ≠ python
        #     (i.e. the bug is still present as documented).  If the assertion
        #     unexpectedly passes, the bug has been silently fixed and the test
        #     fails with a "bug appears fixed" message — ingest a new clean
        #     capture to confirm and clear the knownIssue field.
        if bundle.get("orderedCorners") and detection.corners is not None:
            browser_corners = sorted(
                [c["x"], c["y"]] for c in bundle["orderedCorners"]
            )
            python_corners = sorted(detection.corners.tolist())
            max_delta = float(
                np.abs(np.array(browser_corners) - np.array(python_corners)).max()
            )
            known_issue: str | None = bundle.get("knownIssue") or None
            if known_issue:
                # Canary: assert the browser is STILL producing wrong corners.
                # If it suddenly agrees with Python, the bug may be fixed.
                self.assertGreaterEqual(
                    max_delta,
                    0.15,
                    f"{capture_path.name}: browser corners unexpectedly agree with "
                    f"Python (delta={max_delta:.3f}) — the bug documented in "
                    f"'{known_issue}' may now be fixed!\n"
                    f"If so: clear the 'knownIssue' field in the bundle and "
                    f"ingest a new clean capture to confirm.",
                )
            else:
                # Normal health check: browser should match Python.
                self.assertLess(
                    max_delta,
                    0.15,
                    f"{capture_path.name}: browser corners differ from Python by "
                    f"{max_delta:.3f} — possible regression in the browser pipeline "
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
