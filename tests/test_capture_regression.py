"""Regression tests for captured browser frames.

Drop capture pairs from the web-scanner debug dock into::

    tests/fixtures/captures/

Each pair of files is:

* ``cv_{timestamp}_frame.png``  — raw ``processCanvas`` pixels exported from
  the browser as a lossless PNG (RGBA channels).  ``cv2.imread()`` reads this
  as BGR (drops the alpha channel automatically) so it is ready to feed
  directly into the Python pipeline.
* ``cv_{timestamp}_meta.json`` — sidecar with geometry metadata including the
  reported corners and sharpness from the same frame (optional, but used to
  tighten assertions when present).

Tests are generated dynamically: one test method per PNG file so pytest
reports pass/fail per capture independently.

Usage
-----
1. On the device, open Settings → (debug dock) and tap **Capture**.
2. Two files download: ``cv_…_frame.png`` and ``cv_…_meta.json``.
3. Copy both into ``tests/fixtures/captures/``.
4. Run::

       python -m pytest tests/test_capture_regression.py -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = ROOT / "tests" / "fixtures" / "captures"


def _find_captures() -> list[Path]:
    if not CAPTURES_DIR.exists():
        return []
    return sorted(CAPTURES_DIR.glob("*_frame.png"))


class CaptureRegressionTests(unittest.TestCase):
    """One test per PNG in tests/fixtures/captures/.

    The class has no fixed test_ methods; they are attached dynamically below
    so that pytest counts and reports each capture file individually.
    """

    detector = None

    @classmethod
    def setUpClass(cls) -> None:
        # Only load the heavyweight detector if there are actually captures.
        if not _find_captures():
            return
        import collector_vision as cvg  # noqa: PLC0415
        cls.detector = cvg.NeuralCornerDetector()

    def _run_capture(self, frame_path: Path) -> None:
        if self.detector is None:
            self.skipTest("no detector (no captures found at class setup time)")

        meta_path = frame_path.with_name(
            frame_path.name.replace("_frame.png", "_meta.json")
        )
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        # Canvas toBlob() saves RGBA; cv2.imread() without IMREAD_UNCHANGED
        # drops the alpha channel and returns BGR — which is exactly what the
        # Python NeuralCornerDetector expects.
        bgr = cv2.imread(str(frame_path))
        self.assertIsNotNone(bgr, f"Could not load {frame_path.name}")

        h, w = bgr.shape[:2]
        # Cross-check against the sidecar if it ships processCanvas dimensions.
        if "processCanvas" in meta:
            self.assertEqual(
                w,
                meta["processCanvas"]["width"],
                f"{frame_path.name}: PNG width {w} != meta processCanvas.width "
                f"{meta['processCanvas']['width']}",
            )
            self.assertEqual(
                h,
                meta["processCanvas"]["height"],
                f"{frame_path.name}: PNG height {h} != meta processCanvas.height "
                f"{meta['processCanvas']['height']}",
            )

        detection = self.detector.detect(bgr)

        self.assertTrue(
            detection.card_present,
            f"{frame_path.name}: card not detected "
            f"(python sharpness={detection.sharpness:.3f}, "
            f"browser sharpness={meta.get('sharpness')})",
        )
        self.assertGreater(
            detection.sharpness,
            0.02,
            f"{frame_path.name}: sharpness {detection.sharpness:.3f} below threshold",
        )

        # If the browser reported corners, check they roughly agree with the
        # Python detector (within 0.15 normalised units — allows for model
        # non-determinism and corner-ordering differences).
        if meta.get("orderedCorners") and detection.corners is not None:
            import numpy as np  # noqa: PLC0415

            browser_corners = [
                [c["x"], c["y"]] for c in meta["orderedCorners"]
            ]
            python_corners = detection.corners.tolist()
            max_delta = float(
                np.abs(
                    np.array(sorted(browser_corners)) - np.array(sorted(python_corners))
                ).max()
            )
            self.assertLess(
                max_delta,
                0.15,
                f"{frame_path.name}: corners differ by {max_delta:.3f} "
                f"(browser={browser_corners}, python={python_corners})",
            )


# Dynamically attach one test_ method per capture file.
def _make_test(path: Path):
    def test_method(self: CaptureRegressionTests) -> None:
        self._run_capture(path)

    test_method.__name__ = f"test_capture_{path.stem}"
    test_method.__doc__ = f"Detect card in captured frame: {path.name}"
    return test_method


for _path in _find_captures():
    _test = _make_test(_path)
    setattr(CaptureRegressionTests, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()
