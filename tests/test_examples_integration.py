import contextlib
import io
import json
import re
import runpy
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "examples" / "images"
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


@pytest.mark.hf
class QuickstartIntegrationTests(unittest.TestCase):
    def test_quickstart_script_prints_expected_card(self) -> None:
        real_urlopen = urllib.request.urlopen

        def fake_urlopen(request, *args, **kwargs):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            # Keep the example itself untouched: run the real script, but stub the
            # final Scryfall lookup so this integration test stays deterministic.
            if url == "https://api.scryfall.com/cards/7286819f-6c57-4503-898c-528786ad86e9":
                return _FakeResponse(
                    {
                        "name": "Scrying Glass",
                        "set_name": "Urza's Destiny",
                        "set": "uds",
                        "prices": {"usd": "0.40"},
                    }
                )
            return real_urlopen(request, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with contextlib.redirect_stdout(output):
                runpy.run_path(
                    str(ROOT / "examples" / "quickstart.py"),
                    run_name="__main__",
                )

        stdout = output.getvalue()
        self.assertIn("Detected corner sharpness=", stdout)
        self.assertIn(
            "Top match 7286819f-6c57-4503-898c-528786ad86e9",
            stdout,
        )
        self.assertIn("Name      Scrying Glass", stdout)
        self.assertIn("Set       Urza's Destiny (UDS)", stdout)


@pytest.mark.hf
class SampleImagesIntegrationTests(unittest.TestCase):
    """Integration tests for all three UUID-labelled sample images.

    Loads the catalog once per class to avoid redundant downloads.  Each test
    checks that the pipeline (detect → dewarp → embed → search) returns the
    card ID encoded in the filename as the top-1 match.
    """

    catalog = None
    detector = None

    @classmethod
    def setUpClass(cls) -> None:
        import cv2  # noqa: F401 — confirm OpenCV is importable before loading catalog
        import collector_vision as cvg

        cls.catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
        cls.detector = cvg.NeuralCornerDetector()

    def _assert_image_identifies_correctly(self, filename: str) -> None:
        import cv2

        m = _UUID_RE.search(filename)
        self.assertIsNotNone(m, f"No UUID found in filename: {filename}")
        expected_id = m.group(0).lower()

        path = IMAGES_DIR / filename
        bgr = cv2.imread(str(path))
        self.assertIsNotNone(bgr, f"Could not load image: {path}")

        detection = self.detector.detect(bgr)
        self.assertTrue(
            detection.card_present,
            f"No card detected in {filename} (sharpness={detection.sharpness})",
        )

        crop = detection.dewarp(bgr)
        emb = self.catalog.embedder.embed(crop)
        hits = self.catalog.search(emb, top_k=1)
        _, top_id = hits[0]

        self.assertEqual(
            top_id,
            expected_id,
            f"{filename}: expected {expected_id}, got {top_id}",
        )

    def test_sample_image(self) -> None:
        self._assert_image_identifies_correctly(
            "7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"
        )

    def test_hidden_name_image(self) -> None:
        self._assert_image_identifies_correctly(
            "d0def54b-9f0a-4ab1-9df9-25506a06350c_hidden_name.jpg"
        )

    def test_deep_skew_image(self) -> None:
        self._assert_image_identifies_correctly(
            "f31a6dfd-93d2-49c8-a357-9a707b9c42bd_deep_skew.jpg"
        )


if __name__ == "__main__":
    unittest.main()
