import contextlib
import io
import re
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "examples" / "images"
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# Some samples intentionally obscure edition-specific text.  For those images,
# the embedding model can correctly identify the card while choosing a visually
# indistinguishable reprint from another set.  Keep the strict ID check for most
# samples, but allow known equivalent-print IDs for those ambiguous captures.
_ACCEPTED_EQUIVALENT_PRINTS = {
    "d0def54b-9f0a-4ab1-9df9-25506a06350c": {
        # Rush of Adrenaline — The List reprint of SOI #177, same oracle/art.
        "78fa3708-c56c-41fd-9e1b-5c61996d5dac",
    },
}


@pytest.mark.hf
class QuickstartIntegrationTests(unittest.TestCase):
    def test_quickstart_script_prints_expected_card(self) -> None:
        output = io.StringIO()
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

    def test_rotation_invariant_helper_chooses_strongest_orientation(self) -> None:
        module = runpy.run_path(
            str(ROOT / "examples" / "quickstart_rot_invariant.py"),
            run_name="collectorvision_quickstart_rot_invariant_test",
        )
        search_best_orientation = module["search_best_orientation"]

        class FakeEmbedder:
            def embed(self, images):
                self.images = images
                return ["upright_embedding", "rotated_embedding"]

        class FakeCatalog:
            def __init__(self) -> None:
                self.embedder = FakeEmbedder()

            def search(self, embedding, top_k: int):
                if embedding == "upright_embedding":
                    return [(0.70, "upright-card")]
                return [(0.92, "rotated-card")]

        catalog = FakeCatalog()
        crop = Image.new("RGB", (3, 2))

        result = search_best_orientation(catalog, crop, top_k=1)

        self.assertEqual(result.label, "rotated_180")
        self.assertEqual(result.best_card_id, "rotated-card")
        self.assertEqual(result.best_score, 0.92)
        self.assertEqual(len(catalog.embedder.images), 2)

    def test_eval_accuracy_rot_invariant_search_chooses_strongest_orientation(self) -> None:
        module = runpy.run_path(
            str(ROOT / "examples" / "eval_accuracy.py"),
            run_name="collectorvision_eval_accuracy_test",
        )
        search_hits = module["search_hits"]

        class FakeEmbedder:
            def embed(self, images):
                self.images = images
                return ["upright_embedding", "rotated_embedding"]

        class FakeCatalog:
            def __init__(self) -> None:
                self.embedder = FakeEmbedder()

            def search(self, embedding, top_k: int):
                if embedding == "upright_embedding":
                    return [(0.70, "upright-card")]
                return [(0.92, "rotated-card")]

        catalog = FakeCatalog()
        crop = Image.new("RGB", (3, 2))

        hits = search_hits(catalog, crop, top_k=1, rot_invariant=True)

        self.assertEqual(hits, ["rotated-card"])
        self.assertEqual(len(catalog.embedder.images), 2)

    def test_eval_accuracy_rotation_invariance_defaults_on_with_opt_out(self) -> None:
        module = runpy.run_path(
            str(ROOT / "examples" / "eval_accuracy.py"),
            run_name="collectorvision_eval_accuracy_parser_test",
        )
        parser = module["build_parser"]()

        default_args = parser.parse_args(["images", "--catalog", "catalog.npz"])
        opt_out_args = parser.parse_args(
            ["images", "--catalog", "catalog.npz", "--no-rot-invariant"]
        )
        verbose_args = parser.parse_args(["images", "--catalog", "catalog.npz", "--verbose"])
        debug_args = parser.parse_args(["images", "--catalog", "catalog.npz", "--debug"])

        self.assertTrue(default_args.rot_invariant)
        self.assertFalse(default_args.verbose)
        self.assertFalse(default_args.debug)
        self.assertFalse(opt_out_args.rot_invariant)
        self.assertTrue(verbose_args.verbose)
        self.assertTrue(debug_args.debug)
        self.assertEqual(debug_args.debug_dir, Path("eval_accuracy_debug"))

    def test_eval_accuracy_discovers_supported_image_formats(self) -> None:
        module = runpy.run_path(
            str(ROOT / "examples" / "eval_accuracy.py"),
            run_name="collectorvision_eval_accuracy_image_paths_test",
        )
        files = [
            Path("one.JPG"),
            Path("two.jpeg"),
            Path("three.PNG"),
            Path("four.webp"),
            Path("notes.txt"),
        ]

        with (
            mock.patch.object(Path, "is_dir", return_value=True),
            mock.patch.object(Path, "iterdir", return_value=iter(files)),
        ):
            paths = module["image_paths"](Path("images"))

        self.assertEqual(paths, sorted(files[:4]))

    def test_eval_accuracy_verbose_prints_expected_and_matched_names(self) -> None:
        module = runpy.run_path(
            str(ROOT / "examples" / "eval_accuracy.py"),
            run_name="collectorvision_eval_accuracy_verbose_test",
        )

        cards = {
            "expected-id": {"name": "Scrying Glass", "set_name": "Urza's Destiny", "set": "uds"},
            "matched-id": {"name": "Sol Ring", "set_name": "Commander", "set": "cmd"},
        }

        def fake_fetch(card_id, cache):
            cache[card_id] = cards[card_id]
            return cache[card_id]

        output = io.StringIO()
        with (
            mock.patch.dict(
                module["print_verbose_result"].__globals__, {"fetch_scryfall_card": fake_fetch}
            ),
            contextlib.redirect_stdout(output),
        ):
            module["print_verbose_result"](
                Path("expected-id_sample.jpg"), "expected-id", "matched-id", {}
            )

        self.assertIn("expected-id_sample.jpg", output.getvalue())
        self.assertIn("expected  Scrying Glass / Urza's Destiny (UDS)", output.getvalue())
        self.assertIn("matched   Sol Ring / Commander (CMD)", output.getvalue())

    def test_eval_accuracy_debug_saves_overlay_and_aligned_crop(self) -> None:
        module = runpy.run_path(
            str(ROOT / "examples" / "eval_accuracy.py"),
            run_name="collectorvision_eval_accuracy_debug_test",
        )

        class FakeDetection:
            corners = np.array([[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]], dtype=np.float32)
            card_present = True
            confidence = 0.91
            sharpness = 0.06
            extra = {"presence": 0.99}

            def dewarp(self, bgr):
                return Image.new("RGB", (8, 8), "white")

        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            module["save_debug_artifacts"](
                Path("sample-card.jpg"),
                np.zeros((32, 32, 3), dtype=np.uint8),
                FakeDetection(),
                debug_dir,
            )

            self.assertTrue((debug_dir / "overlays" / "sample-card_corners.jpg").exists())
            self.assertTrue((debug_dir / "aligned" / "sample-card_aligned.png").exists())


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

        accepted_ids = {expected_id, *_ACCEPTED_EQUIVALENT_PRINTS.get(expected_id, set())}
        self.assertIn(
            top_id,
            accepted_ids,
            f"{filename}: expected one of {sorted(accepted_ids)}, got {top_id}",
        )

    def test_sample_image(self) -> None:
        self._assert_image_identifies_correctly("7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")

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
