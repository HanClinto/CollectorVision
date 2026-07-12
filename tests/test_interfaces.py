import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import collector_vision as cvg
from collector_vision.detectors.neural import NeuralCornerDetector, _order_corners
from collector_vision.interfaces import DetectionResult


class DetectionResultTests(unittest.TestCase):
    def test_dewarp_outputs_embedder_sized_square_crop(self) -> None:
        bgr = np.zeros((60, 80, 3), dtype=np.uint8)
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        crop = DetectionResult(corners=corners, card_present=True).dewarp(bgr)

        self.assertEqual(crop.size, (448, 448))


class NeuralDetectorCornerOrderingTests(unittest.TestCase):
    def test_shortest_edge_orientation_uses_original_image_space(self) -> None:
        corners = np.array(
            [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]],
            dtype=np.float32,
        )

        ordered = _order_corners(corners, image_shape=(500, 2000, 3))

        np.testing.assert_allclose(
            ordered,
            np.array(
                [[0.6, 0.2], [0.6, 0.8], [0.4, 0.8], [0.4, 0.2]],
                dtype=np.float32,
            ),
        )

    def test_shortest_edge_becomes_top_from_any_side(self) -> None:
        corners = np.array(
            [[0.0, 0.0], [0.8, 0.0], [0.7, 0.8], [0.0, 0.8]],
            dtype=np.float32,
        )

        ordered = _order_corners(corners, image_shape=(1000, 1000, 3))

        np.testing.assert_allclose(
            ordered,
            np.array(
                [[0.7, 0.8], [0.0, 0.8], [0.0, 0.0], [0.8, 0.0]],
                dtype=np.float32,
            ),
        )

    def test_portrait_orientation_keeps_short_edge_on_top(self) -> None:
        corners = np.array(
            [[0.4, 0.2], [0.6, 0.2], [0.6, 0.8], [0.4, 0.8]],
            dtype=np.float32,
        )

        ordered = _order_corners(corners, image_shape=(2000, 500, 3))

        np.testing.assert_allclose(ordered, corners)


class NeuralDetectorProviderFallbackTests(unittest.TestCase):
    def test_auto_retries_cpu_when_accelerator_inference_fails(self) -> None:
        class FailingAcceleratorSession:
            def get_providers(self) -> list[str]:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

            def run(self, output_names, input_feed):  # noqa: ANN001
                raise RuntimeError("accelerator run failed")

        class CpuSession:
            def get_providers(self) -> list[str]:
                return ["CPUExecutionProvider"]

            def run(self, output_names, input_feed):  # noqa: ANN001
                return [
                    np.array([[0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]], dtype=np.float32),
                    np.array([1.0], dtype=np.float32),
                    np.array([0.06], dtype=np.float32),
                ]

        def fake_load(onnx_path: Path, num_threads: int, provider: str):  # noqa: ARG001
            if provider == "cpu":
                return CpuSession(), "image", 384, True
            return FailingAcceleratorSession(), "image", 384, True

        with mock.patch.object(NeuralCornerDetector, "_load", side_effect=fake_load):
            detector = NeuralCornerDetector(checkpoint=Path(__file__), provider="auto")

            with self.assertWarnsRegex(RuntimeWarning, "falling back to CPUExecutionProvider"):
                result = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertTrue(result.card_present)
        self.assertEqual(detector._sess.get_providers(), ["CPUExecutionProvider"])

    def test_explicit_provider_does_not_retry_inference_failure(self) -> None:
        class FailingCpuSession:
            def get_providers(self) -> list[str]:
                return ["CPUExecutionProvider"]

            def run(self, output_names, input_feed):  # noqa: ANN001
                raise RuntimeError("cpu run failed")

        with mock.patch.object(
            NeuralCornerDetector,
            "_load",
            return_value=(FailingCpuSession(), "image", 384, True),
        ):
            detector = NeuralCornerDetector(checkpoint=Path(__file__), provider="cpu")

            with self.assertRaisesRegex(RuntimeError, "cpu run failed"):
                detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))


class TransformTests(unittest.TestCase):
    def test_rotate_card_180_flips_pixels(self) -> None:
        crop = Image.new("RGB", (2, 2))
        crop.putpixel((0, 0), (255, 0, 0))
        crop.putpixel((1, 0), (0, 255, 0))
        crop.putpixel((0, 1), (0, 0, 255))
        crop.putpixel((1, 1), (255, 255, 255))

        rotated = cvg.rotate_card_180(crop)

        self.assertEqual(rotated.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(rotated.getpixel((1, 0)), (0, 0, 255))
        self.assertEqual(rotated.getpixel((0, 1)), (0, 255, 0))
        self.assertEqual(rotated.getpixel((1, 1)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
