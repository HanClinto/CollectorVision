import unittest
from pathlib import Path
from unittest import mock

from collector_vision.detectors.neural import NeuralCornerDetector
from collector_vision.embedders.neural import NeuralEmbedder
from collector_vision.model_artifacts import resolve_registered_model
from collector_vision.model_registry import get_model


class ModelSelectionTests(unittest.TestCase):
    def test_registered_model_rejects_incompatible_task(self) -> None:
        class Registry:
            def get_model(self, **kwargs):  # noqa: ANN003
                return get_model("milo")

        with mock.patch(
            "collector_vision.model_artifacts.load_model_registry", return_value=Registry()
        ):
            with self.assertRaisesRegex(ValueError, "card-embedding"):
                resolve_registered_model("milo", task="corner-detection")

    def test_corner_detector_resolves_requested_family(self) -> None:
        resolved_path = Path(__file__)
        with (
            mock.patch(
                "collector_vision.model_artifacts.resolve_registered_model",
                return_value=resolved_path,
            ) as resolve,
            mock.patch.object(
                NeuralCornerDetector,
                "_load",
                return_value=(mock.Mock(), "image", 384, True),
            ),
        ):
            NeuralCornerDetector(family="cornelius", channel="testing", offline=True)

        resolve.assert_called_once_with(
            "cornelius",
            task="corner-detection",
            version=None,
            channel="testing",
            cache_dir=None,
            offline=True,
        )

    def test_embedder_resolves_exact_family_version(self) -> None:
        resolved_path = Path(__file__)
        with (
            mock.patch(
                "collector_vision.model_artifacts.resolve_registered_model",
                return_value=resolved_path,
            ) as resolve,
            mock.patch.object(
                NeuralEmbedder,
                "_load",
                return_value=(mock.Mock(), "image", 448),
            ),
        ):
            NeuralEmbedder(family="milo", version="1.0.0")

        resolve.assert_called_once_with(
            "milo",
            task="card-embedding",
            version="1.0.0",
            channel="stable",
            cache_dir=None,
            offline=False,
        )

    def test_checkpoint_and_family_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint or family/version"):
            NeuralCornerDetector(checkpoint=Path(__file__), family="cornelius")


if __name__ == "__main__":
    unittest.main()
