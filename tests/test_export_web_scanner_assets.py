import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from collector_vision.model_registry import get_model

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_web_scanner_assets.py"
_SPEC = importlib.util.spec_from_file_location("export_web_scanner_assets", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


class ExportWebScannerAssetsTests(unittest.TestCase):
    def test_resolves_models_from_requested_registry_channel(self) -> None:
        corner_model = get_model("cornelius")
        embedder_model = get_model("milo")
        corner_path = Path("/cache/cornelius.onnx")
        embedder_path = Path("/cache/milo.onnx")

        class Registry:
            def get_model(self, *, family: str, channel: str):
                self.assertEqual(channel, "testing")
                return {"cornelius": corner_model, "milo": embedder_model}[family]

            def assertEqual(self, actual, expected):  # noqa: ANN001
                test_case.assertEqual(actual, expected)

        test_case = self
        with (
            mock.patch.object(exporter, "load_model_registry", return_value=Registry()),
            mock.patch.object(
                exporter,
                "resolve_model_artifact",
                side_effect=[corner_path, embedder_path],
            ) as resolve,
        ):
            result = exporter._resolve_web_models("testing")

        self.assertEqual(result, (corner_model, corner_path, embedder_model, embedder_path))
        self.assertEqual(
            resolve.call_args_list, [mock.call(corner_model), mock.call(embedder_model)]
        )


if __name__ == "__main__":
    unittest.main()
