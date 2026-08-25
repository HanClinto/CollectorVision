import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import collector_vision as cvg
from collector_vision.model_registry import (
    available_channels,
    available_models,
    get_model,
    load_model_registry,
)


class ModelRegistryTests(unittest.TestCase):
    def test_family_alias_resolves_to_current_default(self) -> None:
        model = get_model("cornelius")

        self.assertEqual(model.id, "cornelius-2.12")
        self.assertEqual(model.family, "cornelius")
        self.assertEqual(model.task, "corner-detection")
        self.assertEqual(model.repository, "HanClinto/cornelius")
        self.assertEqual(len(model.sha256), 64)

    def test_exact_model_id_remains_available(self) -> None:
        model = get_model("milo-1.0.0")

        self.assertEqual(model.version, "1.0.0")
        self.assertEqual(model.architecture, "mobilevit-xxs-arcface")
        self.assertEqual(model.filename, "model.onnx")

    def test_family_and_version_select_an_exact_model(self) -> None:
        model = get_model(family="cornelius", version="2.12")

        self.assertEqual(model.id, "cornelius-2.12")

    def test_channel_selects_a_family_default(self) -> None:
        model = get_model(family="milo", channel="testing")

        self.assertEqual(model.id, "milo-1.0.0")

    def test_channels_are_listed(self) -> None:
        self.assertEqual(available_channels(), ("stable", "testing"))

    def test_aliases_and_exact_ids_are_listed(self) -> None:
        self.assertEqual(
            available_models(),
            ("cornelius", "cornelius-2.12", "milo", "milo-1.0.0"),
        )

    def test_unknown_model_lists_supported_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "cornelius-2.12"):
            get_model("corndog")

    def test_registry_is_available_from_package_root(self) -> None:
        self.assertEqual(cvg.get_model("milo").id, "milo-1.0.0")
        self.assertIn("testing", cvg.available_channels())
        self.assertIn("cornelius", cvg.available_models())

    def test_remote_registry_is_cached_and_selected_by_channel(self) -> None:
        data = {
            "schema_version": 1,
            "channels": {"stable": {"cornelius": "cornelius-9.0"}},
            "models": {
                "cornelius-9.0": {
                    "family": "cornelius",
                    "version": "9.0",
                    "task": "corner-detection",
                    "architecture": "test",
                    "input_size": 384,
                    "repository": "example/cornelius",
                    "revision": "abcdef",
                    "filename": "model.onnx",
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                    "codename": "corndog",
                }
            },
        }

        class Response:
            def read(self) -> bytes:
                return json.dumps(data).encode("utf-8")

            def __enter__(self):  # noqa: ANN201
                return self

            def __exit__(self, *args):  # noqa: ANN002, ANN003
                return None

        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            with mock.patch("urllib.request.urlopen", return_value=Response()) as urlopen:
                registry = load_model_registry(cache_dir=cache_dir, cache_refresh=timedelta(0))

            self.assertEqual(registry.get_model("cornelius").id, "cornelius-9.0")
            self.assertEqual(registry.get_model("cornelius").metadata["codename"], "corndog")
            self.assertTrue((cache_dir / "registry.json").exists())
            urlopen.assert_called_once()

            cached = load_model_registry(cache_dir=cache_dir, offline=True)
            self.assertEqual(cached.get_model("cornelius").id, "cornelius-9.0")


if __name__ == "__main__":
    unittest.main()
