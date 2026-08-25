import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import collector_vision as cvg
from collector_vision.model_artifacts import resolve_model_artifact
from collector_vision.model_registry import get_model


class ModelArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = get_model("cornelius-2.12")
        self.cache_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.cache_dir))

    def test_returns_verified_cached_model_without_hub_dependency(self) -> None:
        cached_model = self.model.__class__(
            **{**self.model.__dict__, "sha256": hashlib.sha256(b"cached model").hexdigest()}
        )
        destination = self.cache_dir / "models" / cached_model.sha256 / cached_model.filename
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"cached model")

        with mock.patch("collector_vision.model_artifacts._download_from_hub") as download:
            resolved = resolve_model_artifact(cached_model, cache_dir=self.cache_dir, offline=True)

        self.assertEqual(resolved, destination)
        download.assert_not_called()

    def test_offline_missing_model_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Disable offline mode"):
            resolve_model_artifact(self.model, cache_dir=self.cache_dir, offline=True)

    def test_downloaded_model_is_verified(self) -> None:
        content = b"downloaded model"
        model = self.model.__class__(
            **{**self.model.__dict__, "sha256": hashlib.sha256(content).hexdigest()}
        )

        def write_model(spec, destination_dir):  # noqa: ANN001
            self.assertEqual(spec, model)
            (destination_dir / spec.filename).write_bytes(content)

        with mock.patch("collector_vision.model_artifacts._download_from_hub", write_model):
            resolved = resolve_model_artifact(model, cache_dir=self.cache_dir)

        self.assertEqual(resolved.read_bytes(), content)

    def test_checksum_mismatch_is_rejected(self) -> None:
        destination = self.cache_dir / "models" / self.model.sha256 / self.model.filename
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"corrupt model")

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            resolve_model_artifact(self.model, cache_dir=self.cache_dir)

    def test_resolver_is_available_from_package_root(self) -> None:
        self.assertIs(cvg.resolve_model_artifact, resolve_model_artifact)


if __name__ == "__main__":
    unittest.main()
