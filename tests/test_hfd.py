import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from collector_vision.hfd import HFD


class HFDTests(unittest.TestCase):
    def test_resolve_downloads_catalog_with_optional_hub_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            hfd = HFD("example/milo", "scryfall-mtg", cache_dir=cache_dir)
            manifest = {"scryfall-mtg": {"latest": "milo1-scryfall-mtg.npz"}}

            def download(**kwargs):  # noqa: ANN003
                self.assertEqual(kwargs["repo_id"], "example/milo")
                self.assertEqual(kwargs["filename"], "milo1-scryfall-mtg.npz")
                self.assertEqual(kwargs["subfolder"], "catalogs")
                destination = Path(kwargs["local_dir"]) / "catalogs" / kwargs["filename"]
                destination.parent.mkdir(parents=True)
                destination.write_bytes(b"catalog")
                return str(destination)

            fake_hub = types.SimpleNamespace(hf_hub_download=download)
            with (
                mock.patch.object(hfd, "_get_manifest", return_value=manifest),
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            ):
                path = hfd.resolve()

            self.assertEqual(path.read_bytes(), b"catalog")
            self.assertEqual(path.parent.name, "catalogs")

    def test_offline_resolve_uses_cached_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            hfd = HFD("example/milo", "scryfall-mtg", cache_dir=cache_dir, offline=True)
            filename = "milo1-scryfall-mtg.npz"
            cached_path = cache_dir / "example_milo" / "scryfall-mtg" / "catalogs" / filename
            cached_path.parent.mkdir(parents=True)
            cached_path.write_bytes(b"catalog")

            with mock.patch.object(
                hfd, "_get_manifest", return_value={"scryfall-mtg": {"latest": filename}}
            ):
                self.assertEqual(hfd.resolve(), cached_path)


if __name__ == "__main__":
    unittest.main()
