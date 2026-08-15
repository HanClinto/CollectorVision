from pathlib import Path
from unittest.mock import patch

import pytest

from collector_vision import Catalog, CatalogV1, CatalogV2
from collector_vision.catalog_v2_downloader import CatalogV2Downloader


@pytest.mark.parametrize(
    "source",
    [
        Path("custom.npz"),
        "./custom.data",
        "custom.npz",
        "hf://HanClinto/milo/scryfall-mtg",
    ],
)
def test_default_catalog_loader_dispatches_v1_sources(source: object) -> None:
    sentinel = object()
    with patch.object(CatalogV1, "load", return_value=sentinel) as load:
        assert Catalog.load(source) is sentinel
    load.assert_called_once_with(source)


def test_default_catalog_loader_dispatches_game_to_v2() -> None:
    sentinel = object()
    with patch.object(CatalogV2, "load", return_value=sentinel) as load:
        assert Catalog.load("mtg", source="tcgplayer") is sentinel
    load.assert_called_once_with("mtg", source="tcgplayer")


def test_default_catalog_loader_does_not_dispatch_by_file_existence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("mtg").touch()
    sentinel = object()
    with patch.object(CatalogV2, "load", return_value=sentinel) as load:
        assert Catalog.load("mtg") is sentinel
    load.assert_called_once_with("mtg")


def test_default_catalog_loader_rejects_v2_options_for_v1_source() -> None:
    with pytest.raises(TypeError, match="Catalog v1 loading does not accept options: source"):
        Catalog.load("custom.npz", source="scryfall")


def test_catalog_constructors_do_not_perform_io() -> None:
    with pytest.raises(TypeError, match=r"Catalog\.load"):
        Catalog("mtg")
    with pytest.raises(TypeError, match=r"CatalogV2\.load"):
        CatalogV2("mtg")


def test_catalog_v2_rejects_unknown_game_before_network_access() -> None:
    with patch.object(CatalogV2Downloader, "install") as install:
        with pytest.raises(ValueError, match="Unknown game"):
            CatalogV2.load("not-a-game")
    install.assert_not_called()
