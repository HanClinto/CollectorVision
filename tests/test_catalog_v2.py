import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import collector_vision.catalog_v2_downloader as downloader_module
from collector_vision import CatalogV2Error as ExportedCatalogV2Error
from collector_vision import catalog_v2_row_key
from collector_vision.catalog_v2 import CatalogV2, CatalogV2Error
from collector_vision.catalog_v2_downloader import CatalogV2Downloader

MODEL_IDENTITY = (
    "collectorvision@9d45a37ebfe40f22ece70507015645de134dc3ec:"
    "milo-1.0.0@sha256:bd13d8d60383c69da04dce261f32e93fdaeaa8fd618fbc991e7385f71b3d45df"
)
FEED_URL = "https://catalog.test/catalog-feed-v2.json"


class PublishedCatalog:
    """A synthetic feed publisher for the combined-record Catalog v2 contract.

    Every base and update carries exactly one `records` JSONL asset (plus an
    optional `embeddings` asset), matching the live
    `catalog-feed-v2.json` contract: no separate recognition/metadata layers.
    """

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.feed = self._build_feed()
        self.publish_feed()

    def asset(self, path: str, decoded: bytes) -> dict[str, object]:
        url = f"https://catalog.test/{path}"
        payload = gzip.compress(decoded, compresslevel=9, mtime=0)
        self.payloads[url] = payload
        return {
            "url": url,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def jsonl(self, path: str, values: list[object]) -> dict[str, object]:
        values = [_with_test_name(value) for value in values]
        return self.asset(
            path,
            b"".join(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for value in values
            ),
        )

    def publish_feed(self) -> None:
        self.payloads[FEED_URL] = json.dumps(self.feed).encode()

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        try:
            return self.payloads[url]
        except KeyError:
            raise AssertionError(f"unexpected request {url}") from None

    def _build_feed(self) -> dict:
        # Base rows are the combined public shape: core recognition fields
        # plus a REQUIRED metadata field (object or null).
        base_records = [
            {
                "id": "a",
                "identifiers": {"scryfall_oracle": "oracle-a"},
                "finishes": ["foil", "nonfoil"],
                "metadata": {"layout": "normal", "name": "Alpha", "promo": False},
            },
            {
                "id": "b",
                "identifiers": {"scryfall_oracle": "oracle-b"},
                "face_index": 1,
                "metadata": {"layout": "transform", "name": "Beta", "promo": False},
            },
        ]
        # v0 -> v1: delete "a" (had metadata), fully update "b" (recognition +
        # metadata), and add "c" with an initial metadata value.
        update_one_records = [
            {"op": "delete", "id": "a"},
            {
                "op": "upsert",
                "record": {
                    "id": "b",
                    "identifiers": {"scryfall_oracle": "oracle-b2"},
                    "face_index": 1,
                    "finishes": ["foil"],
                },
                "metadata": {"layout": "transform", "name": "Beta 2", "promo": False},
                "embedding_index": 0,
            },
            {
                "op": "upsert",
                "record": {
                    "id": "c",
                    "identifiers": {"scryfall_oracle": "oracle-c"},
                },
                "metadata": {"layout": "normal", "name": "Gamma", "promo": False},
                "embedding_index": 1,
            },
        ]
        # v1 -> v2: a metadata-only upsert for "c" that repeats its unchanged
        # core record and omits embedding_index entirely.
        update_two_records = [
            {
                "op": "upsert",
                "record": {
                    "id": "c",
                    "identifiers": {"scryfall_oracle": "oracle-c"},
                },
                "metadata": {"layout": "art_series", "name": "Gamma", "promo": True},
            }
        ]
        base = {
            "version": 0,
            "rows": 2,
            "source_updated_at": "2026-07-24T00:00:00Z",
            "assets": {
                "records": self.jsonl("demo/version/0/base/records.jsonl.gz", base_records),
                "embeddings": self.asset(
                    "demo/version/0/base/embeddings.f16.gz",
                    np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2").tobytes(),
                ),
            },
        }
        updates = {
            "1": {
                "from_version": 0,
                "to_version": 1,
                "rows": {"added": 1, "updated": 1, "deleted": 1},
                "source_updated_at": "2026-07-25T00:00:00Z",
                "recognition_rows": 3,
                "metadata_rows": 3,
                "assets": {
                    "records": self.jsonl(
                        "demo/version/1/delta-from-0/records.jsonl.gz",
                        update_one_records,
                    ),
                    "embeddings": self.asset(
                        "demo/version/1/delta-from-0/embeddings.f16.gz",
                        np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype="<f2").tobytes(),
                    ),
                },
            },
            "2": {
                "from_version": 1,
                "to_version": 2,
                "rows": {"added": 0, "updated": 1, "deleted": 0},
                "source_updated_at": "2026-07-26T00:00:00Z",
                "recognition_rows": 0,
                "metadata_rows": 1,
                "assets": {
                    "records": self.jsonl(
                        "demo/version/2/delta-from-1/records.jsonl.gz",
                        update_two_records,
                    ),
                },
            },
        }
        return {
            "checked_at": "2026-07-26T12:00:00Z",
            "families": {
                "milo1": {
                    "embedding": {
                        "model": MODEL_IDENTITY,
                        "dimensions": 2,
                        "dtype": "float16",
                        "byte_order": "little",
                        "layout": "row-major",
                    },
                    "catalogs": {
                        "test/demo": {
                            "public_name": "demo",
                            "descriptor": {
                                "game": "pokemon",
                                "source": "test",
                                "profile": "printings",
                                "description": "Test catalog.",
                                "result_identifier": "source_card",
                                "recommended": True,
                            },
                            "current_version": 2,
                            "rows": 2,
                            "source_updated_at": "2026-07-26T00:00:00Z",
                            "base": base,
                            "updates": updates,
                        }
                    },
                }
            },
        }


@pytest.fixture
def publication(monkeypatch: pytest.MonkeyPatch) -> PublishedCatalog:
    published = PublishedCatalog()
    monkeypatch.setattr(downloader_module, "_fetch", published.fetch)
    return published


def test_installs_current_catalog_from_feed(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    downloader = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )
    catalog = downloader.load()

    assert catalog.catalog_key == "milo1/test/demo"
    assert catalog.version == 2
    assert catalog.metadata_loaded
    assert catalog.embedding.dimensions == 2
    assert catalog.embeddings.dtype == np.dtype("<f2")
    assert [record.key("test") for record in catalog.records] == [
        "test:b:face:1",
        "test:c",
    ]
    assert catalog.record_for_index(0) == {
        "key": "test:b:face:1",
        "id": "b",
        "name": "b",
        "identifiers": {
            "source_card": "b",
            "scryfall_oracle": "oracle-b2",
        },
        "face_index": 1,
        "finishes": ["foil"],
        "result_identifier": "source_card",
        "card_id": "b",
        "metadata": {"layout": "transform", "name": "Beta 2", "promo": False},
    }
    assert catalog.records[1].metadata == {
        "layout": "art_series",
        "name": "Gamma",
        "promo": True,
    }
    assert np.array_equal(
        catalog.embeddings,
        np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype="<f2"),
    )
    assert catalog.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1) == [(1.0, "c")]


def test_recognition_only_discards_metadata_after_parsing(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    catalog = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=False,
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    ).load()

    assert not catalog.metadata_loaded
    assert all(record.metadata is None for record in catalog.records)
    # The client still downloads the single combined records asset (there is
    # no separate metadata-only asset to skip); it simply discards the parsed
    # metadata content immediately.
    assert any(url.endswith("records.jsonl.gz") for url in publication.calls)
    assert catalog.records[0].finishes == ("foil",)


def test_metadata_only_upsert_without_embedding_index_changes_only_metadata(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    catalog = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        version=1,
        feed_url=FEED_URL,
    ).load()
    embeddings_before = catalog.embeddings.copy()

    upgraded = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        version=2,
        feed_url=FEED_URL,
    ).load()

    assert np.array_equal(upgraded.embeddings, embeddings_before)
    assert upgraded.records[1].metadata == {
        "layout": "art_series",
        "name": "Gamma",
        "promo": True,
    }


def test_rejects_no_embedding_upsert_with_changed_core(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    update = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["updates"]["2"]
    update["assets"]["records"] = publication.jsonl(
        "demo/version/2/delta-from-1/tampered-records.jsonl.gz",
        [
            {
                "op": "upsert",
                "record": {
                    "id": "c",
                    "identifiers": {"scryfall_oracle": "oracle-c-changed"},
                },
                "metadata": {"layout": "art_series", "name": "Gamma", "promo": True},
            }
        ],
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="unchanged core record"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            include_metadata=True,
            cache_dir=tmp_path,
            version=2,
            feed_url=FEED_URL,
        )


def test_explicit_null_metadata_removes_metadata(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    update = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["updates"]["2"]
    update["assets"]["records"] = publication.jsonl(
        "demo/version/2/delta-from-1/null-metadata-records.jsonl.gz",
        [
            {
                "op": "upsert",
                "record": {
                    "id": "c",
                    "identifiers": {"scryfall_oracle": "oracle-c"},
                },
                "metadata": None,
            }
        ],
    )
    publication.publish_feed()

    catalog = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        version=2,
        feed_url=FEED_URL,
    ).load()

    assert catalog.records[1].metadata is None
    assert catalog.record_for_index(1)["metadata"] is None


def test_rejects_added_row_without_embedding_index(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    update = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["updates"]["1"]
    update["assets"]["records"] = publication.jsonl(
        "demo/version/1/delta-from-0/no-index-add-records.jsonl.gz",
        [
            {"op": "delete", "id": "a"},
            {
                "op": "upsert",
                "record": {
                    "id": "b",
                    "identifiers": {"scryfall_oracle": "oracle-b2"},
                    "face_index": 1,
                    "finishes": ["foil"],
                },
                "metadata": {"layout": "transform", "name": "Beta 2", "promo": False},
                "embedding_index": 0,
            },
            {
                "op": "upsert",
                "record": {
                    "id": "c",
                    "identifiers": {"scryfall_oracle": "oracle-c"},
                },
                "metadata": {"layout": "normal", "name": "Gamma", "promo": False},
            },
        ],
    )
    update["assets"]["embeddings"] = publication.asset(
        "demo/version/1/delta-from-0/no-index-add-embeddings.f16.gz",
        np.asarray([[0.5, 0.5]], dtype="<f2").tobytes(),
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="added .* row must include embedding_index"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=1,
            feed_url=FEED_URL,
        )


def test_rejects_upsert_with_neither_metadata_nor_embedding_index(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    update = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["updates"]["2"]
    update["assets"]["records"] = publication.jsonl(
        "demo/version/2/delta-from-1/noop-records.jsonl.gz",
        [
            {
                "op": "upsert",
                "record": {
                    "id": "c",
                    "identifiers": {"scryfall_oracle": "oracle-c"},
                },
            }
        ],
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="must change recognition or metadata"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=2,
            feed_url=FEED_URL,
        )


def test_cached_current_snapshot_uses_only_feed_request(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )
    publication.calls.clear()

    reopened = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )

    assert reopened.version == 2
    assert publication.calls == [FEED_URL]


def test_offline_open_uses_latest_installed_version(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        version=1,
        feed_url=FEED_URL,
    )
    publication.calls.clear()

    offline = CatalogV2Downloader.open(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
    )

    assert offline.version == 1
    assert publication.calls == []


def test_incremental_install_reuses_cached_snapshot(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        version=1,
        feed_url=FEED_URL,
    )
    publication.calls.clear()

    current = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    ).load()

    assert current.version == 2
    assert publication.calls == [
        FEED_URL,
        publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["updates"]["2"]["assets"][
            "records"
        ]["url"],
    ]
    recognition_root = tmp_path / "catalog-v2" / "snapshots" / "milo1--test--demo" / "metadata"
    assert [path.name for path in recognition_root.iterdir()] == ["version-2"]


def test_metadata_upgrade_reuses_current_embeddings(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    recognition = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    ).load()
    publication.calls.clear()

    metadata = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    ).load()

    assert metadata.metadata_loaded
    assert np.array_equal(metadata.embeddings, recognition.embeddings)
    assert not any("embeddings" in url for url in publication.calls)
    assert metadata.records[1].metadata["promo"] is True


def test_explicit_catalog_selection(tmp_path: Path, publication: PublishedCatalog) -> None:
    installed = CatalogV2Downloader.install_catalog(
        "milo1/test/demo",
        cache_dir=tmp_path,
        version=0,
        feed_url=FEED_URL,
    )

    assert installed.catalog_key == "milo1/test/demo"
    assert installed.version == 0


def test_catalog_loader_uses_descriptor_discovery(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    catalog = CatalogV2.load(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )

    assert catalog.catalog_key == "milo1/test/demo"
    assert catalog.version == 2


def test_rejects_tampered_asset(tmp_path: Path, publication: PublishedCatalog) -> None:
    reference = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["base"]["assets"][
        "records"
    ]
    publication.payloads[reference["url"]] += b"tampered"

    with pytest.raises(CatalogV2Error, match="compressed size mismatch"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=0,
            feed_url=FEED_URL,
        )


def test_rejects_noncanonical_identity(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    base = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["base"]
    base["assets"]["records"] = publication.jsonl(
        "invalid-records.jsonl.gz",
        [
            {"id": "a", "identifiers": {}, "face_index": 0, "metadata": None},
            {"id": "b", "identifiers": {}, "face_index": 1, "metadata": None},
        ],
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="omit face_index"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=0,
            feed_url=FEED_URL,
        )


def test_rejects_recognition_row_without_name(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    base = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["base"]
    base["assets"]["records"] = publication.asset(
        "missing-name.jsonl.gz",
        b"".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for value in [
                {"id": "a", "identifiers": {}, "metadata": None},
                {"id": "b", "identifiers": {}, "face_index": 1, "name": "b", "metadata": None},
            ]
        ),
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="invalid fields"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=0,
            feed_url=FEED_URL,
        )


def test_rejects_base_row_missing_metadata_field(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    base = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["base"]
    base["assets"]["records"] = publication.jsonl(
        "missing-metadata.jsonl.gz",
        [
            {"id": "a", "identifiers": {}},
            {"id": "b", "identifiers": {}, "face_index": 1, "metadata": None},
        ],
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="metadata field is required"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=0,
            feed_url=FEED_URL,
        )


def test_rejects_invalid_delta_embedding_indexes(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    update = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["updates"]["1"]
    operations_url = update["assets"]["records"]["url"]
    operations = [
        {"op": "delete", "id": "a"},
        {
            "op": "upsert",
            "record": {"id": "b", "identifiers": {}, "face_index": 1},
            "embedding_index": 1,
        },
        {
            "op": "upsert",
            "record": {"id": "c", "identifiers": {}},
            "embedding_index": 2,
        },
    ]
    update["assets"]["records"] = publication.jsonl(
        operations_url.removeprefix("https://catalog.test/"),
        operations,
    )
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="indexes must be contiguous"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=1,
            feed_url=FEED_URL,
        )


def test_rejects_noncontiguous_feed_route(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    entry = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]
    entry["updates"]["2"]["from_version"] = 0
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="consecutive|contiguous"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            feed_url=FEED_URL,
        )


def test_rejects_ambiguous_descriptor_selection(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    catalogs = publication.feed["families"]["milo1"]["catalogs"]
    catalogs["test/other"] = deepcopy(catalogs["test/demo"])
    catalogs["test/other"]["public_name"] = "other"
    catalogs["test/demo"]["descriptor"]["recommended"] = False
    catalogs["test/other"]["descriptor"]["recommended"] = False
    publication.publish_feed()

    with pytest.raises(CatalogV2Error, match="multiple"):
        CatalogV2Downloader.install(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            feed_url=FEED_URL,
        )


def test_offline_open_rejects_corrupt_snapshot(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    installed = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )
    (installed.snapshot_path / "records.jsonl.gz").write_bytes(b"corrupt")

    with pytest.raises(CatalogV2Error, match="corrupt"):
        CatalogV2Downloader.open(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
        )


def test_online_install_recovers_corrupt_snapshot(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    installed = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )
    (installed.snapshot_path / "records.jsonl.gz").write_bytes(b"corrupt")
    publication.calls.clear()

    recovered = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )

    assert recovered.version == 2
    assert len(recovered.load()) == 2
    assert len(publication.calls) > 1


def test_cached_snapshot_accepts_feed_presentation_changes(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    )
    descriptor = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["descriptor"]
    descriptor["description"] = "Updated description."
    publication.publish_feed()
    publication.calls.clear()

    current = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        feed_url=FEED_URL,
    ).load()

    assert current.descriptor.description == "Updated description."
    assert publication.calls == [FEED_URL]


def test_offline_explicit_version_requires_exact_snapshot(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    CatalogV2Downloader.install(
        "pokemon",
        source="test",
        cache_dir=tmp_path,
        version=0,
        feed_url=FEED_URL,
    )

    with pytest.raises(FileNotFoundError, match="no installed"):
        CatalogV2Downloader.open(
            "pokemon",
            source="test",
            cache_dir=tmp_path,
            version=2,
        )


def test_loaded_null_metadata_is_explicit(
    tmp_path: Path,
    publication: PublishedCatalog,
) -> None:
    base = publication.feed["families"]["milo1"]["catalogs"]["test/demo"]["base"]
    base_records = [
        {"id": "a", "identifiers": {"scryfall_oracle": "oracle-a"}, "metadata": None},
        {
            "id": "b",
            "identifiers": {"scryfall_oracle": "oracle-b"},
            "face_index": 1,
            "metadata": {"name": "Beta"},
        },
    ]
    base["assets"]["records"] = publication.jsonl(
        "demo/version/0/base/null-metadata.jsonl.gz",
        base_records,
    )
    publication.publish_feed()

    catalog = CatalogV2Downloader.install(
        "pokemon",
        source="test",
        include_metadata=True,
        cache_dir=tmp_path,
        version=0,
        feed_url=FEED_URL,
    ).load()

    assert catalog.record_for_index(0)["metadata"] is None


def test_v2_error_and_row_key_are_public() -> None:
    assert ExportedCatalogV2Error is CatalogV2Error
    assert catalog_v2_row_key("scryfall", "card", 1) == "scryfall:card:face:1"


def _with_test_name(value: object) -> object:
    if not isinstance(value, dict):
        return value
    rendered = dict(value)
    record = rendered.get("record")
    if isinstance(record, dict) and isinstance(record.get("id"), str):
        rendered["record"] = {"name": record["id"], **record}
    elif "op" not in rendered and isinstance(rendered.get("id"), str):
        rendered = {"name": rendered["id"], **rendered}
    return rendered


def test_v2_maps_all_published_tcgplayer_games() -> None:
    assert downloader_module._GAME_NAMES["pokemon-japan"] == "pokemon-japan"
    assert downloader_module._GAME_NAMES["union-arena"] == "union-arena"
    assert downloader_module._GAME_NAMES["gundam"] == "gundam-card-game"
    assert downloader_module._GAME_NAMES["riftbound"] == "riftbound"


def test_cold_offline_mode_explains_missing_feed(tmp_path: Path) -> None:
    with pytest.raises(CatalogV2Error, match="install a catalog online"):
        CatalogV2Downloader.open("pokemon", cache_dir=tmp_path)


def test_offline_loader_rejects_feed_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feed_url"):
        CatalogV2.load(
            "pokemon",
            offline=True,
            cache_dir=tmp_path,
            feed_url=FEED_URL,
        )
