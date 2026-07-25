import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from collector_vision.catalog_v2 import CatalogV2, CatalogV2Error
from collector_vision.catalog_v2_release import CatalogV2Release

MODEL_IDENTITY = (
    "collectorvision@9d45a37ebfe40f22ece70507015645de134dc3ec:"
    "milo-1.0.0@sha256:bd13d8d60383c69da04dce261f32e93fdaeaa8fd618fbc991e7385f71b3d45df"
)


def _write_gzip(path: Path, payload: bytes) -> dict[str, object]:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            compresslevel=9,
            mtime=0,
        ) as stream:
            stream.write(payload)
    data = path.read_bytes()
    return {
        "filename": path.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_encoding": "gzip",
    }


def _write_catalog(tmp_path: Path) -> Path:
    recognition = [
        {
            "key": "card:a:face:0",
            "identifiers": {"source_card": "a", "shared_card": "oracle-a"},
        },
        {
            "key": "card:b:face:1",
            "identifiers": {"source_card": "b", "shared_card": "oracle-b"},
            "face_index": 1,
        },
    ]
    metadata = [
        {"key": "card:a:face:0", "metadata": {"name": "Alpha"}},
        {"key": "card:b:face:1", "metadata": {"name": "Beta"}},
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
    assets = {
        "recognition_rows": _write_gzip(
            tmp_path / "demo.recognition.jsonl.gz",
            b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for row in recognition
            ),
        ),
        "recognition_matrix": _write_gzip(
            tmp_path / "demo.recognition.f16.gz", embeddings.tobytes()
        ),
        "metadata_rows": _write_gzip(
            tmp_path / "demo.metadata.jsonl.gz",
            b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for row in metadata
            ),
        ),
    }
    manifest = {
        "schema_version": 2,
        "catalog_key": "milo1/test/demo",
        "version": "catalog-v2-beta.1-2026-07-24",
        "embedding_model": MODEL_IDENTITY,
        "rows": 2,
        "dim": 2,
        "dtype": "float16",
        "descriptor": {
            "game": "demo",
            "source": "test",
            "profile": "printings",
            "description": "Test catalog.",
            "result_identifier": "source_card",
            "recommended": True,
        },
        "assets": assets,
    }
    manifest_path = tmp_path / "demo.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_index(directory: Path, manifest_path: Path, tag: str) -> None:
    manifest_bytes = manifest_path.read_bytes()
    index = {
        "schema_version": 2,
        "release_version": tag,
        "source_updated_at": "2026-07-24T00:00:00Z",
        "catalogs": {
            "milo1/test/demo": {
                "manifest_filename": manifest_path.name,
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "descriptor": json.loads(manifest_bytes)["descriptor"],
                "source_revision": {
                    "type": "test",
                    "name": "test",
                    "updated_at": "2026-07-24T00:00:00Z",
                    "uri": "https://example.test",
                    "identity": "test",
                },
            }
        },
    }
    (directory / "catalog-index-v2.json").write_text(json.dumps(index), encoding="utf-8")


def test_loads_v2_without_touching_v1_catalog(tmp_path: Path) -> None:
    manifest_path = _write_catalog(tmp_path)

    catalog = CatalogV2.load(manifest_path)

    assert catalog.catalog_key == "milo1/test/demo"
    assert catalog.version == "catalog-v2-beta.1-2026-07-24"
    assert catalog.embeddings.dtype == np.dtype("<f2")
    assert catalog.records[0].metadata is None
    assert catalog.search(np.asarray([0.0, 1.0], dtype=np.float32), top_k=1) == [(1.0, "b")]


def test_loads_optional_metadata_and_peer_identifiers(tmp_path: Path) -> None:
    catalog = CatalogV2.load(_write_catalog(tmp_path), include_metadata=True)

    assert catalog.record_for_index(1) == {
        "key": "card:b:face:1",
        "identifiers": {"source_card": "b", "shared_card": "oracle-b"},
        "face_index": 1,
        "result_identifier": "source_card",
        "card_id": "b",
        "metadata": {"name": "Beta"},
    }


def test_rejects_tampered_asset(tmp_path: Path) -> None:
    manifest_path = _write_catalog(tmp_path)
    matrix_path = tmp_path / "demo.recognition.f16.gz"
    matrix_path.write_bytes(matrix_path.read_bytes() + b"tampered")

    with pytest.raises(CatalogV2Error, match="size mismatch"):
        CatalogV2.load(manifest_path)


def test_rejects_asset_path_traversal(tmp_path: Path) -> None:
    manifest_path = _write_catalog(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"]["recognition_rows"]["filename"] = "../outside.jsonl.gz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CatalogV2Error, match="filename must be a basename"):
        CatalogV2.load(manifest_path)


def test_rejects_missing_result_identifier(tmp_path: Path) -> None:
    manifest_path = _write_catalog(tmp_path)
    records_path = tmp_path / "demo.recognition.jsonl.gz"
    asset = _write_gzip(
        records_path,
        b'{"key":"card:a:face:0","identifiers":{"other":"a"}}\n'
        b'{"key":"card:b:face:1","identifiers":{"source_card":"b"}}\n',
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"]["recognition_rows"] = asset
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CatalogV2Error, match="lacks result identifier"):
        CatalogV2.load(manifest_path)


def test_applies_exact_base_delta_with_independent_metadata(tmp_path: Path) -> None:
    previous = CatalogV2.load(_write_catalog(tmp_path), include_metadata=True)
    manifest = json.loads((tmp_path / "demo.manifest.json").read_text())
    manifest["version"] = "catalog-v2-beta.2-2026-07-25"
    manifest["rows"] = 2
    manifest["delta"] = {
        "base_version": previous.version,
        "requires_exact_base": True,
        "operations": 3,
        "metadata_operations": 3,
    }
    operations = [
        {"op": "delete", "key": "card:a:face:0"},
        {
            "op": "upsert",
            "record": {
                "key": "card:b:face:1",
                "identifiers": {"source_card": "b2", "shared_card": "oracle-b"},
                "face_index": 1,
            },
            "embedding_index": 0,
        },
        {
            "op": "upsert",
            "record": {
                "key": "card:c:face:0",
                "identifiers": {"source_card": "c", "shared_card": "oracle-c"},
            },
            "embedding_index": 1,
        },
    ]
    metadata_operations = [
        {"op": "delete", "key": "card:a:face:0"},
        {"op": "upsert", "key": "card:b:face:1", "metadata": {"name": "Beta 2"}},
        {"op": "upsert", "key": "card:c:face:0", "metadata": {"name": "Gamma"}},
    ]
    manifest["assets"]["delta_operations"] = _write_gzip(
        tmp_path / "demo.delta.jsonl.gz",
        b"".join(json.dumps(value).encode() + b"\n" for value in operations),
    )
    manifest["assets"]["delta_matrix"] = _write_gzip(
        tmp_path / "demo.delta.f16.gz",
        np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype="<f2").tobytes(),
    )
    manifest["assets"]["metadata_delta"] = _write_gzip(
        tmp_path / "demo.metadata.delta.jsonl.gz",
        b"".join(json.dumps(value).encode() + b"\n" for value in metadata_operations),
    )
    target_manifest = tmp_path / "demo-v2.manifest.json"
    target_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    current = CatalogV2.apply_delta(previous, target_manifest, include_metadata=True)

    assert current.version == "catalog-v2-beta.2-2026-07-25"
    assert [record.key for record in current.records] == [
        "card:b:face:1",
        "card:c:face:0",
    ]
    assert current.record_for_index(0)["card_id"] == "b2"
    assert current.record_for_index(0)["metadata"] == {"name": "Beta 2"}
    assert np.array_equal(
        current.embeddings,
        np.asarray([[0.5, 0.5], [1.0, 0.0]], dtype="<f2"),
    )


def test_rejects_delta_from_non_exact_base(tmp_path: Path) -> None:
    previous = CatalogV2.load(_write_catalog(tmp_path))
    manifest = json.loads((tmp_path / "demo.manifest.json").read_text())
    manifest["delta"] = {
        "base_version": "another-version",
        "requires_exact_base": True,
        "operations": 0,
        "metadata_operations": 0,
    }
    target_manifest = tmp_path / "demo-v2.manifest.json"
    target_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CatalogV2Error, match="requires base"):
        CatalogV2.apply_delta(previous, target_manifest)


def test_release_installer_uses_separate_v2_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = _write_catalog(source)
    tag = "catalog-v2-beta.1-2026-07-24"
    _write_index(source, manifest_path, tag)
    cache = tmp_path / "cache"

    release = CatalogV2Release.install(
        tag,
        catalog_keys=["milo1/test/demo"],
        include_metadata=True,
        cache_dir=cache,
        base_url=source.as_uri(),
    )

    catalog = release.load("milo1/test/demo")
    assert len(catalog) == 2
    assert catalog.metadata_loaded
    assert (
        cache / "catalog-v2" / "releases" / tag / "metadata" / "demo" / "demo.manifest.json"
    ).is_file()
    assert not (cache / "catalogs").exists()


def test_release_installer_materializes_one_step_delta(tmp_path: Path) -> None:
    base_source = tmp_path / "base-source"
    base_source.mkdir()
    base_manifest = _write_catalog(base_source)
    base_tag = "catalog-v2-beta.1-2026-07-24"
    _write_index(base_source, base_manifest, base_tag)
    cache = tmp_path / "cache"
    CatalogV2Release.install(
        base_tag,
        catalog_keys=["milo1/test/demo"],
        cache_dir=cache,
        base_url=base_source.as_uri(),
    )

    target_source = tmp_path / "target-source"
    target_source.mkdir()
    target_tag = "catalog-v2-beta.2-2026-07-25"
    manifest = json.loads(base_manifest.read_text())
    manifest["version"] = target_tag
    manifest["delta"] = {
        "base_version": base_tag,
        "requires_exact_base": True,
        "operations": 0,
        "metadata_operations": 0,
    }
    manifest["assets"]["delta_operations"] = _write_gzip(target_source / "demo.delta.jsonl.gz", b"")
    manifest["assets"]["delta_matrix"] = _write_gzip(target_source / "demo.delta.f16.gz", b"")
    target_manifest = target_source / "demo.manifest.json"
    target_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    _write_index(target_source, target_manifest, target_tag)

    release = CatalogV2Release.install(
        target_tag,
        catalog_keys=["milo1/test/demo"],
        cache_dir=cache,
        previous_tag=base_tag,
        base_url=target_source.as_uri(),
    )

    current = release.load("milo1/test/demo")
    assert current.version == target_tag
    assert np.array_equal(
        current.embeddings,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2"),
    )


def test_release_installer_falls_back_for_incompatible_exact_base(
    tmp_path: Path,
) -> None:
    base_source = tmp_path / "base-source"
    base_source.mkdir()
    base_manifest = _write_catalog(base_source)
    base_tag = "catalog-v2-beta.1-2026-07-24"
    _write_index(base_source, base_manifest, base_tag)
    cache = tmp_path / "cache"
    CatalogV2Release.install(
        base_tag,
        catalog_keys=["milo1/test/demo"],
        cache_dir=cache,
        base_url=base_source.as_uri(),
    )

    target_source = tmp_path / "target-source"
    target_source.mkdir()
    for asset in base_source.glob("*.gz"):
        (target_source / asset.name).write_bytes(asset.read_bytes())
    target_tag = "catalog-v2-beta.2-2026-07-25"
    manifest = json.loads(base_manifest.read_text())
    manifest["version"] = target_tag
    manifest["embedding_model"] = MODEL_IDENTITY.replace(
        "9d45a37ebfe40f22ece70507015645de134dc3ec",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    manifest["delta"] = {
        "base_version": base_tag,
        "requires_exact_base": True,
        "operations": 1,
        "metadata_operations": 0,
    }
    target_manifest = target_source / "demo.manifest.json"
    target_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    _write_index(target_source, target_manifest, target_tag)

    release = CatalogV2Release.install(
        target_tag,
        catalog_keys=["milo1/test/demo"],
        cache_dir=cache,
        previous_tag=base_tag,
        base_url=target_source.as_uri(),
    )

    assert release.load("milo1/test/demo").embedding_model == manifest["embedding_model"]


def test_release_rejects_tampered_cached_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = _write_catalog(source)
    tag = "catalog-v2-beta.1-2026-07-24"
    _write_index(source, manifest_path, tag)
    cache = tmp_path / "cache"
    release = CatalogV2Release.install(
        tag,
        catalog_keys=["milo1/test/demo"],
        cache_dir=cache,
        base_url=source.as_uri(),
    )
    cached_manifest = (
        cache / "catalog-v2" / "releases" / tag / "recognition" / "demo" / "demo.manifest.json"
    )
    cached_manifest.write_text("{}")

    with pytest.raises(CatalogV2Error, match="checksum mismatch"):
        release.load("milo1/test/demo")


@pytest.mark.parametrize("tag", ["latest", "catalog-v2-beta.0-2026-07-24", "../beta"])
def test_release_requires_explicit_immutable_beta_tag(tmp_path: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="beta tag"):
        CatalogV2Release.open(tag, cache_dir=tmp_path)
