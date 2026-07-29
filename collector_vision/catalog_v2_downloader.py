"""Download and open client assets from explicit Catalog v2 release tags."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from collector_vision.catalog_v2 import CatalogV2, CatalogV2Error

DEFAULT_REPOSITORY = "HanClinto/CollectorVisionCatalog"
DEFAULT_CATALOG_V2_TAG = "catalog-v2-beta.5-2026-07-28"
FEED_FILENAME = "catalog-feed-v2.json"
DEFAULT_FEED_URL = f"https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2/{FEED_FILENAME}"
INDEX_FILENAME = "catalog-index-v2.json"
_USER_AGENT = "CollectorVision-CatalogV2/0.1"
_BETA_TAG = re.compile(r"^catalog-v2-beta\.[1-9][0-9]*-(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})$")


class CatalogV2Downloader:
    """Downloads, verifies, updates, and opens Catalog v2 client assets."""

    def __init__(
        self,
        *,
        tag: str,
        cache_root: Path,
        index: dict,
        include_metadata: bool,
    ) -> None:
        self.tag = tag
        self.cache_root = cache_root
        self.index = index
        self.include_metadata = include_metadata

    @classmethod
    def install(
        cls,
        tag: str,
        *,
        catalog_keys: list[str] | tuple[str, ...],
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        previous_tag: str | None = None,
        repository: str = DEFAULT_REPOSITORY,
        base_url: str | None = None,
    ) -> CatalogV2Downloader:
        """Install selected catalogs from an explicit immutable release tag.

        If ``previous_tag`` is installed in the same cache and is the manifest's
        exact delta base, one-step deltas are used. Otherwise complete snapshots
        are downloaded.
        """
        _validate_tag(tag)
        if not catalog_keys:
            raise ValueError("catalog_keys must contain at least one catalog")
        if len(set(catalog_keys)) != len(catalog_keys):
            raise ValueError("catalog_keys must not contain duplicates")
        root = _cache_root(cache_dir)
        tag_root = root / tag
        index_path = tag_root / INDEX_FILENAME
        release_url = base_url or (
            f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}"
        )
        index_bytes = _fetch(f"{release_url}/{INDEX_FILENAME}")
        index = _parse_index(index_bytes, tag)
        _write_immutable(index_path, index_bytes)

        downloader = cls(
            tag=tag,
            cache_root=root,
            index=index,
            include_metadata=include_metadata,
        )
        for catalog_key in catalog_keys:
            downloader._install_catalog(
                catalog_key,
                release_url=release_url,
                previous_tag=previous_tag,
            )
        return downloader

    @classmethod
    def install_from_feed(
        cls,
        *,
        catalog_key: str,
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        feed_url: str = DEFAULT_FEED_URL,
    ) -> CatalogV2Downloader:
        """Install a catalog from the feed's bounded base-plus-delta chain."""
        feed_bytes = _fetch(feed_url)
        feed = _parse_feed(feed_bytes)
        entry = _feed_entry(feed, catalog_key)
        references = [entry["base"], *entry["deltas"]]
        root = _cache_root(cache_dir)
        previous = None
        downloader = None
        for reference in references:
            tag = reference.get("version", reference.get("to"))
            if not isinstance(tag, str):
                raise CatalogV2Error("catalog feed contains an invalid version")
            index = _index_for_feed_reference(tag, catalog_key, reference)
            index_bytes = json.dumps(
                index,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _write_immutable(root / tag / INDEX_FILENAME, index_bytes)
            downloader = cls(
                tag=tag,
                cache_root=root,
                index=index,
                include_metadata=include_metadata,
            )
            downloader._install_catalog(
                catalog_key,
                release_url="",
                previous_tag=None,
                previous_catalog=previous,
                feed_reference=reference,
            )
            previous = downloader.load(catalog_key)
        if downloader is None:
            raise CatalogV2Error("catalog feed contains no installable release")
        _write_mutable(_feed_cache_path(root, catalog_key), feed_bytes)
        return downloader

    @classmethod
    def open_from_feed(
        cls,
        *,
        catalog_key: str,
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
    ) -> CatalogV2Downloader:
        """Open the latest catalog recorded by the locally cached feed."""
        root = _cache_root(cache_dir)
        feed_path = _feed_cache_path(root, catalog_key)
        if not feed_path.is_file():
            raise FileNotFoundError("Catalog v2 feed is not installed")
        entry = _feed_entry(_parse_feed(feed_path.read_bytes()), catalog_key)
        reference = entry["base"] if not entry["deltas"] else entry["deltas"][-1]
        latest = reference.get("version", reference.get("to"))
        if not isinstance(latest, str):
            raise CatalogV2Error("catalog feed contains an invalid latest version")
        return cls(
            tag=latest,
            cache_root=root,
            index=_index_for_feed_reference(latest, catalog_key, reference),
            include_metadata=include_metadata,
        )

    @classmethod
    def open(
        cls,
        tag: str,
        *,
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
    ) -> CatalogV2Downloader:
        """Open an already installed explicit release without network access."""
        _validate_tag(tag)
        root = _cache_root(cache_dir)
        index_path = root / tag / INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"Catalog v2 release is not installed: {tag}")
        index_bytes = index_path.read_bytes()
        return cls(
            tag=tag,
            cache_root=root,
            index=_parse_index(index_bytes, tag),
            include_metadata=include_metadata,
        )

    @property
    def catalog_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.index["catalogs"]))

    def load(self, catalog_key: str) -> CatalogV2:
        """Load one installed catalog snapshot."""
        entry = self._entry(catalog_key)
        manifest_filename = entry["manifest_filename"]
        manifest_path = self._catalog_dir(manifest_filename) / manifest_filename
        manifest_bytes = manifest_path.read_bytes()
        _verify_bytes(
            manifest_filename,
            manifest_bytes,
            expected_sha256=entry["sha256"],
        )
        return CatalogV2.load(manifest_path, include_metadata=self.include_metadata)

    def _install_catalog(
        self,
        catalog_key: str,
        *,
        release_url: str,
        previous_tag: str | None,
        previous_catalog: CatalogV2 | None = None,
        feed_reference: dict | None = None,
    ) -> None:
        entry = self._entry(catalog_key)
        manifest_filename = entry["manifest_filename"]
        destination = self._catalog_dir(manifest_filename)
        if destination.is_dir():
            self.load(catalog_key)
            return

        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        temporary.mkdir(parents=True)
        try:
            manifest_url = (
                f"{release_url}/{manifest_filename}"
                if feed_reference is None
                else feed_reference["manifest"]["url"]
            )
            manifest_bytes = _fetch(manifest_url)
            _verify_bytes(
                manifest_filename,
                manifest_bytes,
                expected_sha256=entry["sha256"],
                expected_size=(
                    None if feed_reference is None else feed_reference["manifest"]["size"]
                ),
            )
            manifest = _parse_manifest(manifest_bytes, catalog_key, self.tag)
            manifest_path = temporary / manifest_filename
            manifest_path.write_bytes(manifest_bytes)

            previous = (
                previous_catalog
                if previous_catalog is not None
                else self._load_exact_base(catalog_key, manifest, previous_tag)
            )
            if feed_reference is not None:
                _verify_feed_assets(feed_reference, manifest)
                if "to" in feed_reference and previous is None:
                    raise CatalogV2Error("catalog feed delta is missing its exact base")
            if previous is None:
                asset_names = ["identifiers", "embeddings"]
                if self.include_metadata:
                    asset_names.append("metadata")
                if feed_reference is None:
                    _download_assets(
                        release_url,
                        temporary,
                        manifest["assets"],
                        asset_names,
                    )
                else:
                    _download_feed_assets(
                        temporary,
                        feed_reference["assets"],
                        asset_names,
                        self.tag,
                    )
                CatalogV2.load(
                    manifest_path,
                    include_metadata=self.include_metadata,
                )
            else:
                delta = manifest["delta"]
                asset_names = []
                if delta["operations"]:
                    asset_names.append("identifiers_delta")
                    if "embeddings_delta" in manifest["assets"]:
                        asset_names.append("embeddings_delta")
                if self.include_metadata and delta["metadata_operations"]:
                    asset_names.append("metadata_delta")
                if feed_reference is None:
                    _download_assets(
                        release_url,
                        temporary,
                        manifest["assets"],
                        asset_names,
                    )
                else:
                    _download_feed_assets(
                        temporary,
                        feed_reference["assets"],
                        asset_names,
                        self.tag,
                    )
                current = CatalogV2.apply_delta(
                    previous,
                    manifest_path,
                    include_metadata=self.include_metadata,
                )
                _materialize_snapshot(current, manifest, manifest_path)
                CatalogV2.load(
                    manifest_path,
                    include_metadata=self.include_metadata,
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _load_exact_base(
        self,
        catalog_key: str,
        manifest: dict,
        previous_tag: str | None,
    ) -> CatalogV2 | None:
        delta = manifest.get("delta")
        if not isinstance(delta, dict) or delta.get("requires_exact_base") is not True:
            return None
        base_version = delta.get("base_version")
        if not isinstance(base_version, str):
            return None
        if previous_tag is not None and previous_tag != base_version:
            return None
        previous_tag = base_version
        index_path = self.cache_root / previous_tag / INDEX_FILENAME
        if not index_path.is_file():
            return None
        previous_release = CatalogV2Downloader(
            tag=previous_tag,
            cache_root=self.cache_root,
            index=_parse_index(index_path.read_bytes(), previous_tag),
            include_metadata=self.include_metadata,
        )
        try:
            previous = previous_release.load(catalog_key)
        except FileNotFoundError:
            return None
        if not _delta_base_is_compatible(previous, manifest):
            return None
        return previous

    def _entry(self, catalog_key: str) -> dict:
        try:
            entry = self.index["catalogs"][catalog_key]
        except KeyError:
            raise KeyError(f"catalog {catalog_key!r} is not in release {self.tag!r}") from None
        if not isinstance(entry, dict):
            raise CatalogV2Error(f"invalid index entry for catalog {catalog_key!r}")
        return entry

    def _catalog_dir(self, manifest_filename: str) -> Path:
        layer = "metadata" if self.include_metadata else "recognition"
        slug = manifest_filename.removesuffix(".manifest.json")
        return self.cache_root / self.tag / layer / slug


def _materialize_snapshot(
    catalog: CatalogV2,
    manifest: dict,
    manifest_path: Path,
) -> None:
    identifiers_payload = b"".join(
        json.dumps(
            {
                "key": record.key,
                "identifiers": dict(record.identifiers),
                **({"face_index": record.face_index} if record.face_index else {}),
            },
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in catalog.records
    )
    identifiers_asset = manifest["assets"]["identifiers"]
    embeddings_asset = manifest["assets"]["embeddings"]
    identifiers_path = manifest_path.parent / identifiers_asset["filename"]
    embeddings_path = manifest_path.parent / embeddings_asset["filename"]
    generated_identifiers = _write_gzip_asset(
        identifiers_path,
        identifiers_payload,
        content_type="application/x-ndjson",
    )
    generated_embeddings = _write_gzip_asset(
        embeddings_path,
        catalog.embeddings.astype("<f2", copy=False).tobytes(order="C"),
        content_type="application/octet-stream",
    )
    _verify_materialized_asset("identifiers", generated_identifiers, identifiers_asset)
    _verify_materialized_asset("embeddings", generated_embeddings, embeddings_asset)
    if catalog.metadata_loaded:
        metadata_asset = manifest["assets"]["metadata"]
        metadata_payload = b"".join(
            json.dumps(
                {"key": record.key, "metadata": dict(record.metadata)},
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for record in catalog.records
            if record.metadata is not None
        )
        generated_metadata = _write_gzip_asset(
            manifest_path.parent / metadata_asset["filename"],
            metadata_payload,
            content_type="application/x-ndjson",
        )
        _verify_materialized_asset("metadata", generated_metadata, metadata_asset)


def _write_gzip_asset(path: Path, payload: bytes, *, content_type: str) -> dict:
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
        "content_encoding": "gzip",
        "content_type": content_type,
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _verify_materialized_asset(
    asset_name: str,
    generated: dict,
    published: object,
) -> None:
    if not isinstance(published, dict):
        raise CatalogV2Error(f"manifest asset {asset_name!r} must be an object")
    for field in ("filename", "size", "sha256"):
        if generated[field] != published.get(field):
            raise CatalogV2Error(
                f"delta reconstruction does not match published {asset_name!r} asset"
            )


def _delta_base_is_compatible(previous: CatalogV2, manifest: dict) -> bool:
    descriptor = manifest.get("descriptor")
    expected_descriptor = {
        "game": previous.descriptor.game,
        "source": previous.descriptor.source,
        "profile": previous.descriptor.profile,
        "description": previous.descriptor.description,
        "result_identifier": previous.descriptor.result_identifier,
        "recommended": previous.descriptor.recommended,
    }
    return (
        manifest.get("catalog_key") == previous.catalog_key
        and manifest.get("embedding_model") == previous.embedding_model
        and manifest.get("dtype") == "float16"
        and manifest.get("dim") == previous.embeddings.shape[1]
        and descriptor == expected_descriptor
    )


def _download_assets(
    base_url: str,
    destination: Path,
    assets: dict,
    asset_names: list[str],
) -> None:
    for asset_name in asset_names:
        try:
            asset = assets[asset_name]
        except KeyError:
            raise CatalogV2Error(f"manifest is missing {asset_name!r} asset") from None
        if not isinstance(asset, dict):
            raise CatalogV2Error(f"asset {asset_name!r} must be an object")
        filename = asset.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CatalogV2Error(f"asset {asset_name!r} filename must be a basename")
        payload = _fetch(f"{base_url}/{filename}")
        _verify_bytes(
            filename,
            payload,
            expected_sha256=asset.get("sha256"),
            expected_size=asset.get("size"),
        )
        (destination / filename).write_bytes(payload)


def _download_feed_assets(
    destination: Path,
    assets: dict,
    asset_names: list[str],
    version: str,
) -> None:
    for asset_name in asset_names:
        try:
            asset = assets[asset_name]
        except KeyError:
            raise CatalogV2Error(f"catalog feed is missing {asset_name!r} asset") from None
        filename = _validate_file_reference(asset, version)
        payload = _fetch(asset["url"])
        _verify_bytes(
            filename,
            payload,
            expected_sha256=asset["sha256"],
            expected_size=asset["size"],
        )
        (destination / filename).write_bytes(payload)


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _parse_index(payload: bytes, tag: str) -> dict:
    value = _parse_json(payload, "catalog index")
    if value.get("schema_version") != 2:
        raise CatalogV2Error("unsupported Catalog v2 index schema")
    if value.get("release_version") != tag:
        raise CatalogV2Error("catalog index release_version does not match requested tag")
    catalogs = value.get("catalogs")
    if not isinstance(catalogs, dict):
        raise CatalogV2Error("catalog index catalogs must be an object")
    for key, entry in catalogs.items():
        if not isinstance(key, str) or not key or not isinstance(entry, dict):
            raise CatalogV2Error("catalog index contains an invalid entry")
        filename = entry.get("manifest_filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CatalogV2Error(f"catalog index entry {key!r} has an invalid manifest")
        sha256 = entry.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise CatalogV2Error(f"catalog index entry {key!r} has an invalid checksum")
    return value


def _parse_feed(payload: bytes) -> dict:
    value = _parse_json(payload, "catalog feed")
    if value.get("schema_version") != 2:
        raise CatalogV2Error("unsupported Catalog v2 feed schema")
    for name in ("release_version", "checked_at", "source_updated_at"):
        if not isinstance(value.get(name), str):
            raise CatalogV2Error(f"catalog feed {name} must be a string")
    catalogs = value.get("catalogs")
    if not isinstance(catalogs, dict) or not catalogs:
        raise CatalogV2Error("catalog feed catalogs must be a non-empty object")
    return value


def _feed_entry(feed: dict, catalog_key: str) -> dict:
    entry = feed["catalogs"].get(catalog_key)
    if not isinstance(entry, dict):
        raise CatalogV2Error(f"catalog {catalog_key!r} is not present in the Catalog v2 feed")
    if not isinstance(entry.get("source_updated_at"), str):
        raise CatalogV2Error("catalog feed entry source_updated_at must be a string")
    base = entry.get("base")
    deltas = entry.get("deltas")
    if not isinstance(base, dict) or not isinstance(deltas, list):
        raise CatalogV2Error("catalog feed entry must contain a base and delta list")
    expected = base.get("version")
    _validate_feed_stage(base, expected, is_delta=False)
    for delta in deltas:
        if not isinstance(delta, dict) or delta.get("from") != expected:
            raise CatalogV2Error("catalog feed delta chain is not contiguous")
        expected = delta.get("to")
        _validate_feed_stage(delta, expected, is_delta=True)
    return entry


def _validate_feed_stage(reference: dict, version: object, *, is_delta: bool) -> None:
    if not isinstance(version, str):
        raise CatalogV2Error("catalog feed reference has an invalid version")
    _validate_tag(version)
    _validate_file_reference(reference.get("manifest"), version)
    assets = reference.get("assets")
    if not isinstance(assets, dict):
        raise CatalogV2Error("catalog feed stage assets must be an object")
    if is_delta and not assets:
        raise CatalogV2Error("catalog feed delta must contain assets")
    if not is_delta and not {"identifiers", "embeddings"}.issubset(assets):
        raise CatalogV2Error("catalog feed base lacks required assets")
    for asset in assets.values():
        _validate_file_reference(asset, version)


def _validate_file_reference(reference: object, version: str) -> str:
    if not isinstance(reference, dict):
        raise CatalogV2Error("catalog feed file reference must be an object")
    url = reference.get("url")
    if not isinstance(url, str):
        raise CatalogV2Error("catalog feed file reference has an invalid URL")
    parsed = urlparse(url)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or len(parts) < 2
        or parts[-2] != version
        or Path(parts[-1]).name != parts[-1]
    ):
        raise CatalogV2Error("catalog feed file reference has an invalid URL")
    checksum = reference.get("sha256")
    size = reference.get("size")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise CatalogV2Error("catalog feed file reference has an invalid checksum")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise CatalogV2Error("catalog feed file reference has an invalid size")
    return parts[-1]


def _index_for_feed_reference(tag: str, catalog_key: str, reference: dict) -> dict:
    manifest = reference["manifest"]
    return {
        "schema_version": 2,
        "release_version": tag,
        "catalogs": {
            catalog_key: {
                "manifest_filename": _validate_file_reference(manifest, tag),
                "sha256": manifest["sha256"],
            }
        },
    }


def _verify_feed_assets(reference: dict, manifest: dict) -> None:
    names = (
        ("identifiers_delta", "embeddings_delta", "metadata_delta")
        if "to" in reference
        else ("identifiers", "embeddings", "metadata")
    )
    expected = {name for name in names if name in manifest["assets"]}
    assets = reference["assets"]
    if set(assets) != expected:
        raise CatalogV2Error("catalog feed assets do not match the manifest")
    version = reference.get("version", reference.get("to"))
    for name, feed_asset in assets.items():
        filename = _validate_file_reference(feed_asset, version)
        manifest_asset = manifest["assets"][name]
        if (
            filename != manifest_asset["filename"]
            or feed_asset["sha256"] != manifest_asset["sha256"]
            or feed_asset["size"] != manifest_asset["size"]
        ):
            raise CatalogV2Error(f"catalog feed asset {name!r} does not match the manifest")


def _feed_cache_path(root: Path, catalog_key: str) -> Path:
    digest = hashlib.sha256(catalog_key.encode()).hexdigest()
    return root / "feeds" / f"{digest}.json"


def _parse_manifest(payload: bytes, catalog_key: str, tag: str) -> dict:
    value = _parse_json(payload, "catalog manifest")
    if value.get("schema_version") != 2:
        raise CatalogV2Error("unsupported Catalog v2 manifest schema")
    if value.get("catalog_key") != catalog_key:
        raise CatalogV2Error("catalog manifest key does not match its index entry")
    if value.get("version") != tag:
        raise CatalogV2Error("catalog manifest version does not match requested tag")
    if not isinstance(value.get("assets"), dict):
        raise CatalogV2Error("catalog manifest assets must be an object")
    return value


def _parse_json(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogV2Error(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise CatalogV2Error(f"{label} must be an object")
    return value


def _verify_bytes(
    filename: str,
    payload: bytes,
    *,
    expected_sha256: object,
    expected_size: object | None = None,
) -> None:
    if expected_size is not None and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or len(payload) != expected_size
    ):
        raise CatalogV2Error(f"downloaded asset size mismatch: {filename}")
    if (
        not isinstance(expected_sha256, str)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise CatalogV2Error(f"downloaded asset checksum mismatch: {filename}")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise CatalogV2Error(f"immutable cached release file changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_mutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        base = Path(cache_dir)
    else:
        base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser().resolve() / "catalog-v2" / "releases"


def _validate_tag(tag: str) -> None:
    match = _BETA_TAG.fullmatch(tag)
    if match is None:
        raise ValueError("Catalog v2 beta tag must match 'catalog-v2-beta.<number>-YYYY-MM-DD'")
    try:
        date.fromisoformat(match.group("date"))
    except ValueError as error:
        raise ValueError("Catalog v2 beta tag contains an invalid date") from error
