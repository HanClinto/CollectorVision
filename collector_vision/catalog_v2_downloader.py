"""Discovery, verified download, delta updates, and caching for Catalog v2."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from collector_vision.catalog_v2 import (
    CatalogV2,
    CatalogV2Descriptor,
    CatalogV2Embedding,
    CatalogV2Error,
    CatalogV2Record,
    _parse_descriptor,
    _parse_embedding,
    _parse_record,
    _with_metadata,
    catalog_v2_row_key,
)

DEFAULT_FEED_URL = (
    "https://hanclinto.github.io/CollectorVisionCatalog/catalog-v2/catalog-feed-v2.json"
)
_USER_AGENT = "CollectorVision-CatalogV2/0.2"
_SNAPSHOT_SCHEMA = 1
_GAME_NAMES = {
    "mtg": "magic-the-gathering",
    "pokemon": "pokemon",
    "pokemon-japan": "pokemon-japan",
    "yugioh": "yugioh",
    "fab": "flesh-and-blood",
    "lorcana": "lorcana",
    "digimon": "digimon-card-game",
    "onepiece": "one-piece",
    "swu": "star-wars-unlimited",
    "union-arena": "union-arena",
    "gundam": "gundam-card-game",
    "riftbound": "riftbound",
    "dbs": "dragon-ball-super-card-game",
}


@dataclass(frozen=True)
class _Selection:
    family: str
    local_key: str
    catalog_key: str
    embedding: CatalogV2Embedding
    descriptor: CatalogV2Descriptor
    entry: dict[str, Any]


class CatalogV2Downloader:
    """One installed Catalog v2 snapshot selected from the moving feed."""

    def __init__(self, catalog: CatalogV2, snapshot_path: Path) -> None:
        self._catalog = catalog
        self.snapshot_path = snapshot_path

    @property
    def catalog_key(self) -> str:
        return self._catalog.catalog_key

    @property
    def version(self) -> int:
        return self._catalog.version

    @classmethod
    def install(
        cls,
        game: str | Any,
        *,
        source: str | None = None,
        profile: str | None = None,
        family: str = "milo1",
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        version: int | None = None,
        feed_url: str = DEFAULT_FEED_URL,
    ) -> CatalogV2Downloader:
        """Discover and install the recommended catalog for a game."""
        feed_bytes = _fetch(feed_url)
        feed = _parse_feed(feed_bytes)
        selection = _select_catalog(
            feed,
            game=game,
            source=source,
            profile=profile,
            family=family,
        )
        downloader = cls._install_selection(
            selection,
            include_metadata=include_metadata,
            cache_dir=cache_dir,
            version=version,
        )
        _write_mutable(_feed_cache_path(_cache_root(cache_dir)), feed_bytes)
        return downloader

    @classmethod
    def install_catalog(
        cls,
        catalog_key: str,
        *,
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        version: int | None = None,
        feed_url: str = DEFAULT_FEED_URL,
    ) -> CatalogV2Downloader:
        """Install an explicitly selected full catalog key."""
        feed_bytes = _fetch(feed_url)
        feed = _parse_feed(feed_bytes)
        selection = _select_catalog(feed, catalog_key=catalog_key)
        downloader = cls._install_selection(
            selection,
            include_metadata=include_metadata,
            cache_dir=cache_dir,
            version=version,
        )
        _write_mutable(_feed_cache_path(_cache_root(cache_dir)), feed_bytes)
        return downloader

    @classmethod
    def open(
        cls,
        game: str | Any,
        *,
        source: str | None = None,
        profile: str | None = None,
        family: str = "milo1",
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        version: int | None = None,
    ) -> CatalogV2Downloader:
        """Open the newest matching installed snapshot without network access."""
        root = _cache_root(cache_dir)
        feed = _read_cached_feed(root)
        selection = _select_catalog(
            feed,
            game=game,
            source=source,
            profile=profile,
            family=family,
        )
        return cls._open_selection(
            selection,
            include_metadata=include_metadata,
            root=root,
            version=version,
        )

    @classmethod
    def open_catalog(
        cls,
        catalog_key: str,
        *,
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        version: int | None = None,
    ) -> CatalogV2Downloader:
        root = _cache_root(cache_dir)
        feed = _read_cached_feed(root)
        return cls._open_selection(
            _select_catalog(feed, catalog_key=catalog_key),
            include_metadata=include_metadata,
            root=root,
            version=version,
        )

    @classmethod
    def _install_selection(
        cls,
        selection: _Selection,
        *,
        include_metadata: bool,
        cache_dir: str | Path | None,
        version: int | None,
    ) -> CatalogV2Downloader:
        root = _cache_root(cache_dir)
        entry = selection.entry
        target = entry["current_version"] if version is None else _version(version, "version")
        base_version = entry["base"]["version"]
        if target < base_version or target > entry["current_version"]:
            raise CatalogV2Error(
                f"catalog version {target} is outside the advertised "
                f"{base_version}..{entry['current_version']} route"
            )
        updates = _update_chain(entry, base_version, target)

        exact = _load_cached(
            root,
            selection,
            target,
            include_metadata=include_metadata,
            allow_metadata_superset=not include_metadata,
            recover=True,
        )
        if exact is not None:
            catalog, path = exact
            _prune_snapshots(
                root,
                selection.catalog_key,
                include_metadata=include_metadata,
                keep_version=target,
            )
            return cls(catalog, path)

        if include_metadata:
            recognition = _load_cached(
                root,
                selection,
                target,
                include_metadata=False,
                allow_metadata_superset=False,
                recover=True,
            )
            if recognition is not None:
                catalog = _attach_metadata_route(recognition[0], selection, updates)
                path = _write_snapshot(root, catalog)
                _prune_snapshots(
                    root,
                    selection.catalog_key,
                    include_metadata=True,
                    keep_version=target,
                )
                return cls(catalog, path)

        candidate = _latest_reachable_cached(
            root,
            selection,
            base_version=base_version,
            target=target,
            include_metadata=include_metadata,
        )
        if candidate is None:
            catalog = _download_base(selection, include_metadata=include_metadata)
            path = (
                _write_snapshot(root, catalog)
                if catalog.version == target
                else _snapshot_dir(
                    root,
                    catalog.catalog_key,
                    catalog.version,
                    include_metadata=catalog.metadata_loaded,
                )
            )
        else:
            catalog, path = candidate

        for update in updates:
            if update["to_version"] <= catalog.version:
                continue
            if update["from_version"] != catalog.version:
                raise CatalogV2Error("catalog feed update chain does not match cached snapshot")
            catalog = _apply_update(
                catalog,
                update,
                include_metadata=include_metadata,
            )
            if catalog.version == target:
                path = _write_snapshot(root, catalog)
        if catalog.version != target:
            raise CatalogV2Error("catalog feed did not reconstruct the requested version")
        _prune_snapshots(
            root,
            selection.catalog_key,
            include_metadata=include_metadata,
            keep_version=target,
        )
        return cls(catalog, path)

    @classmethod
    def _open_selection(
        cls,
        selection: _Selection,
        *,
        include_metadata: bool,
        root: Path,
        version: int | None,
    ) -> CatalogV2Downloader:
        target = (
            selection.entry["current_version"] if version is None else _version(version, "version")
        )
        candidates = _cached_versions(
            root,
            selection.catalog_key,
            include_metadata=include_metadata,
            allow_metadata_superset=not include_metadata,
        )
        candidates = (
            [target]
            if version is not None and target in candidates
            else [candidate for candidate in candidates if candidate <= target]
            if version is None
            else []
        )
        if not candidates:
            raise FileNotFoundError(
                f"Catalog v2 {selection.catalog_key!r} has no installed "
                f"{'metadata' if include_metadata else 'recognition'} snapshot"
            )
        selected_version = max(candidates)
        loaded = _load_cached(
            root,
            selection,
            selected_version,
            include_metadata=include_metadata,
            allow_metadata_superset=not include_metadata,
            recover=False,
        )
        if loaded is None:
            raise CatalogV2Error("installed Catalog v2 snapshot is unavailable")
        catalog, path = loaded
        return cls(catalog, path)

    def load(self) -> CatalogV2:
        """Return the verified installed snapshot."""
        return self._catalog


def _parse_feed(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogV2Error("invalid Catalog v2 feed JSON") from error
    if not isinstance(value, dict):
        raise CatalogV2Error("Catalog v2 feed must be an object")
    if not isinstance(value.get("checked_at"), str):
        raise CatalogV2Error("Catalog v2 feed checked_at must be a string")
    families = value.get("families")
    if not isinstance(families, dict) or not families:
        raise CatalogV2Error("Catalog v2 feed families must be a non-empty object")
    return value


def _select_catalog(
    feed: dict[str, Any],
    *,
    game: str | Any | None = None,
    source: str | None = None,
    profile: str | None = None,
    family: str | None = None,
    catalog_key: str | None = None,
) -> _Selection:
    families = feed["families"]
    if catalog_key is not None:
        if not isinstance(catalog_key, str) or "/" not in catalog_key:
            raise ValueError("catalog_key must include its family")
        family, local_key = catalog_key.split("/", 1)
        family_payload = families.get(family)
        if not isinstance(family_payload, dict):
            raise CatalogV2Error(f"catalog family {family!r} is not in the feed")
        catalogs = family_payload.get("catalogs")
        entry = catalogs.get(local_key) if isinstance(catalogs, dict) else None
        if not isinstance(entry, dict):
            raise CatalogV2Error(f"catalog {catalog_key!r} is not in the feed")
        return _parse_selection(family, local_key, family_payload, entry)

    if game is None or family is None:
        raise ValueError("game and family are required for catalog discovery")
    from collector_vision.games import GAME_PRIMARY_SOURCE, Game, parse_game

    parsed_game = game if isinstance(game, Game) else parse_game(str(game))
    descriptor_game = _GAME_NAMES.get(parsed_game.value)
    if descriptor_game is None:
        raise ValueError(f"Catalog v2 has no game mapping for {parsed_game.value!r}")
    family_payload = families.get(family)
    if not isinstance(family_payload, dict):
        raise CatalogV2Error(f"catalog family {family!r} is not in the feed")
    catalogs = family_payload.get("catalogs")
    if not isinstance(catalogs, dict):
        raise CatalogV2Error(f"catalog family {family!r} has no catalogs")
    selected_source = source or GAME_PRIMARY_SOURCE[parsed_game]
    candidates: list[_Selection] = []
    for local_key, entry in catalogs.items():
        if not isinstance(local_key, str) or not isinstance(entry, dict):
            raise CatalogV2Error("catalog family contains an invalid catalog entry")
        selection = _parse_selection(family, local_key, family_payload, entry)
        descriptor = selection.descriptor
        if (
            descriptor.game == descriptor_game
            and descriptor.source == selected_source
            and (profile is None or descriptor.profile == profile)
        ):
            candidates.append(selection)
    recommended = [candidate for candidate in candidates if candidate.descriptor.recommended]
    if len(recommended) == 1:
        return recommended[0]
    if len(candidates) == 1:
        return candidates[0]
    criteria = f"game={descriptor_game!r}, source={selected_source!r}, profile={profile!r}"
    if not candidates:
        raise CatalogV2Error(f"no Catalog v2 entry matches {criteria}")
    raise CatalogV2Error(f"multiple Catalog v2 entries match {criteria}; specify source/profile")


def _parse_selection(
    family: str,
    local_key: str,
    family_payload: dict[str, Any],
    entry: dict[str, Any],
) -> _Selection:
    if not family or not local_key or any(part in {"", ".", ".."} for part in local_key.split("/")):
        raise CatalogV2Error("catalog feed contains an invalid catalog key")
    embedding = _parse_embedding(family_payload.get("embedding"))
    descriptor = _parse_descriptor(entry.get("descriptor"))
    public_name = entry.get("public_name")
    if not isinstance(public_name, str) or not public_name:
        raise CatalogV2Error("catalog public_name must be a non-empty string")
    current = _version(entry.get("current_version"), "catalog current_version")
    rows = _non_negative_int(entry.get("rows"), "catalog rows")
    if not isinstance(entry.get("source_updated_at"), str):
        raise CatalogV2Error("catalog source_updated_at must be a string")
    base = _validate_base(entry.get("base"))
    updates = entry.get("updates")
    if not isinstance(updates, dict):
        raise CatalogV2Error("catalog updates must be an object")
    parsed_updates: dict[str, dict[str, Any]] = {}
    for key, update in updates.items():
        if not isinstance(key, str) or not isinstance(update, dict):
            raise CatalogV2Error("catalog updates contains an invalid entry")
        parsed = _validate_update(update)
        if key != str(parsed["to_version"]):
            raise CatalogV2Error("catalog update key must equal to_version")
        parsed_updates[key] = parsed
    parsed_entry = {
        **entry,
        "current_version": current,
        "rows": rows,
        "base": base,
        "updates": parsed_updates,
    }
    chain = _update_chain(parsed_entry, base["version"], current)
    expected_rows = base["rows"]
    for update in chain:
        expected_rows += update["rows"]["added"] - update["rows"]["deleted"]
    if expected_rows != rows:
        raise CatalogV2Error("catalog feed row totals do not reconstruct current rows")
    return _Selection(
        family=family,
        local_key=local_key,
        catalog_key=f"{family}/{local_key}",
        embedding=embedding,
        descriptor=descriptor,
        entry=parsed_entry,
    )


def _validate_base(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogV2Error("catalog base must be an object")
    version = _version(value.get("version"), "catalog base version")
    rows = _non_negative_int(value.get("rows"), "catalog base rows")
    if not isinstance(value.get("source_updated_at"), str):
        raise CatalogV2Error("catalog base source_updated_at must be a string")
    recognition = _layer(value.get("recognition"), "catalog base recognition")
    metadata = _layer(value.get("metadata"), "catalog base metadata")
    if set(recognition["assets"]) != {"embeddings", "identifiers"}:
        raise CatalogV2Error("catalog base recognition assets are incomplete")
    if set(metadata["assets"]) != {"records"}:
        raise CatalogV2Error("catalog base metadata assets are incomplete")
    return {
        **value,
        "version": version,
        "rows": rows,
        "recognition": recognition,
        "metadata": metadata,
    }


def _validate_update(value: dict[str, Any]) -> dict[str, Any]:
    from_version = _version(value.get("from_version"), "catalog update from_version")
    to_version = _version(value.get("to_version"), "catalog update to_version")
    if to_version != from_version + 1:
        raise CatalogV2Error("catalog update versions must be consecutive")
    rows = value.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"added", "updated", "deleted"}:
        raise CatalogV2Error("catalog update rows must classify added, updated, and deleted")
    parsed_rows = {
        name: _non_negative_int(rows.get(name), f"catalog update rows.{name}")
        for name in ("added", "updated", "deleted")
    }
    recognition = _layer(value.get("recognition"), "catalog update recognition", rows=True)
    metadata = _layer(value.get("metadata"), "catalog update metadata", rows=True)
    if recognition["rows"] and "identifiers" not in recognition["assets"]:
        raise CatalogV2Error("catalog update recognition operations asset is missing")
    if not recognition["rows"] and recognition["assets"]:
        raise CatalogV2Error("empty recognition update must not advertise assets")
    if metadata["rows"] and set(metadata["assets"]) != {"records"}:
        raise CatalogV2Error("catalog update metadata operations asset is missing")
    if not metadata["rows"] and metadata["assets"]:
        raise CatalogV2Error("empty metadata update must not advertise assets")
    return {
        **value,
        "from_version": from_version,
        "to_version": to_version,
        "rows": parsed_rows,
        "recognition": recognition,
        "metadata": metadata,
    }


def _layer(value: object, name: str, *, rows: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogV2Error(f"{name} must be an object")
    assets = value.get("assets")
    if not isinstance(assets, dict):
        raise CatalogV2Error(f"{name} assets must be an object")
    parsed = {key: _asset(reference, f"{name} asset {key}") for key, reference in assets.items()}
    result = {**value, "assets": parsed}
    if rows:
        result["rows"] = _non_negative_int(value.get("rows"), f"{name} rows")
    return result


def _asset(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"url", "size", "sha256"}:
        raise CatalogV2Error(f"{name} must contain url, size, and sha256")
    url = value.get("url")
    parsed = urlparse(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise CatalogV2Error(f"{name} URL must be an absolute HTTPS URL")
    size = _non_negative_int(value.get("size"), f"{name} size")
    sha256 = value.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise CatalogV2Error(f"{name} sha256 must be lowercase hexadecimal")
    return {"url": url, "size": size, "sha256": sha256}


def _update_chain(
    entry: dict[str, Any],
    start_version: int,
    target_version: int,
) -> list[dict[str, Any]]:
    updates = entry["updates"]
    chain: list[dict[str, Any]] = []
    current = start_version
    while current < target_version:
        update = updates.get(str(current + 1))
        if not isinstance(update, dict) or update["from_version"] != current:
            raise CatalogV2Error("catalog feed does not contain a contiguous update route")
        chain.append(update)
        current = update["to_version"]
    return chain


def _download_base(selection: _Selection, *, include_metadata: bool) -> CatalogV2:
    base = selection.entry["base"]
    identifiers = _jsonl_asset(
        base["recognition"]["assets"]["identifiers"],
        "base identifiers",
    )
    if len(identifiers) != base["rows"]:
        raise CatalogV2Error("base identifier count does not match advertised rows")
    metadata = (
        _metadata_asset(base["metadata"]["assets"]["records"], base["rows"])
        if include_metadata
        else [None] * base["rows"]
    )
    records = _records(
        identifiers,
        metadata,
        descriptor=selection.descriptor,
        source=selection.descriptor.source,
    )
    embeddings = _embedding_asset(
        base["recognition"]["assets"]["embeddings"],
        rows=base["rows"],
        dimensions=selection.embedding.dimensions,
        label="base embeddings",
    )
    return CatalogV2._from_data(
        embeddings=embeddings,
        records=records,
        catalog_key=selection.catalog_key,
        family=selection.family,
        version=base["version"],
        embedding=selection.embedding,
        descriptor=selection.descriptor,
        metadata_loaded=include_metadata,
    )


def _apply_update(
    previous: CatalogV2,
    update: dict[str, Any],
    *,
    include_metadata: bool,
) -> CatalogV2:
    if previous.version != update["from_version"]:
        raise CatalogV2Error("catalog update requires its exact predecessor")
    source = previous.descriptor.source
    records = {record.key(source): record for record in previous.records}
    embeddings = {
        record.key(source): previous.embeddings[index].copy()
        for index, record in enumerate(previous.records)
    }
    recognition_operations = (
        _jsonl_asset(
            update["recognition"]["assets"]["identifiers"],
            "identifier delta",
        )
        if update["recognition"]["rows"]
        else []
    )
    if len(recognition_operations) != update["recognition"]["rows"]:
        raise CatalogV2Error("identifier delta count does not match feed")
    parsed_operations: list[tuple[str, str, CatalogV2Record | None, int | None]] = []
    upsert_indexes: set[int] = set()
    operated: set[str] = set()
    added: set[str] = set()
    changed: set[str] = set()
    deleted: set[str] = set()
    for operation in recognition_operations:
        if not isinstance(operation, dict):
            raise CatalogV2Error("identifier delta operation must be an object")
        op = operation.get("op")
        if op == "delete":
            if set(operation) - {"op", "id", "face_index"}:
                raise CatalogV2Error("identifier delta delete has invalid fields")
            key = _operation_key(operation, source, "identifier delta delete")
            if key in operated or key not in records:
                raise CatalogV2Error("identifier delta delete targets a missing or duplicate row")
            operated.add(key)
            deleted.add(key)
            parsed_operations.append((op, key, None, None))
        elif op == "upsert":
            if set(operation) != {"op", "record", "embedding_index"}:
                raise CatalogV2Error("identifier delta upsert has invalid fields")
            record = _parse_record(
                operation.get("record"),
                descriptor=previous.descriptor,
            )
            key = record.key(source)
            index = _non_negative_int(
                operation.get("embedding_index"),
                "identifier delta embedding_index",
            )
            if key in operated or index in upsert_indexes:
                raise CatalogV2Error("identifier delta upsert is duplicated")
            operated.add(key)
            upsert_indexes.add(index)
            (changed if key in records else added).add(key)
            parsed_operations.append((op, key, record, index))
        else:
            raise CatalogV2Error(f"unsupported identifier delta operation {op!r}")
    if upsert_indexes != set(range(len(upsert_indexes))):
        raise CatalogV2Error("identifier delta embedding indexes must be contiguous")
    if upsert_indexes:
        asset = update["recognition"]["assets"].get("embeddings")
        if asset is None:
            raise CatalogV2Error("identifier delta embeddings asset is missing")
        delta_embeddings = _embedding_asset(
            asset,
            rows=len(upsert_indexes),
            dimensions=previous.embedding.dimensions,
            label="delta embeddings",
        )
    else:
        if "embeddings" in update["recognition"]["assets"]:
            raise CatalogV2Error("delete-only identifier delta must not contain embeddings")
        delta_embeddings = np.empty((0, previous.embedding.dimensions), dtype="<f2")
    for op, key, record, index in parsed_operations:
        if op == "delete":
            records.pop(key)
            embeddings.pop(key)
            continue
        assert record is not None and index is not None
        existing_metadata = records[key].metadata if key in records else None
        records[key] = _with_metadata(record, existing_metadata)
        embeddings[key] = delta_embeddings[index].copy()

    metadata_changed: set[str] = set()
    if include_metadata:
        metadata_operations = (
            _jsonl_asset(
                update["metadata"]["assets"]["records"],
                "metadata delta",
            )
            if update["metadata"]["rows"]
            else []
        )
        if len(metadata_operations) != update["metadata"]["rows"]:
            raise CatalogV2Error("metadata delta count does not match feed")
        seen_metadata: set[str] = set()
        for operation in metadata_operations:
            if not isinstance(operation, dict):
                raise CatalogV2Error("metadata delta operation must be an object")
            key = _operation_key(operation, source, "metadata delta")
            if key in seen_metadata:
                raise CatalogV2Error("metadata delta contains duplicate operations")
            seen_metadata.add(key)
            op = operation.get("op")
            if op == "delete":
                if set(operation) - {"op", "id", "face_index"}:
                    raise CatalogV2Error("metadata delta delete has invalid fields")
                if key in records:
                    records[key] = _with_metadata(records[key], None)
            elif op == "upsert":
                if set(operation) - {"op", "id", "face_index", "metadata"}:
                    raise CatalogV2Error("metadata delta upsert has invalid fields")
                metadata = operation.get("metadata")
                if not isinstance(metadata, dict) or key not in records:
                    raise CatalogV2Error("metadata delta upsert targets an invalid row")
                records[key] = _with_metadata(records[key], metadata)
            else:
                raise CatalogV2Error(f"unsupported metadata delta operation {op!r}")
            if key not in added and key not in deleted:
                metadata_changed.add(key)

    row_changes = update["rows"]
    if len(added) != row_changes["added"] or len(deleted) != row_changes["deleted"]:
        raise CatalogV2Error("catalog update added/deleted counts do not match operations")
    if include_metadata and len(changed | metadata_changed) != row_changes["updated"]:
        raise CatalogV2Error("catalog update updated count does not match operations")
    expected_rows = len(previous) + row_changes["added"] - row_changes["deleted"]
    if len(records) != expected_rows:
        raise CatalogV2Error("catalog update reconstructed the wrong row count")
    sorted_keys = sorted(records)
    matrix = (
        np.vstack([embeddings[key] for key in sorted_keys]).astype("<f2", copy=False)
        if sorted_keys
        else np.empty((0, previous.embedding.dimensions), dtype="<f2")
    )
    return CatalogV2._from_data(
        embeddings=matrix,
        records=[records[key] for key in sorted_keys],
        catalog_key=previous.catalog_key,
        family=previous.family,
        version=update["to_version"],
        embedding=previous.embedding,
        descriptor=previous.descriptor,
        metadata_loaded=include_metadata,
    )


def _attach_metadata_route(
    recognition: CatalogV2,
    selection: _Selection,
    updates: list[dict[str, Any]],
) -> CatalogV2:
    base = selection.entry["base"]
    base_identifiers = _jsonl_asset(
        base["recognition"]["assets"]["identifiers"],
        "base identifiers",
    )
    base_metadata = _metadata_asset(base["metadata"]["assets"]["records"], base["rows"])
    base_records = _records(
        base_identifiers,
        base_metadata,
        descriptor=selection.descriptor,
        source=selection.descriptor.source,
    )
    metadata = {record.key(selection.descriptor.source): record.metadata for record in base_records}
    for update in updates:
        operations = (
            _jsonl_asset(update["metadata"]["assets"]["records"], "metadata delta")
            if update["metadata"]["rows"]
            else []
        )
        if len(operations) != update["metadata"]["rows"]:
            raise CatalogV2Error("metadata delta count does not match feed")
        seen: set[str] = set()
        for operation in operations:
            key = _operation_key(operation, selection.descriptor.source, "metadata delta")
            if key in seen:
                raise CatalogV2Error("metadata delta contains duplicate operations")
            seen.add(key)
            if operation.get("op") == "delete":
                metadata.pop(key, None)
            elif operation.get("op") == "upsert" and isinstance(operation.get("metadata"), dict):
                metadata[key] = operation["metadata"]
            else:
                raise CatalogV2Error("metadata delta contains an invalid operation")
    records = [
        _with_metadata(record, metadata.get(record.key(selection.descriptor.source)))
        for record in recognition.records
    ]
    return CatalogV2._from_data(
        embeddings=recognition.embeddings.copy(),
        records=records,
        catalog_key=recognition.catalog_key,
        family=recognition.family,
        version=recognition.version,
        embedding=recognition.embedding,
        descriptor=recognition.descriptor,
        metadata_loaded=True,
    )


def _records(
    identifiers: list[Any],
    metadata: list[dict[str, Any] | None],
    *,
    descriptor: CatalogV2Descriptor,
    source: str,
) -> tuple[CatalogV2Record, ...]:
    if len(identifiers) != len(metadata):
        raise CatalogV2Error("base identifiers and metadata are not aligned")
    records = tuple(
        _parse_record(value, descriptor=descriptor, metadata=fields)
        for value, fields in zip(identifiers, metadata, strict=True)
    )
    keys = [record.key(source) for record in records]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise CatalogV2Error("base identifiers must be uniquely sorted by derived identity")
    return records


def _metadata_asset(reference: dict[str, Any], rows: int) -> list[dict[str, Any] | None]:
    values = _jsonl_asset(reference, "base metadata", allow_null=True)
    if len(values) != rows:
        raise CatalogV2Error("base metadata count does not match advertised rows")
    if any(value is not None and not isinstance(value, dict) for value in values):
        raise CatalogV2Error("base metadata rows must be objects or null")
    return values


def _jsonl_asset(
    reference: dict[str, Any],
    label: str,
    *,
    allow_null: bool = False,
) -> list[Any]:
    decoded = _gzip_asset(reference, label)
    values: list[Any] = []
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogV2Error(f"invalid JSON in {label} at line {line_number}") from error
        if value is None and allow_null:
            values.append(None)
        elif not isinstance(value, dict):
            raise CatalogV2Error(f"{label} line {line_number} must be an object")
        else:
            values.append(value)
    return values


def _embedding_asset(
    reference: dict[str, Any],
    *,
    rows: int,
    dimensions: int,
    label: str,
) -> np.ndarray:
    decoded = _gzip_asset(reference, label)
    expected = rows * dimensions * np.dtype("<f2").itemsize
    if len(decoded) != expected:
        raise CatalogV2Error(f"{label} decoded size does not match its matrix shape")
    return np.frombuffer(decoded, dtype="<f2").reshape(rows, dimensions).copy()


def _gzip_asset(reference: dict[str, Any], label: str) -> bytes:
    payload = _fetch(reference["url"])
    _verify_payload(payload, reference, label)
    try:
        return gzip.decompress(payload)
    except (OSError, EOFError) as error:
        raise CatalogV2Error(f"{label} is not valid gzip") from error


def _verify_payload(payload: bytes, reference: dict[str, Any], label: str) -> None:
    if len(payload) != reference["size"]:
        raise CatalogV2Error(f"{label} compressed size mismatch")
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise CatalogV2Error(f"{label} checksum mismatch")


def _operation_key(operation: object, source: str, label: str) -> str:
    if not isinstance(operation, dict):
        raise CatalogV2Error(f"{label} must be an object")
    primary_id = operation.get("id")
    if not isinstance(primary_id, str) or not primary_id or ":" in primary_id:
        raise CatalogV2Error(f"{label} id is invalid")
    face_index = operation.get("face_index", 0)
    if not isinstance(face_index, int) or isinstance(face_index, bool) or face_index < 0:
        raise CatalogV2Error(f"{label} face_index is invalid")
    if face_index == 0 and "face_index" in operation:
        raise CatalogV2Error(f"{label} must omit face_index for face 0")
    return catalog_v2_row_key(source, primary_id, face_index)


def _without_metadata(catalog: CatalogV2) -> CatalogV2:
    if not catalog.metadata_loaded:
        return catalog
    return CatalogV2._from_data(
        embeddings=catalog.embeddings.copy(),
        records=[_with_metadata(record, None) for record in catalog.records],
        catalog_key=catalog.catalog_key,
        family=catalog.family,
        version=catalog.version,
        embedding=catalog.embedding,
        descriptor=catalog.descriptor,
        metadata_loaded=False,
    )


def _write_snapshot(root: Path, catalog: CatalogV2) -> Path:
    destination = _snapshot_dir(
        root,
        catalog.catalog_key,
        catalog.version,
        include_metadata=catalog.metadata_loaded,
    )
    if destination.is_dir():
        loaded = _load_snapshot(destination)
        _assert_compatible(loaded, catalog)
        return destination
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        identifier_rows = []
        for record in catalog.records:
            value: dict[str, Any] = {
                "id": record.id,
                "identifiers": dict(sorted(record.identifiers.items())),
            }
            if record.face_index:
                value["face_index"] = record.face_index
            if record.finishes:
                value["finishes"] = list(record.finishes)
            identifier_rows.append(value)
        assets = {
            "identifiers": _write_gzip_jsonl(
                temporary / "identifiers.jsonl.gz",
                identifier_rows,
            ),
            "embeddings": _write_gzip(
                temporary / "embeddings.f16.gz",
                catalog.embeddings.astype("<f2", copy=False).tobytes(order="C"),
            ),
        }
        if catalog.metadata_loaded:
            assets["metadata"] = _write_gzip_jsonl(
                temporary / "metadata.jsonl.gz",
                [record.metadata for record in catalog.records],
            )
        receipt = {
            "schema": _SNAPSHOT_SCHEMA,
            "catalog_key": catalog.catalog_key,
            "family": catalog.family,
            "version": catalog.version,
            "embedding": asdict(catalog.embedding),
            "descriptor": asdict(catalog.descriptor),
            "metadata_loaded": catalog.metadata_loaded,
            "rows": len(catalog),
            "assets": assets,
        }
        (temporary / "snapshot.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.is_dir():
                raise
            _assert_compatible(_load_snapshot(destination), catalog)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _load_snapshot(
    path: Path,
    *,
    include_metadata: bool | None = None,
) -> CatalogV2:
    receipt_path = path / "snapshot.json"
    if not receipt_path.is_file():
        raise CatalogV2Error(f"Catalog v2 snapshot receipt is missing: {path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CatalogV2Error("invalid cached Catalog v2 snapshot receipt") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != _SNAPSHOT_SCHEMA:
        raise CatalogV2Error("unsupported cached Catalog v2 snapshot schema")
    catalog_key = receipt.get("catalog_key")
    family = receipt.get("family")
    version = receipt.get("version")
    stored_metadata = receipt.get("metadata_loaded")
    rows = receipt.get("rows")
    if (
        not isinstance(catalog_key, str)
        or not isinstance(family, str)
        or not isinstance(version, int)
        or isinstance(version, bool)
        or not isinstance(stored_metadata, bool)
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows < 0
    ):
        raise CatalogV2Error("cached Catalog v2 snapshot identity is invalid")
    embedding = _parse_embedding(receipt.get("embedding"))
    descriptor = _parse_descriptor(receipt.get("descriptor"))
    assets = receipt.get("assets")
    if include_metadata is True and not stored_metadata:
        raise CatalogV2Error("cached Catalog v2 snapshot does not contain metadata")
    metadata_loaded = stored_metadata if include_metadata is None else include_metadata
    expected_assets = {"identifiers", "embeddings", *(["metadata"] if stored_metadata else [])}
    if not isinstance(assets, dict) or set(assets) != expected_assets:
        raise CatalogV2Error("cached Catalog v2 snapshot assets are invalid")
    decoded: dict[str, bytes] = {}
    for name, reference in assets.items():
        if name == "metadata" and not metadata_loaded:
            continue
        if not isinstance(reference, dict):
            raise CatalogV2Error("cached Catalog v2 asset reference is invalid")
        filename = reference.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CatalogV2Error("cached Catalog v2 asset filename is invalid")
        try:
            payload = (path / filename).read_bytes()
        except FileNotFoundError as error:
            raise CatalogV2Error(f"cached Catalog v2 {name} asset is missing") from error
        if len(payload) != reference.get("size") or hashlib.sha256(
            payload
        ).hexdigest() != reference.get("sha256"):
            raise CatalogV2Error(f"cached Catalog v2 {name} asset is corrupt")
        try:
            decoded[name] = gzip.decompress(payload)
        except (OSError, EOFError) as error:
            raise CatalogV2Error(f"cached Catalog v2 {name} asset is invalid gzip") from error
    identifiers = _parse_jsonl_bytes(decoded["identifiers"], "cached identifiers")
    metadata = (
        _parse_jsonl_bytes(decoded["metadata"], "cached metadata", allow_null=True)
        if metadata_loaded
        else [None] * rows
    )
    if len(identifiers) != rows or len(metadata) != rows:
        raise CatalogV2Error("cached Catalog v2 row count is invalid")
    records = _records(
        identifiers,
        metadata,
        descriptor=descriptor,
        source=descriptor.source,
    )
    expected_matrix = rows * embedding.dimensions * np.dtype("<f2").itemsize
    if len(decoded["embeddings"]) != expected_matrix:
        raise CatalogV2Error("cached Catalog v2 embedding matrix size is invalid")
    matrix = (
        np.frombuffer(decoded["embeddings"], dtype="<f2").reshape(rows, embedding.dimensions).copy()
    )
    return CatalogV2._from_data(
        embeddings=matrix,
        records=records,
        catalog_key=catalog_key,
        family=family,
        version=version,
        embedding=embedding,
        descriptor=descriptor,
        metadata_loaded=metadata_loaded,
    )


def _parse_jsonl_bytes(payload: bytes, label: str, *, allow_null: bool = False) -> list[Any]:
    values = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogV2Error(f"invalid {label} JSON at line {line_number}") from error
        if value is None and allow_null:
            values.append(None)
        elif not isinstance(value, dict):
            raise CatalogV2Error(f"{label} line {line_number} must be an object")
        else:
            values.append(value)
    return values


def _write_gzip_jsonl(path: Path, values: list[Any]) -> dict[str, Any]:
    payload = b"".join(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for value in values
    )
    return _write_gzip(path, payload)


def _write_gzip(path: Path, payload: bytes) -> dict[str, Any]:
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    path.write_bytes(compressed)
    return {
        "filename": path.name,
        "size": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def _load_cached(
    root: Path,
    selection: _Selection,
    version: int,
    *,
    include_metadata: bool,
    allow_metadata_superset: bool,
    recover: bool,
) -> tuple[CatalogV2, Path] | None:
    modes = [include_metadata]
    if allow_metadata_superset and not include_metadata:
        modes.append(True)
    for mode in modes:
        path = _snapshot_dir(root, selection.catalog_key, version, include_metadata=mode)
        if not path.is_dir():
            continue
        try:
            catalog = _load_snapshot(path, include_metadata=include_metadata)
            _assert_selection(catalog, selection)
        except CatalogV2Error:
            if not recover:
                raise
            shutil.rmtree(path)
            continue
        catalog = _refresh_selection(catalog, selection)
        if not include_metadata:
            catalog = _without_metadata(catalog)
        return catalog, path
    return None


def _latest_reachable_cached(
    root: Path,
    selection: _Selection,
    *,
    base_version: int,
    target: int,
    include_metadata: bool,
) -> tuple[CatalogV2, Path] | None:
    versions = _cached_versions(
        root,
        selection.catalog_key,
        include_metadata=include_metadata,
        allow_metadata_superset=not include_metadata,
    )
    for version in sorted(
        (value for value in versions if base_version <= value <= target),
        reverse=True,
    ):
        loaded = _load_cached(
            root,
            selection,
            version,
            include_metadata=include_metadata,
            allow_metadata_superset=not include_metadata,
            recover=True,
        )
        if loaded is not None:
            return loaded
    return None


def _cached_versions(
    root: Path,
    catalog_key: str,
    *,
    include_metadata: bool,
    allow_metadata_superset: bool,
) -> list[int]:
    slug = _catalog_slug(catalog_key)
    modes = ["metadata" if include_metadata else "recognition"]
    if allow_metadata_superset and "metadata" not in modes:
        modes.append("metadata")
    versions: set[int] = set()
    for mode in modes:
        parent = root / "snapshots" / slug / mode
        if not parent.is_dir():
            continue
        for path in parent.glob("version-*"):
            try:
                versions.add(int(path.name.removeprefix("version-")))
            except ValueError:
                continue
    return sorted(versions)


def _snapshot_dir(
    root: Path,
    catalog_key: str,
    version: int,
    *,
    include_metadata: bool,
) -> Path:
    mode = "metadata" if include_metadata else "recognition"
    return root / "snapshots" / _catalog_slug(catalog_key) / mode / f"version-{version}"


def _catalog_slug(catalog_key: str) -> str:
    parts = catalog_key.split("/")
    if any(not part or part in {".", ".."} or "--" in part for part in parts):
        raise CatalogV2Error("catalog key cannot be represented safely in the cache")
    return "--".join(parts)


def _assert_selection(catalog: CatalogV2, selection: _Selection) -> None:
    if (
        catalog.catalog_key != selection.catalog_key
        or catalog.family != selection.family
        or catalog.embedding != selection.embedding
        or catalog.descriptor.game != selection.descriptor.game
        or catalog.descriptor.source != selection.descriptor.source
        or catalog.descriptor.profile != selection.descriptor.profile
        or catalog.descriptor.result_identifier != selection.descriptor.result_identifier
    ):
        raise CatalogV2Error("cached Catalog v2 snapshot is incompatible with the feed")


def _refresh_selection(catalog: CatalogV2, selection: _Selection) -> CatalogV2:
    return CatalogV2._from_data(
        embeddings=catalog.embeddings,
        records=catalog.records,
        catalog_key=catalog.catalog_key,
        family=catalog.family,
        version=catalog.version,
        embedding=catalog.embedding,
        descriptor=selection.descriptor,
        metadata_loaded=catalog.metadata_loaded,
    )


def _prune_snapshots(
    root: Path,
    catalog_key: str,
    *,
    include_metadata: bool,
    keep_version: int,
) -> None:
    mode = "metadata" if include_metadata else "recognition"
    parent = root / "snapshots" / _catalog_slug(catalog_key) / mode
    if not parent.is_dir():
        return
    for path in parent.glob("version-*"):
        if path.name != f"version-{keep_version}" and path.is_dir():
            shutil.rmtree(path)


def _assert_compatible(left: CatalogV2, right: CatalogV2) -> None:
    if (
        left.catalog_key != right.catalog_key
        or left.version != right.version
        or left.embedding != right.embedding
        or left.descriptor != right.descriptor
        or left.metadata_loaded != right.metadata_loaded
        or left.records != right.records
        or not np.array_equal(left.embeddings, right.embeddings)
    ):
        raise CatalogV2Error("immutable cached Catalog v2 snapshot changed")


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _write_mutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _feed_cache_path(root: Path) -> Path:
    return root / "feed.json"


def _read_cached_feed(root: Path) -> dict[str, Any]:
    path = _feed_cache_path(root)
    if not path.is_file():
        raise CatalogV2Error(
            "Catalog v2 feed is not cached; install a catalog online before using offline mode"
        )
    return _parse_feed(path.read_bytes())


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        base = Path(cache_dir)
    else:
        base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser().resolve() / "catalog-v2"


def _version(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CatalogV2Error(f"{name} must be a non-negative integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CatalogV2Error(f"{name} must be a non-negative integer")
    return value
