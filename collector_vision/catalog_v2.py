"""Catalog v2 client.

This module is intentionally independent from :mod:`collector_vision.catalog`.
Catalog v1 continues to use NPZ/Hugging Face artifacts while v2 uses versioned
GitHub Release manifests, aligned JSONL records, and raw FP16 matrices.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collector_vision.games import Game

_GAME_NAMES = {
    "mtg": "magic-the-gathering",
    "pokemon": "pokemon",
    "yugioh": "yugioh",
    "fab": "flesh-and-blood",
    "lorcana": "lorcana",
    "digimon": "digimon-card-game",
    "onepiece": "one-piece",
    "swu": "star-wars-unlimited",
}

_MODEL_IDENTITY = re.compile(
    r"^collectorvision@[0-9a-f]{40}:(?P<model>[a-z0-9.-]+)@sha256:(?P<sha256>[0-9a-f]{64})$"
)


class CatalogV2Error(ValueError):
    """Raised when a Catalog v2 artifact violates the client contract."""


@dataclass(frozen=True)
class CatalogV2Descriptor:
    game: str
    source: str
    profile: str
    description: str
    result_identifier: str
    recommended: bool


@dataclass(frozen=True)
class CatalogV2Record:
    key: str
    identifiers: Mapping[str, str]
    face_index: int = 0
    metadata: Mapping[str, Any] | None = None


class CatalogV2:
    """Loaded Catalog v2 recognition snapshot.

    Use :meth:`load` with a downloaded v2 manifest. This explicit class keeps
    beta v2 behavior separate from the stable :class:`collector_vision.Catalog`
    API and its v1 cache.
    """

    def __init__(
        self,
        *,
        embeddings: np.ndarray,
        records: tuple[CatalogV2Record, ...],
        catalog_key: str,
        version: str,
        embedding_model: str,
        descriptor: CatalogV2Descriptor,
        metadata_loaded: bool = False,
    ) -> None:
        self.embeddings = embeddings
        self.records = records
        self.catalog_key = catalog_key
        self.version = version
        self.embedding_model = embedding_model
        self.descriptor = descriptor
        self.metadata_loaded = metadata_loaded
        self._embedder = None

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        include_metadata: bool = False,
    ) -> CatalogV2:
        """Load and verify a complete v2 snapshot beside its manifest."""
        path = Path(manifest_path)
        manifest = _load_json_object(path)
        if manifest.get("schema_version") != 2:
            raise CatalogV2Error("unsupported Catalog v2 manifest schema")

        catalog_key = _required_string(manifest, "catalog_key", "manifest")
        version = _required_string(manifest, "version", "manifest")
        embedding_model = _required_string(manifest, "embedding_model", "manifest")
        rows = _required_non_negative_int(manifest, "rows", "manifest")
        dim = _required_positive_int(manifest, "dim", "manifest")
        if manifest.get("dtype") != "float16":
            raise CatalogV2Error("Catalog v2 requires dtype 'float16'")

        descriptor = _parse_descriptor(manifest.get("descriptor"))
        assets = manifest.get("assets")
        if not isinstance(assets, dict):
            raise CatalogV2Error("manifest assets must be an object")

        recognition_rows = _asset_path(path.parent, assets, "recognition_rows")
        recognition_matrix = _asset_path(path.parent, assets, "recognition_matrix")
        _verify_asset(recognition_rows, assets["recognition_rows"])
        _verify_asset(recognition_matrix, assets["recognition_matrix"])

        raw_records = _read_jsonl_gzip(recognition_rows)
        if len(raw_records) != rows:
            raise CatalogV2Error(
                f"recognition row count mismatch: expected {rows}, found {len(raw_records)}"
            )
        metadata_by_key: dict[str, Mapping[str, Any]] = {}
        if include_metadata:
            metadata_rows = _asset_path(path.parent, assets, "metadata_rows")
            _verify_asset(metadata_rows, assets["metadata_rows"])
            metadata_by_key = _parse_metadata(_read_jsonl_gzip(metadata_rows))

        records = _parse_records(
            raw_records,
            result_identifier=descriptor.result_identifier,
            metadata_by_key=metadata_by_key,
        )
        unknown_metadata = metadata_by_key.keys() - {record.key for record in records}
        if unknown_metadata:
            raise CatalogV2Error(
                f"metadata contains unknown recognition key {min(unknown_metadata)!r}"
            )

        with gzip.open(recognition_matrix, "rb") as stream:
            matrix_bytes = stream.read()
        expected_bytes = rows * dim * np.dtype("<f2").itemsize
        if len(matrix_bytes) != expected_bytes:
            raise CatalogV2Error(
                f"recognition matrix size mismatch: expected {expected_bytes}, "
                f"found {len(matrix_bytes)}"
            )
        embeddings = np.frombuffer(matrix_bytes, dtype="<f2").reshape(rows, dim).copy()

        return cls(
            embeddings=embeddings,
            records=records,
            catalog_key=catalog_key,
            version=version,
            embedding_model=embedding_model,
            descriptor=descriptor,
            metadata_loaded=include_metadata,
        )

    @classmethod
    def for_game(
        cls,
        game: str | Game,
        *,
        source: str | None = None,
        profile: str | None = None,
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        version: str | None = None,
    ) -> CatalogV2:
        """Download or open a ready-to-search catalog for one game.

        Defaults mirror :meth:`collector_vision.Catalog.for_game`: choose the
        game's primary source and its recommended catalog. ``profile="cards"``
        selects the compact Scryfall MTG catalog.
        """
        from collector_vision.catalog_v2_downloader import (
            DEFAULT_CATALOG_V2_TAG,
            CatalogV2Downloader,
        )
        from collector_vision.games import GAME_PRIMARY_SOURCE, Game, parse_game

        parsed_game = game if isinstance(game, Game) else parse_game(str(game))
        descriptor_game = _GAME_NAMES.get(parsed_game.value)
        if descriptor_game is None:
            raise ValueError(f"Catalog v2 does not publish a catalog for {parsed_game.value!r}")
        selected_source = source or GAME_PRIMARY_SOURCE[parsed_game]
        selected_version = version or DEFAULT_CATALOG_V2_TAG
        if offline:
            downloader = CatalogV2Downloader.open(
                selected_version,
                include_metadata=include_metadata,
                cache_dir=cache_dir,
            )
            catalog_key = downloader.catalog_for_game(
                game=descriptor_game,
                source=selected_source,
                profile=profile,
            )
        else:
            downloader, catalog_key = CatalogV2Downloader.install_for_game(
                selected_version,
                game=descriptor_game,
                source=selected_source,
                profile=profile,
                include_metadata=include_metadata,
                cache_dir=cache_dir,
            )
        return downloader.load(catalog_key)

    @classmethod
    def apply_delta(
        cls,
        previous: CatalogV2,
        manifest_path: str | Path,
        *,
        include_metadata: bool = False,
    ) -> CatalogV2:
        """Apply one exact-base v2 delta to an already loaded snapshot."""
        path = Path(manifest_path)
        manifest = _load_json_object(path)
        if manifest.get("schema_version") != 2:
            raise CatalogV2Error("unsupported Catalog v2 manifest schema")
        catalog_key = _required_string(manifest, "catalog_key", "manifest")
        version = _required_string(manifest, "version", "manifest")
        embedding_model = _required_string(manifest, "embedding_model", "manifest")
        rows = _required_non_negative_int(manifest, "rows", "manifest")
        dim = _required_positive_int(manifest, "dim", "manifest")
        if manifest.get("dtype") != "float16":
            raise CatalogV2Error("Catalog v2 requires dtype 'float16'")
        descriptor = _parse_descriptor(manifest.get("descriptor"))
        delta = manifest.get("delta")
        if not isinstance(delta, dict):
            raise CatalogV2Error("manifest delta must be an object")
        base_version = _required_string(delta, "base_version", "manifest delta")
        if not delta.get("requires_exact_base"):
            raise CatalogV2Error("versioned delta must require its exact base")
        if previous.version != base_version:
            raise CatalogV2Error(f"delta requires base {base_version!r}, not {previous.version!r}")
        if (
            previous.catalog_key != catalog_key
            or previous.embedding_model != embedding_model
            or previous.descriptor != descriptor
            or previous.embeddings.shape[1] != dim
        ):
            raise CatalogV2Error("delta is incompatible with the previous catalog")
        if include_metadata and not previous.metadata_loaded:
            raise CatalogV2Error("metadata delta requires metadata in the previous catalog")

        assets = manifest.get("assets")
        if not isinstance(assets, dict):
            raise CatalogV2Error("manifest assets must be an object")
        operations_path = _asset_path(path.parent, assets, "delta_operations")
        matrix_path = _asset_path(path.parent, assets, "delta_matrix")
        _verify_asset(operations_path, assets["delta_operations"])
        _verify_asset(matrix_path, assets["delta_matrix"])
        operations = _read_jsonl_gzip(operations_path)
        expected_operations = _required_non_negative_int(delta, "operations", "manifest delta")
        if len(operations) != expected_operations:
            raise CatalogV2Error(
                f"delta operation count mismatch: expected {expected_operations}, "
                f"found {len(operations)}"
            )

        with gzip.open(matrix_path, "rb") as stream:
            matrix_bytes = stream.read()
        upsert_count = sum(operation.get("op") == "upsert" for operation in operations)
        expected_matrix_bytes = upsert_count * dim * np.dtype("<f2").itemsize
        if len(matrix_bytes) != expected_matrix_bytes:
            raise CatalogV2Error(
                f"delta matrix size mismatch: expected {expected_matrix_bytes}, "
                f"found {len(matrix_bytes)}"
            )
        delta_embeddings = np.frombuffer(matrix_bytes, dtype="<f2").reshape(upsert_count, dim)
        embedding_indexes = {
            operation.get("embedding_index")
            for operation in operations
            if operation.get("op") == "upsert"
        }
        if embedding_indexes != set(range(upsert_count)):
            raise CatalogV2Error("delta embedding indexes must be contiguous and unique")

        current_records = {record.key: record for record in previous.records}
        current_embeddings = {
            record.key: previous.embeddings[index].copy()
            for index, record in enumerate(previous.records)
        }
        seen_operations: set[str] = set()
        for operation in operations:
            op = operation.get("op")
            if op == "delete":
                key = _required_string(operation, "key", "delta delete")
                if key in seen_operations:
                    raise CatalogV2Error(f"duplicate delta operation for key {key!r}")
                seen_operations.add(key)
                if key not in current_records:
                    raise CatalogV2Error(f"delta deletes missing key {key!r}")
                del current_records[key]
                del current_embeddings[key]
            elif op == "upsert":
                raw_record = operation.get("record")
                if not isinstance(raw_record, dict):
                    raise CatalogV2Error("delta upsert record must be an object")
                parsed = _parse_records(
                    [raw_record],
                    result_identifier=descriptor.result_identifier,
                    metadata_by_key={},
                )[0]
                if parsed.key in seen_operations:
                    raise CatalogV2Error(f"duplicate delta operation for key {parsed.key!r}")
                seen_operations.add(parsed.key)
                embedding_index = _required_non_negative_int(
                    operation, "embedding_index", "delta upsert"
                )
                if embedding_index >= len(delta_embeddings):
                    raise CatalogV2Error(
                        f"delta upsert {parsed.key!r} references missing embedding"
                    )
                prior_metadata = current_records.get(parsed.key)
                current_records[parsed.key] = CatalogV2Record(
                    key=parsed.key,
                    identifiers=parsed.identifiers,
                    face_index=parsed.face_index,
                    metadata=None if prior_metadata is None else prior_metadata.metadata,
                )
                current_embeddings[parsed.key] = delta_embeddings[embedding_index].copy()
            else:
                raise CatalogV2Error(f"unsupported recognition delta operation {op!r}")

        if include_metadata:
            metadata_path = _asset_path(path.parent, assets, "metadata_delta")
            _verify_asset(metadata_path, assets["metadata_delta"])
            metadata_operations = _read_jsonl_gzip(metadata_path)
            expected_metadata_operations = _required_non_negative_int(
                delta, "metadata_operations", "manifest delta"
            )
            if len(metadata_operations) != expected_metadata_operations:
                raise CatalogV2Error(
                    "metadata delta operation count mismatch: "
                    f"expected {expected_metadata_operations}, "
                    f"found {len(metadata_operations)}"
                )
            for operation in metadata_operations:
                key = _required_string(operation, "key", "metadata delta")
                if operation.get("op") != "delete" and key not in current_records:
                    raise CatalogV2Error(
                        f"metadata delta references missing recognition key {key!r}"
                    )
                if operation.get("op") == "delete":
                    metadata = None
                elif operation.get("op") == "upsert":
                    metadata = operation.get("metadata")
                    if not isinstance(metadata, dict):
                        raise CatalogV2Error(
                            f"metadata delta upsert {key!r} must contain an object"
                        )
                else:
                    raise CatalogV2Error(
                        f"unsupported metadata delta operation {operation.get('op')!r}"
                    )
                if key not in current_records:
                    continue
                record = current_records[key]
                current_records[key] = CatalogV2Record(
                    key=record.key,
                    identifiers=record.identifiers,
                    face_index=record.face_index,
                    metadata=metadata,
                )

        sorted_keys = sorted(current_records)
        if len(sorted_keys) != rows:
            raise CatalogV2Error(
                f"delta reconstructed {len(sorted_keys)} rows but manifest expects {rows}"
            )
        embeddings = (
            np.vstack([current_embeddings[key] for key in sorted_keys]).astype("<f2", copy=False)
            if sorted_keys
            else np.empty((0, dim), dtype="<f2")
        )
        return cls(
            embeddings=embeddings,
            records=tuple(current_records[key] for key in sorted_keys),
            catalog_key=catalog_key,
            version=version,
            embedding_model=embedding_model,
            descriptor=descriptor,
            metadata_loaded=include_metadata,
        )

    @property
    def embedder(self):
        """Construct the exact registered embedder required by this snapshot."""
        if self._embedder is None:
            match = _MODEL_IDENTITY.fullmatch(self.embedding_model)
            if match is None:
                raise CatalogV2Error(
                    f"unsupported Catalog v2 embedding model {self.embedding_model!r}"
                )
            model_id = match.group("model")
            from collector_vision.embedders.neural import NeuralEmbedder
            from collector_vision.model_artifacts import resolve_model_artifact
            from collector_vision.model_registry import get_model

            model = get_model(model_id)
            if model.sha256 != match.group("sha256"):
                raise CatalogV2Error(
                    f"installed model registry does not match {self.embedding_model!r}"
                )
            self._embedder = NeuralEmbedder(checkpoint=resolve_model_artifact(model))
        return self._embedder

    @property
    def card_ids(self) -> list[str]:
        """Primary card IDs, matching the Catalog v1 compatibility attribute."""
        identifier = self.descriptor.result_identifier
        return [record.identifiers[identifier] for record in self.records]

    @property
    def oracle_ids(self) -> list[str] | None:
        """Scryfall Oracle IDs when present, matching the Catalog v1 attribute."""
        values = [record.identifiers.get("scryfall_oracle", "") for record in self.records]
        return values if any(values) else None

    @property
    def source(self) -> str:
        """The catalog's upstream source."""
        return self.descriptor.source

    @property
    def algo_key(self) -> str:
        """Stable embedding algorithm identifier, such as ``"milo1"``."""
        return self.catalog_key.split("/", 1)[0]

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[float, str]]:
        """Return compatibility search results using the descriptor's result ID."""
        from collector_vision import retrieval

        _validate_top_k(top_k)
        raw = retrieval.cosine_search(embedding, self.embeddings, top_k=top_k)
        identifier = self.descriptor.result_identifier
        return [(score, self.records[index].identifiers[identifier]) for score, index in raw]

    def search_records(self, embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return scored records with peer identifiers and optional metadata."""
        from collector_vision import retrieval

        _validate_top_k(top_k)
        raw = retrieval.cosine_search(embedding, self.embeddings, top_k=top_k)
        return [self.record_for_index(index, score=score) for score, index in raw]

    def record_for_index(self, index: int, score: float | None = None) -> dict[str, Any]:
        record = self.records[index]
        result: dict[str, Any] = {
            "key": record.key,
            "identifiers": dict(record.identifiers),
            "face_index": record.face_index,
            "result_identifier": self.descriptor.result_identifier,
            "card_id": record.identifiers[self.descriptor.result_identifier],
        }
        if record.metadata is not None:
            result["metadata"] = dict(record.metadata)
        if score is not None:
            result["score"] = score
        return result

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"CatalogV2(catalog_key={self.catalog_key!r}, version={self.version!r}, "
            f"profile={self.descriptor.profile!r}, n={len(self)})"
        )


def _parse_descriptor(value: object) -> CatalogV2Descriptor:
    if not isinstance(value, dict):
        raise CatalogV2Error("manifest descriptor must be an object")
    recommended = value.get("recommended")
    if not isinstance(recommended, bool):
        raise CatalogV2Error("catalog descriptor recommended must be a boolean")
    return CatalogV2Descriptor(
        game=_required_string(value, "game", "catalog descriptor"),
        source=_required_string(value, "source", "catalog descriptor"),
        profile=_required_string(value, "profile", "catalog descriptor"),
        description=_required_string(value, "description", "catalog descriptor"),
        result_identifier=_required_string(value, "result_identifier", "catalog descriptor"),
        recommended=recommended,
    )


def _parse_records(
    values: list[dict[str, Any]],
    *,
    result_identifier: str,
    metadata_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[CatalogV2Record, ...]:
    records: list[CatalogV2Record] = []
    keys: set[str] = set()
    for value in values:
        key = _required_string(value, "key", "recognition record")
        if key in keys:
            raise CatalogV2Error(f"duplicate recognition key {key!r}")
        keys.add(key)
        identifiers = value.get("identifiers")
        if not isinstance(identifiers, dict) or not identifiers:
            raise CatalogV2Error(f"recognition record {key!r} identifiers must be an object")
        parsed_identifiers = {
            name: identifier
            for name, identifier in identifiers.items()
            if isinstance(name, str) and name and isinstance(identifier, str) and identifier
        }
        if len(parsed_identifiers) != len(identifiers):
            raise CatalogV2Error(f"recognition record {key!r} contains an invalid identifier")
        if result_identifier not in parsed_identifiers:
            raise CatalogV2Error(
                f"recognition record {key!r} lacks result identifier {result_identifier!r}"
            )
        face_index = value.get("face_index", 0)
        if not isinstance(face_index, int) or isinstance(face_index, bool) or face_index < 0:
            raise CatalogV2Error(f"recognition record {key!r} has invalid face_index")
        records.append(
            CatalogV2Record(
                key=key,
                identifiers=parsed_identifiers,
                face_index=face_index,
                metadata=metadata_by_key.get(key),
            )
        )
    return tuple(records)


def _parse_metadata(values: list[dict[str, Any]]) -> dict[str, Mapping[str, Any]]:
    metadata: dict[str, Mapping[str, Any]] = {}
    for value in values:
        key = _required_string(value, "key", "metadata record")
        fields = value.get("metadata")
        if not isinstance(fields, dict):
            raise CatalogV2Error(f"metadata record {key!r} metadata must be an object")
        if key in metadata:
            raise CatalogV2Error(f"duplicate metadata key {key!r}")
        metadata[key] = fields
    return metadata


def _asset_path(
    manifest_dir: Path,
    assets: Mapping[str, object],
    asset_name: str,
) -> Path:
    value = assets.get(asset_name)
    if not isinstance(value, dict):
        raise CatalogV2Error(f"manifest is missing {asset_name!r} asset")
    filename = _required_string(value, "filename", f"asset {asset_name!r}")
    if Path(filename).name != filename:
        raise CatalogV2Error(f"asset {asset_name!r} filename must be a basename")
    return manifest_dir / filename


def _verify_asset(path: Path, asset: object) -> None:
    if not isinstance(asset, dict):
        raise CatalogV2Error("asset descriptor must be an object")
    if not path.is_file():
        raise FileNotFoundError(f"Catalog v2 asset not found: {path}")
    expected_size = _required_non_negative_int(asset, "size", "asset")
    expected_sha256 = _required_string(asset, "sha256", "asset")
    if path.stat().st_size != expected_size:
        raise CatalogV2Error(f"Catalog v2 asset size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise CatalogV2Error(f"Catalog v2 asset checksum mismatch: {path.name}")


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CatalogV2Error(
                    f"invalid JSON in {path.name} at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise CatalogV2Error(
                    f"JSON record in {path.name} at line {line_number} must be an object"
                )
            records.append(value)
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Catalog v2 manifest not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CatalogV2Error(f"invalid Catalog v2 manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise CatalogV2Error("Catalog v2 manifest must be an object")
    return value


def _required_string(value: Mapping[str, object], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise CatalogV2Error(f"{context} {key} must be a non-empty string")
    return result


def _required_non_negative_int(
    value: Mapping[str, object],
    key: str,
    context: str,
) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise CatalogV2Error(f"{context} {key} must be a non-negative integer")
    return result


def _required_positive_int(
    value: Mapping[str, object],
    key: str,
    context: str,
) -> int:
    result = _required_non_negative_int(value, key, context)
    if result == 0:
        raise CatalogV2Error(f"{context} {key} must be positive")
    return result


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
