"""Searchable CollectorVision Catalog v2 snapshots."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collector_vision.games import Game

_MODEL_IDENTITY = re.compile(
    r"^collectorvision@[0-9a-f]{40}:(?P<model>[a-z0-9.-]+)@sha256:(?P<sha256>[0-9a-f]{64})$"
)


class CatalogV2Error(ValueError):
    """Raised when Catalog v2 data violates the active client contract."""


@dataclass(frozen=True)
class CatalogV2Embedding:
    """Immutable embedding contract shared by one catalog family."""

    model: str
    dimensions: int
    dtype: str
    byte_order: str
    layout: str


@dataclass(frozen=True)
class CatalogV2Descriptor:
    """Discovery and result semantics for one catalog."""

    game: str
    source: str
    profile: str
    description: str
    result_identifier: str
    recommended: bool


@dataclass(frozen=True)
class CatalogV2Record:
    """One recognition row and its optional metadata."""

    id: str
    name: str
    identifiers: Mapping[str, str]
    face_index: int = 0
    finishes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def key(self, source: str) -> str:
        """Return the derived catalog-local identity used for delta application."""
        return catalog_v2_row_key(source, self.id, self.face_index)


class CatalogV2:
    """A loaded, searchable Catalog v2 snapshot.

    Constructing by game discovers the recommended matching catalog in the
    moving feed. Catalog v1 remains independent.
    """

    def __init__(
        self,
        game: str | Game,
        *,
        source: str | None = None,
        profile: str | None = None,
        family: str = "milo1",
        include_metadata: bool = False,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        version: int | None = None,
        feed_url: str | None = None,
    ) -> None:
        from collector_vision.catalog_v2_downloader import (
            DEFAULT_FEED_URL,
            CatalogV2Downloader,
        )

        if offline and feed_url is not None:
            raise ValueError("feed_url cannot be used when offline=True")
        downloader = (
            CatalogV2Downloader.open(
                game,
                source=source,
                profile=profile,
                family=family,
                include_metadata=include_metadata,
                cache_dir=cache_dir,
                version=version,
            )
            if offline
            else CatalogV2Downloader.install(
                game,
                source=source,
                profile=profile,
                family=family,
                include_metadata=include_metadata,
                cache_dir=cache_dir,
                version=version,
                feed_url=feed_url or DEFAULT_FEED_URL,
            )
        )
        self._initialize_from(downloader.load())

    def _initialize_from(self, loaded: CatalogV2) -> None:
        self.embeddings = loaded.embeddings
        self.records = loaded.records
        self.catalog_key = loaded.catalog_key
        self.family = loaded.family
        self.version = loaded.version
        self.embedding = loaded.embedding
        self.descriptor = loaded.descriptor
        self.metadata_loaded = loaded.metadata_loaded
        self._embedder = None

    @classmethod
    def _from_data(
        cls,
        *,
        embeddings: np.ndarray,
        records: Sequence[CatalogV2Record],
        catalog_key: str,
        family: str,
        version: int,
        embedding: CatalogV2Embedding,
        descriptor: CatalogV2Descriptor,
        metadata_loaded: bool,
    ) -> CatalogV2:
        if embeddings.dtype != np.dtype("<f2"):
            raise CatalogV2Error("Catalog v2 embeddings must use little-endian float16")
        if embeddings.shape != (len(records), embedding.dimensions):
            raise CatalogV2Error(
                "Catalog v2 matrix shape does not match its records and embedding contract"
            )
        catalog = cls.__new__(cls)
        catalog.embeddings = embeddings
        catalog.records = tuple(records)
        catalog.catalog_key = catalog_key
        catalog.family = family
        catalog.version = version
        catalog.embedding = embedding
        catalog.descriptor = descriptor
        catalog.metadata_loaded = metadata_loaded
        catalog._embedder = None
        return catalog

    @property
    def embedding_model(self) -> str:
        """Exact model identity required by this catalog family."""
        return self.embedding.model

    @property
    def embedder(self):
        """Construct the exact registered embedder required by this snapshot."""
        if self._embedder is None:
            match = _MODEL_IDENTITY.fullmatch(self.embedding.model)
            if match is None:
                raise CatalogV2Error(
                    f"unsupported Catalog v2 embedding model {self.embedding.model!r}"
                )
            from collector_vision.embedders.neural import NeuralEmbedder
            from collector_vision.model_artifacts import resolve_model_artifact
            from collector_vision.model_registry import get_model

            model = get_model(match.group("model"))
            if model.sha256 != match.group("sha256"):
                raise CatalogV2Error(
                    f"installed model registry does not match {self.embedding.model!r}"
                )
            self._embedder = NeuralEmbedder(checkpoint=resolve_model_artifact(model))
        return self._embedder

    @property
    def card_ids(self) -> list[str]:
        """Primary result IDs, matching the Catalog v1 compatibility attribute."""
        return [record.id for record in self.records]

    @property
    def oracle_ids(self) -> list[str] | None:
        """Scryfall Oracle IDs when present, matching the Catalog v1 attribute."""
        values = [record.identifiers.get("scryfall_oracle", "") for record in self.records]
        return values if any(values) else None

    @property
    def source(self) -> str:
        return self.descriptor.source

    @property
    def algo_key(self) -> str:
        return self.family

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[float, str]]:
        """Return compatibility results using the catalog's primary ID."""
        from collector_vision import retrieval

        _validate_top_k(top_k)
        raw = retrieval.cosine_search(embedding, self.embeddings, top_k=top_k)
        return [(score, self.records[index].id) for score, index in raw]

    def search_records(self, embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return scored records with all identifiers and optional metadata."""
        from collector_vision import retrieval

        _validate_top_k(top_k)
        raw = retrieval.cosine_search(embedding, self.embeddings, top_k=top_k)
        return [self.record_for_index(index, score=score) for score, index in raw]

    def record_for_index(self, index: int, score: float | None = None) -> dict[str, Any]:
        record = self.records[index]
        identifiers = {
            self.descriptor.result_identifier: record.id,
            **record.identifiers,
        }
        result: dict[str, Any] = {
            "key": record.key(self.descriptor.source),
            "id": record.id,
            "name": record.name,
            "identifiers": identifiers,
            "face_index": record.face_index,
            "finishes": list(record.finishes),
            "result_identifier": self.descriptor.result_identifier,
            "card_id": record.id,
        }
        if self.metadata_loaded:
            result["metadata"] = None if record.metadata is None else dict(record.metadata)
        if score is not None:
            result["score"] = score
        return result

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"CatalogV2(catalog_key={self.catalog_key!r}, version={self.version}, "
            f"profile={self.descriptor.profile!r}, n={len(self)})"
        )


def catalog_v2_row_key(source: str, primary_id: str, face_index: int = 0) -> str:
    """Derive the stable identity omitted from compact public records."""
    for name, value in (("source", source), ("id", primary_id)):
        if not isinstance(value, str) or not value or ":" in value:
            raise CatalogV2Error(f"Catalog v2 {name} must be a non-empty string without ':'")
    if not isinstance(face_index, int) or isinstance(face_index, bool) or face_index < 0:
        raise CatalogV2Error("Catalog v2 face_index must be a non-negative integer")
    base = f"{source}:{primary_id}"
    return base if face_index == 0 else f"{base}:face:{face_index}"


def _parse_embedding(value: object) -> CatalogV2Embedding:
    if not isinstance(value, dict):
        raise CatalogV2Error("catalog family embedding must be an object")
    embedding = CatalogV2Embedding(
        model=_required_string(value, "model", "family embedding"),
        dimensions=_required_positive_int(value, "dimensions", "family embedding"),
        dtype=_required_string(value, "dtype", "family embedding"),
        byte_order=_required_string(value, "byte_order", "family embedding"),
        layout=_required_string(value, "layout", "family embedding"),
    )
    if (
        embedding.dtype != "float16"
        or embedding.byte_order != "little"
        or embedding.layout != "row-major"
    ):
        raise CatalogV2Error("Catalog v2 requires little-endian, row-major float16 embeddings")
    if _MODEL_IDENTITY.fullmatch(embedding.model) is None:
        raise CatalogV2Error("catalog family contains an unsupported model identity")
    return embedding


def _parse_descriptor(value: object) -> CatalogV2Descriptor:
    if not isinstance(value, dict):
        raise CatalogV2Error("catalog descriptor must be an object")
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


def _parse_record(
    value: object,
    *,
    descriptor: CatalogV2Descriptor,
) -> CatalogV2Record:
    """Parse the core recognition fields of a record, without metadata.

    Used both for base rows (after their required ``metadata`` field is
    stripped off) and for the ``record`` payload embedded in update upsert
    operations, which never carries a ``metadata`` key of its own.
    """
    if not isinstance(value, dict):
        raise CatalogV2Error("recognition record must be an object")
    allowed = {"id", "name", "identifiers", "face_index", "finishes"}
    if set(value) - allowed or not {"id", "name", "identifiers"}.issubset(value):
        raise CatalogV2Error("recognition record has invalid fields")
    primary_id = _required_string(value, "id", "recognition record")
    if ":" in primary_id:
        raise CatalogV2Error("recognition record id must not contain ':'")
    identifiers = value.get("identifiers")
    if not isinstance(identifiers, dict):
        raise CatalogV2Error("recognition record identifiers must be an object")
    parsed_identifiers: dict[str, str] = {}
    for name, identifier in identifiers.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(identifier, str)
            or not identifier
        ):
            raise CatalogV2Error("recognition record contains an invalid peer identifier")
        parsed_identifiers[name] = identifier
    if descriptor.result_identifier in parsed_identifiers:
        raise CatalogV2Error("recognition record repeats its compact primary identifier")
    face_index = value.get("face_index", 0)
    if not isinstance(face_index, int) or isinstance(face_index, bool) or face_index < 0:
        raise CatalogV2Error("recognition record face_index must be a non-negative integer")
    if face_index == 0 and "face_index" in value:
        raise CatalogV2Error("recognition record must omit face_index for face 0")
    raw_finishes = value.get("finishes", [])
    if not isinstance(raw_finishes, list) or any(
        not isinstance(finish, str) or not finish for finish in raw_finishes
    ):
        raise CatalogV2Error("recognition record finishes must be a string list")
    finishes = tuple(raw_finishes)
    if finishes != tuple(sorted(set(finishes))):
        raise CatalogV2Error("recognition record finishes must be sorted and unique")
    return CatalogV2Record(
        id=primary_id,
        name=_required_string(value, "name", "recognition record"),
        identifiers=parsed_identifiers,
        face_index=face_index,
        finishes=finishes,
        metadata=None,
    )


def _parse_base_row(value: object, *, descriptor: CatalogV2Descriptor) -> CatalogV2Record:
    """Parse one combined base/cache record row, with its required metadata field."""
    if not isinstance(value, dict):
        raise CatalogV2Error("catalog record must be an object")
    if "metadata" not in value:
        raise CatalogV2Error("catalog record metadata field is required")
    metadata = value["metadata"]
    if metadata is not None and not isinstance(metadata, dict):
        raise CatalogV2Error("catalog record metadata must be an object or null")
    core = {key: item for key, item in value.items() if key != "metadata"}
    return _with_metadata(_parse_record(core, descriptor=descriptor), metadata)


def _core_equal(left: CatalogV2Record, right: CatalogV2Record) -> bool:
    """Compare two records ignoring metadata, used to validate no-op recognition upserts."""
    return (
        left.id == right.id
        and left.name == right.name
        and left.identifiers == right.identifiers
        and left.face_index == right.face_index
        and left.finishes == right.finishes
    )


def _with_metadata(
    record: CatalogV2Record,
    metadata: Mapping[str, Any] | None,
) -> CatalogV2Record:
    return CatalogV2Record(
        id=record.id,
        name=record.name,
        identifiers=record.identifiers,
        face_index=record.face_index,
        finishes=record.finishes,
        metadata=metadata,
    )


def _required_string(value: Mapping[str, object], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise CatalogV2Error(f"{context} {key} must be a non-empty string")
    return result


def _required_positive_int(value: Mapping[str, object], key: str, context: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise CatalogV2Error(f"{context} {key} must be a positive integer")
    return result


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
