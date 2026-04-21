"""Gallery: pre-built card index used for nearest-neighbour retrieval.

A Gallery is the primary user-facing artifact of CollectorVision.  It bundles:
  - the card embedding/hash matrix
  - card identity metadata (ids, names, set codes)
  - the embedder specification needed to embed query images consistently

Users select a gallery file; the library derives everything else from it.
Embedder parameters (algorithm, hash size, sigma, model checkpoint, …) are
properties of the gallery, not user-controlled knobs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collector_vision.interfaces import Embedder


# Gallery NPZ keys written by the gallery builder
_KEY_EMBEDDINGS  = "embeddings"   # (N, D) float32  or  (N, B) uint8
_KEY_CARD_IDS    = "card_ids"     # (N,) str — primary key (e.g. Scryfall UUID)
_KEY_IDS_JSON    = "ids_json"     # (N,) str — JSON-encoded per-card ids dicts
_KEY_CARD_NAMES  = "card_names"   # (N,) str
_KEY_SET_CODES   = "set_codes"    # (N,) str
_KEY_SOURCE      = "source"       # scalar str — "scryfall", "tcgplayer", …
_KEY_MODE        = "mode"         # scalar str — "embedding" | "hash"
_KEY_EMBEDDER    = "embedder_spec"# scalar str — JSON embedder specification


class Gallery:
    """Loaded card gallery ready for retrieval.

    Obtain via Gallery.load() — do not construct directly.

    The embedder required to query this gallery is available via
    Gallery.embedder.  identify() uses this automatically; most users
    never need to access it directly.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        card_ids: list[str],
        ids: list[dict],
        card_names: list[str],
        set_codes: list[str],
        source: str,
        mode: str,
        embedder_spec: dict,
    ) -> None:
        self.embeddings = embeddings
        self.card_ids = card_ids
        self.ids = ids                  # parallel list of per-card id dicts
        self.card_names = card_names
        self.set_codes = set_codes
        self.source = source
        self.mode = mode                # "embedding" | "hash"
        self.embedder_spec = embedder_spec  # raw spec dict from the NPZ

        self._embedder: "Embedder | None" = None  # lazy, created on first access

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        """Load a gallery from a CollectorVision NPZ file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Gallery file not found: {path}")

        data = np.load(path, allow_pickle=False)

        n = len(data[_KEY_CARD_IDS])

        if _KEY_IDS_JSON in data.files:
            ids = [json.loads(s) for s in data[_KEY_IDS_JSON].tolist()]
        else:
            # Legacy: reconstruct a minimal ids dict from card_ids alone
            raw = data[_KEY_CARD_IDS].tolist()
            src = str(data[_KEY_SOURCE]) if _KEY_SOURCE in data.files else "unknown"
            pk = _source_primary_key(src)
            ids = [{pk: cid} for cid in raw]

        embedder_spec: dict = {}
        if _KEY_EMBEDDER in data.files:
            embedder_spec = json.loads(str(data[_KEY_EMBEDDER]))

        return cls(
            embeddings=data[_KEY_EMBEDDINGS],
            card_ids=data[_KEY_CARD_IDS].tolist(),
            ids=ids,
            card_names=data[_KEY_CARD_NAMES].tolist() if _KEY_CARD_NAMES in data.files else [""] * n,
            set_codes=data[_KEY_SET_CODES].tolist() if _KEY_SET_CODES in data.files else [""] * n,
            source=str(data[_KEY_SOURCE]) if _KEY_SOURCE in data.files else "unknown",
            mode=str(data[_KEY_MODE]) if _KEY_MODE in data.files else "embedding",
            embedder_spec=embedder_spec,
        )

    @property
    def embedder(self) -> "Embedder":
        """The embedder that must be used to query this gallery.

        Constructed lazily from embedder_spec on first access.
        """
        if self._embedder is None:
            self._embedder = _embedder_from_spec(self.embedder_spec)
        return self._embedder

    @property
    def algo_key(self) -> str | None:
        """Stable algorithm identifier, e.g. 'phash_16' or 'neural_e15'."""
        return self.embedder_spec.get("algo_key")

    def __len__(self) -> int:
        return len(self.card_ids)

    def __repr__(self) -> str:
        return (
            f"Gallery(source={self.source!r}, mode={self.mode!r}, "
            f"n={len(self)}, algo={self.algo_key!r})"
        )


# ---------------------------------------------------------------------------
# Embedder spec helpers
# ---------------------------------------------------------------------------

def _source_primary_key(source: str) -> str:
    """Best-guess primary key field name for a given source."""
    return {
        "scryfall": "scryfall_id",
        "tcgplayer": "tcgplayer_id",
        "pokemontcg": "pokemontcg_id",
    }.get(source, "card_id")


def _embedder_from_spec(spec: dict) -> "Embedder":
    """Reconstruct an Embedder from the spec dict stored in the gallery."""
    kind = spec.get("kind")

    if kind == "hash":
        from collector_vision.embedders.hash import HashEmbedder
        algo = spec.get("algo_key", "")
        if algo.startswith("phash_"):
            return HashEmbedder.phash(hash_size=int(spec["hash_size"]))
        if algo.startswith("dhash_"):
            return HashEmbedder.dhash(hash_size=int(spec["hash_size"]))
        if algo.startswith("whash_"):
            return HashEmbedder.whash(hash_size=int(spec["hash_size"]))
        if algo.startswith("marr_hildreth_"):
            return HashEmbedder.marr_hildreth(
                hash_size=int(spec["hash_size"]),
                sigma=float(spec["sigma"]),
            )
        raise ValueError(f"Unknown hash algo_key in gallery: {algo!r}")

    if kind == "neural":
        from collector_vision.embedders.neural import NeuralEmbedder
        return NeuralEmbedder(
            image_size=int(spec.get("image_size", 448)),
        )

    raise ValueError(
        f"Unknown embedder kind {kind!r} in gallery spec.\n"
        f"Full spec: {spec}\n"
        "This gallery may have been built with a newer version of CollectorVision."
    )


def make_embedder_spec_hash(algo_key: str, hash_size: int, **kwargs) -> dict:
    """Build the embedder_spec dict to store in a hash gallery NPZ."""
    return {"kind": "hash", "algo_key": algo_key, "hash_size": hash_size, **kwargs}


def make_embedder_spec_neural(image_size: int, checkpoint_sha256: str | None = None) -> dict:
    """Build the embedder_spec dict to store in a neural gallery NPZ."""
    spec = {"kind": "neural", "image_size": image_size}
    if checkpoint_sha256:
        spec["checkpoint_sha256"] = checkpoint_sha256
    return spec
