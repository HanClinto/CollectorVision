"""Catalog: pre-built card embedding index used for nearest-neighbour retrieval.

A Catalog bundles:
  - the card embedding matrix
  - card IDs (one per row — callers look up names/metadata from the source)
  - the embedder specification needed to embed query images consistently

Users select a catalog file; the library derives everything else from it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collector_vision.interfaces import Embedder
    from collector_vision.games import Game, Embedding


# NPZ keys
_KEY_EMBEDDINGS  = "embeddings"    # (N, D) float32  or  (N, B) uint8
_KEY_CARD_IDS    = "card_ids"      # (N,) str — primary key (e.g. Scryfall UUID)
_KEY_SOURCE      = "source"        # scalar str — "scryfall", "tcgplayer", …
_KEY_MODE        = "mode"          # scalar str — "embedding" | "hash"
_KEY_EMBEDDER    = "embedder_spec" # scalar str — JSON embedder specification


class Catalog:
    """Loaded card catalog ready for retrieval.

    Obtain via :meth:`Catalog.load` or :meth:`Catalog.for_game` — do not
    construct directly.

    The embedder required to query this catalog is available via
    :attr:`Catalog.embedder`.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        card_ids: list[str],
        source: str,
        mode: str,
        embedder_spec: dict,
    ) -> None:
        self.embeddings = embeddings
        self.card_ids = card_ids
        self.source = source
        self.mode = mode        # "embedding" | "hash"
        self.embedder_spec = embedder_spec

        self._embedder: "Embedder | None" = None

    @classmethod
    def load(cls, source: "str | Path | HFD") -> "Catalog":
        """Load a catalog from a local NPZ file or a HuggingFace reference.

        Parameters
        ----------
        source:
            - Local path: ``"./milo1-scryfall-mtg-2026-04.npz"`` or a ``Path``
            - HuggingFace URI: ``"hf://HanClinto/milo/scryfall-mtg"``
              (format: ``hf://{user}/{repo}/{catalog-key}``)
            - :class:`~collector_vision.hfd.HFD` instance

        Examples
        --------
        ::

            catalog = Catalog.load("hf://HanClinto/milo/scryfall-mtg")
            catalog = Catalog.load("./milo1-scryfall-mtg-2026-04.npz")
        """
        from collector_vision.hfd import HFD as _HFD

        if isinstance(source, _HFD):
            path = source.resolve()
        elif isinstance(source, str) and source.startswith("hf://"):
            rest = source[len("hf://"):]
            parts = rest.split("/", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid hf:// URI {source!r}. "
                    "Expected format: hf://user/repo/catalog-key"
                )
            repo = f"{parts[0]}/{parts[1]}"
            path = _HFD(repo, parts[2]).resolve()
        else:
            path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        data = np.load(path, allow_pickle=False)

        embedder_spec: dict = {}
        if _KEY_EMBEDDER in data.files:
            embedder_spec = json.loads(str(data[_KEY_EMBEDDER]))

        return cls(
            embeddings=data[_KEY_EMBEDDINGS],
            card_ids=data[_KEY_CARD_IDS].tolist(),
            source=str(data[_KEY_SOURCE]) if _KEY_SOURCE in data.files else "unknown",
            mode=str(data[_KEY_MODE]) if _KEY_MODE in data.files else "embedding",
            embedder_spec=embedder_spec,
        )

    @property
    def embedder(self) -> "Embedder":
        """The embedder that must be used to query this catalog.

        Constructed lazily from :attr:`embedder_spec` on first access.
        """
        if self._embedder is None:
            self._embedder = _embedder_from_spec(self.embedder_spec)
        return self._embedder

    @property
    def algo_key(self) -> str | None:
        """Stable algorithm identifier stored in the catalog, e.g. ``'milo1'``."""
        return self.embedder_spec.get("algo_key")

    @classmethod
    def for_game(
        cls,
        game: "Game",
        embedding: "Embedding | None" = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> "Catalog":
        """Download (or load from cache) the latest catalog for a single game.

        Parameters
        ----------
        game:
            A :class:`~collector_vision.games.Game` enum value.
        embedding:
            Which embedding algorithm to use.  Defaults to
            :attr:`~collector_vision.games.Embedding.MILO`.
        cache_dir:
            Override the default cache root (``~/.cache/collectorvision/``).
        offline:
            If ``True``, never make network calls.  Raises if not cached.

        Example::

            from collector_vision.games import Game
            catalog = Catalog.for_game(Game.MTG)
        """
        from collector_vision.games import Embedding as _Embedding, GAME_PRIMARY_SOURCE
        from collector_vision.hfd import HFD

        embedding = embedding or _Embedding.MILO
        source = GAME_PRIMARY_SOURCE.get(game, "unknown")
        repo = f"HanClinto/{embedding.family}"
        catalog_key = f"{source}-{game.value}"
        return cls.load(HFD(repo, catalog_key, cache_dir=cache_dir, offline=offline).resolve())

    @classmethod
    def for_games(
        cls,
        *games: "Game",
        embedding: "Embedding | None" = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> "Catalog":
        """Download and merge catalogs for multiple games.

        All games must use the same embedding so query vectors are compatible.

        Example::

            from collector_vision.games import Game
            catalog = Catalog.for_games(Game.MTG, Game.POKEMON)
        """
        loaded = [
            cls.for_game(g, embedding=embedding, cache_dir=cache_dir, offline=offline)
            for g in games
        ]
        return loaded[0] if len(loaded) == 1 else cls._merge(loaded)

    @classmethod
    def _merge(cls, catalogs: "list[Catalog]") -> "Catalog":
        """Concatenate multiple compatible catalogs into one."""
        ref_spec = catalogs[0].embedder_spec
        for c in catalogs[1:]:
            if c.embedder_spec != ref_spec:
                raise ValueError(
                    f"Cannot merge catalogs with different embedder specs:\n"
                    f"  {catalogs[0].source}: {ref_spec}\n"
                    f"  {c.source}: {c.embedder_spec}"
                )
        return cls(
            embeddings=np.concatenate([c.embeddings for c in catalogs], axis=0),
            card_ids=sum((c.card_ids for c in catalogs), []),
            source="+".join(c.source for c in catalogs),
            mode=catalogs[0].mode,
            embedder_spec=ref_spec,
        )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[float, str]]:
        """Find the closest cards to an embedding vector.

        Parameters
        ----------
        embedding:
            Query vector — ``(D,)`` float32 for neural catalogs,
            ``(B,)`` uint8 for hash catalogs.  Must match the catalog's
            own embedder output (use :attr:`embedder` to produce it).
        top_k:
            Number of results to return.

        Returns
        -------
        List of ``(score, card_id)`` tuples sorted by descending score.
        Score is cosine similarity for neural catalogs, normalised Hamming
        similarity for hash catalogs (both in the range [0, 1]).

        Example::

            catalog = Catalog.load("hf://HanClinto/milo/scryfall-mtg")
            crop = detection.dewarp(image)
            emb = catalog.embedder.embed(crop)
            hits = catalog.search(emb, top_k=3)
            score, card_id = hits[0]
        """
        from collector_vision import retrieval

        if self.mode == "hash":
            raw = retrieval.hamming_search(embedding, self.embeddings, top_k=top_k)
        else:
            raw = retrieval.cosine_search(embedding, self.embeddings, top_k=top_k)
        return [(score, self.card_ids[idx]) for score, idx in raw]

    def __len__(self) -> int:
        return len(self.card_ids)

    def __repr__(self) -> str:
        return (
            f"Catalog(source={self.source!r}, mode={self.mode!r}, "
            f"n={len(self)}, algo={self.algo_key!r})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_primary_key(source: str) -> str:
    """Return the canonical ID field name for a catalog source."""
    return {
        "scryfall":   "scryfall_id",
        "tcgplayer":  "tcgplayer_id",
        "pokemontcg": "pokemontcg_id",
    }.get(source, "card_id")


def _embedder_from_spec(spec: dict) -> "Embedder":
    """Reconstruct an Embedder from the spec dict stored in the catalog."""
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
        raise ValueError(f"Unknown hash algo_key in catalog: {algo!r}")

    if kind == "neural":
        from collector_vision.embedders.neural import NeuralEmbedder
        return NeuralEmbedder()

    raise ValueError(
        f"Unknown embedder kind {kind!r} in catalog spec.\n"
        f"Full spec: {spec}\n"
        "This catalog may have been built with a newer version of CollectorVision."
    )
