"""Gallery: pre-built card embedding index used for nearest-neighbour retrieval.

A Gallery is the primary user-facing artifact of CollectorVision.  It bundles:
  - the card embedding matrix
  - card IDs (one per row — callers look up names/metadata from the source catalog)
  - the embedder specification needed to embed query images consistently

Users select a gallery file; the library derives everything else from it.
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


class Gallery:
    """Loaded card gallery ready for retrieval.

    Obtain via :meth:`Gallery.load` or :meth:`Gallery.for_game` — do not
    construct directly.

    The embedder required to query this gallery is available via
    :attr:`Gallery.embedder`.  :meth:`~collector_vision.Identifier.identify`
    uses this automatically; most users never need to access it directly.
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
    def load(cls, path: str | Path) -> "Gallery":
        """Load a gallery from a CollectorVision NPZ file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Gallery file not found: {path}")

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
        """The embedder that must be used to query this gallery.

        Constructed lazily from :attr:`embedder_spec` on first access.
        """
        if self._embedder is None:
            self._embedder = _embedder_from_spec(self.embedder_spec)
        return self._embedder

    @property
    def algo_key(self) -> str | None:
        """Stable algorithm identifier stored in the gallery, e.g. ``'milo1'``."""
        return self.embedder_spec.get("algo_key")

    @classmethod
    def for_game(
        cls,
        game: "Game",
        embedding: "Embedding | None" = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> "Gallery":
        """Download (or load from cache) the latest gallery for a single game.

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
            gallery = Gallery.for_game(Game.MTG)
        """
        from collector_vision.games import Embedding as _Embedding, GAME_PRIMARY_SOURCE
        from collector_vision.hfd import HFD

        embedding = embedding or _Embedding.MILO
        source = GAME_PRIMARY_SOURCE.get(game, "unknown")
        repo = f"HanClinto/{embedding.family}"
        gallery_key = f"{source}-{game.value}"
        return cls.load(HFD(repo, gallery_key, cache_dir=cache_dir, offline=offline).resolve())

    @classmethod
    def for_games(
        cls,
        *games: "Game",
        embedding: "Embedding | None" = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> "Gallery":
        """Download and merge galleries for multiple games.

        All games must use the same embedding so query vectors are compatible.

        Example::

            from collector_vision.games import Game
            gallery = Gallery.for_games(Game.MTG, Game.POKEMON)
        """
        loaded = [
            cls.for_game(g, embedding=embedding, cache_dir=cache_dir, offline=offline)
            for g in games
        ]
        return loaded[0] if len(loaded) == 1 else cls._merge(loaded)

    @classmethod
    def _merge(cls, galleries: "list[Gallery]") -> "Gallery":
        """Concatenate multiple compatible galleries into one."""
        ref_spec = galleries[0].embedder_spec
        for g in galleries[1:]:
            if g.embedder_spec != ref_spec:
                raise ValueError(
                    f"Cannot merge galleries with different embedder specs:\n"
                    f"  {galleries[0].source}: {ref_spec}\n"
                    f"  {g.source}: {g.embedder_spec}"
                )
        return cls(
            embeddings=np.concatenate([g.embeddings for g in galleries], axis=0),
            card_ids=sum((g.card_ids for g in galleries), []),
            source="+".join(g.source for g in galleries),
            mode=galleries[0].mode,
            embedder_spec=ref_spec,
        )

    def __len__(self) -> int:
        return len(self.card_ids)

    def __repr__(self) -> str:
        return (
            f"Gallery(source={self.source!r}, mode={self.mode!r}, "
            f"n={len(self)}, algo={self.algo_key!r})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_primary_key(source: str) -> str:
    """Return the canonical ID field name for a gallery source."""
    return {
        "scryfall":   "scryfall_id",
        "tcgplayer":  "tcgplayer_id",
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
        return NeuralEmbedder()

    raise ValueError(
        f"Unknown embedder kind {kind!r} in gallery spec.\n"
        f"Full spec: {spec}\n"
        "This gallery may have been built with a newer version of CollectorVision."
    )
