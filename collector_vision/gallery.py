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
    from collector_vision.games import Game


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

    @classmethod
    def for_game(
        cls,
        game: "str | Game",
        variant: str | None = None,
        cache_dir: "Path | None" = None,
        offline: bool = False,
    ) -> "Gallery":
        """Load the latest published gallery for a single game.

        Parameters
        ----------
        game:
            Game identifier, e.g. ``"magic"``, ``"pokemon"``, or a
            ``Game`` enum value.
        variant:
            Model variant override, e.g. ``"phash16"``.  Defaults to the
            manifest's ``default_variant`` (currently ``"milo1"``).
        cache_dir:
            Local directory for downloaded galleries.  Defaults to
            ``~/.cache/collectorvision/``.
        offline:
            If True, never make network calls — use only locally cached
            files.  Raises if the gallery is not cached.
        """
        from collector_vision.games import parse_game
        from collector_vision.manifest import Manifest, _default_cache_dir

        game = parse_game(str(game)) if not isinstance(game, Game) else game
        cache_dir = cache_dir or _default_cache_dir()
        manifest = Manifest.bundled() if offline else Manifest.fetch(cache_dir)

        filename = manifest.resolve(game, variant)
        local_path = cache_dir / filename

        if not local_path.exists():
            if offline:
                raise FileNotFoundError(
                    f"Gallery not cached locally: {local_path}\n"
                    "Run without offline=True to download it."
                )
            _download(manifest.url_for(game, variant), local_path)

        return cls.load(local_path)

    @classmethod
    def for_games(
        cls,
        *games: "str | Game",
        variant: str | None = None,
        cache_dir: "Path | None" = None,
        offline: bool = False,
    ) -> "Gallery":
        """Load and merge galleries for multiple games.

        All games must use the same variant (and therefore the same embedder)
        so that query embeddings are compatible with the merged gallery.
        Raises ``ValueError`` if the loaded galleries have incompatible
        embedder specs.

        Example::

            gallery = Gallery.for_games("magic", "pokemon", "yugioh")
        """
        loaded = [
            cls.for_game(g, variant=variant, cache_dir=cache_dir, offline=offline)
            for g in games
        ]
        if len(loaded) == 1:
            return loaded[0]
        return cls._merge(loaded)

    @classmethod
    def _merge(cls, galleries: "list[Gallery]") -> "Gallery":
        """Concatenate multiple compatible galleries into one."""
        # Validate embedder compatibility
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
            ids=sum((g.ids for g in galleries), []),
            card_names=sum((g.card_names for g in galleries), []),
            set_codes=sum((g.set_codes for g in galleries), []),
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


def _download(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest* with a progress indicator."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        print(f"Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"  saved to {dest}")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


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
