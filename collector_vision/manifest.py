"""Gallery manifest: resolves game names to published gallery filenames.

The canonical manifest lives on HuggingFace Datasets at
``CollectorVision/galleries`` and is fetched + cached locally on first use.
A bundled fallback manifest covers the galleries shipped with the current
package version so the library works offline without a network call.

Gallery filename convention: {game}-{source}-{algo}-{YYYY-MM}.npz
  game   — canonical game id, e.g. "magic", "pokemon"
  source — data provider, e.g. "scryfall", "tcgplayer"
  algo   — model/hash variant, e.g. "milo1", "phash16"
  date   — YYYY-MM of the bulk data snapshot

Examples:
  magic-scryfall-milo1-2026-04.npz
  magic-scryfall-phash16-2026-04.npz
  pokemon-tcgplayer-milo1-2026-04.npz

Manifest format (JSON):
{
  "version": "2026-04",
  "default_variant": "milo1",
  "games": {
    "magic": {
      "milo1":   "magic-scryfall-milo1-2026-04.npz",
      "phash16": "magic-scryfall-phash16-2026-04.npz"
    },
    "pokemon": {
      "milo1":   "pokemon-tcgplayer-milo1-2026-04.npz",
      "phash16": "pokemon-tcgplayer-phash16-2026-04.npz"
    }
  }
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from collector_vision.games import Game, parse_game


# Bundled fallback — updated with each package release.
# Maps game → variant → filename (no download URL; the fetcher adds the base).
_BUNDLED_MANIFEST: dict = {
    "version": "0.1.0.dev0",
    "default_variant": "milo1",
    "games": {
        # Entries added here as galleries are published.
        # Format: game → { variant → filename }
        # Filename convention: {game}-{source}-{algo}-{YYYY-MM}.npz
    },
}

# HuggingFace Datasets base URL for gallery files.
HF_GALLERY_BASE_URL = (
    "https://huggingface.co/datasets/CollectorVision/galleries/resolve/main/"
)


class Manifest:
    """Resolved gallery manifest."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self._default_variant: str = data.get("default_variant", "milo1")
        self._games: dict[str, dict[str, str]] = data.get("games", {})

    @classmethod
    def bundled(cls) -> "Manifest":
        """Return the manifest bundled with the installed package."""
        return cls(_BUNDLED_MANIFEST)

    @classmethod
    def from_file(cls, path: str | Path) -> "Manifest":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def fetch(cls, cache_dir: Path | None = None) -> "Manifest":
        """Fetch the latest manifest from HuggingFace, with local cache.

        Falls back to the bundled manifest if the network is unavailable.
        """
        import urllib.request

        url = HF_GALLERY_BASE_URL + "manifest.json"
        cache_path = (cache_dir or _default_cache_dir()) / "manifest.json"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return cls(data)
        except Exception:
            if cache_path.exists():
                return cls.from_file(cache_path)
            return cls.bundled()

    def supported_games(self) -> list[Game]:
        """Games with at least one published gallery."""
        games = []
        for key in self._games:
            try:
                games.append(parse_game(key))
            except ValueError:
                pass
        return games

    def variants_for(self, game: Game) -> list[str]:
        """Available variant names for a game, e.g. ['milo1', 'phash16']."""
        return list(self._games.get(game.value, {}).keys())

    def resolve(self, game: Game, variant: str | None = None) -> str:
        """Return the gallery filename for a game + variant.

        Raises KeyError if the game or variant is not in the manifest.
        """
        variant = variant or self._default_variant
        game_entry = self._games.get(game.value)
        if not game_entry:
            supported = [g.value for g in self.supported_games()]
            raise KeyError(
                f"No gallery published for {game!r}. "
                f"Supported games: {supported or '(none yet)'}"
            )
        filename = game_entry.get(variant)
        if not filename:
            raise KeyError(
                f"No variant {variant!r} for {game!r}. "
                f"Available: {list(game_entry)}"
            )
        return filename

    def url_for(self, game: Game, variant: str | None = None) -> str:
        """Return the full HF download URL for a gallery file."""
        return HF_GALLERY_BASE_URL + self.resolve(game, variant)


def _default_cache_dir() -> Path:
    """Per-user cache directory for downloaded galleries and manifests."""
    import os
    base = Path(os.environ.get("COLLECTORVISION_CACHE", "~/.cache/collectorvision"))
    return base.expanduser()
