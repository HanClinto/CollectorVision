"""Supported collectible card games, embedding algorithms, and their registry.

``Game`` and ``Embedding`` are the primary user-facing enums.

Adding a new game requires:
  1. A new ``Game`` entry here
  2. A catalog built and published under the naming convention
     ``{algo}-{source}-{game}-{YYYY-MM}.npz``
  3. The manifest on HF Datasets updated to point at the new file
"""
from __future__ import annotations

import re
from enum import Enum


class Embedding(str, Enum):
    """Embedding algorithm used to represent and compare cards.

    MILO   — ArcFace neural embedding (MobileViT-XXS backbone).  Best
              accuracy for edition (exact printing) identification.  Requires
              a GPU or Apple Silicon for reasonable speed.

    PHASH  — Perceptual hash (16×16 DCT).  Runs on any CPU, no GPU needed.
              Excellent for artwork identification; weaker on edition.
              Catalog is ~30× smaller than MILO (32 B vs 512 B per card).
    """

    MILO  = "milo1"    # neural ArcFace, default
    PHASH = "phash16"  # perceptual hash 16×16

    def __str__(self) -> str:
        return self.value

    @property
    def family(self) -> str:
        """Model family name for HF repo naming — strips version suffix.

        ``Embedding.MILO.family`` → ``"milo"``   (milo1, milo2 → same repo)
        ``Embedding.PHASH.family`` → ``"phash16"`` (variant is stable)
        """
        return re.sub(r"\d+$", "", self.value) or self.value


class Game(str, Enum):
    """Canonical game identifiers used throughout CollectorVision.

    Values are lowercase strings safe for use in filenames and API calls.
    ``str(Game.MTG)`` → ``"mtg"``.
    """

    # -----------------------------------------------------------------------
    # Currently supported (catalogs published)
    # -----------------------------------------------------------------------
    MTG     = "mtg"       # Magic: The Gathering  (source: Scryfall)
    POKEMON = "pokemon"   # Pokémon TCG           (source: TCGplayer / PokémonTCG.io)

    # -----------------------------------------------------------------------
    # Planned (catalogs not yet published)
    # -----------------------------------------------------------------------
    YUGIOH  = "yugioh"    # Yu-Gi-Oh!             (source: TCGplayer)
    FAB     = "fab"       # Flesh and Blood        (source: TCGplayer)
    LORCANA = "lorcana"   # Disney Lorcana         (source: TCGplayer)
    DIGIMON = "digimon"   # Digimon Card Game      (source: TCGplayer)
    ONEPIECE = "onepiece" # One Piece Card Game    (source: TCGplayer)
    SWU     = "swu"       # Star Wars: Unlimited   (source: TCGplayer)
    DBS     = "dbs"       # Dragon Ball Super CG   (source: TCGplayer)

    def __str__(self) -> str:
        return self.value


# Human-readable display names for UI / error messages
GAME_DISPLAY_NAMES: dict[Game, str] = {
    Game.MTG:    "Magic: The Gathering",
    Game.POKEMON:  "Pokémon TCG",
    Game.YUGIOH:   "Yu-Gi-Oh!",
    Game.FAB:      "Flesh and Blood",
    Game.LORCANA:  "Disney Lorcana",
    Game.DIGIMON:  "Digimon Card Game",
    Game.ONEPIECE: "One Piece Card Game",
    Game.SWU:      "Star Wars: Unlimited",
    Game.DBS:      "Dragon Ball Super Card Game",
}

# Primary data source for each game.
# This is informational — the catalog filename encodes the source explicitly.
GAME_PRIMARY_SOURCE: dict[Game, str] = {
    Game.MTG:    "scryfall",
    Game.POKEMON:  "tcgplayer",
    Game.YUGIOH:   "tcgplayer",
    Game.FAB:      "tcgplayer",
    Game.LORCANA:  "tcgplayer",
    Game.DIGIMON:  "tcgplayer",
    Game.ONEPIECE: "tcgplayer",
    Game.SWU:      "tcgplayer",
    Game.DBS:      "tcgplayer",
}


def parse_embedding(value: str) -> Embedding:
    """Case-insensitive parse of an embedding identifier string."""
    try:
        return Embedding(value.lower().strip())
    except ValueError:
        known = ", ".join(e.value for e in Embedding)
        raise ValueError(
            f"Unknown embedding {value!r}. Supported embeddings: {known}"
        ) from None


def parse_game(value: str) -> Game:
    """Case-insensitive parse of a game identifier string.

    Raises ValueError with a helpful message on unknown input.
    """
    try:
        return Game(value.lower().strip())
    except ValueError:
        known = ", ".join(g.value for g in Game)
        raise ValueError(
            f"Unknown game {value!r}. Supported games: {known}"
        ) from None
