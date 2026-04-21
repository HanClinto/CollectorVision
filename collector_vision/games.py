"""Supported collectible card games and their gallery registry.

``Game`` is the primary user-facing identifier.  Users refer to games by
these canonical short names; the manifest maps them to published gallery
filenames.

Adding a new game requires:
  1. A new ``Game`` entry here
  2. A gallery built and published under the naming convention
     ``{game}-{source}-{YYYY-MM}-{variant}.npz``
  3. The manifest on HF Datasets updated to point at the new file
"""
from __future__ import annotations

from enum import Enum


class Game(str, Enum):
    """Canonical game identifiers used throughout CollectorVision.

    Values are lowercase strings safe for use in filenames and API calls.
    ``str(Game.MAGIC)`` → ``"magic"``.
    """

    # -----------------------------------------------------------------------
    # Currently supported (galleries published)
    # -----------------------------------------------------------------------
    MAGIC   = "magic"     # Magic: The Gathering  (source: Scryfall)
    POKEMON = "pokemon"   # Pokémon TCG           (source: TCGplayer / PokémonTCG.io)

    # -----------------------------------------------------------------------
    # Planned (galleries not yet published)
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
    Game.MAGIC:    "Magic: The Gathering",
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
# This is informational — the gallery filename encodes the source explicitly.
GAME_PRIMARY_SOURCE: dict[Game, str] = {
    Game.MAGIC:    "scryfall",
    Game.POKEMON:  "tcgplayer",
    Game.YUGIOH:   "tcgplayer",
    Game.FAB:      "tcgplayer",
    Game.LORCANA:  "tcgplayer",
    Game.DIGIMON:  "tcgplayer",
    Game.ONEPIECE: "tcgplayer",
    Game.SWU:      "tcgplayer",
    Game.DBS:      "tcgplayer",
}


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
