"""Base types shared across all card catalog adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CardResult:
    """Identification result from a card gallery lookup.

    ``ids`` is an open-ended mapping from key to value, populated with
    whatever identifiers the gallery was built from.  Well-known keys:

        scryfall_id           — Scryfall UUID
        oracle_id             — Scryfall oracle UUID (groups all printings of
                                the same card text)
        illustration_id       — Scryfall illustration UUID (groups reprints
                                sharing the same artwork)
        tcgplayer_id          — TCGplayer product ID
        tcgplayer_etched_id   — TCGplayer etched-foil variant product ID
        cardmarket_id         — Cardmarket (MKM) product ID
        mtgo_id               — Magic Online card ID
        arena_id              — MTG Arena card ID
        multiverse_ids        — list of Gatherer multiverse IDs (may be a
                                JSON-encoded list string)
        pokemontcg_id         — Pokémon TCG API card ID (e.g. "base1-4")

    Keys present depend entirely on what the gallery builder recorded.
    Missing keys are simply absent from the dict rather than None.
    """
    card_name: str
    set_code: str
    confidence: float           # cosine similarity (embedding) or 1 - hamming_norm (hash)

    # Open-ended identifier mapping — whatever the gallery source provides
    ids: dict[str, str] = field(default_factory=dict)

    # Convenience accessor
    def get_id(self, key: str) -> str | None:
        """Return the identifier for *key*, or None if not present."""
        return self.ids.get(key)

    # Top-k alternatives (same structure, lower confidence)
    alternatives: list["CardResult"] = field(default_factory=list)

    # Per-frame results when identify() was called with multiple images.
    # Each entry is the best match for that individual frame before voting.
    # Empty when identify() was called with a single image.
    frame_results: list["CardResult"] = field(default_factory=list)

    # Raw retrieval metadata for advanced users
    extra: dict[str, Any] = field(default_factory=dict)
