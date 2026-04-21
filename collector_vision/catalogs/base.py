"""Base types shared across all card catalog adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CardResult:
    """Identification result returned by :meth:`~collector_vision.Identifier.identify`.

    The library's job is identification — returning stable IDs for the matched
    card.  All human-readable metadata (name, set, price, legality, image URL)
    should be fetched from the authoritative source using these IDs.

    Well-known keys in ``ids``:

        scryfall_id           — Scryfall UUID (uniquely identifies a printing)
        oracle_id             — Scryfall oracle UUID (groups all printings of
                                the same card text)
        illustration_id       — Scryfall illustration UUID (groups reprints
                                sharing the same artwork)
        tcgplayer_id          — TCGplayer product ID
        tcgplayer_etched_id   — TCGplayer etched-foil variant product ID
        cardmarket_id         — Cardmarket (MKM) product ID
        mtgo_id               — Magic Online card ID
        arena_id              — MTG Arena card ID
        multiverse_ids        — Gatherer multiverse IDs (JSON-encoded list)
        pokemontcg_id         — Pokémon TCG API card ID (e.g. ``"base1-4"``)

    Which keys are present depends entirely on what the gallery was built from.
    Missing keys are absent from the dict rather than ``None``.

    Fetching metadata
    -----------------
    Given a ``scryfall_id``, fetch full card data from Scryfall::

        import urllib.request, json
        sid = result.ids["scryfall_id"]
        url = f"https://api.scryfall.com/cards/{sid}"
        card = json.loads(urllib.request.urlopen(url).read())
        print(card["name"], card["set"])

    Given a ``tcgplayer_id``, look up pricing via the TCGplayer API or a
    third-party price service.
    """

    ids: dict[str, str] = field(default_factory=dict)
    """Stable identifiers for the matched card. See class docstring for keys."""

    confidence: float = 0.0
    """Match quality. Cosine similarity for neural embeddings (0–1, higher is
    better); normalised Hamming similarity for hash embeddings (0–1)."""

    alternatives: list["CardResult"] = field(default_factory=list)
    """Next-best matches in descending confidence order."""

    frame_results: list["CardResult"] = field(default_factory=list)
    """Per-frame results when :meth:`~collector_vision.Identifier.identify`
    was called with multiple images.  Empty for single-image calls."""

    def get_id(self, key: str) -> str | None:
        """Return the identifier for *key*, or ``None`` if not present."""
        return self.ids.get(key)
