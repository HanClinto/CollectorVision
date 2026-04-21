"""Base types shared across all card catalog adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CardResult:
    """Identification result from any catalog source.

    Fields common to all sources are always populated.  Source-specific IDs
    (scryfall_id, tcgplayer_product_id, …) are set only when the gallery was
    built from that source.
    """
    card_name: str
    set_code: str
    confidence: float                        # cosine similarity or 1 - hamming_norm

    # Source-specific identifiers — None when not applicable
    scryfall_id: str | None = None
    oracle_id: str | None = None
    illustration_id: str | None = None
    tcgplayer_product_id: str | None = None

    # Top-k alternatives (same structure, lower confidence)
    alternatives: list["CardResult"] = field(default_factory=list)

    # Raw retrieval metadata for advanced users
    extra: dict[str, Any] = field(default_factory=dict)
