"""Gallery: pre-built card index used for nearest-neighbour retrieval."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Gallery:
    """Loaded card gallery ready for retrieval.

    Wraps either a pre-computed embedding matrix (float32, cosine similarity)
    or a perceptual-hash matrix (uint8, Hamming distance), plus the card ID
    metadata needed to map hits back to card identities.

    Obtain a Gallery via Gallery.load() rather than constructing directly.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        card_ids: list[str],
        card_names: list[str],
        set_codes: list[str],
        source: str,
        mode: str,              # "embedding" | "hash"
        algo_key: str | None,   # e.g. "phash_32", "marr_hildreth_32_s2p5"
        extra: dict | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.card_ids = card_ids
        self.card_names = card_names
        self.set_codes = set_codes
        self.source = source
        self.mode = mode
        self.algo_key = algo_key
        self.extra = extra or {}

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        """Load a gallery from a CollectorVision NPZ file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        mode = str(data["mode"]) if "mode" in data.files else "embedding"
        return cls(
            embeddings=data["embeddings"],
            card_ids=data["card_ids"].tolist(),
            card_names=data["card_names"].tolist() if "card_names" in data.files else [""] * len(data["card_ids"]),
            set_codes=data["set_codes"].tolist() if "set_codes" in data.files else [""] * len(data["card_ids"]),
            source=str(data["source"]) if "source" in data.files else "unknown",
            mode=mode,
            algo_key=str(data["algo_key"]) if "algo_key" in data.files else None,
        )

    def __len__(self) -> int:
        return len(self.card_ids)

    def __repr__(self) -> str:
        return (
            f"Gallery(source={self.source!r}, mode={self.mode!r}, "
            f"n={len(self)}, algo={self.algo_key!r})"
        )
