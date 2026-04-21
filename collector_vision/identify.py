"""Top-level identification API."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from collector_vision.catalogs.base import CardResult
from collector_vision.gallery import Gallery


def identify(
    image: str | Path,
    *,
    gallery: Gallery | None = None,
    top_k: int = 5,
    device: str | None = None,
) -> CardResult:
    """Identify a single card image.

    Parameters
    ----------
    image:    Path to the card photo (JPEG, PNG, …).
    gallery:  Pre-loaded Gallery.  If None, the default Scryfall embedding
              gallery is downloaded on first use and cached locally.
    top_k:    Number of alternatives to include in CardResult.alternatives.
    device:   "cpu", "cuda", "mps", or None (auto-detect).

    Returns
    -------
    CardResult with the best match and up to top_k-1 alternatives.
    """
    raise NotImplementedError("identify() not yet implemented — stub only")


def identify_batch(
    images: Sequence[str | Path],
    *,
    gallery: Gallery | None = None,
    top_k: int = 5,
    device: str | None = None,
    batch_size: int = 16,
) -> list[CardResult]:
    """Identify multiple card images in a single batched pass."""
    raise NotImplementedError("identify_batch() not yet implemented — stub only")
