"""Identifier — the primary user-facing API for card identification.

Typical usage::

    import collector_vision as cvg

    cvid = cvg.Identifier(cvg.Game.MAGIC)
    result = cvid.identify("photo.jpg")
    print(result.card_name, result.set_code)

The Identifier downloads and caches the appropriate gallery on first use.
Heavy objects (detector, embedder, gallery) are loaded lazily and reused
across calls — create one Identifier per process, not one per image.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from collector_vision.games import Game, Embedding
from collector_vision.catalogs.base import CardResult

if TYPE_CHECKING:
    from collector_vision.interfaces import CornerDetector
    from collector_vision.gallery import Gallery


_SENTINEL = object()  # distinguishes "not passed" from explicit None


class Identifier:
    """Loaded, ready-to-use card identifier.

    Parameters
    ----------
    *games:
        One or more :class:`Game` values.  Galleries for each are downloaded
        on first use and merged into a single search index.  At least one game
        is required unless *gallery* is supplied.
    embedding:
        Which embedding algorithm to use.  Defaults to
        :attr:`Embedding.MILO` (neural ArcFace).  Use
        :attr:`Embedding.PHASH` for CPU-only / no-GPU environments.
    detector:
        Corner detector instance.  Defaults to the bundled
        :class:`~collector_vision.detectors.neural.NeuralCornerDetector`.
        Pass ``detector=None`` to skip detection and treat the full image as
        the card (useful when the input is already a clean crop).
    gallery:
        Supply a pre-loaded :class:`~collector_vision.gallery.Gallery`
        directly.  When provided, *games* and *embedding* are ignored.
        Intended for power users loading a local gallery file.
    offline:
        If ``True``, never make network calls — use only locally cached
        galleries.  Raises if a required gallery is not cached.
    cache_dir:
        Override the default cache directory
        (``~/.cache/collectorvision/``).

    Examples
    --------
    Default (neural detector + neural embedding, Magic)::

        cvid = cvg.Identifier(cvg.Game.MAGIC)

    Hash embedding, no GPU required::

        cvid = cvg.Identifier(cvg.Game.MAGIC, embedding=cvg.Embedding.PHASH)

    Multi-game::

        cvid = cvg.Identifier(cvg.Game.MAGIC, cvg.Game.POKEMON)

    Canny detector (faster on clean backgrounds)::

        from collector_vision.detectors import CannyCornerDetector
        cvid = cvg.Identifier(cvg.Game.MAGIC, detector=CannyCornerDetector())

    No detection — input is already a card crop::

        cvid = cvg.Identifier(cvg.Game.MAGIC, detector=None)

    Local gallery file::

        cvid = cvg.Identifier(gallery=cvg.Gallery.load("my_gallery.npz"))
    """

    def __init__(
        self,
        *games: Game,
        embedding: Embedding = Embedding.MILO,
        detector: "CornerDetector | None" = _SENTINEL,  # type: ignore[assignment]
        gallery: "Gallery | None" = None,
        offline: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        if gallery is None and not games:
            raise ValueError(
                "Provide at least one Game, e.g. Identifier(Game.MAGIC), "
                "or supply a pre-loaded gallery via gallery=Gallery.load(...)"
            )

        self._games = games
        self._embedding = embedding
        self._offline = offline
        self._cache_dir = cache_dir

        # Detector sentinel: _SENTINEL means "use default neural detector"
        # None means "no detection, treat full image as card"
        self._detector_arg = detector

        # Lazy-loaded heavy objects
        self._gallery: Gallery | None = gallery
        self._detector: CornerDetector | None | _NotLoaded = _NOT_LOADED
        self._embedder = None  # derived from gallery.embedder on first use

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_gallery(self) -> "Gallery":
        if self._gallery is None:
            from collector_vision.gallery import Gallery
            self._gallery = Gallery.for_games(
                *self._games,
                embedding=self._embedding,
                offline=self._offline,
                cache_dir=self._cache_dir,
            )
        return self._gallery

    def _get_detector(self) -> "CornerDetector | None":
        if self._detector is _NOT_LOADED:
            if self._detector_arg is _SENTINEL:
                # Default: neural detector
                from collector_vision.detectors.neural import NeuralCornerDetector
                self._detector = NeuralCornerDetector()
            else:
                # Explicit: either a detector instance or None
                self._detector = self._detector_arg  # type: ignore[assignment]
        return self._detector  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify(
        self,
        image: "str | Path | np.ndarray",
        *,
        top_k: int = 5,
    ) -> CardResult:
        """Identify a single card image.

        Parameters
        ----------
        image:
            Path to the card photo, or a BGR numpy array (as returned by
            ``cv2.imread``).
        top_k:
            Number of alternatives to include in
            :attr:`~collector_vision.catalogs.base.CardResult.alternatives`.

        Returns
        -------
        :class:`~collector_vision.catalogs.base.CardResult` with the best
        match and up to *top_k*-1 alternatives.
        """
        raise NotImplementedError("Identifier.identify() — not yet wired up")

    def identify_batch(
        self,
        images: "list[str | Path | np.ndarray]",
        *,
        top_k: int = 5,
        batch_size: int = 16,
    ) -> "list[CardResult]":
        """Identify multiple card images in a single batched pass.

        Parameters
        ----------
        images:
            List of paths or BGR numpy arrays.
        top_k:
            Alternatives per result.
        batch_size:
            Images per embedding forward pass.
        """
        raise NotImplementedError("Identifier.identify_batch() — not yet wired up")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        games = ", ".join(g.value for g in self._games)
        det = (
            "neural" if self._detector_arg is _SENTINEL
            else "none" if self._detector_arg is None
            else type(self._detector_arg).__name__
        )
        return (
            f"Identifier(games=[{games}], embedding={self._embedding.value}, "
            f"detector={det})"
        )


class _NotLoaded:
    """Sentinel type for unloaded lazy objects."""


_NOT_LOADED = _NotLoaded()
