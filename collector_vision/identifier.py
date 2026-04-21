"""Identifier — the primary user-facing API for card identification.

Typical usage::

    import collector_vision as cvg

    # Local gallery file
    cvid = cvg.Identifier("./magic-scryfall-phash16-2026-04.npz")

    # Auto-download from HuggingFace (re-checks for updates every 7 days)
    cvid = cvg.Identifier(cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"))

    result = cvid.identify("photo.jpg")
    print(result.card_name, result.set_code)

Create one Identifier per process. Heavy objects (detector, gallery) are
loaded lazily on first call and reused across subsequent calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from collector_vision.catalogs.base import CardResult

if TYPE_CHECKING:
    from collector_vision.interfaces import CornerDetector
    from collector_vision.gallery import Gallery
    from collector_vision.hfd import HFD

# A gallery source is either a local path (str or Path) or an HFD reference
GallerySource = Union[str, Path, "HFD", "Gallery"]

_SENTINEL = object()  # distinguishes "not passed" from explicit None


class Identifier:
    """Loaded, ready-to-use card identifier.

    Parameters
    ----------
    *galleries:
        One or more gallery sources — any combination of:

        - ``str`` or ``Path`` — path to a local ``.npz`` gallery file
        - :class:`~collector_vision.hfd.HFD` — auto-download from HuggingFace
        - :class:`~collector_vision.gallery.Gallery` — pre-loaded gallery object

        When multiple galleries are supplied they are merged into a single
        search index.  All galleries must use the same embedding algorithm
        (encoded in the file's ``embedder_spec``).
    detector:
        Corner detector instance.  Defaults to the bundled
        :class:`~collector_vision.detectors.neural.NeuralCornerDetector`.
        Pass ``detector=None`` to skip detection and treat the full image as
        the card (useful when the input is already a clean crop).

    Examples
    --------
    Local file::

        cvid = cvg.Identifier("./magic-scryfall-phash16-2026-04.npz")

    Auto-download from HuggingFace::

        cvid = cvg.Identifier(cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"))

    Multi-gallery (must share the same embedding algorithm)::

        cvid = cvg.Identifier(
            cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"),
            cvg.HFD("CollectorVision/galleries", "pokemon-tcgplayer-phash16"),
        )

    Canny detector (faster on clean backgrounds, no GPU needed)::

        from collector_vision.detectors import CannyCornerDetector
        cvid = cvg.Identifier(
            cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"),
            detector=CannyCornerDetector(),
        )

    No detection — input is already a card crop::

        cvid = cvg.Identifier("./magic-scryfall-phash16-2026-04.npz", detector=None)
    """

    def __init__(
        self,
        *galleries: GallerySource,
        detector: "CornerDetector | None" = _SENTINEL,  # type: ignore[assignment]
    ) -> None:
        if not galleries:
            raise ValueError(
                "Provide at least one gallery source, e.g.:\n"
                "  Identifier('./magic-scryfall-phash16-2026-04.npz')\n"
                "  Identifier(HFD('CollectorVision/galleries', 'magic-scryfall-phash16'))"
            )

        self._sources = galleries
        self._detector_arg = detector  # _SENTINEL | CornerDetector | None

        # Lazy-loaded
        self._gallery: Gallery | None = None
        self._detector: object = _NOT_LOADED  # CornerDetector | None once loaded

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_gallery(self) -> "Gallery":
        if self._gallery is None:
            from collector_vision.gallery import Gallery
            from collector_vision.hfd import HFD

            loaded: list[Gallery] = []
            for src in self._sources:
                if isinstance(src, Gallery):
                    loaded.append(src)
                elif isinstance(src, HFD):
                    loaded.append(Gallery.load(src.resolve()))
                else:
                    loaded.append(Gallery.load(Path(src)))

            self._gallery = loaded[0] if len(loaded) == 1 else Gallery._merge(loaded)
        return self._gallery

    def _get_detector(self) -> "CornerDetector | None":
        if self._detector is _NOT_LOADED:
            if self._detector_arg is _SENTINEL:
                from collector_vision.detectors.neural import NeuralCornerDetector
                self._detector = NeuralCornerDetector()
            else:
                self._detector = self._detector_arg  # type: ignore[assignment]
        return self._detector  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify(
        self,
        *images: "str | Path | np.ndarray",
        top_k: int = 5,
    ) -> CardResult:
        """Identify a card from one or more images.

        Parameters
        ----------
        *images:
            One or more paths or BGR numpy arrays (``cv2.imread`` output).

            **Single image** — standard identification; returns the best match.

            **Multiple images** — treated as frames of the same physical card
            (e.g. consecutive video frames or multiple photos of the same card).
            Similarity scores are summed across frames before ranking, giving a
            more confident result than any single frame alone.  The individual
            per-frame results are available in
            :attr:`~collector_vision.catalogs.base.CardResult.frame_results`.

        top_k:
            Number of alternatives to include in the result.

        Returns
        -------
        :class:`~collector_vision.catalogs.base.CardResult`

        Examples
        --------
        Single image::

            result = cvid.identify("photo.jpg")

        Vote across video frames::

            result = cvid.identify("frame1.jpg", "frame2.jpg", "frame3.jpg")
            print(result.card_name)          # aggregate winner
            print(result.frame_results)      # per-frame breakdown
        """
        if not images:
            raise ValueError("At least one image is required.")
        raise NotImplementedError("Identifier.identify() — not yet wired up")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        srcs = ", ".join(repr(s) for s in self._sources)
        det = (
            "neural (default)" if self._detector_arg is _SENTINEL
            else "none" if self._detector_arg is None
            else repr(self._detector_arg)
        )
        return f"Identifier({srcs}, detector={det})"


class _NotLoaded:
    pass


_NOT_LOADED = _NotLoaded()
