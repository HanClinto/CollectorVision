"""Identifier — the primary user-facing API for card identification.

Typical usage::

    import collector_vision as cvg

    # Local gallery file
    cvid = cvg.Identifier("./milo1-scryfall-mtg-2026-04.npz")

    # Auto-download from HuggingFace (re-checks for updates every 7 days)
    cvid = cvg.Identifier(cvg.HFD("HanClinto/milo", "scryfall-mtg"))

    result = cvid.identify("photo.jpg")
    print(result.ids, result.confidence)

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

        cvid = cvg.Identifier("./milo1-scryfall-mtg-2026-04.npz")

    Auto-download from HuggingFace::

        cvid = cvg.Identifier(cvg.HFD("HanClinto/milo", "scryfall-mtg"))

    Multi-gallery (must share the same embedding algorithm)::

        cvid = cvg.Identifier(
            cvg.HFD("HanClinto/milo", "scryfall-mtg"),
            cvg.HFD("HanClinto/milo", "tcgplayer-pokemon"),
        )

    Canny detector (faster on clean backgrounds, no GPU needed)::

        from collector_vision.detectors import CannyCornerDetector
        cvid = cvg.Identifier(
            cvg.HFD("HanClinto/milo", "scryfall-mtg"),
            detector=CannyCornerDetector(),
        )

    No detection — input is already a card crop::

        cvid = cvg.Identifier("./milo1-scryfall-mtg-2026-04.npz", detector=None)
    """

    def __init__(
        self,
        *galleries: GallerySource,
        detector: "CornerDetector | None" = _SENTINEL,  # type: ignore[assignment]
    ) -> None:
        if not galleries:
            raise ValueError(
                "Provide at least one gallery source, e.g.:\n"
                "  Identifier('./milo1-scryfall-mtg-2026-04.npz')\n"
                "  Identifier(HFD('HanClinto/milo', 'scryfall-mtg'))"
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
            print(result.frame_results)      # per-frame breakdown
        """
        if not images:
            raise ValueError("At least one image is required.")

        import cv2
        from PIL import Image

        from collector_vision import retrieval

        gallery = self._get_gallery()
        detector = self._get_detector()
        embedder = gallery.embedder

        # ── Step 1: load, detect, dewarp ──────────────────────────────────
        pil_crops: list[Image.Image] = []
        for img_src in images:
            bgr = _load_image(img_src)

            if detector is not None:
                detection = detector.detect(bgr)
                if detection.card_present and detection.corners is not None:
                    bgr_crop = _dewarp(bgr, detection.corners)
                else:
                    bgr_crop = bgr  # no card detected — use full frame
            else:
                bgr_crop = bgr  # caller guarantees a clean crop

            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            pil_crops.append(Image.fromarray(rgb))

        # ── Step 2: embed (single batched forward pass) ───────────────────
        embs = embedder.embed(pil_crops)  # (n_frames, D) or (n_frames, B)

        # ── Step 3: per-frame nearest-neighbour search ────────────────────
        is_hash = gallery.mode == "hash"
        frame_hits: list[list[tuple[float, int]]] = []
        for emb in embs:
            if is_hash:
                hits = retrieval.hamming_search(emb, gallery.embeddings, top_k=top_k)
            else:
                hits = retrieval.cosine_search(emb, gallery.embeddings, top_k=top_k)
            frame_hits.append(hits)

        # ── Step 4: aggregate across frames ───────────────────────────────
        if len(frame_hits) == 1:
            agg_hits = frame_hits[0]
        else:
            from collections import defaultdict
            score_map: dict[int, float] = defaultdict(float)
            for hits in frame_hits:
                for score, idx in hits:
                    score_map[idx] += score
            agg_hits = [
                (score, idx)
                for idx, score in sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            ][:top_k]

        # ── Step 5: build result ──────────────────────────────────────────
        frame_results: list[CardResult] = []
        if len(images) > 1:
            frame_results = [_hits_to_result(h, gallery, top_k) for h in frame_hits]

        result = _hits_to_result(agg_hits, gallery, top_k)
        result.frame_results = frame_results
        return result

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


# ---------------------------------------------------------------------------
# Image loading and dewarping helpers
# ---------------------------------------------------------------------------

# Dewarp output size — card aspect ratio ~63.5 × 88.9 mm, generous resolution
# so the embedder's internal resize doesn't upscale from a tiny source.
_DEWARP_W = 252  # 4 × 63
_DEWARP_H = 352  # 4 × 88


def _load_image(image: "str | Path | np.ndarray") -> np.ndarray:
    """Load an image from a path or return a BGR ndarray as-is."""
    if isinstance(image, np.ndarray):
        return image
    import cv2
    bgr = cv2.imread(str(image))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return bgr


def _dewarp(bgr: np.ndarray, corners_norm: np.ndarray) -> np.ndarray:
    """Perspective-warp a card to a flat rectangle given normalised corners."""
    import cv2
    h, w = bgr.shape[:2]
    src = corners_norm * np.array([w, h], dtype=np.float32)
    dst = np.array(
        [[0, 0], [_DEWARP_W - 1, 0], [_DEWARP_W - 1, _DEWARP_H - 1], [0, _DEWARP_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (_DEWARP_W, _DEWARP_H))


def _hits_to_result(
    hits: "list[tuple[float, int]]",
    gallery: "Gallery",
    top_k: int,
) -> CardResult:
    """Build a CardResult from a ranked (score, gallery_index) list."""
    if not hits:
        return CardResult()

    from collector_vision.gallery import _source_primary_key
    pk = _source_primary_key(gallery.source)

    def _make_ids(idx: int) -> dict:
        return {pk: gallery.card_ids[idx]}

    best_score, best_idx = hits[0]
    alts = [
        CardResult(ids=_make_ids(idx), confidence=float(score))
        for score, idx in hits[1:]
    ]
    return CardResult(
        ids=_make_ids(best_idx),
        confidence=float(best_score),
        alternatives=alts,
    )


class _NotLoaded:
    pass


_NOT_LOADED = _NotLoaded()
