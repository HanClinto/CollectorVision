"""Top-level identification API."""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Sequence

from collector_vision.catalogs.base import CardResult
from collector_vision.gallery import Gallery
from collector_vision.interfaces import CornerDetector, Embedder, DetectionResult

# Dewarp output size (card aspect ratio ~63.5 × 88.9 mm)
_DEWARP_W = 224
_DEWARP_H = 312


def _dewarp(bgr: np.ndarray, corners_norm: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    src = corners_norm * np.array([w, h], dtype=np.float32)
    dst = np.array(
        [[0, 0], [_DEWARP_W - 1, 0], [_DEWARP_W - 1, _DEWARP_H - 1], [0, _DEWARP_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (_DEWARP_W, _DEWARP_H))


def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    bgr = cv2.imread(str(image))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return bgr


def _default_detector() -> CornerDetector:
    from collector_vision.detectors.neural import NeuralCornerDetector
    return NeuralCornerDetector()


def _default_embedder() -> Embedder:
    from collector_vision.embedders.neural import NeuralEmbedder
    return NeuralEmbedder()


def identify(
    image: str | Path | np.ndarray,
    *,
    gallery: Gallery | None = None,
    detector: CornerDetector | None = None,
    embedder: Embedder | None = None,
    corners: np.ndarray | None = None,
    top_k: int = 5,
    device: str | None = None,
) -> CardResult:
    """Identify a single card image.

    Parameters
    ----------
    image:
        Path to the card photo, or a BGR numpy array (as returned by
        cv2.imread).
    gallery:
        Pre-loaded Gallery.  If None, the default Scryfall embedding gallery
        is downloaded on first use and cached locally.
    detector:
        Corner detector to use.  Defaults to the bundled NeuralCornerDetector.
        Use CannyCornerDetector for clean-background images without a GPU, or
        FixedCornerDetector for setups with known card placement.
        Ignored when *corners* is supplied.
    embedder:
        Embedder to use.  Defaults to the bundled NeuralEmbedder.  Any object
        satisfying the Embedder protocol (including HashEmbedder) works.
    corners:
        Optional (4, 2) float32 array of normalised [0, 1] corner coordinates
        in TL, TR, BR, BL order.  When supplied, corner detection is skipped
        entirely — useful for robots or jigs with fixed card placement, or
        when corners are provided by an external source.
    top_k:
        Number of alternatives to include in CardResult.alternatives.
    device:
        "cpu", "cuda", "mps", or None (auto-detect).  Passed to the default
        detector and embedder if they support it; ignored for custom objects.

    Returns
    -------
    CardResult with the best match and up to top_k-1 alternatives.
    """
    raise NotImplementedError("identify() not yet implemented — stub only")


def identify_batch(
    images: Sequence[str | Path | np.ndarray],
    *,
    gallery: Gallery | None = None,
    detector: CornerDetector | None = None,
    embedder: Embedder | None = None,
    corners: Sequence[np.ndarray | None] | None = None,
    top_k: int = 5,
    device: str | None = None,
    batch_size: int = 16,
) -> list[CardResult]:
    """Identify multiple card images in a single batched pass.

    ``corners`` may be a parallel sequence of per-image corner arrays (or
    None entries to use the detector for those images), allowing mixed use
    of manual and detected corners in a single batch.
    """
    raise NotImplementedError("identify_batch() not yet implemented — stub only")
