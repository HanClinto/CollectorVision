"""Protocols defining the pluggable detector and embedder interfaces.

Any object satisfying these protocols can be used in the pipeline — no
subclassing required.  The bundled NeuralCornerDetector and NeuralEmbedder
satisfy them, and users can supply their own.

To bypass detection entirely, either crop the image yourself or construct
a DetectionResult with your own corners and call dewarp() on it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# Dewarped card output size — card aspect ratio ~63.5 × 88.9 mm
_DEWARP_W = 252  # 4 × 63
_DEWARP_H = 352  # 4 × 88


@runtime_checkable
class CornerDetector(Protocol):
    """Locates the four corners of a card in a raw image.

    Corners are returned in normalised image coordinates [0, 1] in the order
    top-left, top-right, bottom-right, bottom-left.  When no card is detected
    ``card_present`` should be False and ``corners`` may be None or arbitrary.
    """

    def detect(self, image: np.ndarray) -> "DetectionResult": ...


class DetectionResult:
    """Output of a CornerDetector.detect() call."""

    __slots__ = ("corners", "card_present", "confidence", "sharpness", "extra")

    def __init__(
        self,
        corners: np.ndarray | None,     # (4, 2) float32, normalised [0, 1]
        card_present: bool = True,
        confidence: float = 1.0,
        sharpness: float | None = None,
        extra: dict | None = None,
    ) -> None:
        self.corners = corners
        self.card_present = card_present
        self.confidence = confidence
        self.sharpness = sharpness
        self.extra = extra or {}

    def dewarp(self, bgr: np.ndarray) -> Image:
        """Perspective-warp the card region to a flat rectangle.

        Parameters
        ----------
        bgr:
            Full-frame BGR image (as returned by ``cv2.imread``).

        Returns
        -------
        PIL Image (RGB) of the dewarped card, sized ``252 × 352`` px.
        Raises ``ValueError`` if no card was detected (``card_present`` is False
        or ``corners`` is None).
        """
        if not self.card_present or self.corners is None:
            raise ValueError(
                "Cannot dewarp: no card detected. Check card_present before calling dewarp()."
            )
        import cv2
        from PIL import Image

        h, w = bgr.shape[:2]
        src = self.corners * np.array([w, h], dtype=np.float32)
        dst = np.array(
            [[0, 0], [_DEWARP_W - 1, 0], [_DEWARP_W - 1, _DEWARP_H - 1], [0, _DEWARP_H - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(bgr, M, (_DEWARP_W, _DEWARP_H))
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


@runtime_checkable
class Embedder(Protocol):
    """Produces a fixed-length vector representation of a dewarped card image.

    Input is a single PIL Image or a list of PIL Images (already dewarped).

    - Single image → (D,) float32 vector.
    - List of images → (N, D) float32 array, one row per image.

    Vectors should be L2-normalised for cosine similarity retrieval.
    """

    def embed(self, images: Image | list[Image]) -> np.ndarray: ...
