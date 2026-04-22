"""Protocols defining the pluggable detector and embedder interfaces.

Any object satisfying these protocols can be passed to identify() — no
subclassing required.  The bundled implementations (NeuralCornerDetector,
CannyCornerDetector, NeuralEmbedder, HashEmbedder) all satisfy them, and
users can supply their own without touching CollectorVision internals.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


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

    __slots__ = ("corners", "card_present", "confidence", "extra")

    def __init__(
        self,
        corners: np.ndarray | None,     # (4, 2) float32, normalised [0, 1]
        card_present: bool = True,
        confidence: float = 1.0,
        extra: dict | None = None,
    ) -> None:
        self.corners = corners
        self.card_present = card_present
        self.confidence = confidence
        self.extra = extra or {}


@runtime_checkable
class Embedder(Protocol):
    """Produces a fixed-length vector representation of a dewarped card image.

    Input is a single PIL Image or a list of PIL Images (already dewarped).

    - Single image → (D,) float32 vector.
    - List of images → (N, D) float32 array, one row per image.

    Vectors should be L2-normalised for cosine similarity retrieval, or
    packed uint8 bits for Hamming distance retrieval.
    """

    def embed(self, images: "Image | list[Image]") -> np.ndarray: ...
