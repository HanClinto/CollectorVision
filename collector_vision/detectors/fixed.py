"""FixedCornerDetector — for setups where card placement is known in advance.

Useful for:
  - Robots or jigs that always place a card at the same position
  - Flatbed scanners with a fixed card window
  - Any pipeline where corner detection is handled externally

Usage::

    import numpy as np
    from collector_vision.detectors import FixedCornerDetector

    # Corners in normalised [0, 1] coords: TL, TR, BR, BL
    detector = FixedCornerDetector(
        corners=np.array([
            [0.05, 0.04],   # top-left
            [0.95, 0.04],   # top-right
            [0.95, 0.96],   # bottom-right
            [0.05, 0.96],   # bottom-left
        ], dtype=np.float32)
    )

    # Per-call corners can also be supplied directly to identify():
    #   cv.identify(image, corners=my_corners)
    # which bypasses the detector entirely.
"""
from __future__ import annotations

import numpy as np

from collector_vision.interfaces import DetectionResult


class FixedCornerDetector:
    """Returns the same pre-specified corners for every image."""

    def __init__(self, corners: np.ndarray) -> None:
        """
        Parameters
        ----------
        corners:
            (4, 2) float32 array of normalised [0, 1] corner coordinates in
            TL, TR, BR, BL order.
        """
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(f"corners must be shape (4, 2), got {corners.shape}")
        self._corners = corners

    def detect(self, image: np.ndarray) -> DetectionResult:
        return DetectionResult(corners=self._corners.copy(), card_present=True, confidence=1.0)
