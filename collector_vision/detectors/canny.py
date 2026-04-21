"""CannyCornerDetector — contour-based card detection using OpenCV.

Finds the largest roughly-rectangular contour in the image and returns its
four corners.  Works well for cards on clean or high-contrast backgrounds
(flatbed scans, card-sorting rigs with a plain mat).  Falls back gracefully
when no clear rectangle is found.

No model weights required — runs entirely on CPU with standard OpenCV.
"""
from __future__ import annotations

import cv2
import numpy as np

from collector_vision.interfaces import DetectionResult


class CannyCornerDetector:
    """Contour + Canny edge based rectangular card detector.

    Parameters
    ----------
    canny_low, canny_high:
        Hysteresis thresholds for cv2.Canny.
    min_area_fraction:
        Minimum fraction of the image area a contour must cover to be
        considered a card candidate.  Rejects small noise contours.
    approx_epsilon_fraction:
        Fraction of contour perimeter used as epsilon for
        cv2.approxPolyDP.  Controls how aggressively corners are
        simplified.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area_fraction: float = 0.05,
        approx_epsilon_fraction: float = 0.02,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_fraction = min_area_fraction
        self.approx_epsilon_fraction = approx_epsilon_fraction

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        image_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_fraction * image_area:
                break  # remaining contours are smaller — stop searching

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon_fraction * peri, True)

            if len(approx) != 4:
                continue

            pts = approx.reshape(4, 2).astype(np.float32)
            pts = self._order_corners(pts)
            pts_norm = pts / np.array([w, h], dtype=np.float32)
            return DetectionResult(corners=pts_norm, card_present=True, confidence=area / image_area)

        return DetectionResult(corners=None, card_present=False, confidence=0.0)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Reorder four points to TL, TR, BR, BL."""
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        return np.array([
            pts[np.argmin(s)],   # TL: smallest x+y
            pts[np.argmin(d)],   # TR: smallest x-y
            pts[np.argmax(s)],   # BR: largest x+y
            pts[np.argmax(d)],   # BL: largest x-y
        ], dtype=np.float32)
