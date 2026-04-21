"""NeuralCornerDetector — SimCC-based learned corner detector.

Loads the bundled weights from collector_vision.weights by default.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from collector_vision.interfaces import DetectionResult


class NeuralCornerDetector:
    """MobileViT-XXS + SimCC learned corner detector.

    Parameters
    ----------
    checkpoint:
        Path to a .pt checkpoint.  Defaults to the bundled weights.
    device:
        "cpu", "cuda", "mps", or None (auto-detect).
    presence_threshold:
        Minimum card-presence score to treat a detection as valid.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        presence_threshold: float = 0.5,
    ) -> None:
        import torch
        from collector_vision import weights as _w

        if checkpoint is None:
            checkpoint = _w.CORNER_DETECTOR
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Corner detector weights not found: {checkpoint}\n"
                "Install the full package or supply a checkpoint path."
            )

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._device = device
        self._threshold = presence_threshold
        self._model = self._load(checkpoint, device)

    def _load(self, checkpoint: Path, device: str):
        # Import deferred so the package is importable without torch installed
        # (e.g. when only using FixedCornerDetector or CannyCornerDetector).
        raise NotImplementedError(
            "NeuralCornerDetector._load() — to be wired up when weights are finalised"
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        raise NotImplementedError(
            "NeuralCornerDetector.detect() — to be wired up when weights are finalised"
        )
