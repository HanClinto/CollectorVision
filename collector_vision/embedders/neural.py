"""NeuralEmbedder — ArcFace MobileViT-XXS learned card embedder.

Produces L2-normalised float32 vectors for cosine-similarity retrieval.
Loads the bundled weights by default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class NeuralEmbedder:
    """MobileViT-XXS + linear projection, trained with ArcFace loss.

    Parameters
    ----------
    checkpoint:
        Path to a .pt checkpoint.  Defaults to the bundled weights.
    device:
        "cpu", "cuda", "mps", or None (auto-detect).
    image_size:
        Input resolution.  Must match what the checkpoint was trained with
        (448 for the bundled model).
    batch_size:
        Number of images to embed per forward pass.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        image_size: int = 448,
        batch_size: int = 16,
    ) -> None:
        import torch
        from collector_vision import weights as _w

        if checkpoint is None:
            checkpoint = _w.EMBEDDER
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Embedder weights not found: {checkpoint}\n"
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
        self._image_size = image_size
        self._batch_size = batch_size
        self._model = self._load(checkpoint, device)

    def _load(self, checkpoint: Path, device: str):
        raise NotImplementedError(
            "NeuralEmbedder._load() — to be wired up when weights are finalised"
        )

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL Images.

        Returns (n, dim) float32 array of L2-normalised vectors.
        """
        raise NotImplementedError(
            "NeuralEmbedder.embed() — to be wired up when weights are finalised"
        )

    def __repr__(self) -> str:
        return f"NeuralEmbedder(device={self._device!r}, image_size={self._image_size})"
