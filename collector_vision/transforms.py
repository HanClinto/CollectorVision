"""Image transforms for dewarped card crops."""

from __future__ import annotations

from PIL import Image


def rotate_card_180(crop: Image.Image) -> Image.Image:
    """Return a 180-degree rotated copy of a dewarped card crop."""
    return crop.transpose(Image.Transpose.ROTATE_180)
