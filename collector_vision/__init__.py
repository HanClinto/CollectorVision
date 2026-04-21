"""CollectorVision — card identification library for collectible card games."""

import numpy as np

from collector_vision.identify import identify, identify_batch
from collector_vision.gallery import Gallery
from collector_vision.games import Game

__all__ = ["identify", "identify_batch", "Gallery", "Game", "FULL_IMAGE_CORNERS"]
__version__ = "0.1.0.dev0"

# Pass as `corners` to skip corner detection entirely and treat the whole
# image as the card.  Useful for pre-cropped images or camera setups where
# framing is handled externally.
#
#   result = cv.identify("crop.jpg", gallery=gallery, corners=cv.FULL_IMAGE_CORNERS)
#
FULL_IMAGE_CORNERS = np.array([
    [0.0, 0.0],  # top-left
    [1.0, 0.0],  # top-right
    [1.0, 1.0],  # bottom-right
    [0.0, 1.0],  # bottom-left
], dtype=np.float32)
