"""CollectorVision — card identification library for collectible card games."""

from collector_vision.identifier import Identifier
from collector_vision.hfd import HFD
from collector_vision.gallery import Gallery
from collector_vision.games import Game, Embedding  # Embedding available but not advertised

__all__ = ["Identifier", "HFD", "Gallery", "Game"]
__version__ = "0.1.0.dev0"
