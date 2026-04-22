"""CollectorVision — card identification library for collectible card games."""

from collector_vision.hfd import HFD
from collector_vision.gallery import Gallery
from collector_vision.games import Game, Embedding
from collector_vision.interfaces import DetectionResult
from collector_vision.detectors import NeuralCornerDetector, CannyCornerDetector

__all__ = [
    "HFD",
    "Gallery",
    "Game",
    "Embedding",
    "DetectionResult",
    "NeuralCornerDetector",
    "CannyCornerDetector",
]
__version__ = "0.1.0.dev0"
