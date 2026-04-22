"""CollectorVision — card identification library for collectible card games."""

from collector_vision.hfd import HFD
from collector_vision.catalog import Catalog
from collector_vision.games import Game, Embedding
from collector_vision.interfaces import DetectionResult
from collector_vision.detectors import NeuralCornerDetector

__all__ = [
    "HFD",
    "Catalog",
    "Game",
    "Embedding",
    "DetectionResult",
    "NeuralCornerDetector",
]
__version__ = "0.1.0.dev0"
