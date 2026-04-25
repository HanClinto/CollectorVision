"""CollectorVision — card identification library for collectible card games."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from collector_vision.catalog import Catalog
from collector_vision.detectors import NeuralCornerDetector
from collector_vision.embedders import NeuralEmbedder
from collector_vision.games import Embedding, Game
from collector_vision.hfd import HFD
from collector_vision.interfaces import DetectionResult

try:
    __version__: str = version("collectorvision")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "Catalog",
    "DetectionResult",
    "Embedding",
    "Game",
    "HFD",
    "NeuralCornerDetector",
    "NeuralEmbedder",
    "__version__",
]
