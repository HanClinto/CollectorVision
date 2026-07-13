"""CollectorVision — card identification library for collectible card games."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from collector_vision.catalog import Catalog
from collector_vision.detectors import NeuralCornerDetector
from collector_vision.embedders import NeuralEmbedder
from collector_vision.games import Embedding, Game
from collector_vision.geometry import DEFAULT_MIN_CORNER_QUALITY, QuadQuality, quad_quality
from collector_vision.hfd import HFD
from collector_vision.interfaces import DetectionResult
from collector_vision.transforms import rotate_card_180

try:
    __version__: str = version("collectorvision")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "Catalog",
    "DEFAULT_MIN_CORNER_QUALITY",
    "DetectionResult",
    "Embedding",
    "Game",
    "HFD",
    "NeuralCornerDetector",
    "NeuralEmbedder",
    "QuadQuality",
    "quad_quality",
    "rotate_card_180",
    "__version__",
]
