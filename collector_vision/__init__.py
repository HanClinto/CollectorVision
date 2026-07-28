"""CollectorVision — card identification library for collectible card games."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from collector_vision.catalog import Catalog
from collector_vision.catalog_v2 import CatalogV2
from collector_vision.catalog_v2_downloader import CatalogV2Downloader
from collector_vision.detectors import NeuralCornerDetector
from collector_vision.embedders import NeuralEmbedder
from collector_vision.games import Embedding, Game
from collector_vision.hfd import HFD
from collector_vision.interfaces import DetectionResult
from collector_vision.model_artifacts import resolve_model_artifact
from collector_vision.model_registry import (
    ModelSpec,
    available_channels,
    available_models,
    get_model,
    load_model_registry,
)
from collector_vision.transforms import rotate_card_180

try:
    __version__: str = version("collectorvision")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "Catalog",
    "CatalogV2",
    "CatalogV2Downloader",
    "DetectionResult",
    "Embedding",
    "Game",
    "HFD",
    "ModelSpec",
    "NeuralCornerDetector",
    "NeuralEmbedder",
    "available_channels",
    "available_models",
    "get_model",
    "load_model_registry",
    "resolve_model_artifact",
    "rotate_card_180",
    "__version__",
]
