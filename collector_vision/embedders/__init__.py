"""Built-in embedder implementations."""

from collector_vision.embedders.hash import HashEmbedder
from collector_vision.embedders.neural import NeuralEmbedder

__all__ = [
    "HashEmbedder",
    "NeuralEmbedder",
]
