"""Built-in embedder implementations."""

from collector_vision.embedders.neural import NeuralEmbedder

try:
    from collector_vision.embedders.hash import HashEmbedder
except ImportError:
    HashEmbedder = None  # type: ignore[assignment,misc]

__all__ = [
    "NeuralEmbedder",
    "HashEmbedder",
]
