"""Built-in corner detector implementations."""

from collector_vision.detectors.fixed import FixedCornerDetector
from collector_vision.detectors.neural import NeuralCornerDetector

__all__ = [
    "FixedCornerDetector",
    "NeuralCornerDetector",
]
