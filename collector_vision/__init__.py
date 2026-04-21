"""CollectorVision — card identification library for collectible card games."""

from collector_vision.identify import identify, identify_batch
from collector_vision.gallery import Gallery

__all__ = ["identify", "identify_batch", "Gallery"]
__version__ = "0.1.0.dev0"
