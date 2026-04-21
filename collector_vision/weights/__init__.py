"""Bundled model weights — resolved at import time to absolute paths."""
from pathlib import Path

_WEIGHTS_DIR = Path(__file__).parent

CORNER_DETECTOR = _WEIGHTS_DIR / "corner_detector.pt"
EMBEDDER = _WEIGHTS_DIR / "embedder.pt"


def check() -> dict[str, bool]:
    """Return which bundled weight files are present."""
    return {
        "corner_detector": CORNER_DETECTOR.exists(),
        "embedder": EMBEDDER.exists(),
    }
