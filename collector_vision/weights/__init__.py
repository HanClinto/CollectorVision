"""Bundled model weights — resolved at import time to absolute paths.

Both models are single-file ONNX — no paired .data file required.

reggie.onnx  (8.2 MB) — Reggie, corner detector
    MobileViT-XXS + SimCC, trained on CCG card corners.
    Input:   (1, 3, 384, 384) float32, ImageNet-normalised
    Outputs: corners (1, 8)    — normalised [0,1] TL/TR/BR/BL x0,y0…x3,y3
             presence (1,)     — raw card-presence logit (unreliable; use sharpness)
             sharpness (1,)    — mean peak of 8 SimCC softmax distributions

milo.onnx  (5.0 MB) — Milo, card embedder
    MobileViT-XXS + ArcFace, multi-task (illustration_id + set_code), epoch 15.
    Input:  (1, 3, 448, 448) float32, ImageNet-normalised
    Output: embedding (1, 128) float32, L2-normalised
"""
from pathlib import Path

_WEIGHTS_DIR = Path(__file__).parent

CORNER_DETECTOR = _WEIGHTS_DIR / "reggie.onnx"   # Reggie
EMBEDDER        = _WEIGHTS_DIR / "milo.onnx"     # Milo


def check() -> dict[str, bool]:
    """Return which bundled weight files are present."""
    return {
        "reggie (corner_detector)": CORNER_DETECTOR.exists(),
        "milo (embedder)":          EMBEDDER.exists(),
    }
