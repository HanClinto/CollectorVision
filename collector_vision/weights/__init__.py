"""Bundled model weights — resolved at import time to absolute paths.

Corner detector codename: Reggie
  MobileViT-XXS + SimCC, trained on CCG card corners, 384×384 input.

Embedder codename: Milo
  MobileViT-XXS + ArcFace, multi-task (illustration_id + set_code), 448×448 input.
  Produces L2-normalised 128-d embeddings.

Both models use ONNX external data format: the .onnx file contains the graph;
the .onnx.data file contains the weight tensors.  They must remain in the same
directory and the .onnx file references its .data file by its exact filename.
"""
from pathlib import Path

_WEIGHTS_DIR = Path(__file__).parent

# Reggie — corner detector (codename; the file keeps its training-run name)
CORNER_DETECTOR = (
    _WEIGHTS_DIR / "detector_mvit_simcc_lc5_img384_ph10_seedin_blr10_fz2_e45.onnx"
)

# Milo — card embedder (codename)
EMBEDDER = (
    _WEIGHTS_DIR / "identifier_mobilevit_xxs_multitask_illustration_id"
    "+set_code_shared_128d+128d_mvitxxs_shared2h_arcface_v2light_img448_ph10_e15.onnx"
)


def check() -> dict[str, bool]:
    """Return which bundled weight files are present."""
    return {
        "reggie (corner_detector)": CORNER_DETECTOR.exists(),
        "milo (embedder)": EMBEDDER.exists(),
    }
