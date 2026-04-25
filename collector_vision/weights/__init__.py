"""Bundled model weights — paths and version metadata.

Both models are single-file ONNX (no paired .data file).

cornelius.onnx  (8.2 MB) — Cornelius v1.164, card corner detector
    MobileViT-XXS + SimCC, epoch 164, test_iou=0.7234.
    Input:   (1, 3, 384, 384) float32, ImageNet-normalised
    Outputs: corners (1, 8) normalised [0,1] TL/TR/BR/BL x0,y0…
             presence (1,) raw logit — use sharpness instead
             sharpness (1,) mean SimCC softmax peak, range [0, 1]

milo.onnx  (5.0 MB) — Milo, card embedder
    MobileViT-XXS + ArcFace, multi-task (illustration_id + set_code).
    Input:  (1, 3, 448, 448) float32, ImageNet-normalised
    Output: embedding (1, 128) float32, L2-normalised

Version metadata is embedded inside each ONNX file (readable via
``onnxruntime`` session.get_modelmeta().custom_metadata_map) and
reflected in the constants below.  Update both when swapping a file.
"""

from pathlib import Path

_WEIGHTS_DIR = Path(__file__).parent

# --- Paths ------------------------------------------------------------------

CORNER_DETECTOR = _WEIGHTS_DIR / "cornelius.onnx"  # Cornelius
EMBEDDER = _WEIGHTS_DIR / "milo.onnx"  # Milo

# --- Versions ---------------------------------------------------------------
# These mirror the 'version' key in each model's ONNX metadata_props.
# Bump here (and rebuild the wheel) whenever the .onnx file is replaced.

CORNELIUS_VERSION = "1.164"
MILO_VERSION = "1.0.0"

BUNDLED_VERSIONS: dict[str, str] = {
    "cornelius": CORNELIUS_VERSION,
    "milo": MILO_VERSION,
}


# --- Diagnostics ------------------------------------------------------------


def check() -> dict[str, dict]:
    """Return presence, size, and ONNX-embedded metadata for each bundled model.

    Example output::

        {
            'cornelius': {
                'present': True,
                'version': '1.0.0',
                'task': 'card-corner-detection',
                'size_mb': 8.2,
            },
            'milo': {
                'present': True,
                'version': '1.0.0',
                'task': 'card-embedding',
                'size_mb': 5.0,
            },
        }

    Reads ONNX metadata via ``onnxruntime`` on first call (takes ~0.5 s).
    """
    result: dict[str, dict] = {}
    for name, path in [("cornelius", CORNER_DETECTOR), ("milo", EMBEDDER)]:
        if not path.exists():
            result[name] = {"present": False}
            continue

        info: dict = {
            "present": True,
            "size_mb": round(path.stat().st_size / 1_000_000, 1),
        }
        try:
            import onnxruntime as ort  # noqa: PLC0415

            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            info.update(sess.get_modelmeta().custom_metadata_map)
        except Exception as exc:
            info["metadata_error"] = str(exc)

        result[name] = info
    return result
