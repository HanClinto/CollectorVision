"""NeuralCornerDetector — ONNX-based learned card corner detector (Reggie).

Runs entirely on CPU via onnxruntime — no PyTorch dependency required.

Architecture: MobileViT-XXS backbone + SimCC coordinate heads trained on
CCG card corners.  Input 384×384 RGB, outputs four normalised (x, y) corner
coordinates plus a card-presence logit.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from collector_vision.interfaces import DetectionResult

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 ndarray → (1, 3, size, size) float32, ImageNet-normalised."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x   = rgb.astype(np.float32) / 255.0
    x   = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,H,W)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Reorder four (x,y) points to canonical TL, TR, BR, BL order."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmin(d)],   # TR: smallest x-y
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmax(d)],   # BL: largest x-y
    ], dtype=np.float32)


class NeuralCornerDetector:
    """Reggie — SimCC card corner detector, runs via onnxruntime.

    Parameters
    ----------
    checkpoint:
        Path to the ``.onnx`` file.  The ``.onnx.data`` weight file must sit
        in the same directory.  Defaults to the bundled Reggie weights.
    presence_threshold:
        Minimum sigmoid(presence_logit) to treat a detection as valid.
        Below this the full image is returned as the crop.
    num_threads:
        Number of intra-op threads for onnxruntime.  Defaults to 4 (good
        for Pi; reduce to 1 for profiling).
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        presence_threshold: float = 0.5,
        num_threads: int = 4,
    ) -> None:
        from collector_vision import weights as _w

        if checkpoint is None:
            checkpoint = _w.CORNER_DETECTOR
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Corner detector weights not found: {checkpoint}\n"
                "Install the full package or supply a checkpoint path."
            )

        self._threshold = presence_threshold
        self._sess, self._input_name, self._input_size = self._load(checkpoint, num_threads)

    @staticmethod
    def _load(onnx_path: Path, num_threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        shape = input_meta.shape
        # Shape is (1, 3, H, W); H may be an int or a symbolic string
        input_size = int(shape[2]) if isinstance(shape[2], int) else 384
        return sess, input_name, input_size

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect card corners in a BGR uint8 image.

        Parameters
        ----------
        image:
            BGR uint8 ndarray as returned by ``cv2.imread``.

        Returns
        -------
        DetectionResult with normalised (x, y) corners in TL, TR, BR, BL order.
        ``card_present`` is False when the presence score is below the threshold.
        """
        x = _preprocess(image, self._input_size)
        outs = self._sess.run(None, {self._input_name: x})

        corners_flat = np.clip(outs[0].squeeze(), 0.0, 1.0)  # (8,)
        presence_logit = float(outs[1].squeeze())
        presence = float(1.0 / (1.0 + np.exp(-presence_logit)))  # sigmoid

        corners = _order_corners(corners_flat.reshape(4, 2).astype(np.float32))

        extra: dict = {"presence_logit": presence_logit}
        if len(outs) > 2:
            extra["sharpness"] = float(outs[2].squeeze())

        return DetectionResult(
            corners=corners,
            card_present=presence >= self._threshold,
            confidence=presence,
            extra=extra,
        )

    def __repr__(self) -> str:
        return f"NeuralCornerDetector(input_size={self._input_size}, threshold={self._threshold})"
