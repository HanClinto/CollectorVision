"""NeuralCornerDetector — ONNX-based learned card corner detector (Cornelius).

Runs entirely on CPU via onnxruntime — no PyTorch dependency required.

Architecture: MobileViT-XXS backbone + SimCC coordinate heads trained on
CCG card corners.  Input 384×384 RGB, outputs four normalised (x, y) corner
coordinates, a card-presence logit, and a SimCC sharpness scalar.

Card presence heuristic
-----------------------
For Cornelius (SimCC architecture) the exported ONNX model includes a *sharpness*
output — the mean peak of the eight softmax coordinate distributions (4 corners
× 2 axes).  A high peak means the model has a sharp, confident prediction for
each axis; a low peak means the distributions are flat (no card in view).

In practice, the raw presence logit is unreliable: it fires strongly even on
blank images.  Sharpness is a much better gate.  When the model emits a
sharpness output, ``card_present`` is determined by ``sharpness >= min_sharpness``
and the presence logit is recorded in ``extra`` for diagnostics only.

When the model does not emit sharpness (older checkpoints or non-SimCC models),
the detector falls back to ``sigmoid(presence_logit) >= presence_threshold``.
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
    """Cornelius — SimCC card corner detector, runs via onnxruntime.

    Parameters
    ----------
    checkpoint:
        Path to the ``.onnx`` file.  The ``.onnx.data`` weight file must sit
        in the same directory.  Defaults to the bundled Cornelius weights.
    min_sharpness:
        Minimum SimCC mean-peak sharpness to treat a detection as valid.
        Range [0, 1]; 0.0 (default) disables the gate — all frames are
        treated as card-present regardless of sharpness.  Only used when
        the model emits a sharpness output (Cornelius does).

        Tune this for card-in-scene video pipelines where you want to skip
        frames with no card visible.  For Cornelius, blank/no-card frames
        typically score ≤ 0.01; valid card frames typically score 0.03–0.07.
        A value around 0.02 cleanly separates the two populations.
    presence_threshold:
        Fallback gate used when the model does not emit a sharpness output.
        Minimum ``sigmoid(presence_logit)`` to treat a detection as valid.
    num_threads:
        Number of intra-op threads for onnxruntime.  Defaults to 4 (good
        for Pi; reduce to 1 for profiling).
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        min_sharpness: float = 0.0,
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

        self._min_sharpness = min_sharpness
        self._presence_threshold = presence_threshold
        self._sess, self._input_name, self._input_size, self._has_sharpness = (
            self._load(checkpoint, num_threads)
        )

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
        input_size = int(shape[2]) if isinstance(shape[2], int) else 384
        out_names = {o.name for o in sess.get_outputs()}
        has_sharpness = "sharpness" in out_names
        return sess, input_name, input_size, has_sharpness

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect card corners in a BGR uint8 image.

        Parameters
        ----------
        image:
            BGR uint8 ndarray as returned by ``cv2.imread``.

        Returns
        -------
        DetectionResult with normalised (x, y) corners in TL, TR, BR, BL order.
        ``card_present`` is False when sharpness (or presence) is below threshold.
        ``extra`` contains ``sharpness`` and ``presence`` for diagnostics.
        """
        x = _preprocess(image, self._input_size)
        outs = self._sess.run(None, {self._input_name: x})

        corners_flat = np.clip(outs[0].squeeze(), 0.0, 1.0)  # (8,)
        presence_logit = float(outs[1].squeeze())
        presence = float(1.0 / (1.0 + np.exp(-presence_logit)))  # sigmoid

        extra: dict = {"presence": presence}

        if self._has_sharpness:
            sharpness = float(outs[2].squeeze())
            extra["sharpness"] = sharpness
            card_present = sharpness >= self._min_sharpness
            confidence = sharpness
        else:
            # Fallback for models without a sharpness output
            card_present = presence >= self._presence_threshold
            confidence = presence

        corners = _order_corners(corners_flat.reshape(4, 2).astype(np.float32))

        return DetectionResult(
            corners=corners,
            card_present=card_present,
            confidence=confidence,
            extra=extra,
        )

    def __repr__(self) -> str:
        gate = (
            f"min_sharpness={self._min_sharpness}"
            if self._has_sharpness
            else f"presence_threshold={self._presence_threshold}"
        )
        return f"NeuralCornerDetector(input_size={self._input_size}, {gate})"
