"""NeuralEmbedder — ONNX-based ArcFace card embedder (Milo).

Runs entirely on CPU via onnxruntime — no PyTorch dependency required.

Architecture: MobileViT-XXS backbone + shared linear projection trained with
multi-task ArcFace loss (illustration_id + set_code).  Input 448×448 RGB,
outputs a L2-normalised 128-d float32 embedding.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_pil(img: Image.Image, size: int) -> np.ndarray:
    """PIL Image (any mode) → (1, 3, size, size) float32, ImageNet-normalised."""
    rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x   = np.array(rgb, dtype=np.float32) / 255.0
    x   = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,H,W)


class NeuralEmbedder:
    """Milo — ArcFace MobileViT-XXS card embedder, runs via onnxruntime.

    Parameters
    ----------
    checkpoint:
        Path to the ``.onnx`` file.  The ``.onnx.data`` weight file must sit
        in the same directory.  Defaults to the bundled Milo weights.
    batch_size:
        Images to process per ONNX session call.  The default (1) keeps
        latency low for single-image use; increase for throughput-oriented
        gallery building (though gallery building typically runs in the
        CollectorVision-Pipeline project, not here).
    num_threads:
        Number of intra-op threads for onnxruntime.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        batch_size: int = 1,
        num_threads: int = 4,
    ) -> None:
        from collector_vision import weights as _w

        if checkpoint is None:
            checkpoint = _w.EMBEDDER
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Embedder weights not found: {checkpoint}\n"
                "Install the full package or supply a checkpoint path."
            )

        self._batch_size = batch_size
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
        input_size = int(shape[2]) if isinstance(shape[2], int) else 448
        return sess, input_name, input_size

    def embed(self, images: "Image.Image | list[Image.Image]") -> np.ndarray:
        """Embed one or more PIL Images.

        Parameters
        ----------
        images:
            A single PIL Image or a list of PIL Images (any mode; converted
            to RGB internally).

        Returns
        -------
        Single image: (128,) float32 vector.
        List:         (n, 128) float32 array of L2-normalised embeddings.
        """
        single = isinstance(images, Image.Image)
        if single:
            images = [images]

        if not images:
            return np.zeros((0, 128), dtype=np.float32)

        all_embs: list[np.ndarray] = []

        for i in range(0, len(images), self._batch_size):
            batch = images[i : i + self._batch_size]
            # ONNX model was exported for batch_size=1; run each image separately
            # and stack the results.
            for img in batch:
                x   = _preprocess_pil(img, self._input_size)
                out = self._sess.run(None, {self._input_name: x})[0]
                emb = out.squeeze().astype(np.float32)  # (128,)
                norm = float(np.linalg.norm(emb))
                if norm > 1e-8:
                    emb = emb / norm
                all_embs.append(emb)

        result = np.stack(all_embs, axis=0)  # (n, 128)
        return result[0] if single else result

    def __repr__(self) -> str:
        return f"NeuralEmbedder(input_size={self._input_size})"
