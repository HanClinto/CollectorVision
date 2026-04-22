"""HashEmbedder — perceptual-hash based card embedder.

Produces packed uint8 bit vectors for Hamming-distance retrieval.
The Catalog must have been built with the same algorithm and size.

Convenience constructors::

    from collector_vision.embedders import HashEmbedder

    # DCT perceptual hash (fast, good on clean crops)
    embedder = HashEmbedder.phash(hash_size=16)

    # Marr-Hildreth LoG hash (more robust to spatial noise)
    embedder = HashEmbedder.marr_hildreth(hash_size=32, sigma=2.5)

    # Bring your own: any callable (PIL.Image → bool array of length *bits*)
    embedder = HashEmbedder(hash_fn=my_fn, bits=256, algo_key="my_algo")
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image


# Type alias: (PIL.Image) → bool ndarray of length *bits*
HashFn = Callable[[Image.Image], np.ndarray]


class HashEmbedder:
    """Wraps any perceptual hash function as an Embedder.

    Parameters
    ----------
    hash_fn:
        Callable that accepts a PIL.Image and returns a bool numpy array of
        exactly *bits* elements.
    bits:
        Number of bits produced by hash_fn.  Must match the catalog.
    algo_key:
        Stable string identifier (used in cache filenames and Catalog
        metadata).  Change it whenever hash_fn behaviour changes.
    """

    def __init__(self, hash_fn: HashFn, bits: int, algo_key: str) -> None:
        self._hash_fn = hash_fn
        self._bits = bits
        self._bytes = (bits + 7) // 8
        self.algo_key = algo_key

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def phash(cls, hash_size: int = 16) -> "HashEmbedder":
        """DCT-based perceptual hash.  hash_size² bits total."""
        import imagehash

        def _fn(img: Image.Image) -> np.ndarray:
            return imagehash.phash(img, hash_size=hash_size).hash.flatten().astype(bool)

        return cls(_fn, bits=hash_size * hash_size, algo_key=f"phash_{hash_size}")

    @classmethod
    def dhash(cls, hash_size: int = 16) -> "HashEmbedder":
        """Gradient difference hash."""
        import imagehash

        def _fn(img: Image.Image) -> np.ndarray:
            return imagehash.dhash(img, hash_size=hash_size).hash.flatten().astype(bool)

        return cls(_fn, bits=hash_size * hash_size, algo_key=f"dhash_{hash_size}")

    @classmethod
    def whash(cls, hash_size: int = 16) -> "HashEmbedder":
        """Haar wavelet hash."""
        import imagehash

        def _fn(img: Image.Image) -> np.ndarray:
            return imagehash.whash(img, hash_size=hash_size).hash.flatten().astype(bool)

        return cls(_fn, bits=hash_size * hash_size, algo_key=f"whash_{hash_size}")

    @classmethod
    def marr_hildreth(cls, hash_size: int = 32, sigma: float = 2.5) -> "HashEmbedder":
        """Marr-Hildreth (Laplacian-of-Gaussian) hash.

        More robust to small spatial misalignments than DCT-based hashes;
        larger sigma → coarser edges → greater noise tolerance.
        """
        from scipy.ndimage import gaussian_laplace

        sig_key = f"{sigma:.1f}".replace(".", "p")

        def _fn(img: Image.Image) -> np.ndarray:
            big = hash_size * 4
            gray = np.array(
                img.convert("L").resize((big, big), Image.LANCZOS), dtype=np.float32
            )
            log = gaussian_laplace(gray, sigma=sigma)
            log_ds = log.reshape(hash_size, 4, hash_size, 4).mean(axis=(1, 3))
            return log_ds.flatten() > 0

        return cls(_fn, bits=hash_size * hash_size,
                   algo_key=f"marr_hildreth_{hash_size}_s{sig_key}")

    # ------------------------------------------------------------------
    # Embedder protocol
    # ------------------------------------------------------------------

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Hash a list of PIL Images.  Returns (n, bytes) uint8 array."""
        out = np.zeros((len(images), self._bytes), dtype=np.uint8)
        for i, img in enumerate(images):
            try:
                bits_arr = self._hash_fn(img.convert("RGB"))
                padded = np.zeros(self._bytes * 8, dtype=np.uint8)
                padded[:len(bits_arr)] = bits_arr.astype(np.uint8)
                out[i] = np.packbits(padded)
            except Exception:
                pass  # leaves row as zeros
        return out

    def __repr__(self) -> str:
        return f"HashEmbedder(algo_key={self.algo_key!r}, bits={self._bits})"
