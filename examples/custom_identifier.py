#!/usr/bin/env python3
"""Custom identifier example — swap out the embedding step.

This example shows how to replace Milo (the built-in neural embedder) with
a perceptual hash algorithm.  The corner detection step (Cornelius) and the
pipeline structure stay identical; only the embedding and search differ.

This is useful if you want:
  - Zero ML-framework dependencies (imagehash is pure Python + Pillow)
  - Faster embedding at the cost of some accuracy
  - A starting point for plugging in any other embedding function

Prerequisites
-------------
    pip install imagehash

Building a hash catalog is left as an exercise — the structure is:
    embeddings: (N, 32) uint8   packed bit vectors (256-bit phash)
    card_ids:   (N,)    str
    source:     str
    np.savez("my_phash_catalog.npz", embeddings=..., card_ids=..., source=...)

Usage
-----
    python examples/custom_identifier.py <image.jpg> --catalog my_phash_catalog.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Hamming search — 5 lines, no library needed
# ---------------------------------------------------------------------------

def hamming_search(
    query: np.ndarray,
    catalog: np.ndarray,
    top_k: int = 5,
) -> list[tuple[float, int]]:
    """Return top-k (similarity, index) pairs from a packed-bit catalog."""
    n_bits = query.shape[0] * 8
    xor = np.bitwise_xor(catalog, query[np.newaxis, :])
    distances = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
    scores = 1.0 - distances / n_bits
    if top_k >= len(scores):
        idxs = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        idxs = part[np.argsort(scores[part])[::-1]]
    return [(float(scores[i]), int(i)) for i in idxs]


# ---------------------------------------------------------------------------
# pHash embedder — wraps imagehash.phash as a plain function
# ---------------------------------------------------------------------------

def phash_embed(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Return a (hash_size²/8,) uint8 packed bit vector."""
    import imagehash
    bits = imagehash.phash(image.convert("RGB"), hash_size=hash_size).hash.flatten()
    padded = np.zeros((hash_size * hash_size + 7) // 8 * 8, dtype=np.uint8)
    padded[: len(bits)] = bits.astype(np.uint8)
    return np.packbits(padded)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def identify(image_path: str, catalog_path: str, top_k: int = 5) -> None:
    import cv2
    import collector_vision as cvg

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Could not read {image_path}", file=sys.stderr)
        return

    # Step 1 — detect corners (Cornelius, unchanged)
    detector = cvg.NeuralCornerDetector()
    detection = detector.detect(bgr)
    if not detection.card_present:
        print("No card detected.")
        return

    # Step 2 — dewarp to aligned crop (unchanged)
    crop = detection.dewarp(bgr)

    # Step 3 — embed with pHash instead of Milo
    query = phash_embed(crop)

    # Step 4 — load catalog and search with Hamming distance
    data = np.load(catalog_path, allow_pickle=False)
    embeddings = data["embeddings"]   # (N, B) uint8
    card_ids = data["card_ids"].tolist()

    hits = hamming_search(query, embeddings, top_k=top_k)

    print(f"Top {top_k} matches:")
    for rank, (score, idx) in enumerate(hits, 1):
        print(f"  {rank}. {card_ids[idx]}  (similarity={score:.3f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Image files to identify")
    parser.add_argument("--catalog", required=True, help="Path to phash NPZ catalog")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    for path in args.images:
        print(f"\n=== {path} ===")
        identify(path, args.catalog, top_k=args.top_k)


if __name__ == "__main__":
    main()
