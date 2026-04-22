#!/usr/bin/env python3
"""Custom identifier example — swap out either pipeline component.

The CollectorVision pipeline has two pluggable steps:

  1. Corner detector  — anything with a detect(bgr) -> DetectionResult signature
  2. Embedder         — anything that produces vectors you can search against

This file shows both swaps:

  --detector canny   replaces Cornelius with a plain Canny contour detector
                     (no ML weights; works well on clean/high-contrast backgrounds)

  --embedder phash   replaces Milo with a perceptual hash (requires: pip install imagehash)
                     (no ONNX runtime; faster but weaker on edition identification)

Building a hash catalog is left as an exercise — the NPZ structure is:
    embeddings: (N, 32) uint8   packed 256-bit phash vectors
    card_ids:   (N,)    str

Usage
-----
    python examples/custom_identifier.py image.jpg --catalog my_catalog.npz
    python examples/custom_identifier.py image.jpg --catalog my_catalog.npz --detector canny
    python examples/custom_identifier.py image.jpg --catalog my_catalog.npz --embedder phash
"""
import argparse
import sys

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Custom detector — Canny contour, no ML weights required
# ---------------------------------------------------------------------------

def detect_canny(bgr: np.ndarray):
    """Return a DetectionResult using Canny edge contours.

    Works well on cards against clean or high-contrast backgrounds.
    Falls back to card_present=False when no clear rectangle is found.
    """
    from collector_vision.interfaces import DetectionResult

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < 0.05 * h * w:
            break
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
        ordered = np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                            pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)
        return DetectionResult(corners=ordered / [w, h], card_present=True,
                               confidence=area / (h * w))

    return DetectionResult(corners=None, card_present=False, confidence=0.0)


# ---------------------------------------------------------------------------
# Custom embedder — pHash, no ONNX runtime required
# ---------------------------------------------------------------------------

def phash_embed(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Return a packed uint8 bit vector using DCT perceptual hash."""
    import imagehash
    bits = imagehash.phash(image.convert("RGB"), hash_size=hash_size).hash.flatten()
    padded = np.zeros((hash_size * hash_size + 7) // 8 * 8, dtype=np.uint8)
    padded[: len(bits)] = bits.astype(np.uint8)
    return np.packbits(padded)


def hamming_search(query: np.ndarray, embeddings: np.ndarray, top_k: int) -> list[tuple[float, int]]:
    n_bits = query.shape[0] * 8
    scores = 1.0 - np.unpackbits(np.bitwise_xor(embeddings, query), axis=1).sum(axis=1) / n_bits
    if top_k >= len(scores):
        idxs = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        idxs = part[np.argsort(scores[part])[::-1]]
    return [(float(scores[i]), int(i)) for i in idxs]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def identify(image_path: str, catalog_path: str, use_canny: bool, use_phash: bool, top_k: int) -> None:
    import collector_vision as cvg

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Could not read {image_path}", file=sys.stderr)
        return

    # Step 1 — detect corners
    if use_canny:
        detection = detect_canny(bgr)
    else:
        detection = cvg.NeuralCornerDetector().detect(bgr)

    if not detection.card_present:
        print("No card detected.")
        return

    # Step 2 — dewarp
    crop = detection.dewarp(bgr)

    # Step 3 — embed
    data = np.load(catalog_path, allow_pickle=False)
    card_ids = data["card_ids"].tolist()

    if use_phash:
        query = phash_embed(crop)
        hits = hamming_search(query, data["embeddings"], top_k)
    else:
        catalog = cvg.Catalog.load(catalog_path)
        query = catalog.embedder.embed([crop])[0]
        hits = [(score, catalog.card_ids.index(cid)) for score, cid in catalog.search(query, top_k)]
        card_ids = catalog.card_ids

    print(f"Top {top_k} matches:")
    for rank, (score, idx) in enumerate(hits, 1):
        print(f"  {rank}. {card_ids[idx]}  (score={score:.3f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Image files to identify")
    parser.add_argument("--catalog", required=True, help="Path to NPZ catalog file")
    parser.add_argument("--detector", choices=["cornelius", "canny"], default="cornelius")
    parser.add_argument("--embedder", choices=["milo", "phash"], default="milo")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    for path in args.images:
        print(f"\n=== {path} ===")
        identify(path, args.catalog,
                 use_canny=(args.detector == "canny"),
                 use_phash=(args.embedder == "phash"),
                 top_k=args.top_k)


if __name__ == "__main__":
    main()
