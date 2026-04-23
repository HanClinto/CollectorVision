#!/usr/bin/env python3
"""Precompute actual pipeline values from the sample image for hardcoding into animation.

Run from the collector_vision project root:
    python docs/animation/precompute.py

Outputs: docs/animation/precomputed.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    sample = Path(__file__).parent.parent.parent / "examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg"
    if not sample.exists():
        print(f"Sample image not found: {sample}")
        sys.exit(1)

    bgr = cv2.imread(str(sample))
    h, w = bgr.shape[:2]
    print(f"Image size: {w}×{h}")

    import collector_vision as cvg
    from collector_vision.catalog import Catalog

    # --- Corner detection ---
    detector = cvg.NeuralCornerDetector()
    detection = detector.detect(bgr)
    print(f"card_present: {detection.card_present}")
    print(f"sharpness:    {detection.sharpness:.4f}")
    print(f"corners (normalized):")
    for label, corner in zip(["TL", "TR", "BR", "BL"], detection.corners):
        print(f"  {label}: ({corner[0]:.4f}, {corner[1]:.4f})")

    # --- Dewarp ---
    crop = detection.dewarp(bgr)

    # --- Embed ---
    catalog_path = Path(__file__).parent.parent.parent / "examples"
    # Find any npz catalog nearby
    npz_paths = list(catalog_path.glob("**/*.npz")) + list(Path(".").glob("*.npz"))
    if not npz_paths:
        print("No catalog NPZ found; skipping embed step. Run with a catalog.")
        embedding_values = None
        top_hits = None
    else:
        catalog = Catalog.load(npz_paths[0])
        emb = catalog.embedder.embed([crop])[0]
        embedding_values = emb.tolist()
        hits = catalog.search(emb, top_k=5)
        top_hits = [(float(score), cid) for score, cid in hits]
        print(f"\nTop-5 hits:")
        for i, (score, cid) in enumerate(top_hits):
            print(f"  {i+1}. {cid}  score={score:.4f}")

    result = {
        "image_size": [w, h],
        "corners_normalized": detection.corners.tolist(),  # [[x,y], ...] TL TR BR BL
        "sharpness": float(detection.sharpness) if detection.sharpness is not None else None,
        "confidence": float(detection.confidence),
        "card_present": bool(detection.card_present),
        "dewarp_size": [252, 352],
        "embedding": embedding_values,
        "top_hits": top_hits,
    }

    out = Path(__file__).parent / "precomputed.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
