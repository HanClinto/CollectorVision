#!/usr/bin/env python3
"""Measure top-1 / top-k identification accuracy on a labelled image directory.

Filenames must embed the Scryfall UUID:
    {scryfall_uuid}_{anything}.jpg

Runs the full pipeline on every image, compares the returned card_id
against the UUID in the filename, and reports detection rate and accuracy.

Usage
-----
    python examples/eval_accuracy.py <image_dir> --catalog hf://HanClinto/milo/scryfall-mtg
    python examples/eval_accuracy.py <image_dir> --catalog ./my_catalog.npz --top-k 5
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import collector_vision as cvg

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-sharpness", type=float, default=0.02)
    args = parser.parse_args()

    images = [p for p in sorted(args.image_dir.glob("*.jpg"))
              if UUID_RE.search(p.name)]
    if not images:
        sys.exit(f"No UUID-labelled images found in {args.image_dir}")

    catalog = cvg.Catalog.load(args.catalog)
    detector = cvg.NeuralCornerDetector()
    catalog_ids = set(catalog.card_ids)

    detected = top1 = topk = total = 0

    for img_path in images:
        true_id = UUID_RE.search(img_path.name).group(0).lower()
        if true_id not in catalog_ids:
            continue
        total += 1

        bgr = cv2.imread(str(img_path))
        detection = detector.detect(bgr, min_sharpness=args.min_sharpness)
        if not detection.card_present:
            continue
        detected += 1

        crop = detection.dewarp(bgr)
        emb = catalog.embedder.embed([crop])[0]
        hits = [cid for _, cid in catalog.search(emb, top_k=args.top_k)]

        if hits[0] == true_id:
            top1 += 1
        if true_id in hits:
            topk += 1

    def pct(n, d): return f"{100*n/d:.1f}%" if d else "n/a"

    print(f"Dataset:    {args.image_dir.name}  ({total} images in catalog)")
    print(f"Detected:   {detected}/{total}  ({pct(detected, total)})")
    print(f"Top-1:      {top1}/{detected}  ({pct(top1, detected)})")
    print(f"Top-{args.top_k}:      {topk}/{detected}  ({pct(topk, detected)})")


if __name__ == "__main__":
    main()
