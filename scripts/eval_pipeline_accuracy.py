#!/usr/bin/env python3
"""Evaluate full pipeline accuracy on a directory of labelled card images.

Card UUID is extracted from the filename using the standard naming convention:
    {scryfall_uuid}_{anything}.jpg

For each image the script runs:
    detect → (skip if no card) → dewarp → embed → catalog.search()

Then reports:
    - Detection rate   — fraction of images where a card was found
    - Top-1 accuracy   — exact UUID match in first result
    - Top-3 accuracy   — exact UUID match in top-3 results
    - Top-1 name match — any result shares the same card name (artwork-level)

Usage
-----
    python scripts/eval_pipeline_accuracy.py <image_dir> --catalog hf://HanClinto/milo/scryfall-mtg
    python scripts/eval_pipeline_accuracy.py <image_dir> --catalog ./my_catalog.npz
    python scripts/eval_pipeline_accuracy.py <image_dir> --catalog hf://... --min-sharpness 0.01
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


def extract_uuid(filename: str) -> str | None:
    m = UUID_RE.search(filename)
    return m.group(0).lower() if m else None


def run(image_dir: Path, catalog_source: str, min_sharpness: float, top_k: int) -> None:
    import cv2
    import collector_vision as cvg

    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    images = [p for p in images if extract_uuid(p.name)]
    if not images:
        print(f"No labelled images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading catalog: {catalog_source}")
    catalog = cvg.Catalog.load(catalog_source)
    catalog_ids = set(catalog.card_ids)
    print(f"Catalog: {len(catalog)} cards  ({catalog.source})")

    detector = cvg.NeuralCornerDetector()

    n_total = len(images)
    n_detected = 0
    n_in_catalog = 0
    top1_exact = 0
    top3_exact = 0

    misses_by_card: dict[str, int] = defaultdict(int)
    errors_by_card: dict[str, int] = defaultdict(int)

    for img_path in images:
        true_id = extract_uuid(img_path.name)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        detection = detector.detect(bgr, min_sharpness=min_sharpness)
        if not detection.card_present:
            misses_by_card[true_id] += 1
            continue

        n_detected += 1

        if true_id not in catalog_ids:
            continue
        n_in_catalog += 1

        crop = detection.dewarp(bgr)
        emb = catalog.embedder.embed([crop])[0]
        hits = catalog.search(emb, top_k=top_k)
        result_ids = [cid for _, cid in hits]

        if result_ids[0] == true_id:
            top1_exact += 1
        if true_id in result_ids:
            top3_exact += 1
        else:
            errors_by_card[true_id] += 1

    # Summary
    print(f"\n{'─'*50}")
    print(f"Dataset:          {image_dir.name}")
    print(f"Total images:     {n_total}")
    print(f"Detected:         {n_detected}/{n_total}  ({pct(n_detected, n_total)})")
    print(f"In catalog:       {n_in_catalog}/{n_detected}  ({pct(n_in_catalog, n_detected)})")
    print(f"Top-1 exact:      {top1_exact}/{n_in_catalog}  ({pct(top1_exact, n_in_catalog)})")
    print(f"Top-{top_k} exact:      {top3_exact}/{n_in_catalog}  ({pct(top3_exact, n_in_catalog)})")

    if misses_by_card:
        total_missed = sum(misses_by_card.values())
        print(f"\nDetection misses: {total_missed} frames across {len(misses_by_card)} cards")

    if errors_by_card:
        print(f"\nID errors (top-{top_k} miss) by card ({len(errors_by_card)} cards):")
        for cid, count in sorted(errors_by_card.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cid}  {count} frames wrong")


def pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--catalog", required=True,
                        help="hf://user/repo/key or path to .npz")
    parser.add_argument("--min-sharpness", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    run(args.image_dir, args.catalog, args.min_sharpness, args.top_k)


if __name__ == "__main__":
    main()
