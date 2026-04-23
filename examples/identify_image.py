#!/usr/bin/env python3
"""Identify a Magic: The Gathering card from a photo.

Usage
-----
    python examples/identify_image.py
    python examples/identify_image.py path/to/card.jpg
    python examples/identify_image.py path/to/card.jpg --catalog ./milo1-scryfall-mtg-2026-04.npz
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import collector_vision as cvg

SAMPLE_IMAGE = Path("examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=SAMPLE_IMAGE,
                        help="Card image to identify (default: sample Scrying Glass)")
    parser.add_argument("--catalog", metavar="PATH",
                        help="Local .npz or hf:// URI (default: auto-download from HuggingFace)")
    args = parser.parse_args()

    # 1. Load catalog
    catalog = cvg.Catalog.load(args.catalog or "hf://HanClinto/milo/scryfall-mtg")

    # 2. Detect card corners
    image = cv2.imread(str(args.image))
    if image is None:
        sys.exit(f"Could not read image: {args.image}")

    detector = cvg.NeuralCornerDetector()
    detection = detector.detect(image)

    if not detection.card_present:
        sys.exit("No card detected in image.")

    print(f"Detected  sharpness={detection.extra['sharpness']:.3f}")

    # 3. Dewarp to aligned crop
    crop = detection.dewarp(image)

    # 4. Embed
    emb = catalog.embedder.embed(crop)

    # 5. Search
    hits = catalog.search(emb, top_k=5)
    score, card_id = hits[0]
    print(f"Top match {card_id}  score={score:.4f}")

    if len(hits) > 1:
        for s, cid in hits[1:]:
            print(f"          {cid}  score={s:.4f}")

    # 6. Metadata lookup via Scryfall API
    try:
        with urllib.request.urlopen(f"https://api.scryfall.com/cards/{card_id}", timeout=10) as r:
            card = json.loads(r.read())
        print(f"Name      {card['name']}")
        print(f"Set       {card['set_name']} ({card['set'].upper()})")
        usd = card.get("prices", {}).get("usd")
        print(f"Price     {'$' + usd if usd else 'n/a'}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
