#!/usr/bin/env python3
"""Identify a card from a photo — explicit pipeline walkthrough.

Shows every step:
  1. Load image
  2. Detect card corners (NeuralCornerDetector / Cornelius)
  3. Dewarp to aligned crop
  4. Embed (NeuralEmbedder / Milo)
  5. Nearest-neighbour search against the catalog
  6. Metadata lookup via Scryfall API

Usage
-----
    python examples/identify_image.py <image.jpg> [image2.jpg ...]

    # Use a local catalog file instead of downloading
    python examples/identify_image.py --catalog ./milo1-scryfall-mtg-2026-04.npz <image.jpg>

Multiple images of the same physical card are treated as frames — scores are
summed across frames before ranking.

Smoke-test (no card image required)
------------------------------------
    python examples/identify_image.py --smoke-test
    python examples/identify_image.py --smoke-test --catalog ./milo1-scryfall-mtg-2026-04.npz
"""
import argparse
import json
import sys
import urllib.request

import cv2
import numpy as np

import collector_vision as cvg


def load_catalog(args) -> cvg.Catalog:
    if args.catalog:
        return cvg.Catalog.load(args.catalog)
    return cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")


def lookup_scryfall(scryfall_id: str) -> dict:
    url = f"https://api.scryfall.com/cards/{scryfall_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def identify_images(image_paths: list[str], catalog: cvg.Catalog) -> None:
    detector = cvg.NeuralCornerDetector()

    # ── Steps 1–3: load, detect, dewarp ──────────────────────────────────────
    crops = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read image: {path}", file=sys.stderr)
            sys.exit(1)

        detection = detector.detect(image)  # default min_sharpness=0.02
        sharpness = detection.extra.get("sharpness", 0.0)
        print(f"  {path}: sharpness={sharpness:.3f}  card_present={detection.card_present}")

        if not detection.card_present:
            print(f"  (skipping — no card detected)")
            continue

        crops.append(detection.dewarp(image))   # PIL Image, 252×352 px

    if not crops:
        print("No card detected in any frame.")
        sys.exit(1)

    # ── Step 4: embed ─────────────────────────────────────────────────────────
    embeddings = catalog.embedder.embed(crops)   # (n_frames, 128) float32

    # ── Step 5: nearest-neighbour search, aggregate across frames ────────────
    if len(crops) == 1:
        hits = catalog.search(embeddings[0], top_k=5)
    else:
        from collections import defaultdict
        score_map: dict[str, float] = defaultdict(float)
        for emb in embeddings:
            for score, card_id in catalog.search(emb, top_k=5):
                score_map[card_id] += score
        hits = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:5]
        hits = [(score, card_id) for card_id, score in hits]

    best_score, best_id = hits[0]
    print(f"\nTop match   {best_id}  confidence={best_score:.4f}")

    if len(hits) > 1:
        print("Alternatives:")
        for score, card_id in hits[1:4]:
            print(f"  {card_id}  confidence={score:.4f}")

    # ── Step 6: metadata lookup ───────────────────────────────────────────────
    try:
        card = lookup_scryfall(best_id)
        print(f"Name        {card['name']}")
        print(f"Set         {card['set_name']} ({card['set'].upper()})")
        usd = card.get("prices", {}).get("usd")
        print(f"Price (USD) {'$' + usd if usd else 'n/a'}")
    except Exception as exc:
        print(f"Scryfall lookup failed: {exc}")


def smoke_test(catalog: cvg.Catalog) -> None:
    print(f"Catalog: {catalog}")
    assert len(catalog) > 0, "Catalog is empty"
    assert catalog.card_ids[0], "First card_id is blank"

    detector = cvg.NeuralCornerDetector()
    blank = np.zeros((800, 600, 3), dtype=np.uint8)

    detection = detector.detect(blank)
    assert isinstance(detection.card_present, bool)
    assert isinstance(detection.extra.get("sharpness", 0.0), float)

    from PIL import Image
    crop = Image.fromarray(blank[..., ::-1])  # skip dewarp on blank — no corners

    emb = catalog.embedder.embed(crop)
    assert emb.shape == (128,), f"Expected (128,), got {emb.shape}"

    hits = catalog.search(emb, top_k=3)
    assert len(hits) == 3
    assert all(isinstance(score, float) and isinstance(cid, str) for score, cid in hits)

    print(f"Top hit: {hits[0][1]}  score={hits[0][0]:.4f}")
    print("Smoke test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", metavar="IMAGE")
    parser.add_argument("--catalog", metavar="PATH",
                        help="Local .npz catalog file (default: auto-download from HuggingFace)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a headless pipeline check instead of identifying a card")
    args = parser.parse_args()

    catalog = load_catalog(args)

    if args.smoke_test:
        smoke_test(catalog)
    elif args.images:
        identify_images(args.images, catalog)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
