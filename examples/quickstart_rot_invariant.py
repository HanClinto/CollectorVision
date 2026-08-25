#!/usr/bin/env python3
"""Identify a Magic: The Gathering card, trying upright and upside-down crops.

Run from the repo root:
    python examples/quickstart_rot_invariant.py

This is a temporary workaround for embedding algorithms that are sensitive to
180-degree rotation. It dewarps the detected card once, embeds both the crop and
the crop rotated 180 degrees, then keeps whichever orientation has the strongest
nearest-neighbor result.
"""

import json
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
from PIL import Image

import collector_vision as cvg

IMAGE = Path("examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")


class OrientationResult(NamedTuple):
    label: str
    hits: list[tuple[float, str]]

    @property
    def best_score(self) -> float:
        return self.hits[0][0]

    @property
    def best_card_id(self) -> str:
        return self.hits[0][1]


def search_best_orientation(
    catalog: cvg.CatalogLike, crop: Image.Image, top_k: int = 5
) -> OrientationResult:
    """Search upright and 180-degree orientations, returning the stronger result."""
    candidates = [("upright", crop), ("rotated_180", cvg.rotate_card_180(crop))]
    embeddings = catalog.embedder.embed([image for _, image in candidates])

    results = [
        OrientationResult(label=label, hits=catalog.search(embedding, top_k=top_k))
        for (label, _), embedding in zip(candidates, embeddings)
    ]
    return max(results, key=lambda result: result.best_score)


def main() -> None:
    # Catalog assets are downloaded once and then updated from the local cache.
    catalog = cvg.Catalog.load("mtg")

    image = cv2.imread(str(IMAGE))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE}")

    detector = cvg.NeuralCornerDetector()
    detection = detector.detect(image)
    print(f"Detected corner sharpness={detection.sharpness:.3f}")

    crop = detection.dewarp(image)

    result = search_best_orientation(catalog, crop, top_k=5)
    score, card_id = result.hits[0]

    print(f"Best orientation {result.label}")
    print(f"Top match {card_id}  score={score:.4f}")
    for s, cid in result.hits[1:]:
        print(f"          {cid}  score={s:.4f}")

    # 7. Metadata lookup via Scryfall API (optional)
    #    Since the catalog only contains the card's Scryfall ID, if we want any more data (like its name, set, color, flavor text, or price)
    #    then we need to look it up from another data source -- in this case, Scryfall's public API.
    req = urllib.request.Request(
        f"https://api.scryfall.com/cards/{card_id}",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        card = json.loads(r.read())

    print(f"Name      {card['name']}")
    print(f"Set       {card['set_name']} ({card['set'].upper()})")
    usd = card.get("prices", {}).get("usd")
    print(f"Price     {'$' + usd if usd else 'n/a'}")


if __name__ == "__main__":
    main()
