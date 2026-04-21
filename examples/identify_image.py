#!/usr/bin/env python3
"""Minimal CollectorVision example — identify a single card image.

Usage
-----
    python examples/identify_image.py <gallery.npz> <image.jpg> [image2.jpg ...]

Download a gallery from https://huggingface.co/datasets/CollectorVision/galleries
or point at a local .npz file you built yourself.

Multiple images of the same physical card are treated as frames and votes
are summed before ranking, giving a more confident result.
"""
import json
import sys
import urllib.request
from pathlib import Path

import collector_vision as cvg


def lookup_scryfall(scryfall_id: str) -> dict:
    """Fetch card metadata from the Scryfall API."""
    url = f"https://api.scryfall.com/cards/{scryfall_id}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    gallery_path = sys.argv[1]
    image_paths  = sys.argv[2:]

    # One Identifier per process — lazy-loads gallery and detector on first call
    cvid = cvg.Identifier(gallery_path)

    result = cvid.identify(*image_paths)

    print(f"Top match  : confidence={result.confidence:.4f}")
    print(f"IDs        : {result.ids}")

    # Look up human-readable metadata from Scryfall (if a scryfall_id is present)
    sfid = result.ids.get("scryfall_id")
    if sfid:
        try:
            card = lookup_scryfall(sfid)
            print(f"Card name  : {card['name']}")
            print(f"Set        : {card['set_name']} ({card['set']})")
            usd = card.get("prices", {}).get("usd")
            print(f"Price (USD): {'$' + usd if usd else 'n/a'}")
        except Exception as exc:
            print(f"Scryfall lookup failed: {exc}")

    if result.alternatives:
        print("\nAlternatives:")
        for alt in result.alternatives[:3]:
            print(f"  {alt.ids}  confidence={alt.confidence:.4f}")

    if result.frame_results:
        print(f"\nPer-frame results ({len(result.frame_results)} frames):")
        for i, fr in enumerate(result.frame_results):
            print(f"  frame {i}: {fr.ids.get('scryfall_id', '?')[:8]}...  confidence={fr.confidence:.4f}")


if __name__ == "__main__":
    main()
