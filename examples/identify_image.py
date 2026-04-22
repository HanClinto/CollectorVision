#!/usr/bin/env python3
"""Identify a card from one or more images.

Usage
-----
    # Local gallery file
    python examples/identify_image.py <gallery.npz> <image.jpg> [image2.jpg ...]

    # Auto-download from HuggingFace
    python examples/identify_image.py --hf scryfall-mtg <image.jpg>

Multiple images of the same physical card are treated as frames — scores are
summed across frames before ranking, giving a more confident result.

Smoke-test mode (no card images required)
------------------------------------------
    python examples/identify_image.py --smoke-test <gallery.npz>

Runs the full pipeline on a blank image and checks the result structure.
Useful for verifying the install and that the gallery loads correctly.
"""
import json
import sys
import urllib.request


def lookup_scryfall(scryfall_id: str) -> dict:
    url = f"https://api.scryfall.com/cards/{scryfall_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def smoke_test(gallery_path: str) -> None:
    import numpy as np
    import collector_vision as cvg

    print(f"Loading gallery: {gallery_path}")
    gallery = cvg.Gallery.load(gallery_path)
    print(f"  {gallery}")
    assert len(gallery) > 0, "Gallery is empty"
    assert gallery.card_ids[0], "First card_id is blank"

    print("Running full pipeline on a blank image ...")
    cvid = cvg.Identifier(gallery)

    blank = np.zeros((800, 600, 3), dtype=np.uint8)
    result = cvid.identify(blank)

    assert result is not None, "identify() returned None"
    assert isinstance(result.confidence, float), "confidence is not a float"
    assert isinstance(result.alternatives, list), "alternatives is not a list"
    assert isinstance(result.ids, dict), "ids is not a dict"

    print(f"  Result: {result.ids}  confidence={result.confidence:.4f}")
    print("Smoke test passed.")


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(1)

    if args[0] == "--smoke-test":
        if len(args) < 2:
            print("Usage: identify_image.py --smoke-test <gallery.npz>")
            sys.exit(1)
        smoke_test(args[1])
        return

    if args[0] == "--hf":
        if len(args) < 3:
            print("Usage: identify_image.py --hf <gallery-key> <image.jpg> [...]")
            print("  gallery-key examples: scryfall-mtg, tcgplayer-pokemon")
            sys.exit(1)
        import collector_vision as cvg
        gallery_key = args[1]
        image_paths = args[2:]
        cvid = cvg.Identifier(cvg.HFD("HanClinto/milo", gallery_key))
    else:
        import collector_vision as cvg
        gallery_path = args[0]
        image_paths  = args[1:]
        if not image_paths:
            print("No images provided. Use --smoke-test to verify without images.")
            sys.exit(1)
        cvid = cvg.Identifier(gallery_path)

    result = cvid.identify(*image_paths)

    print(f"Top match   confidence={result.confidence:.4f}")
    print(f"IDs         {result.ids}")

    sfid = result.ids.get("scryfall_id")
    if sfid:
        try:
            card = lookup_scryfall(sfid)
            print(f"Name        {card['name']}")
            print(f"Set         {card['set_name']} ({card['set'].upper()})")
            usd = card.get("prices", {}).get("usd")
            print(f"Price (USD) {'$' + usd if usd else 'n/a'}")
        except Exception as exc:
            print(f"Scryfall lookup failed: {exc}")

    if result.alternatives:
        print("\nAlternatives:")
        for alt in result.alternatives[:3]:
            print(f"  {alt.ids}  confidence={alt.confidence:.4f}")

    if result.frame_results:
        print(f"\nPer-frame results ({len(result.frame_results)} frames):")
        for i, fr in enumerate(result.frame_results):
            sfid = fr.ids.get("scryfall_id", "?")
            print(f"  frame {i}: {sfid[:8]}...  confidence={fr.confidence:.4f}")


if __name__ == "__main__":
    main()
