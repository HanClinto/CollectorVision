#!/usr/bin/env python3
"""Identify a Magic: The Gathering card from a photo.

Run from the repo root:
    python examples/quickstart.py
"""

from pathlib import Path

import cv2

import collector_vision as cvg

IMAGE = Path("examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")

# 1. Discover and load the current Scryfall MTG catalog. Catalog v2 caches the
#    materialized snapshot and applies incremental updates on later runs.
catalog = cvg.CatalogV2("mtg", include_metadata=True)

# 2. Load the image you want to identify. Can be a photo from your phone, or a scan from a webcam feed.
image = cv2.imread(str(IMAGE))

# 3. Detect card corners within image, and get a sharpness score (0-1) indicating confidence in the detection.
#    If sharpness is low, try retaking the photo with better lighting, less blur, or a clearer view of the card.
detector = cvg.NeuralCornerDetector()
detection = detector.detect(image)
print(f"Detected corner sharpness={detection.sharpness:.3f}")

# 4. Dewarp to aligned crop using detected corners and perspective transform.
#    This gives us a clean, squared-up, card-only image to feed into the embedding model.
crop = detection.dewarp(image)

# 5. Convert the cropped image to an embedding vector using the same model used to create the catalog.
#    This ensures that the search in step 6 is comparing apples to apples.
emb = catalog.embedder.embed(crop)

# 6. Search for nearest neighbors in embedding space.
#    We search for the reference card embedding that is most similar to our input card's embedding.
#    The returned score is a number between 0 and 1 indicating similarity, with 1 being a perfect match.
hits = catalog.search_records(emb, top_k=5)
match = hits[0]

# 7. Print results
print(f"Top match {match['card_id']}  score={match['score']:.4f}")
print(f"Name      {match['name']}")
if match["metadata"]:
    set_name = match["metadata"].get("set_name", "n/a")
    set_code = match["metadata"].get("set")
    print(f"Set       {set_name}{f' ({set_code.upper()})' if set_code else ''}")
for candidate in hits[1:]:
    print(f"          {candidate['card_id']}  {candidate['name']}  score={candidate['score']:.4f}")
