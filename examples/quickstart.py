#!/usr/bin/env python3
"""Identify a Magic: The Gathering card from a photo.

Run from the repo root:
    python examples/quickstart.py
"""

import json
import urllib.request
from pathlib import Path

import cv2

import collector_vision as cvg

IMAGE = Path("examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")

# 1. Download and load catalog of reference image embeddings (29mb, cached after first download)
catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

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
hits = catalog.search(emb, top_k=5)
score, card_id = hits[0]

# 7. Print results
print(f"Top match {card_id}  score={score:.4f}")
for s, cid in hits[1:]:
    print(f"          {cid}  score={s:.4f}")

# 8. Metadata lookup via Scryfall API (optional)
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
