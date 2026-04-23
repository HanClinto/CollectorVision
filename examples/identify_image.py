#!/usr/bin/env python3
"""Identify a Magic: The Gathering card from a photo.

Run from the repo root:
    python examples/identify_image.py
"""
import json
import urllib.request
from pathlib import Path

import cv2
import collector_vision as cvg

IMAGE = Path("examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")

# 1. Load catalog (~29 MB, cached after first download)
catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

# 2. Detect card corners
image     = cv2.imread(str(IMAGE))
detector  = cvg.NeuralCornerDetector()
detection = detector.detect(image)
print(f"Detected  sharpness={detection.sharpness:.3f}")

# 3. Dewarp to aligned crop
crop = detection.dewarp(image)

# 4. Embed
emb = catalog.embedder.embed(crop)

# 5. Search
hits = catalog.search(emb, top_k=5)
score, card_id = hits[0]
print(f"Top match {card_id}  score={score:.4f}")
for s, cid in hits[1:]:
    print(f"          {cid}  score={s:.4f}")

# 6. Metadata lookup via Scryfall API
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
