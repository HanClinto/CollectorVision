# CollectorVision

Card identification library for collectible card games. Feed it a photo, get back a card identity.

Supports Magic: The Gathering today, with Pokémon and others on the way.

---

## Install

> **Not yet on PyPI.** Install directly from GitHub:

```bash
pip install git+https://github.com/HanClinto/CollectorVision.git
```

Requires Python 3.10+. No GPU required — inference runs on CPU via ONNX Runtime.

---

## Quickstart

```python
import cv2
import collector_vision as cvg

# Load catalog (downloads ~54 MB on first run, cached locally after that)
catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

# 1. Detect card corners
image = cv2.imread("photo.jpg")
detector = cvg.NeuralCornerDetector()
detection = detector.detect(image)

# 2. Dewarp to aligned crop
crop = detection.dewarp(image)          # PIL Image, 252×352 px

# 3. Embed + search
emb = catalog.embedder.embed(crop)      # (128,) float32
hits = catalog.search(emb, top_k=5)    # [(score, card_id), ...]

score, card_id = hits[0]
print(card_id, score)   # "abc123-...", 0.94
```

---

## Local catalog file

Pass a local path and nothing touches the network:

```python
catalog = cvg.Catalog.load("./milo1-scryfall-mtg-2026-04.npz")
```

Catalog files are available at [HuggingFace](https://huggingface.co/HanClinto/milo/tree/main/catalogs).

---

## Multiple frames, one card

Embed each frame separately, then sum scores before ranking:

```python
embeddings = catalog.embedder.embed([crop1, crop2, crop3])  # (3, 128)

from collections import defaultdict
score_map = defaultdict(float)
for emb in embeddings:
    for score, card_id in catalog.search(emb, top_k=5):
        score_map[card_id] += score

best_id = max(score_map, key=score_map.get)
```

---

## Pre-cropped images

If your input is already a clean card crop, skip detection and embed directly:

```python
from PIL import Image
crop = Image.open("crop.jpg")
emb = catalog.embedder.embed(crop)
hits = catalog.search(emb)
```

---

## Available catalogs

| Game | Source | Catalog key | Size |
|---|---|---|---|
| Magic: The Gathering | Scryfall | `scryfall-mtg` | ~54 MB |

Browse at **https://huggingface.co/HanClinto/milo/tree/main/catalogs**

Catalogs are updated monthly. Filename format: `{algo}-{source}-{game}-{YYYY-MM}.npz`

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
