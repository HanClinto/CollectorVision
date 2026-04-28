# CollectorVision

Card identification library for collectible card games. Feed it a photo, get back a card identity.

Supports Magic: The Gathering today. Pokémon, Yu-Gi-Oh, and others are planned.

## Try online

Experimental javascript port version hosted here: https://hanclinto.github.io/CollectorVision/


---

## Install

> **Not yet on PyPI.** Install directly from GitHub:

```bash
uv pip install git+https://github.com/HanClinto/CollectorVision.git
```

Or with plain `pip`:

```bash
pip install git+https://github.com/HanClinto/CollectorVision.git
```

Requires Python 3.10+. No GPU required — inference runs on CPU via ONNX Runtime.

---

## How it works

Given a photo of a card held in hand (or on a table, in a sleeve, etc.), CollectorVision finds the four corners, dewarps the card to a canonical crop, and produces a compact 128-d embedding vector:

![Pipeline: photo → corner detection → dewarped crop → 128-d embedding](docs/pipeline.jpg)

That embedding is then matched against a catalog of ~108k reference embeddings using nearest-neighbour search:

![Search: query embedding matched against catalog candidates](docs/pipeline_search.jpg)

The full pipeline runs end-to-end in under 100ms on a laptop CPU.

---

## Quickstart

```python
import cv2
import collector_vision as cvg

# Load catalog (downloads ~29 MB on first run, cached locally after that)
catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

# 1. Detect card corners
image = cv2.imread("examples/images/7286819f-6c57-4503-898c-528786ad86e9_sample.jpg")
detector = cvg.NeuralCornerDetector()
detection = detector.detect(image)

# 2. Dewarp to aligned crop
crop = detection.dewarp(image)          # PIL Image, 448×448 px

# 3. Embed + search
emb = catalog.embedder.embed(crop)      # (128,) float32
hits = catalog.search(emb, top_k=5)    # [(score, card_id), ...]

score, card_id = hits[0]
print(card_id, score)   # "abc123-...", 0.94
```

---

## Local catalog file

Catalog files are simple NumPy archives containing card IDs and their corresponding reference embeddings (image "fingerprints").

Build your own catalog of IDs + reference images, or use our pre-build catalog files available at [HuggingFace](https://huggingface.co/HanClinto/milo/tree/main/catalogs).

Pass a local path and nothing touches the network:

```python
catalog = cvg.Catalog.load("./milo1-scryfall-mtg-2026-04.npz")
```

Or point at our HuggingFace repository and always download the latest: 

```python
catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
```

Catalog files are cached locally after the first download, and update automatically when a new version is released. Default cache directory is `~/.cache/collectorvision/` but can be overridden with the `COLLECTORVISION_CACHE_DIR` environment variable.


---

## Multiple frames, one card

Useful when scanning from live video feeds, pass in the last N frames to get a democratic vote across multiple images. Embed each frame separately, then sum scores before ranking:

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
| Magic: The Gathering | Scryfall | `scryfall-mtg` | ~29 MB |

Browse at **https://huggingface.co/HanClinto/milo/tree/main/catalogs**

Catalogs are updated monthly. Filename format: `{algo}-{source}-{game}-{YYYY-MM}.npz`

Other games and sources coming soon. To request a specific game/source, open an issue or reach out on Twitter @HanClinto.

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

## Web Scanner

The mobile-first browser scanner lives in `examples/web_scanner`.

GitHub Pages deployment is wired to publish that folder directly from `main`.
Once Pages is enabled for the repo, the scanner should be available at:

`https://hanclinto.github.io/CollectorVision/`
