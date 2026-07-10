# CollectorVision

Card identification library for collectible card games. Feed it a photo, get back a card identity.

Magic: The Gathering is the primary supported catalog today. Additional non-MTG catalogs are available as highly experimental previews, and user feedback is welcome.

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

### Hardware acceleration

CollectorVision uses ONNX Runtime providers through a small, simple API:

```python
cvg.NeuralCornerDetector(provider="auto")  # default: accelerator if available, then CPU
cvg.NeuralEmbedder(provider="auto")

cvg.NeuralEmbedder(provider="cpu")         # force CPU
cvg.NeuralEmbedder(provider="cuda")        # require CUDA, error if unavailable
```

The default `provider="auto"` prefers installed accelerator providers and falls
back to CPU. For CUDA, install the optional GPU runtime in an environment with
compatible NVIDIA drivers:

```bash
pip install "collectorvision[cuda] @ git+https://github.com/HanClinto/CollectorVision.git"
```

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

# Load catalog (downloads ~53 MB on first run, cached locally after that)
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

## Available catalogs

Catalogs are simple databases that include card embeddings (fingerprints) with keys (such as ScryFall IDs or TCGplayerIDs). You can [build your catalog yourself](catalog), or point at our HuggingFace repository and always download the official latest version: 

```python
catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
```

To use another published catalog, change only the catalog key at the end of the URI. The rest of the pipeline stays the same:

```python
# Pokemon TCG
catalog = cvg.Catalog.load("hf://HanClinto/milo/tcgplayer-pokemon")

# Star Wars: Unlimited
catalog = cvg.Catalog.load("hf://HanClinto/milo/tcgplayer-swu")
```

Returned `card_id` values come from the selected catalog's source. For example, TCGplayer catalogs return TCGplayer-style IDs rather than Scryfall UUIDs.

Catalog files are cached locally after the first download, and update automatically when a new version is released. Default cache directory is `~/.cache/collectorvision/` but can be overridden with the `COLLECTORVISION_CACHE_DIR` environment variable.

All catalogs below are official snapshots from the CollectorVision Hugging Face repository. Magic: The Gathering catalogs are the primary supported path; all non-MTG catalogs are highly experimental and need user feedback from real-world scans.

| Game | Source | Catalog key | Description | Size |
|---|---|---|---|---|
| Magic: The Gathering | Scryfall | `scryfall-mtg` | Primary MTG catalog from Scryfall reference images and card IDs. | ~53 MB |
| Magic: The Gathering | Scryfall | `scryfall-mtg-es` | Experimental Spanish-language MTG catalog from Scryfall; Milo has not been trained to distinguish English vs. Spanish printings. | ~21 MB |
| Magic: The Gathering | TCGplayer | `tcgplayer-mtg` | MTG catalog built from TCGplayer product/reference images. | ~50 MB |
| Digimon Card Game | TCGplayer | `tcgplayer-digimon` | Highly experimental Digimon catalog; feedback wanted. | ~3.9 MB |
| Flesh and Blood | TCGplayer | `tcgplayer-fab` | Highly experimental Flesh and Blood catalog; feedback wanted. | ~4.3 MB |
| Disney Lorcana | TCGplayer | `tcgplayer-lorcana` | Highly experimental Lorcana catalog; feedback wanted. | ~1.3 MB |
| One Piece Card Game | TCGplayer | `tcgplayer-onepiece` | Highly experimental One Piece catalog; feedback wanted. | ~3.0 MB |
| Pokémon TCG | TCGplayer | `tcgplayer-pokemon` | Highly experimental Pokémon catalog; feedback wanted. | ~13 MB |
| Star Wars: Unlimited | TCGplayer | `tcgplayer-swu` | Highly experimental Star Wars: Unlimited catalog; feedback wanted. | ~3.2 MB |
| Yu-Gi-Oh! | TCGplayer | `tcgplayer-yugioh` | Highly experimental Yu-Gi-Oh! catalog; feedback wanted. | ~21 MB |

Browse at **https://huggingface.co/HanClinto/milo/tree/main/catalogs**

Catalogs are published as dated snapshots. Filename format: `{algo}-{source}-{game}-{YYYY-MM-DD}.npz`

To share results, request a specific game/source, or report a catalog issue, open an issue or reach out [on Discord](https://discord.gg/ds8SMCRFZp) or [Twitter @HanClinto](https://x.com/HanClinto).

---

## Local catalog file

Catalog files are simple NumPy archives containing card IDs and their corresponding reference embeddings (image "fingerprints").

[Build your own catalog](catalog) of IDs + reference images, or use our pre-built catalog files available at [Hugging Face](https://huggingface.co/HanClinto/milo/tree/main/catalogs).

Pass a local path and nothing touches the network:

```python
catalog = cvg.Catalog.load("./milo1-scryfall-mtg-2026-05-07.npz")
```

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

## Upside-down cards

Current embeddings can be sensitive to 180-degree rotation. For a temporary rotation-invariant workaround, see [examples/quickstart_rot_invariant.py](examples/quickstart_rot_invariant.py). [examples/eval_accuracy.py](examples/eval_accuracy.py) uses this workaround by default; pass `--no-rot-invariant` to measure upright-only accuracy, `--verbose` to print the expected and matched card name / set for each image, or `--debug` to save aligned crops and corner-detector overlays. The web scanner also enables this by default with the "Scan upside-down cards" setting. These rotation-invariant paths dewarp the card once, embed the crop and a 180-degree rotated copy, search both, and keep the orientation with the strongest top match.

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

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

## Integrations

If you build something with CollectorVision, an announcement is welcome.

- Open an issue if you want to say that your project uses CollectorVision.
- Open a PR if you want to add your project to a future integrations list.

For hobby and noncommercial projects, this request is appreciated but not a
condition of the license.

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

## Playground

Demonstration:

https://www.youtube.com/watch?v=MHieOcmC7Dw

https://hanclinto.github.io/CollectorVision/applet_example.html


## Discord

Join our Discord to discuss all things CollectorVision, open-source, and computer vision:

https://discord.gg/ds8SMCRFZp
