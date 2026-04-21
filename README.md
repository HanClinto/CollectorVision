# CollectorVision

Card identification library for collectible card games. Feed it an image, get back a card identity.

Supports Magic: The Gathering today, with Pokémon and others on the way.

---

## Install

```bash
pip install collectorvision
```

Requires Python 3.10+. Model weights are bundled — no separate download needed.
PyTorch is a dependency; install the GPU-accelerated variant for your platform first if you want faster inference.

---

## Quickstart

Download a gallery file for your game from [HuggingFace](https://huggingface.co/datasets/CollectorVision/galleries), then:

```python
import collector_vision as cvg

cvid = cvg.Identifier("./magic-scryfall-phash16-2026-04.npz")
result = cvid.identify("photo.jpg")

print(result.card_name, result.set_code)
print(result.ids)  # {"scryfall_id": "...", "tcgplayer_id": "...", ...}
```

Create one `Identifier` per process — it holds the gallery in memory and reuses it across calls.

---

## Auto-download with HFD

`HFD` resolves the latest gallery for a given name and caches it locally. It checks for updates at most once every 7 days, so repeated calls cost nothing after the first.

```python
cvid = cvg.Identifier(
    cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16")
)
```

The gallery is saved to `~/.cache/collectorvision/`. When a newer version is published, `HFD` downloads it and removes the old file automatically. You can also pin the refresh window:

```python
from datetime import timedelta

# Never re-check (pin forever)
cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16", cache_refresh=timedelta(days=365))

# Always check (useful in CI)
cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16", cache_refresh=timedelta(0))
```

---

## Multiple games

Galleries must share the same embedding algorithm. Mix and match freely within that constraint:

```python
cvid = cvg.Identifier(
    cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"),
    cvg.HFD("CollectorVision/galleries", "pokemon-tcgplayer-phash16"),
)
result = cvid.identify("photo.jpg")
```

---

## Embedding algorithms

Two algorithms are available. The gallery file you download determines which one is used — the `Identifier` reads this from the file automatically.

| Gallery name suffix | Algorithm | GPU needed | Accuracy |
|---|---|---|---|
| `phash16` | Perceptual hash 16×16 | No | Good for artwork ID |
| `milo1` | ArcFace neural (Milo) | Recommended | Best for exact printing ID |

```python
# Hash — any CPU, ~32 bytes/card
cvid = cvg.Identifier(cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"))

# Neural — GPU recommended, ~512 bytes/card
cvid = cvg.Identifier(cvg.HFD("CollectorVision/galleries", "magic-scryfall-milo1"))
```

---

## Corner detection

By default CollectorVision uses the bundled neural detector. Two alternatives:

**Canny** — no GPU required, works well on clean or high-contrast backgrounds:

```python
from collector_vision.detectors import CannyCornerDetector

cvid = cvg.Identifier(
    cvg.HFD("CollectorVision/galleries", "magic-scryfall-phash16"),
    detector=CannyCornerDetector(),
)
```

**No detection** — if the image is already a clean card crop:

```python
cvid = cvg.Identifier("./magic-scryfall-phash16-2026-04.npz", detector=None)
```

---

## Batch identification

```python
results = cvid.identify_batch(["a.jpg", "b.jpg", "c.jpg"])
```

---

## Offline use

Pass a local path and HFD will never touch the network:

```python
cvid = cvg.Identifier("./magic-scryfall-phash16-2026-04.npz")
```

HFD also falls back to its local cache automatically if the network is unavailable.

---

## Available galleries

Browse and download gallery files at:
**https://huggingface.co/datasets/CollectorVision/galleries**

Galleries are updated monthly. Filename format: `{game}-{source}-{algorithm}-{YYYY-MM}.npz`

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
