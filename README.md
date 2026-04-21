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
import urllib.request, json
import collector_vision as cvg

cvid = cvg.Identifier("./magic-scryfall-milo1-2026-04.npz")
result = cvid.identify("photo.jpg")

print(result.ids)        # {"scryfall_id": "...", "tcgplayer_id": "...", ...}
print(result.confidence) # 0.94

# Look up card metadata from Scryfall
sid = result.ids["scryfall_id"]
card = json.loads(urllib.request.urlopen(f"https://api.scryfall.com/cards/{sid}").read())
print(card["name"], card["set_name"], card["prices"]["usd"])
```

Create one `Identifier` per process — it holds the gallery in memory and reuses it across calls.

---

## Auto-download with HFD

`HFD` resolves the latest gallery for a given name and caches it locally. It checks for updates at most once every 7 days, so repeated calls cost nothing after the first.

```python
cvid = cvg.Identifier(
    cvg.HFD("CollectorVision/galleries", "magic-scryfall-milo1")
)
```

The gallery is saved to `~/.cache/collectorvision/`. When a newer version is published, `HFD` downloads it and removes the old file automatically. You can also pin the refresh window:

```python
from datetime import timedelta

# Never recheck — use whatever is cached (pin indefinitely)
cvg.HFD("CollectorVision/galleries", "magic-scryfall-milo1", cache_refresh=None)

# Always check (useful in CI)
cvg.HFD("CollectorVision/galleries", "magic-scryfall-milo1", cache_refresh=timedelta(0))
```

---

## Multiple games

Can detect against multiple games at once. Just load the galleries for multiple games at the same time.

```python
cvid = cvg.Identifier(
    cvg.HFD("CollectorVision/galleries", "magic-scryfall-milo1"),
    cvg.HFD("CollectorVision/galleries", "pokemon-tcgplayer-milo1"),
)
result = cvid.identify("photo.jpg")
```

---

## Multiple frames, one card

Pass several images of the same physical card and `identify()` votes across
them — similarity scores are summed before ranking, giving a more confident
result than any single frame:

```python
result = cvid.identify("frame1.jpg", "frame2.jpg", "frame3.jpg")
print(result.card_name)      # aggregate winner
print(result.confidence)     # combined confidence

# Individual per-frame results are also available
for frame in result.frame_results:
    print(frame.card_name, frame.confidence)
```

---

## Pre-cropped images

If your image is already a clean crop of just the card — no background, no perspective — pass `detector=None` to skip corner detection:

```python
cvid = cvg.Identifier("./magic-scryfall-milo1-2026-04.npz", detector=None)
result = cvid.identify("crop.jpg")
```

---

## Offline use

Pass a local path and nothing will touch the network:

```python
cvid = cvg.Identifier("./magic-scryfall-milo1-2026-04.npz")
```

`HFD` also falls back to its local cache automatically if the network is unavailable.

---

## Available galleries

Browse and download gallery files at:
**https://huggingface.co/datasets/CollectorVision/galleries**

Galleries are updated monthly. Filename format: `{game}-{source}-{algorithm}-{YYYY-MM}.npz`

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
