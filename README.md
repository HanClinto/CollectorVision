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
import collector_vision as cvg

# Downloads the gallery on first run (~54 MB), cached locally after that
cvid = cvg.Identifier(cvg.HFD("HanClinto/milo", "scryfall-mtg"))
result = cvid.identify("photo.jpg")

print(result.ids)         # {"scryfall_id": "abc123..."}
print(result.confidence)  # 0.94
```

`Identifier` loads the gallery once and reuses it across calls. Create one per process.

---

## Local gallery file

Download a gallery from [HuggingFace](https://huggingface.co/HanClinto/milo/tree/main/galleries)
and pass the path directly — nothing touches the network at runtime:

```python
cvid = cvg.Identifier("./milo1-scryfall-mtg-2026-04.npz")
result = cvid.identify("photo.jpg")
```

---

## Multiple frames, one card

Pass several images of the same physical card to vote across frames — similarity scores
are summed before ranking, giving a more confident result than any single frame:

```python
result = cvid.identify("frame1.jpg", "frame2.jpg", "frame3.jpg")
print(result.confidence)     # combined confidence
print(result.frame_results)  # per-frame breakdown
```

---

## Pre-cropped images

If your input is already a clean card crop with no background or perspective distortion,
skip corner detection:

```python
cvid = cvg.Identifier("./milo1-scryfall-mtg-2026-04.npz", detector=None)
```

---

## Available galleries

| Game | Source | Gallery key | Size |
|---|---|---|---|
| Magic: The Gathering | Scryfall | `scryfall-mtg` | ~54 MB |

Browse at **https://huggingface.co/HanClinto/milo/tree/main/galleries**

Galleries are updated monthly. Filename format: `{algo}-{source}-{game}-{YYYY-MM}.npz`

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
