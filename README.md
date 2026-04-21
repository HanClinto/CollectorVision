# CollectorVision

Card identification library for collectible card games. Feed it an image, get back a card identity.

Supports Magic: The Gathering today, with Pokémon and others on the way. Works against Scryfall and TCGplayer reference data.

---

## Install

```bash
pip install collectorvision
```

Requires Python 3.10+. PyTorch is a dependency; if you want GPU acceleration install the appropriate torch variant for your platform first.

---

## Quickstart

```python
import collector_vision as cvg

cvid = cvg.Identifier(cvg.Game.MAGIC)
result = cvid.identify("photo.jpg")

print(result.card_name, result.set_code)
print(result.ids)  # {"scryfall_id": "...", "tcgplayer_id": "...", ...}
```

The gallery downloads on first use and is cached in `~/.cache/collectorvision/`. Subsequent calls load from disk. Create one `Identifier` per process — it holds the gallery in memory and reuses it across calls.

---

## Multiple games

```python
cvid = cvg.Identifier(cvg.Game.MAGIC, cvg.Game.POKEMON)
result = cvid.identify("photo.jpg")
```

---

## Embedding algorithms

The default embedding uses the bundled neural model (codename "Milo"), which gives the best accuracy and benefits from a GPU or Apple Silicon:

```python
# default — Milo neural embedding
cvid = cvg.Identifier(cvg.Game.MAGIC)
```

A perceptual hash variant runs on any hardware with no GPU required, at the cost of some accuracy on exact-printing identification:

```python
# PHASH — no GPU required, excellent for artwork identification
cvid = cvg.Identifier(cvg.Game.MAGIC, embedding=cvg.Embedding.PHASH)
```

---

## Corner detection

By default CollectorVision locates the card in the image using the bundled neural detector:

```python
# default — neural detector
cvid = cvg.Identifier(cvg.Game.MAGIC)
result = cvid.identify("photo.jpg")
```

**Canny** — no GPU required, works well on clean or high-contrast backgrounds:

```python
from collector_vision.detectors import CannyCornerDetector

cvid = cvg.Identifier(cvg.Game.MAGIC, detector=CannyCornerDetector())
result = cvid.identify("scan.jpg")
```

**No detection** — if the image is already a clean crop of just the card, pass `detector=None` to skip detection entirely:

```python
cvid = cvg.Identifier(cvg.Game.MAGIC, detector=None)
result = cvid.identify("crop.jpg")
```

---

## Batch identification

```python
results = cvid.identify_batch(["a.jpg", "b.jpg", "c.jpg"])
```

---

## Offline use

Once galleries are cached locally, pass `offline=True` to prevent any network calls:

```python
cvid = cvg.Identifier(cvg.Game.MAGIC, offline=True)
```

---

## Power user: local gallery file

If you have a gallery NPZ file on disk (e.g. built by CollectorVision-Pipeline):

```python
cvid = cvg.Identifier(gallery=cvg.Gallery.load("my_gallery.npz"))
```

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
