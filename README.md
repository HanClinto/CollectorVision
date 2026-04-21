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
import collector_vision as cv

gallery = cv.Gallery.for_game("magic")
result = cv.identify("photo.jpg", gallery=gallery)

print(result.card_name, result.set_code)
print(result.ids)  # {"scryfall_id": "...", "cardmarket_id": "...", ...}
```

The gallery downloads on first use and is cached in `~/.cache/collectorvision/`. Subsequent calls load from disk.

---

## Multiple games

```python
gallery = cv.Gallery.for_games("magic", "pokemon")
result = cv.identify("photo.jpg", gallery=gallery)
```

---

## Corner detection

By default CollectorVision locates the card automatically using the bundled neural detector. Two alternatives are available:

**Canny** — no GPU required, works well on clean or high-contrast backgrounds:

```python
from collector_vision.detectors import CannyCornerDetector

result = cv.identify("scan.jpg", gallery=gallery, detector=CannyCornerDetector())
```

**Fixed corners** — for robots, jigs, or scanners where the card is always in the same position:

```python
import numpy as np
from collector_vision.detectors import FixedCornerDetector

detector = FixedCornerDetector(corners=np.array([
    [0.05, 0.04],  # top-left
    [0.95, 0.04],  # top-right
    [0.95, 0.96],  # bottom-right
    [0.05, 0.96],  # bottom-left
]))
result = cv.identify("frame.jpg", gallery=gallery, detector=detector)
```

**Manual corners** — if you have corners from an external source, pass them directly and skip detection entirely:

```python
result = cv.identify("frame.jpg", gallery=gallery, corners=my_corners)
```

---

## Gallery variants

The default gallery uses the neural embedder (Milo). A perceptual hash variant is also available — no GPU required at query time, slightly lower accuracy:

```python
gallery = cv.Gallery.for_game("magic", variant="phash16")
```

---

## Batch identification

```python
results = cv.identify_batch(["a.jpg", "b.jpg", "c.jpg"], gallery=gallery)
```

---

## Offline use

Once galleries are cached locally, pass `offline=True` to prevent any network calls:

```python
gallery = cv.Gallery.for_game("magic", offline=True)
```

---

## License

AGPL-3.0. Commercial licenses available — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).
