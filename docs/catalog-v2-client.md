# Catalog v2 client

Catalog v2 keeps the parts that work well in Catalog v1: choose a game, receive
a ready-to-search catalog, and use the catalog's matching embedder.

```python
from PIL import Image
import collector_vision as cv

catalog = cv.CatalogV2("mtg")

with Image.open("card.jpg") as image:
    embedding = catalog.embedder.embed(image.convert("RGB"))

score, card_id = catalog.search(embedding, top_k=1)[0]
```

`Game.MTG` is accepted too. Construction downloads the catalog when needed and
reuses its cache afterward. The default is the recommended source and profile.

Choose the compact one-card-per-Oracle catalog when exact printings do not
matter:

```python
catalog = cv.CatalogV2("mtg", profile="cards")
```

Load names, sets, languages, finishes, and peer IDs only when needed:

```python
catalog = cv.CatalogV2("mtg", include_metadata=True)
match = catalog.search_records(embedding, top_k=1)[0]
print(match["identifiers"])
print(match["metadata"])
```

The familiar v1 attributes `card_ids`, `oracle_ids`, `source`, `algo_key`,
`embeddings`, and `embedder` remain available. `offline=True` opens the pinned
beta from the separate v2 cache without network access.

Release tags, catalog keys, checksums, cache layout, and exact-base deltas are
managed internally. `CatalogV2Downloader` remains available for applications
that need explicit control. Catalog v1 remains unchanged and can run beside v2.

## Browser

The browser API follows the same game-first shape:

```javascript
import {
  BrowserCatalogV2,
} from "https://hanclinto.github.io/CollectorVision/lib/collectorvision-catalog-v2.mjs";

const catalog = await BrowserCatalogV2.forGame("mtg", {
  profile: "cards",
  includeMetadata: true,
});

const [[score, cardId]] = catalog.search(queryEmbedding, 1);
```

`queryEmbedding` is the normalized `Float32Array` from the existing Milo
inference pipeline. The catalog keeps its matrix packed as FP16. Advanced
applications can use `CatalogV2BrowserClient` and `CatalogV2IndexedDbCache`
directly for explicit versions, mirrors, and persistent snapshots.
