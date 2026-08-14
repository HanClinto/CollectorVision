# Catalog v2 client

Catalog v2 keeps the parts that work well in Catalog v1: choose a game, receive
a ready-to-search catalog, and use the catalog's matching embedder.

Existing Catalog v1 applications should start with the
[v1 to v2 migration guide](catalog-v2-migration.md).

```python
from PIL import Image
import collector_vision as cv

catalog = cv.CatalogV2("mtg")

with Image.open("card.jpg") as image:
    embedding = catalog.embedder.embed(image.convert("RGB"))

score, card_id = catalog.search(embedding, top_k=1)[0]
```

`Game.MTG` is accepted too. Construction downloads the catalog when needed and
reuses its cache afterward. The default source is Scryfall for MTG and
TCGplayer for the other supported games. A small discovery feed selects a
bounded catalog-local base-plus-update route. The feed is the complete client
contract: it includes immutable family embedding details, catalog descriptors,
integer versions, absolute asset URLs, compressed sizes, and checksums. Clients
do not fetch release indexes or per-version manifests.

Names, finishes, and peer IDs are always downloaded as part of the catalog's
combined records. Load sets, languages, rarity, and other display/filter
metadata only when needed:

```python
catalog = cv.CatalogV2("mtg", include_metadata=True)
match = catalog.search_records(embedding, top_k=1)[0]
print(match["identifiers"])
print(match["name"])
print(match["metadata"])
```

The primary result appears as `id` and under the descriptor's namespace in
`identifiers`. Recognition-level `name` and `finishes` are available without
metadata.
Current Scryfall metadata includes `promo` and canonical `layout`, including
`layout == "art_series"` for art-card filtering.

The familiar v1 attributes `card_ids`, `oracle_ids`, `source`, `algo_key`,
`embeddings`, and `embedder` remain available. `offline=True` opens the latest
compatible locally installed version from the separate v2 cache without network
access:

```python
catalog = cv.CatalogV2("mtg", include_metadata=True, offline=True)
```

Catalog keys, checksums, cache layout, and exact-predecessor updates are managed
internally. Cached snapshots are materialized atomically so subsequent updates
start from the newest compatible local integer version. Only the latest snapshot
per catalog and metadata mode is retained. Adding metadata to an installed
recognition-only snapshot replays the base and update combined records to
extract metadata, without redownloading embeddings.

`CatalogV2Downloader` remains available for explicit catalog keys or versions:

```python
download = cv.CatalogV2Downloader.install_catalog(
    "milo1/scryfall/mtg",
    include_metadata=True,
    version=2,
)
catalog = download.load()
```

Catalog v1 remains unchanged and can run beside v2.

## Browser

The browser API follows the same game-first shape:

```javascript
import {
  BrowserCatalogV2,
} from "https://hanclinto.github.io/CollectorVision/lib/collectorvision-catalog-v2.mjs";

const catalog = await BrowserCatalogV2.forGame("mtg", {
  includeMetadata: true,
});

const [[score, cardId]] = catalog.search(queryEmbedding, 1);
```

`queryEmbedding` is the normalized `Float32Array` from the existing Milo
inference pipeline. The catalog keeps its matrix packed as FP16 and persists
the newest compatible snapshot in IndexedDB by default. Pass `cache: null` to
disable persistent caching. Advanced applications can use `CatalogV2FeedClient`
and `CatalogV2IndexedDbCache` directly for explicit catalog keys,
family/profile selection, mirrors, and custom cache management.

A runnable browser loading example is published at
<https://hanclinto.github.io/CollectorVision/catalog_v2_example.html>. The main
camera scanner also uses Catalog v2 by default; append `?catalog=v1` to its URL
to exercise the bundled Catalog v1 compatibility path. The standalone page
exercises the finalized v2 feed, combined records, metadata-retention option,
and IndexedDB cache directly.
