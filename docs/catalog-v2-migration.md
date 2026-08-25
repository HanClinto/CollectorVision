# Migrating from Catalog v1 to Catalog v2

Catalog v2 is the recommended API for hosted catalogs. It adds game-first
discovery, incremental updates, browser-native FP16 storage, required names and
peer identifiers, optional retained metadata, and catalogs for more games.

Catalog v1 remains supported for custom NPZ files. Its cache is separate from
v2, so both APIs can run in one application during rollout.

## Choose a migration path

| Current use | Recommendation |
|---|---|
| Hosted Hugging Face NPZ catalog in Python | Replace `Catalog.load("hf://...")` with `Catalog.load(game)` |
| Custom/local NPZ catalog in Python | Keep `Catalog.load(path)` unless the catalog is published as a v2 feed |
| Browser with bundled FP16, IDs, and a manifest | Replace the catalog loader and search helper with `BrowserCatalogV2` |
| Browser needing a gradual rollout | Make v2 the default and retain v1 behind a temporary feature flag |

## Python: hosted catalog

### Before: Catalog v1

```python
from PIL import Image
import collector_vision as cv

catalog = cv.Catalog.load("hf://HanClinto/milo/scryfall-mtg")

with Image.open("card.jpg") as image:
    embedding = catalog.embedder.embed(image.convert("RGB"))

score, card_id = catalog.search(embedding, top_k=1)[0]
```

### After: Catalog v2

```python
from PIL import Image
import collector_vision as cv

catalog = cv.Catalog.load("mtg")

with Image.open("card.jpg") as image:
    embedding = catalog.embedder.embed(image.convert("RGB"))

score, card_id = catalog.search(embedding, top_k=1)[0]
```

The embedding and tuple-based `search()` path is intentionally compatible. The
loader now discovers the recommended source and newest catalog-local
version, installs a base or exact-predecessor deltas, and reuses a separate v2
cache afterward.

Common hosted source migrations are:

| Catalog v1 source | Catalog v2 |
|---|---|
| `hf://HanClinto/milo/scryfall-mtg` | `cv.Catalog.load("mtg")` |
| `hf://HanClinto/milo/tcgplayer-mtg` | `cv.Catalog.load("mtg", source="tcgplayer")` |
| `hf://HanClinto/milo/tcgplayer-pokemon` | `cv.Catalog.load("pokemon")` |
| `hf://HanClinto/milo/tcgplayer-swu` | `cv.Catalog.load("swu")` |

The default source is Scryfall for MTG and TCGplayer for other games. Pass
`source="scryfall"` or `source="tcgplayer"` when source identity matters.

## Python: rich results and metadata

Catalog v1 commonly required aligned side arrays or a provider API request
after recognition. Catalog v2 can return the complete local recognition record.

### Before: Catalog v1 plus provider lookup

```python
import json
import urllib.request

score, card_id = catalog.search(embedding, top_k=1)[0]

with urllib.request.urlopen(f"https://api.scryfall.com/cards/{card_id}") as response:
    card = json.load(response)

print(card["name"], card["set_name"])
```

### After: Catalog v2 local record

```python
catalog = cv.Catalog.load("mtg")
match = catalog.search_records(embedding, top_k=1)[0]

print(match["id"])  # selected source's primary result ID
print(match["name"])  # always available
print(match["identifiers"])  # primary and peer source IDs
print(match["face_index"])  # 0 for the front face
print(match["finishes"])  # recognition-time physical finishes
print(match["metadata"]["set_name"])
```

Names, identifiers, faces, and finishes are core recognition fields. Extended
metadata includes fields such as set, collector number, language, rarity,
colors, promo, and layout. Provider data that changes frequently, such as live
market prices, should still be fetched from its authoritative API.

Metadata is retained by default. Pass `include_metadata=False` for a
recognition-only in-memory snapshot; the client still downloads and validates
the combined records stream, then discards metadata. This opt-out reduces
steady-state memory and persistent cache use, not network transfer.

### Result-field mapping

| Catalog v1 concept | Catalog v2 |
|---|---|
| `card_id` or `primary_key` value | `match["id"]` or compatibility alias `match["card_id"]` |
| `primary_key_name` | `match["result_identifier"]` |
| `secondary_key` or named peer IDs | `match["identifiers"]` |
| `matched_face` string | `match["face_index"]` (`0` is front) |
| provider lookup for card name | `match["name"]` |
| custom metadata sidecar | `match["metadata"]` |

For Scryfall, use `match["identifiers"]["scryfall_oracle"]` to group printings
of the same underlying card. TCGplayer does not currently provide a comparable
cross-printing ID; use exact `match["name"]` as the practical fallback when
edition differences do not matter. A TCGplayer product ID identifies a specific
marketplace product and is not an equivalence ID.

## Python: custom NPZ catalogs

Do not migrate a local/custom NPZ merely to use the default loader:

```python
catalog = cv.Catalog.load("./my-custom-catalog.npz")
```

`Catalog.load()` dispatches deterministically: game names use v2, while local
paths, `.npz` filenames, `hf://` references, and `HFD` objects use v1. It does
not catch v2 download or validation failures and retry them as v1. Use
`CatalogV1.load(path)` or `CatalogV2.load(game)` to select a generation
explicitly.

Catalog v2 is feed-driven and does not reinterpret arbitrary NPZ files. Keep
Catalog v1 for custom NPZ catalogs, or publish the catalog through the v2
producer contract before changing clients.

## JavaScript: bundled browser catalog

There was no standalone Catalog v1 browser client. Browser integrations
typically loaded the scanner manifest, packed FP16 matrix, and aligned IDs
themselves.

### Before: Catalog v1 application-owned loading

```javascript
const manifest = await fetch("./assets/manifest.json").then((response) =>
  response.json(),
);

const [embeddingBuffer, cardIds] = await Promise.all([
  fetch(`./assets/${manifest.catalog.embeddings}`).then((response) =>
    response.arrayBuffer(),
  ),
  fetch(`./assets/${manifest.catalog.card_ids}`).then((response) =>
    response.json(),
  ),
]);

const embeddings = new Uint16Array(embeddingBuffer);
const { score, index } = searchPackedFp16(
  queryEmbedding,
  embeddings,
  manifest.catalog.rows,
  manifest.catalog.dims,
);
const cardId = cardIds[index];
```

`searchPackedFp16` represents the application-specific search loop required by
the v1 bundle. Existing applications may use different helper names.

### After: Catalog v2 managed loading and search

```javascript
import {
  BrowserCatalogV2,
} from "https://hanclinto.github.io/CollectorVision/lib/collectorvision-catalog-v2.mjs";

const catalog = await BrowserCatalogV2.forGame("mtg");

const [match] = catalog.searchRecords(queryEmbedding, 1);
console.log(match.score, match.card_id, match.name);
console.log(match.identifiers, match.face_index, match.metadata);
```

Keep the existing Milo inference pipeline: `queryEmbedding` remains a
normalized `Float32Array`. The v2 client discovers the catalog, verifies sizes
and checksums, reconstructs updates, keeps embeddings packed as FP16, and
persists the newest compatible snapshot in IndexedDB.

If only `(score, cardId)` is needed, use the compact compatibility shape:

```javascript
const [[score, cardId]] = catalog.search(queryEmbedding, 1);
```

Pass `cache: null` only when persistent caching is undesirable. Advanced
applications can use `CatalogV2FeedClient` and `CatalogV2IndexedDbCache` for
explicit catalog keys, family/profile selection, mirrors, or custom cache
management. Metadata is retained by default; pass `includeMetadata: false` for
a recognition-only in-memory snapshot.

## JavaScript: staged rollout

Run both paths against the same detector, dewarp, embedder, thresholds, and
confirmation logic. Change only the catalog loader/search backend:

```javascript
const catalogMode =
  new URLSearchParams(location.search).get("catalog") ?? "v2";

if (catalogMode === "v1") {
  // Load the existing bundled manifest, FP16 matrix, and aligned IDs.
} else if (catalogMode === "v2") {
  // Load BrowserCatalogV2.forGame(...).
} else {
  throw new Error(`Unsupported catalog mode: ${catalogMode}`);
}
```

The live CollectorVision scanner follows this pattern: Catalog v2 is the
default, while `?catalog=v1` exercises the bundled compatibility path.

## Caches and updates

- V1 and v2 use separate caches; do not copy or rename v1 cache files.
- Python `offline=True` opens the newest compatible installed v2 snapshot.
- Browser v2 snapshots use a separate IndexedDB database.
- Clients apply exact-predecessor deltas automatically and fall back to the
  advertised base when a compatible predecessor is unavailable.
- Current clients ignore incompatible beta cache schemas automatically.
- Applications that consumed discarded beta files such as separate
  `identifiers.jsonl.gz` and `metadata.jsonl.gz` must move to combined
  `records.jsonl.gz`; there is no compatibility parser for those prototypes.

## Rollout checklist

1. Replace one hosted v1 catalog construction/loading path with v2.
2. Keep Milo embedding generation and tuple-based searches unchanged.
3. Migrate rich results to `name`, `identifiers`, `face_index`, and `metadata`.
4. Choose Oracle ID or exact TCGplayer name for repeated-scan grouping.
5. Exercise first install, cached startup, incremental update, and offline mode.
6. Compare match quality and search latency against v1 using the same inputs.
7. Retain v1 only where custom NPZ compatibility or rollback is still needed.

See the [Catalog v2 client reference](catalog-v2-client.md) and
[live browser example](https://hanclinto.github.io/CollectorVision/catalog_v2_example.html)
for the complete API and a runnable integration.
