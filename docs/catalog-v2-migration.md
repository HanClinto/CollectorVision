# Migrating from Catalog v1 to Catalog v2

Catalog v2 is a parallel beta API. Catalog v1 remains supported, uses its
existing cache, and does not need to be removed before adopting v2.

## Choose the migration boundary

Use Catalog v2 when an application benefits from hosted multi-game discovery,
always-available card names and peer identifiers, optional retained metadata,
browser-friendly FP16 storage, or incremental updates.

Keep Catalog v1 for custom NPZ catalogs or applications that do not need those
features yet. One process can load both versions.

## Python

Replace the hosted Catalog v1 source:

```python
catalog = cv.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
```

with game-first Catalog v2 discovery:

```python
catalog = cv.CatalogV2("mtg")
```

The common search path is intentionally compatible:

```python
embedding = catalog.embedder.embed(crop)
score, card_id = catalog.search(embedding, top_k=1)[0]
```

Use `search_records()` for v2 data:

```python
catalog = cv.CatalogV2("mtg", include_metadata=True)
match = catalog.search_records(embedding, top_k=1)[0]

print(match["id"])                    # selected source's primary result ID
print(match["name"])                  # always available
print(match["identifiers"])           # primary and peer source IDs
print(match["face_index"])            # 0 for the front face
print(match["finishes"])              # recognition-time physical finishes
print(match["metadata"])              # present when include_metadata=True
```

`card_id` remains an alias for `id` in search results. Catalog v1 fields such as
`primary_key`, `secondary_key`, and `matched_face` do not define the v2 record
contract. Use `result_identifier`, `identifiers`, and `face_index` instead.

The default source is Scryfall for MTG and TCGplayer for other games. Select a
source explicitly when needed:

```python
scryfall = cv.CatalogV2("mtg", source="scryfall")
tcgplayer = cv.CatalogV2("mtg", source="tcgplayer")
```

## Metadata behavior

Names, peer identifiers, faces, and finishes are core recognition fields.
Extended metadata includes fields such as set, collector number, language,
rarity, colors, promo, and layout.

Base and delta records combine core fields and metadata in one compressed
JSONL stream. `include_metadata=False` still downloads and validates that
stream, then discards metadata. It reduces steady-state memory and persistent
cache use, not network transfer.

Changing an existing installation from recognition-only to
`include_metadata=True` replays cached/downloaded record assets without
redownloading embeddings.

## Equivalence and repeated-scan grouping

For Scryfall, prefer `match["identifiers"]["scryfall_oracle"]` when grouping
printings of the same underlying card. For TCGplayer catalogs, no equivalent
cross-printing ID is currently available; use the required exact `name` as a
practical fallback when edition-level differences do not matter.

Do not treat a TCGplayer product ID as a card-equivalence ID. It identifies the
specific marketplace product returned by the catalog.

## Browser

Replace custom Catalog v1 manifest/asset loading with the feed client:

```javascript
import {
  BrowserCatalogV2,
} from "https://hanclinto.github.io/CollectorVision/lib/collectorvision-catalog-v2.mjs";

const catalog = await BrowserCatalogV2.forGame("mtg", {
  includeMetadata: true,
});

const [match] = catalog.searchRecords(queryEmbedding, 1);
console.log(match.card_id, match.name, match.identifiers, match.metadata);
```

The browser client stores the newest compatible materialized snapshot in
IndexedDB and applies exact-predecessor deltas automatically. Pass `cache: null`
only when persistent caching is undesirable.

## Caches and beta schema changes

Catalog v2 uses a cache separate from v1. Current clients version their v2 cache
schema and ignore incompatible beta snapshots, so normal users do not need a
manual cache migration.

Applications that directly consumed an earlier Catalog v2 beta prototype must
switch to the moving feed and combined assets:

```text
base/records.jsonl.gz
base/embeddings.f16.gz
delta-from-N/records.jsonl.gz
delta-from-N/embeddings.f16.gz   # only when recognition changed
```

There is no compatibility parser for discarded beta shapes such as separate
`identifiers.jsonl.gz` and `metadata.jsonl.gz` files.

## Rollout checklist

1. Change one catalog construction path to `CatalogV2`.
2. Keep tuple-based `search()` calls unchanged.
3. Update rich-result code to the v2 record fields.
4. Decide whether each application surface should retain metadata.
5. Update repeated-scan grouping to Oracle ID or the TCGplayer name fallback.
6. Exercise first install, cached startup, incremental update, and offline mode.
7. Remove the v1 path only after the application no longer needs NPZ/custom
   catalog behavior.
