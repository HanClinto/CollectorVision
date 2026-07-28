# Catalog v2 Python client

Catalog v2 is a beta, explicit-tag API. It operates independently from the
existing `Catalog` class:

- `Catalog` continues to load Catalog v1 NPZ files from local paths or
  Hugging Face.
- `CatalogV2` loads the v2 FP16/JSONL artifact contract.
- `CatalogV2Release` installs immutable GitHub Release assets beneath a
  separate `catalog-v2/` cache directory.

Both versions can be loaded in the same process without changing or migrating
the v1 cache.

## Install an explicit beta

During beta, select a reviewed release tag rather than a moving latest release:

```python
import collector_vision as cvg

release = cvg.CatalogV2Release.install(
    "catalog-v2-beta.2-2026-07-27",
    catalog_keys=["milo1/scryfall/mtg"],
)
catalog = release.load("milo1/scryfall/mtg")
```

The compatibility `search()` method returns the identifier selected by the
catalog descriptor:

```python
embedding = catalog.embedder.embed(card_image)
score, card_id = catalog.search(embedding, top_k=1)[0]
```

Use `search_records()` to receive the stable row key, all peer identifiers,
face index, result identifier, and score.

## Optional metadata

Metadata remains an independent download and cache layer:

```python
release = cvg.CatalogV2Release.install(
    "catalog-v2-beta.2-2026-07-27",
    catalog_keys=["milo1/scryfall/mtg"],
    include_metadata=True,
)
catalog = release.load("milo1/scryfall/mtg")
```

Recognition-only and metadata installations do not overwrite one another.

## One-step updates

Pass the currently installed release when moving to the immediately following
release:

```python
updated = cvg.CatalogV2Release.install(
    "catalog-v2-beta.2-2026-07-27",
    catalog_keys=["milo1/scryfall/mtg"],
    previous_tag="catalog-v2-beta.1-2026-07-24",
)
```

The installer uses the delta only when the target manifest requires that exact
base and the prior catalog is installed in the same cache layer. It verifies
the delta assets, materializes a complete local snapshot, and can therefore
apply the next release as another single step. If the exact base is unavailable
or incompatible, it downloads the target release's full snapshot instead.

The default v2 location is:

```text
~/.cache/collectorvision/catalog-v2/releases/
```

`COLLECTORVISION_CACHE` or the `cache_dir` argument changes the common
CollectorVision cache root while preserving the separate `catalog-v2/`
namespace.

## Browser client

The browser client is also separate from the existing v1 scanner catalog:

```javascript
import {
  CatalogV2BrowserClient,
} from "./lib/collectorvision-catalog-v2.mjs";

const client = new CatalogV2BrowserClient({
  releaseBaseUrl: "https://hanclinto.github.io/CollectorVision/catalog-v2/",
});
const catalog = await client.load(
  "catalog-v2-beta.2-2026-07-27",
  "milo1/scryfall/mtg",
);

const [[score, cardId]] = catalog.search(queryEmbedding, 1);
```

The browser keeps embeddings packed as little-endian FP16 and converts values
during dot products. This avoids expanding the catalog to float32 in memory.
It uses the browser's native `fetch`, Web Crypto SHA-256, and
`DecompressionStream` implementations without adding a package dependency.
`releaseBaseUrl` must point to a same-origin or CORS-enabled mirror organized
as `<base>/<tag>/<release asset>`. GitHub Release download responses do not
currently permit cross-origin browser reads, so the module deliberately does
not present the GitHub release URL as a working browser default. The official Pages deployment mirrors only client assets—never builder
state—under the URL shown above. It promotes the highest valid published beta
on a weekly schedule after the catalog release workflow. This mirror is
independent from the scanner's bundled v1 catalog.

Pass the currently loaded catalog to use the next release's exact-base delta:

```javascript
const updated = await client.load(
  "catalog-v2-beta.2-2026-07-27",
  "milo1/scryfall/mtg",
  { previous: catalog },
);
```

An absent or incompatible base automatically selects the complete target
snapshot. Applications can rely on normal HTTP caching or persist release
assets separately; the beta module does not alter the v1 scanner's bundled
catalog storage.
