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
    "catalog-v2-beta.1-2026-07-24",
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
    "catalog-v2-beta.1-2026-07-24",
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
    "catalog-v2-beta.2-2026-07-25",
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
