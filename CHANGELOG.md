# Changelog

## Unreleased

- `Catalog.load("game")` now selects the recommended Catalog v2 feed, while
  local paths, `.npz` files, `hf://` references, and `HFD` objects retain the
  Catalog v1 loader. Explicit `CatalogV1.load()` and `CatalogV2.load()` entry
  points are also available; catalog constructors no longer perform I/O.
- Catalog v2 adds game-first discovery, packed float16 embeddings, combined
  recognition records and metadata, exact-predecessor incremental updates, and
  atomic local snapshots. Metadata is retained by default.
- The browser scanner now uses Catalog v2 by default, with `?catalog=v1` kept
  as a compatibility path. Browser snapshots are cached in IndexedDB.
- Default corner detection now uses Cornelius 2.12 from Hugging Face, superseding 1.221 with the new global-token SimCC detector. This keeps the rotated-card quality fix tracked in [issue #24](https://github.com/HanClinto/CollectorVision/issues/24), where soft-argmax averaging could place corners between competing peaks.
- The Python library and generated web scanner assets now share the same bundled `collector_vision/weights/cornelius.onnx` default, so web scanner refreshes will publish Cornelius 2.12 automatically.
