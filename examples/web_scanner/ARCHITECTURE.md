# Web Scanner Architecture

The scanner is a static application. Camera capture and UI run on the main
thread; detection, dewarping, embedding, and catalog search run in
`scanner.worker.mjs`.

## Recognition pipeline

1. `getUserMedia()` captures the back camera when available.
2. The manifest-selected detector (`assets/models/detector.onnx`) predicts four
   normalized corners.
3. JavaScript perspective warp produces a `448x448` Milo crop.
4. Milo emits a normalized 128-dimensional embedding.
5. Search compares the query with the packed float16 catalog matrix without
   expanding the full matrix to float32.
6. Confirmed matches are shown immediately using the catalog record name.
   Scryfall enrichment supplies current set and price data asynchronously.

The worker can also embed a 180-degree rotation and keep the stronger match.

## Catalog modes

### Catalog v2 (default)

`BrowserCatalogV2.forGame("mtg")` reads the moving feed from
`hanclinto.github.io/CollectorVisionCatalog`. The feed identifies immutable
base and delta assets with sizes and checksums. Reconstructed snapshots are
cached in IndexedDB.

The scanner passes `includeMetadata: false`: it retains names, identifiers,
faces, and finishes but discards extended metadata after parsing. This saves
steady-state memory without changing record-download bandwidth.

### Catalog v1 (compatibility)

`?catalog=v1` loads the prepared static bundle from `./assets/catalog/`.
Embeddings, card IDs, optional secondary IDs, and face data are aligned by row.
This path remains available for rollback and local bundle testing.

## Models and generated assets

The runtime manifest describes model files, dimensions, checksums, and the v1
catalog assets. `scripts/export_web_scanner_assets.py` generates `assets/` and
vendors the required ONNX Runtime files. Generated directories are ignored by
Git.

Heavy stable and testing bundles are published as GitHub release assets. The
Pages workflow downloads a prepared bundle, stamps build-ID placeholders
throughout web sources, and deploys static files. The model/catalog refresh and
Pages deployment workflows are independent.

## Caching

- Model and v1 bundle assets use the scanner asset IndexedDB store.
- Catalog v2 snapshots use their own versioned IndexedDB store.
- The service worker and build-ID query parameters prevent stale source modules
  from mixing with a newer deployment.

## Inference backends

WASM is the default. WebGPU is opt-in because ONNX Runtime Web has produced
valid-looking but incorrect output on Android ARM for both detector and
embedder models (issues #9 and #12). Firefox WebGPU is disabled for the current
runtime because it can generate invalid Metal shaders. Benchmark backend
changes separately before enabling WebGPU on another platform.

The scanner uses `ort.webgpu.min.mjs`, not the legacy JSEP
`ort.all.min.mjs` bundle.

## Non-goals

- server-side recognition
- pHash or Canny fallback paths
- a separate desktop UI
