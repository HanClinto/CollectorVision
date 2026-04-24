# CollectorVision Web Scanner

Browser-only scanner scaffold for a GitHub Pages deployment.

Primary path only:
- mobile-first UI
- WebGPU required
- Cornelius via `onnxruntime-web`
- dewarp via local JS perspective warp
- Milo via `onnxruntime-web`
- local gallery search in JS
- live metadata enrichment from Scryfall

No fallback modes are planned here. If the browser cannot run the main path,
the app should say so and stop.

## Product Shape

The app should feel like a handheld scanner, not a desktop dashboard.

Layout:
- dark app bar on top
- camera view on top, with small / expanded modes
- running scan list on the bottom
- one compact actions row for copy / CSV / settings
- a lightweight settings sheet instead of a dense control panel

Desktop should still render the mobile layout rather than switching to a
separate desktop UX.

## Runtime

Pipeline:

1. `getUserMedia()` captures a frame from the back camera.
2. Cornelius predicts normalized card corners.
3. Local JS dewarp warps into the canonical `252x352` crop.
4. Milo emits a `128`-d embedding.
5. Browser code runs cosine search against a local embedding gallery.
6. The winning `card_id` is enriched with static metadata and optional live
   Scryfall data.
7. The confirmed card is appended to the running list and later exported as
   text or CSV.

Runtime pieces:
- vendored `onnxruntime-web`
- `IndexedDB` for cached ONNX + catalog assets
- `fetch()` for live Scryfall lookup
- plain ES modules and static files for GitHub Pages

The browser runtime should read only local `./assets/...` files. Treat Hugging
Face as a publish-time sync source, not a live browser dependency. The same
goes for the browser runtime files: ship them with the app instead of loading
from a CDN.

## Asset Contract

Expected static assets:
- `assets/models/cornelius.onnx`
- `assets/models/milo.onnx`
- `assets/catalog/scryfall-mtg-embeddings.f16.bin`
- `assets/catalog/scryfall-mtg-card-ids.json`
- `assets/manifest.json`

Recommended shape:
- embeddings: raw float16 matrix, row-major, already L2-normalized
- card IDs: JSON string array aligned to embedding rows

For GitHub Pages, a `20-30 MB` catalog asset is acceptable. The main concern is
first-load time on phones, not whether the browser can handle it.

## Caching

Cache these in browser storage after first launch:
- Cornelius ONNX
- Milo ONNX
- gallery embedding file
- card ID table

Use `IndexedDB` with a manifest version to invalidate old assets cleanly.

## Deploy Model

The intended long-term deploy shape is:

- app source stays on `main`
- generated scanner assets are published as a GitHub release bundle
- normal Pages deploys fetch that prepared bundle from GitHub
- the HF catalog is only consulted by a separate asset refresh workflow

See:

- [ASSET_DEPLOY_PLAN.md](./ASSET_DEPLOY_PLAN.md)
- [assets.bundle.json](./assets.bundle.json)

## Local Test Loop

The generated `assets/` and `vendor/` folders are not tracked in git anymore.
For local development, regenerate them first.

Fast path:

```bash
./scripts/run_web_scanner_local.sh
```

This will:

- rebuild the local scanner bundle
- serve `examples/web_scanner` on `http://localhost:8040`

Manual path:

```bash
uv run python scripts/export_web_scanner_assets.py
cd examples/web_scanner
uv run python -m http.server 8040
```

Open `http://localhost:8040`.

If you are testing on desktop before camera permissions or mobile WebGPU are
sorted out, use `Run Bundled Sample` from the settings sheet to exercise the
real inference path on a known card image.

## Nice To Have Later

- use `sounds/scan.wav` for successful scan confirmation
- use `sounds/pickup_mid.wav` for cards over `$0.25`
- use `sounds/pickup_high.wav` for cards over `$5`
- tiny crop thumbnail beside each confirmed scan
- card count badges for duplicates
- better offline startup messaging
