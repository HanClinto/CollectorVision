# CollectorVision Web Scanner

Static, browser-only card scanner used by the GitHub Pages demo.

## Runtime

The scanner captures camera frames, detects and dewarps a card, embeds the
crop, and searches a packed float16 catalog in a worker. WASM is the safe
default inference backend. WebGPU is opt-in and disabled where known runtime
bugs make its output unreliable.

Catalog v2 is the default:

- `BrowserCatalogV2.forGame("mtg")` follows the live Catalog v2 feed.
- Combined records and packed embeddings are cached in IndexedDB.
- Card names are available immediately; Scryfall supplies current prices after
  confirmation.

Append `?catalog=v1` to use the prepared static catalog bundle. This is the
compatibility path, not the default. `?channel=testing` selects the separately
published testing model bundle.

The standalone [`catalog_v2_example.html`](./catalog_v2_example.html) loads any
published game catalog and displays its first record.

## Generated assets

`assets/` and `vendor/` are generated and ignored by Git. The prepared stable
and testing bundles are published as GitHub release assets. Pages downloads
those bundles during deployment; it does not build models or the v1 catalog.

The asset refresh workflow runs monthly and can also be started manually.
Pages deploys on relevant source changes, its own schedule, or manual dispatch.
These workflows are independent.

## Local development

From the repository root:

```bash
./scripts/run_web_scanner_local.sh
```

This rebuilds the generated bundle and serves the scanner at
<http://localhost:8040>.

To run the steps manually:

```bash
uv run python scripts/export_web_scanner_assets.py
cd examples/web_scanner
uv run python -m http.server 8040
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the runtime and deployment
boundaries.

## Embeddable applet

The experimental applet reuses the scanner worker without the full demo UI:

- [`lib/collectorvision-scanner-applet.mjs`](./lib/collectorvision-scanner-applet.mjs)
- [`applet_example.html`](./applet_example.html)

```js
import { createCollectorVisionScannerApplet } from "./lib/collectorvision-scanner-applet.mjs";

const scanner = await createCollectorVisionScannerApplet({
  target: "#collectorvision",
  matchThreshold: 0.5,
  consecutiveMatches: 2,
  scanIntervalMs: 900,
  groupBySecondaryId: true,
  overlay: true,
  onCardDetected(card) {
    console.log(card.cardId, card.secondaryId, card.score);
  },
});
```

The applet expects the standard `assets/`, `vendor/`, and
`scanner.worker.mjs` layout on the same origin.

## Diagnostics

Use **Run Bundled Sample** in Settings to exercise the inference path without a
camera. Model benchmarks are available at:

- WASM: `model_benchmark.html?backend=wasm&download=md`
- WebGPU: `model_benchmark.html?backend=webgpu&download=md`
