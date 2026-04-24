# Web Scanner Architecture

## Goal

Ship a static scanner demo that runs fully in the browser and can be hosted on
GitHub Pages.

Primary path only:
- camera -> Cornelius -> JS dewarp -> Milo -> local cosine search
- mobile-first UI -> running scan list -> exportable results

## Browser Pipeline

### 1. Capture

- Use `navigator.mediaDevices.getUserMedia`.
- Prefer `facingMode: "environment"`.
- Draw the visible video region into an offscreen canvas.
- Design around portrait mode first.

### 2. Detection

- Load `cornelius.onnx` with `onnxruntime-web`.
- Use the WebGPU execution provider only.
- Match Python preprocessing:
  - BGR/RGB handling
  - resize to model input
  - ImageNet mean/std normalization
- Output:
  - `corners`
  - `sharpness`
  - `card_present`

### 3. Dewarp

- Convert normalized corners to pixel coordinates.
- Solve the perspective transform in JS.
- Warp the crop in JS into the canonical output.
- Emit the canonical `252x352` crop.

### 4. Embedding

- Load `milo.onnx` with `onnxruntime-web`.
- Match Python preprocessing.
- Emit `128`-d float32 embedding.
- L2-normalize in JS even if the model already does so.

### 5. Search

- Load embeddings into a `Float32Array`.
- Compute cosine similarity in a tight JS loop.
- Track only the top `k` scores during the scan.
- Return the aligned card ID and metadata row.

### 6. Metadata Enrichment

- Live metadata can be fetched from Scryfall after a card is confirmed.
- Scryfall should enrich the UI, not block the recognition loop.

### 7. Audio Hooks

- play a confirmation sound as soon as a card is confirmed
- later, after Scryfall returns price data:
  - play `pickup_high.wav` for cards above `$5`
  - play `pickup_mid.wav` for cards above `$0.25`

## UI Shape

- top half: live camera with overlay
- top app bar and camera block, with a toggle between compact and expanded camera
- bottom half: running list of confirmed scans
- sticky action row:
  - copy text
  - download CSV
  - clear list
- settings live in a bottom sheet, not in the main scanning surface

Even on desktop, keep the mobile layout centered and narrow.

## Asset Layout

Use a manifest so the app can stay dumb and static:

```json
{
  "version": "0.1.0",
  "models": {
    "cornelius": "models/cornelius.onnx",
    "milo": "models/milo.onnx"
  },
  "catalog": {
    "embeddings": "catalog/scryfall-mtg-embeddings.f16.bin",
    "card_ids": "catalog/scryfall-mtg-card-ids.json",
    "rows": 108354,
    "dims": 128,
    "dtype": "float16"
  },
  "sample_frame": "samples/mtg-sample.jpg"
}
```

The browser app should fetch those files from local `./assets/...` URLs. Keep
Hugging Face as the source used by the export step, not as a direct runtime
dependency for GitHub Pages.

Do the same for the browser runtimes:
- `vendor/onnxruntime-web/ort.all.min.mjs`
- `vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs`
- `vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm`

## Caching Strategy

- First load fetches the manifest and large assets over HTTP.
- Persist heavy assets in `IndexedDB`.
- On later loads, revalidate with manifest version/hash and reuse the cache.

This avoids paying the full gallery/model download cost on every visit.

Suggested stores:
- `assets`

## Publish Shape

- No framework required.
- Static files only.
- Host from `examples/web_scanner/` or a built `docs/` output later.
- GitHub Pages serves:
  - `index.html`
  - `style.css`
  - `app.js`
  - asset files

Long-term deploy model:

- `main` holds app source
- a separate asset refresh workflow builds the heavy generated bundle
- that bundle is published as a GitHub release asset
- Pages deploy fetches the prepared bundle from GitHub, not from HF

See:

- [ASSET_DEPLOY_PLAN.md](./ASSET_DEPLOY_PLAN.md)
- [assets.bundle.json](./assets.bundle.json)

## Non-Goals

- no server lookup
- no pHash
- no Canny fallback
- no non-WebGPU backend
- no desktop-first layout
