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
- **Always run on WASM** — see Lessons Learned below.
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
- `vendor/onnxruntime-web/ort.webgpu.min.mjs` — new WebGPU EP (not the legacy JSEP bundle)
- `vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs` — asyncify WASM for WebGPU EP fallback
- `vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm`

> **Do NOT use** `ort.all.min.mjs` or `ort-wasm-simd-threaded.jsep.*` — see Lessons Learned.

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
- no desktop-first layout

---

## Lessons Learned: onnxruntime-web Execution Providers

### Do NOT use the legacy JSEP backend (`ort.all.min.mjs`)

The legacy bundle (`ort.all.min.mjs` + `ort-wasm-simd-threaded.jsep.wasm`) uses
the JSEP (JavaScript EP) backend for WebGPU.  It silently returns all-zeros for
Conv operator outputs across **all ort-web versions 1.20–1.24.3** on Android
(replicated on Chrome/Chromium on ARM, Adreno, and Mali GPUs).  The outputs look
valid (non-NaN, non-Inf) so inference appears to succeed while producing completely
wrong results.  Fixed by switching to the *new* WebGPU EP (see below).

Commit: `aa0f88f fix(webgpu): switch to new WebGPU EP (ort.webgpu.min.mjs)`

### Use `ort.webgpu.min.mjs` (new WebGPU EP) for the embedder

The new EP (`ort.webgpu.min.mjs` + `ort-wasm-simd-threaded.asyncify.wasm`) fixes
the JSEP Conv bug.  It works correctly for `milo.onnx` (embedder) on all tested
devices and gives a significant speedup on desktop and high-end mobile GPUs.

The new EP requires the **asyncify** WASM variant, not the jsep variant:
```
ort.webgpu.min.mjs                         ← new EP entry point
ort-wasm-simd-threaded.asyncify.mjs/.wasm  ← required WASM fallback
```

### Always run `cornelius.onnx` (corner detector) on WASM

Even with the new WebGPU EP, `cornelius.onnx` produces numerically wrong
outputs on Android ARM GPU architectures (confirmed armv81 Chrome 147,
ort-web 1.24.3).  The outputs are coherent (non-zero, convex quads that
pass `isUsableQuad`) but point to an entirely wrong region of the frame.
Wrong corners cascade silently through dewarp → embed → search with no
error signal.

This is distinct from the JSEP all-zeros bug; it is a model-specific
numerical issue with the new WebGPU EP on ARM.

`cornelius` is small enough that WASM completes comfortably within the
900 ms scan interval.  Always use `executionProviders: ["wasm"]` for it.

Commit: `c8defb1 fix: force corner detector (cornelius) to WASM-only`
Diagnosed from: issue #9 capture bundle (build 7ed8f8f)
