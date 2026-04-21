# CollectorVision — Road to v0.1.0

Checklist for turning the scaffold into a shippable library.

---

## 1. Core library (stubs → real code)

### 1a. identify()
- [ ] Implement `identify()` in `collector_vision/identify.py`
  - Load image (path or ndarray)
  - Run detector (or skip if detector=None, treating full image as card)
  - Dewarp card to fixed resolution
  - Call `gallery.embedder.embed()` on the dewarped card
  - Nearest-neighbour lookup in the gallery (cosine for floats, Hamming for hashes)
  - Return `CardResult` (best match + alternatives)
- [ ] Implement `identify_batch()` (batched detect + embed, single NN pass)

### 1b. NeuralCornerDetector
- [ ] Wire up `_load()` with the trained SimCC checkpoint
  - Build / import the model architecture from the ccg_card_id corner detector
  - Strip ArcFace / training heads; keep backbone + SimCC head
  - Load state dict, set eval mode
- [ ] Wire up `detect()` — preprocess → forward → decode SimCC heatmaps → normalised corners
- [ ] Export a clean inference-only checkpoint (~10 MB target)
- [ ] Place checkpoint at `collector_vision/weights/corner_detector.pt`
- [ ] Update `collector_vision/weights/__init__.py` to expose `CORNER_DETECTOR` path

### 1c. NeuralEmbedder (Milo)
- [ ] Wire up `_load()` — backbone + linear projection, eval mode
- [ ] Wire up `embed()` — resize → normalize → batched forward → L2 normalise → ndarray
- [ ] Export inference-only checkpoint (~12 MB target; quantize INT8 if needed)
- [ ] Place checkpoint at `collector_vision/weights/embedder.pt`
- [ ] Update `collector_vision/weights/__init__.py` to expose `EMBEDDER` path

### 1d. CannyCornerDetector
- [ ] Implement real contour-based card detection in `detectors/canny.py`
  - Canny edges → findContours → largest quadrilateral → normalised corners
  - Return DetectionResult with `found=False` if no valid quad found

### 1e. Nearest-neighbour retrieval helper
- [ ] Add `collector_vision/retrieval.py`
  - `cosine_search(query_vec, gallery_embeddings)` → sorted (score, idx)
  - `hamming_search(query_bits_u8, gallery_bits_u8)` → sorted (distance, idx)
  - Used internally by `identify()`; not part of the public API

### 1f. Catalog helpers
- [ ] Implement `collector_vision/catalogs/scryfall.py` — lightweight wrapper to
  resolve scryfall_id → human-readable card data (optional, for richer CardResult)
- [ ] Consider whether catalog lookups should be online-optional or purely from
  metadata already embedded in the gallery NPZ

---

## 2. Model weights

- [ ] **Finalize corner detector checkpoint** from ccg_card_id training
  - Pick best epoch (currently training, ~epoch 131–155)
  - Strip optimizer / scheduler state; keep only model weights
  - Verify inference on a sample image before bundling
- [ ] **Finalize Milo embedding checkpoint**
  - Current best: `mobilevit_xxs_ft_illustration_id+set_code_e15_128d` (v2light_img448_ph10 epoch 15)
  - Strip optimizer state
  - Target ≤ 12 MB (quantize if needed)
- [ ] **Mirror both weights to HuggingFace Hub**
  - Repo: `CollectorVision/milo` (model hub, not datasets)
  - Files: `corner_detector.pt`, `embedder.pt`
  - Include model card describing architecture, training data, license
- [ ] **Bundle both weights in the PyPI package** (via `MANIFEST.in` + `package_data`)
  - Verify `collector_vision/weights/*.pt` is included in the wheel
  - Run `python -c "import collector_vision; print(collector_vision.weights.EMBEDDER)"` after install

---

## 3. Gallery format (CollectorVision's side of the contract)

CollectorVision is a *consumer* of gallery NPZ files — it does not build them.
Gallery construction, data scraping, and publishing live in the companion project
**CollectorVision-Pipeline** (see section 14). This section covers only what
CollectorVision itself needs to know about the format.

- [ ] **Document the NPZ format** in `collector_vision/gallery.py` module docstring
  - All required keys (`embeddings`, `card_ids`, `ids_json`, `embedder_spec`, …)
  - `embedder_spec` JSON schema with all supported `kind` values
  - Versioning / forward-compatibility story (unknown keys must be ignored)
- [ ] **Spec test** — `tests/test_gallery_format.py`
  - Synthesize a minimal compliant NPZ and verify `Gallery.load()` round-trips cleanly
  - Verify missing optional keys are handled gracefully (backward compatibility)
- [ ] **Update bundled manifest** in `collector_vision/manifest.py` (`_BUNDLED_MANIFEST`)
  once Pipeline has published the first real gallery files to HF Datasets
- [ ] Confirm `Manifest.fetch()` → `Identifier(Game.MAGIC)` → `identify()` works end-to-end
  against live HF Datasets files

---

## 4. HuggingFace setup

- [ ] Create HF organization `CollectorVision`
- [ ] Create HF Datasets repo `CollectorVision/galleries`
  - Set license (AGPL-3.0 for derived embeddings; check Scryfall ToS re: derived works)
  - README explaining the gallery NPZ format, embedder_spec, and how to use with the library
  - Actual gallery uploads happen via CollectorVision-Pipeline (see section 14)
- [ ] Create HF Hub repo `CollectorVision/milo`
  - Write model card: architecture (MobileViT-XXS), training data (Scryfall),
    input spec (448×448 RGB, L2-normalised 128-d output), license
- [ ] Set repo visibility (public after initial galleries are uploaded)
- [ ] Confirm `Manifest.fetch()` resolves and downloads correctly end-to-end

---

## 5. PyPI publishing

### 5a. pyproject.toml polish
- [ ] Pin dependency lower bounds to tested minimum, not latest
- [ ] Add `[project.urls]` — Homepage, Repository, Bug Tracker
- [ ] Add `classifiers` — Development Status, Intended Audience, Topic, License,
      Programming Language
- [ ] Add `readme = "README.md"` under `[project]`
- [ ] Add `[project.optional-dependencies]`
  - `cpu` — `torch` CPU-only variant instructions (document in README; not pip-installable)
  - `hash` — only `Pillow`, `imagehash`, `scipy` (no torch dependency)
  - `dev` — `pytest`, `ruff`, `build`, `twine`
- [ ] Verify `python -m build` produces a clean sdist + wheel

### 5b. First publish
- [ ] Create PyPI account / org for CollectorVision
- [ ] Publish to Test PyPI first: `twine upload --repository testpypi dist/*`
- [ ] Smoke-test `pip install --index-url https://test.pypi.org/simple/ collectorvision`
- [ ] Publish to PyPI: `twine upload dist/*`
- [ ] Verify `pip install collectorvision` works from scratch in a fresh venv

### 5c. GitHub Actions — publish on tag
- [ ] `.github/workflows/publish.yml`
  - Trigger: push of `v*` tags
  - Steps: checkout → build → twine upload (using PyPI trusted publisher / OIDC)
  - Gate: only publish if tests pass

---

## 6. CI/CD

### 6a. Test and lint on push
- [ ] `.github/workflows/ci.yml`
  - Trigger: push to main, pull requests
  - Matrix: Python 3.10, 3.11, 3.12
  - Steps: install deps (no torch GPU) → ruff → pytest

### 6b. Monthly gallery refresh
- [ ] `.github/workflows/gallery_refresh.yml`
  - Trigger: schedule (1st of each month) + manual `workflow_dispatch`
  - Steps:
    - Sync Scryfall `default_cards.json`
    - Sync Pokémon TCG API
    - Run `galleries/build_gallery.py` for each game×variant
    - Upload resulting NPZ files to HF Datasets
    - Update `manifest.json` on HF Datasets
    - Open a PR to update `_BUNDLED_MANIFEST` in `manifest.py`
  - Note: requires HF token secret + secrets management

### 6c. Dependabot / renovate
- [ ] Enable Dependabot for `pyproject.toml` dependencies
- [ ] Enable GitHub security advisories / Dependabot alerts

---

## 7. Testing

### 7a. Unit tests (`tests/`)
- [ ] `test_games.py` — `parse_game()` happy + error paths
- [ ] `test_manifest.py` — bundled manifest, `resolve()`, error on unknown game/variant
- [ ] `test_gallery.py` — `Gallery.load()` with synthetic NPZ, `_merge()`,
      incompatible spec rejection
- [ ] `test_hash_embedder.py` — phash, dhash, marr_hildreth on a 1×1 test image
- [ ] `test_fixed_detector.py` — FixedCornerDetector returns supplied corners as-is
- [ ] `test_canny_detector.py` — CannyCornerDetector on a synthetic card image
- [ ] `test_identify_stubs.py` — verify NotImplementedError raised cleanly
      (to be flipped to real tests once identify() is implemented)

### 7b. Integration tests
- [ ] `tests/integration/test_identify.py`
  - Requires a small bundled test card image + synthetic gallery NPZ
  - End-to-end: `Identifier(Game.MAGIC).identify(img)` returns the correct card
  - Run only when weights are present (`pytest -m integration`)

### 7c. Smoke test (post-install)
- [ ] `tests/smoke/test_install.py`
  - `import collector_vision as cvg` — imports without error
  - `cvg.__version__` is a string
  - `cvg.Embedding` and `cvg.Game` enums are importable
  - No-GPU, no-network, no weights required

---

## 8. Documentation

- [ ] **API reference** — add/complete docstrings on all public classes and functions
- [ ] **Quickstart tutorial** in README (already started; expand with images/outputs)
- [ ] **How-to: add a new game** — gallery naming, source adapter, manifest entry
- [ ] **How-to: train a custom embedder** — points to ccg_card_id training scripts
- [ ] **How-to: build a gallery** — run `build_gallery.py`, upload to HF
- [ ] **CONTRIBUTING.md** — dev setup, test commands, PR process
- [ ] **CHANGELOG.md** — start at 0.1.0.dev0, commit-linked
- [ ] Consider hosting generated API docs on ReadTheDocs or GitHub Pages
      (Sphinx + autodoc, or mkdocs-material)

---

## 9. Legal / licensing

- [ ] Add contact details to COMMERCIAL_LICENSE.md once email / form is ready
- [ ] Add SPDX license header comment to each Python source file
  (`# SPDX-License-Identifier: AGPL-3.0-or-later`)
- [ ] Decide on gallery data license (gallery NPZs contain embeddings of
  Scryfall images — check Scryfall ToS re: derived works)
- [ ] Add `LICENSE` file (full AGPL-3.0 text) if not already present
- [ ] Verify HF Datasets repo license is set correctly

---

## 10. Polish and UX

- [ ] **Progress bars** — use `tqdm` during gallery downloads (optional dependency)
- [ ] **Logging** — replace `print()` in `_download()` with `logging.getLogger(__name__)`
- [ ] **Better errors** — if gallery download fails, suggest `offline=True` and cache path
- [ ] **Version check** — warn if installed package is older than the manifest version
  (gallery format may have changed)
- [ ] **Device string validation** — friendly error if user passes `device="gpu"`
- [ ] **Type stubs / py.typed marker** — add `collector_vision/py.typed` (PEP 561) so
  type checkers know the package ships inline types

---

## 11. Evaluation and benchmarks

The goal: reproducible, public numbers so users can see what accuracy to expect
and compare variants against each other and against alternatives.

### 11a. Benchmark dataset
- [ ] **Define and publish a small public benchmark corpus**
  - Target: ~500–1000 card images covering a range of capture conditions
    (phone camera, flatbed scan, video frame, various lighting/backgrounds)
  - Split by capture type: clean scan / phone clear-bg / phone cluttered-bg / video
  - Cover multiple card games (Magic, Pokémon at minimum)
  - Include ground-truth card IDs in a manifest CSV
  - Upload to HF Datasets as `CollectorVision/benchmark-v1`
  - License images carefully (CC-BY or original photographer consent)
- [ ] **Define evaluation metrics**
  - Top-1 and Top-3 artwork accuracy (matches illustration_id)
  - Top-1 and Top-3 edition accuracy (matches exact card_id / printing)
  - Per-capture-condition breakdowns
  - Query latency (ms/card, CPU and GPU)
  - Gallery size (bytes/card)

### 11b. Eval harness
- [ ] **`eval/benchmark.py`** — standalone CLI evaluation script
  - Downloads benchmark dataset from HF if not cached
  - Accepts `--gallery magic` or `--gallery-file path.npz`
  - Accepts `--embedding phash milo` (sweep)
  - Runs `identify()` on each benchmark image
  - Reports per-condition and overall accuracy + latency table
  - Writes `results.csv` and `results.md`
- [ ] **Results reproducibility** — pin gallery version (YYYY-MM) in results so
  comparisons are against the same reference set
- [ ] **Baseline comparisons to include**
  - `phash16` (hash, no GPU)
  - `milo1` (neural, GPU recommended)
  - Canny detector vs neural detector (ablation on detection quality)
  - Optionally: Ximilar / other commercial APIs as reference points
    (require user to supply their own API key)

### 11c. Published results
- [ ] **Results table in README** — top-1 edition accuracy by Embedding × condition
  - Keep updated with each gallery release
  - Mark GPU/CPU requirement per Embedding
- [ ] **HuggingFace Model Card** (`CollectorVision/milo`) — embed results table
- [ ] **HuggingFace Space — live demo**
  - Gradio app: upload an image → shows detected corners → identified card + confidence
  - Dropdown to select Game and Embedding
  - Hosted on HF Spaces (free tier for initial launch)
  - Link from README and PyPI page
- [ ] **Versioned results archive** in `CollectorVision/galleries` dataset repo
  - `eval_results/benchmark-v1/{embedding}-{YYYY-MM}.json` for each gallery release

---

## 12. API server

> **Reference implementation:** `ccg_card_id/07_web_scanner` is a working
> FastAPI server with an Ximilar-compatible API, browser client, SSL support,
> and ScanBucket deduplication. The CollectorVision server is a port of this,
> replacing ccg_card_id's data paths with `Identifier` / `Gallery.for_game()`.
> Do not redesign the API shape — adopt it wholesale.

### 12a. API format (Ximilar-compatible — keep as-is from web_scanner)

```
POST /v1/identify
  body: {"records": [{"_base64": "...", "detector": "canny", "embedding": "phash"}]}
  response: {"records": [{...result...}], "_status": {"code": 200, "text": "OK"}}

GET /v1/health
GET /v1/detectors
GET /v1/embeddings   (replaces web_scanner's /v1/identifiers)
GET /v1/defaults
GET /v1/memory
```

Benefits of Ximilar compatibility: client code is swappable between
CollectorVision and Ximilar Visual AI without modification.

### 12b. Port from web_scanner
- [ ] Copy `07_web_scanner/server/app.py` → `server/app.py`
- [ ] Copy `07_web_scanner/client/` → `server/client/` (browser UI + ScanBucket)
- [ ] Replace `GallerySearchManager` (disk-scan of ccg_card_id paths) with
  `Identifier(game, embedding=embedding)` — gallery downloads from HF automatically
- [ ] Replace `CardLookup` (SQLite) with metadata already in the gallery NPZ
  (`card_names`, `set_codes`, `ids_json`)
  - Price data: omit for now (YAGNI until there's a plan for keeping it fresh)
- [ ] Remove `sys.path` hacks and `ccg_card_id.config` dependency
- [ ] Rename `/v1/identifiers` → `/v1/embeddings` to match enum rename
- [ ] Keep SSL self-signed cert generation (required for LAN camera access)

### 12c. Packaging
- [ ] **`pyproject.toml` extras** — `pip install collectorvision[server]`
  adds `fastapi`, `uvicorn`, `python-multipart`
- [ ] **Entry point** — `collectorvision-server` CLI
  - `collectorvision-server --game magic --embedding phash --port 8080`
- [ ] **Docker image** — `Dockerfile` in `server/`
- [ ] **Publish to GHCR** on `v*` tags via GitHub Actions

### 12d. Hosted demo
- [ ] **HuggingFace Space** (`CollectorVision/demo`) — Gradio or the browser
  client pointing at a hosted instance; rate-limited

---

## 13. Mobile

Two strategies, in ascending complexity. Start with the API approach; add
on-device later as demand warrants.

### Strategy A — API-backed (ship now, works immediately)

- [ ] **Document the REST API** clearly so mobile developers can integrate
  against a self-hosted or hosted instance
- [ ] **Reference mobile clients** (thin wrappers, not full apps)
  - [ ] **React Native** — `packages/react-native-collectorvision/`
    - `identify(imageUri, options)` → Promise<CardResult>
    - Handles multipart upload to configured server URL
    - Typed with TypeScript definitions
    - Published to npm as `react-native-collectorvision`
  - [ ] **Flutter** — `packages/flutter_collectorvision/`
    - `CollectorVision.identify(File image, {String game})` → Future<CardResult>
    - Published to pub.dev as `collectorvision`
  - [ ] **Swift / iOS native** — `CollectorVisionClient.swift`
    - Thin URLSession wrapper, available as a Swift package
  - [ ] **Kotlin / Android native** — thin OkHttp wrapper, published to Maven

### Strategy B — On-device inference (future, heavier lift)

On-device removes the server dependency and latency, enabling offline use and
reducing privacy concerns. Requires model conversion work.

#### B1. ONNX export (prerequisite for both platforms)
- [ ] **Export corner detector to ONNX**
  - `torch.onnx.export(model, dummy_input, "corner_detector.onnx", opset_version=17)`
  - Verify output matches PyTorch reference on a set of test images (< 1% diff)
  - Run `onnxsim` to simplify the graph
  - Target: < 10 MB after simplification
- [ ] **Export Milo embedder to ONNX**
  - Same process; verify L2-normalised output matches reference
  - Target: < 15 MB
- [ ] **Upload ONNX files to HF Hub** (`CollectorVision/milo`) alongside `.pt`
- [ ] **Hash embedder** — no ONNX needed; port DCT/wavelet logic natively per platform

#### B2. Android
- [ ] **ONNX Runtime for Android** — add to `android/` module
  - `implementation("com.microsoft.onnxruntime:onnxruntime-android:...")`
  - Preprocess: Bitmap → float32 tensor, normalise
  - Run corner detector → decode SimCC → warp ROI
  - Run embedder → L2 normalise → cosine search against bundled gallery
- [ ] **Bundle gallery** — include phash16 gallery NPZ in assets for offline use
  (milo1 gallery is too large; phash16 is ~3 MB for Magic)
- [ ] **Android Archive (AAR)** — publishable library
  - Publish to Maven Central or GitHub Packages
  - Artifact: `com.collectorvision:collectorvision-android`
- [ ] **Sample Android app** — demonstrates camera capture → identify → display result

#### B3. iOS
- [ ] **CoreML conversion** (preferred over ONNX Runtime on iOS for ANE access)
  - `coremltools.convert(onnx_model, ...)` → `CornerDetector.mlpackage`
  - `coremltools.convert(...)` → `MiloEmbedder.mlpackage`
  - Verify outputs match ONNX reference
- [ ] **Swift package** — `CollectorVisionKit`
  - Wraps CoreML model inference
  - `CollectorVisionKit.identify(pixelBuffer:) async throws -> CardResult`
  - Published via Swift Package Manager (GitHub URL)
- [ ] **XCFramework** — for CocoaPods / Carthage users
- [ ] **Sample iOS app** — AVFoundation camera → identify → display result

#### B4. Cross-platform (optional, higher reach)
- [ ] **React Native on-device** using ONNX Runtime React Native
  - `ort-react-native` package for model inference
  - Single JS API for both platforms
- [ ] **Flutter on-device** using `onnxruntime` Flutter package
- [ ] **Capacitor / Ionic plugin** for web-app-style mobile apps

#### B5. On-device gallery considerations
- [ ] **Gallery format for mobile** — the standard NPZ works but may be slow to load
  - Consider a flat binary format: header (N, D, dtype) + raw matrix
  - Or SQLite with a BLOB column (easy random access)
- [ ] **Gallery size tiers**
  - phash16 Magic: ~3.4 MB — fine to bundle in app
  - milo1 Magic: ~54 MB — too large; stream on first use, cache locally
  - On-device default should be phash16 unless device has Neural Engine / GPU
- [ ] **Incremental gallery updates** — download only new/changed cards between
  gallery versions rather than re-downloading the full NPZ

---

## 14. CollectorVision-Pipeline (separate project)

> **This work belongs in a separate repository**, not here. Most CollectorVision
> users never need it — it is maintainer tooling for keeping galleries up to date.
> Tracked here only for cross-project visibility.
>
> Suggested repo: `github.com/HanClinto/CollectorVision-Pipeline`
> Suggested PyPI package: `cvg-pipeline` (dev/maintainer install only, not end-user)

The Pipeline project is responsible for the full data → gallery lifecycle:
- Scraping card data from upstream APIs
- Downloading reference images
- Building and publishing gallery NPZ files
- Updating the manifest on HF Datasets
- Automated monthly refresh

### 14a. Data sources

#### Magic — Scryfall
- [ ] Sync `default_cards.json` (English printings, ~80k cards)
- [ ] Sync `all_cards.json` (all languages, ~517k cards) — optional, for multilingual galleries
- [ ] Download reference card images (PNG fronts, ~108k files, ~60 GB)
- [ ] Store locally in a structured cache; track ETags for incremental sync
- [ ] Respect Scryfall rate limits and bulk-data guidelines

#### Pokémon — TCGplayer / Pokémon TCG API
- [ ] Sync all-cards JSON from Pokémon TCG API (~18k cards, paginated)
- [ ] Download card images (large PNG, ~18k files)
- [ ] Handle API pagination and rate limits
- [ ] Build SQLite catalog with synthetic `illustration_id` (sha1 of name + artist)

#### Future game sources
- [ ] Yu-Gi-Oh — YGOPRODeck API or Konami official data
- [ ] Flesh and Blood — Legend Story Studios official card data
- [ ] Lorcana — Ravensburger official / community sources
- [ ] Digimon, One Piece, Dragon Ball Super — community APIs

### 14b. Gallery builder
- [ ] **`pipeline/build_gallery.py`** — generic CLI
  - Args: `--game magic --source scryfall --embedding milo1 --date 2026-04`
  - Reads local image cache built by the sync step
  - Instantiates the appropriate CollectorVision embedder (hash or neural)
  - Embeds all reference images in batches
  - Writes `{game}-{source}-{algo}-{YYYY-MM}.npz` with full metadata
    including `embedder_spec`, `ids_json`, `card_names`, `set_codes`
- [ ] **Checkpointing** — resume interrupted builds (gallery for Magic = ~108k images,
  several hours on CPU)
- [ ] **Verification step** — after build, sample 100 random gallery entries and
  confirm retrieval of known cards before uploading
- [ ] **Size budget** — warn if a gallery exceeds 100 MB (check mobile implications)

### 14c. Publishing
- [ ] **`pipeline/upload_gallery.py`** — upload NPZ + update manifest on HF Datasets
  - Requires `HUGGINGFACE_TOKEN` env var with write access to `CollectorVision/galleries`
  - Atomically updates `manifest.json` after upload succeeds
  - Archives the previous month's gallery (keep last 3 versions)
- [ ] **Trigger bundled manifest PR** — after upload, open a PR against
  CollectorVision repo to update `_BUNDLED_MANIFEST` in `manifest.py`
- [ ] **Gallery changelog** — record count of new/changed/removed cards per release

### 14d. Automation
- [ ] **GitHub Actions — monthly refresh**
  - Trigger: schedule (1st of month) + manual `workflow_dispatch`
  - Steps: sync data → download new images → build galleries → verify → upload → PR
  - Requires: HF token secret, sufficient Actions storage/compute (large runner or
    self-hosted for GPU embedding)
  - Estimated runtime: ~6 hours for Magic milo1 on a mid-range GPU
- [ ] **Incremental mode** — on monthly runs, only re-embed cards whose source
  image has changed (use ETag / mtime tracking)
- [ ] **Failure alerting** — notify (GitHub issue or email) if monthly build fails

### 14e. Separation of concerns checklist
Things that must stay in CollectorVision-Pipeline (not in CollectorVision):
- [ ] Any scraping or API-calling code for Scryfall, TCGplayer, etc.
- [ ] SQLite card catalogs (source of truth for metadata during build)
- [ ] Raw image storage and download management
- [ ] `build_gallery.py` and `upload_gallery.py`
- [ ] HF Datasets write credentials
- [ ] Training scripts (those live in ccg_card_id for now)

Things that must stay in CollectorVision (not in Pipeline):
- [ ] Gallery NPZ format spec and `Gallery.load()`
- [ ] `Manifest` class (read-only consumer of the manifest JSON)
- [ ] All inference code (identify, embedders, detectors)
- [ ] `_BUNDLED_MANIFEST` (updated via PR from Pipeline, not auto-committed)

---

## Milestone summary

| Milestone | Key items |
|---|---|
| **M0 — Code complete** | identify() + batch, Canny detector, retrieval helper |
| **M1 — Weights finalized** | Corner + embedder checkpoints exported, bundled, mirrored to HF Hub |
| **M2 — First gallery** | magic-scryfall-phash16 built + uploaded + downloadable |
| **M3 — End-to-end works** | `pip install`, `Identifier(Game.MAGIC).identify()` returns a card |
| **M4 — Full gallery set** | magic milo1, pokemon phash16 + milo1 all live |
| **M5 — PyPI v0.1.0** | CI green, tests pass, published to PyPI |
| **M6 — Automated** | Monthly gallery refresh CI, dependabot, docs site |
| **M6p — Pipeline v1** | CollectorVision-Pipeline repo; Scryfall + Pokémon sync; first galleries built and uploaded |
| **M7 — Benchmark** | Public benchmark dataset on HF, eval harness, results in README |
| **M8 — API server** | FastAPI server, Docker image on GHCR, HF Space demo live |
| **M9 — Mobile (API)** | React Native + Flutter packages published, API-backed |
| **M10 — Mobile (on-device)** | ONNX export, Android AAR, iOS Swift package |
