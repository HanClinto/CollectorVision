# CollectorVision — Road to v0.1.0

Checklist for turning the scaffold into a shippable library.

---

## 1. Core library

### 1a. identify() ✅
- [x] `Identifier.identify(*images)` with single-image and multi-frame voting
- [x] Load/detect/dewarp/embed pipeline
- [x] Per-frame `frame_results` returned when multiple images are passed
- [x] Cosine and Hamming retrieval in `collector_vision/retrieval.py`

### 1b. NeuralCornerDetector (Cornelius) ✅
- [x] ONNX-based inference via `onnxruntime` — no PyTorch at runtime
- [x] SimCC sharpness gate (mean peak of 8 softmax distributions) instead of
      unreliable presence logit
- [x] Bundled as `collector_vision/weights/cornelius.onnx` (8.2 MB, single file)

### 1c. NeuralEmbedder (Milo) ✅
- [x] ONNX-based inference via `onnxruntime`
- [x] 128-d L2-normalised embeddings from 448×448 input
- [x] Bundled as `collector_vision/weights/milo.onnx` (5.0 MB, single file)

### 1d. CannyCornerDetector ✅
- [x] Contour-based detection in `collector_vision/detectors/canny.py`
- [x] Returns `DetectionResult(found=False)` when no valid quadrilateral found

### 1e. Retrieval helpers ✅
- [x] `collector_vision/retrieval.py` — `cosine_search()` and `hamming_search()`

### 1f. CardResult — crop image (optional, per-call)
- [ ] Add `include_crop: bool = False` parameter to `Identifier.identify()`
  - When True, attach the dewarped card BGR image (or a JPEG bytes) to `CardResult`
  - Useful for server responses, debugging, and the ScanBucket UI preview
  - Return the intermediate `DetectionResult` as well (similar to how multi-image
    calls return per-frame `frame_results`)
  - Keep it optional and off by default — most library users don't need it

### 1g. Metadata lookup (future module)
`identify()` returns IDs only. A thin lookup helper is planned but not blocking v0.1.0:

- [ ] `sources/scryfall.py` — `get(scryfall_id) -> dict` via Scryfall REST API,
  with local SQLite cache
- [ ] `sources/tcgplayer.py` — `get(tcgplayer_id) -> dict` with price data

---

## 2. Model weights ✅

- [x] Corner detector (Cornelius) — `cornelius.onnx` (8.2 MB, merged from export)
- [x] Embedder (Milo) — `milo.onnx` (5.0 MB, merged from export)
- [x] Both are single-file ONNX, no paired `.data` file
- [x] Bundled in `collector_vision/weights/`; `package_data` configured in `pyproject.toml`
- [ ] Mirror both to HuggingFace Hub (`CollectorVision/models` or similar)
  - Write model card: architecture, training data, input spec, license, accuracy table
- [ ] Verify weights survive `python -m build` → wheel → fresh venv install

---

## 3. Catalog format

CollectorVision consumes catalog NPZ files — it does not build them.
Catalog construction lives in **CollectorVision-Pipeline** (section 14).

**Required NPZ keys:**

| Key | Shape / type | Description |
|---|---|---|
| `embeddings` | (N, D) float32 or (N, B) uint8 | Embedding matrix |
| `card_ids` | (N,) str | Primary key per row (e.g. Scryfall UUID) |
| `source` | scalar str | `"scryfall"`, `"tcgplayer"`, … |
| `mode` | scalar str | `"embedding"` or `"hash"` |
| `embedder_spec` | scalar str | JSON spec for reconstructing the embedder |

Card names and metadata are not stored in the catalog — callers use the returned
ID to look up metadata (e.g. via Scryfall API or a local catalog).

- [ ] Document the NPZ format fully in `collector_vision/catalog.py` module docstring
- [ ] `tests/test_catalog.py` — `Catalog.load()` round-trips a synthetic NPZ; missing
      optional keys handled gracefully; `_merge()` rejects incompatible specs
- [x] Confirm `HFD` → `Catalog.load()` → `identify()` works end-to-end against live HF

---

## 4. HuggingFace setup

- [x] `HanClinto/milo` — model repo hosting Milo weights + catalogs (`catalogs/*.npz`)
- [ ] Upload `cornelius.onnx` and `milo.onnx` to HF Hub with model cards
- [ ] Write model cards: architecture, training data, input spec, license, accuracy table
- [x] `HFD("HanClinto/milo", "scryfall-mtg").resolve()` confirmed end-to-end ✅

---

## 5. PyPI publishing

### 5a. pyproject.toml polish
- [ ] `[project.urls]` — Homepage, Repository, Bug Tracker
- [ ] `classifiers` — Development Status, Intended Audience, Topic, License,
      Programming Language
- [ ] `readme = "README.md"` under `[project]`
- [ ] Add `python-multipart` to `[server]` extra (required by FastAPI file uploads)
- [ ] Add `build`, `twine` to `[dev]` extra
- [ ] Verify `python -m build` produces clean sdist + wheel

### 5b. First publish
- [ ] Create PyPI account / org
- [ ] Publish to Test PyPI first
- [ ] Smoke-test from scratch in a fresh venv
- [ ] Publish to PyPI

### 5c. GitHub Actions — publish on tag
- [ ] `.github/workflows/publish.yml` — trigger on `v*` tags, build + twine upload

---

## 6. CI/CD

### 6a. Test and lint on push
- [ ] `.github/workflows/ci.yml`
  - Trigger: push to main, pull requests
  - Matrix: Python 3.10, 3.11, 3.12
  - Steps: install deps → ruff → pytest

### 6b. Dependabot
- [ ] Enable Dependabot for `pyproject.toml`
- [ ] Enable GitHub security advisories

---

## 7. Testing

### 7a. Unit tests
- [ ] `test_hfd.py` — mock manifest; stale/fresh cache; `cache_refresh=None`; eviction
- [ ] `test_games.py` — `parse_game()` happy + error paths; enum values
- [ ] `test_catalog.py` — synthetic NPZ load, `_merge()`, incompatible spec rejection
- [ ] `test_retrieval.py` — cosine and hamming search correctness + top-k ordering
- [ ] `test_canny_detector.py` — CannyCornerDetector on a synthetic card image

### 7b. Integration tests
- [ ] `tests/integration/test_identify.py`
  - Synthetic catalog NPZ + test card image (small, checked in)
  - `Catalog.load("test_catalog.npz")` returns correct ID
  - Multi-image voting with `frame_results`
  - `include_crop=True` returns a crop image
  - Gated by `pytest -m integration` (requires bundled weights)

### 7c. Smoke test (post-install)
- [ ] `tests/smoke/test_install.py`
  - `import collector_vision as cvg` — no error
  - `cvg.__version__` is a string
  - `cvg.Game.MTG` accessible
  - `cvg.HFD` callable
  - `cvg.weights.check()` returns expected keys

---

## 8. Documentation

- [ ] Expand README quickstart with real output examples
- [ ] API reference — docstrings on all public classes
- [ ] CONTRIBUTING.md — dev setup, test commands, PR process
- [ ] CHANGELOG.md — start at 0.1.0.dev0
- [ ] How-to: build a catalog (points to Pipeline)
- [ ] Consider ReadTheDocs or GitHub Pages

---

## 9. Legal / licensing

- [ ] Add contact details to COMMERCIAL_LICENSE.md
- [ ] SPDX license headers in each Python source file
- [ ] Decide on catalog data license (check Scryfall ToS re: derived works)
- [ ] Verify `LICENSE` file (full AGPL-3.0 text) exists

---

## 10. Polish and UX

- [ ] Progress bars in `hfd.py` downloads (optional `tqdm`)
- [ ] Replace `print()` in `hfd.py` with `logging`
- [ ] Better error if `HFD` download fails with no local cache
- [ ] `collector_vision/py.typed` (PEP 561 type stubs marker)

---

## 11. Evaluation and benchmarks

### 11a. Benchmark dataset
- [ ] ~500–1000 card images (varied: phone, flatbed, video, backgrounds)
  uploaded to `CollectorVision/benchmark-v1` on HF Datasets
- [ ] Ground-truth manifest CSV (`scryfall_id` or `pokemontcg_id` per image)

### 11b. Eval harness
- [ ] `eval/benchmark.py` — CLI; runs `Identifier.identify()` on each image;
  reports top-1/top-3 accuracy + latency; writes `results.csv`

### 11c. Published results
- [ ] Results table in README
- [ ] Embed in Milo model card on HF Hub
- [ ] HF Space — live demo (upload image → identified card)

---

## 12. API server

> Reference: `ccg_card_id/07_web_scanner`. A minimal version is already in
> `examples/server/server.py`. The production port below adds multi-catalog,
> browser UI, and Docker.

### 12a. API format (Ximilar-compatible)
```
POST /v1/identify
  body:     {"records": [{"_base64": "...", "catalog": "scryfall-mtg"}]}
  response: {"records": [{...}], "_status": {"code": 200, "text": "OK"}}

GET /v1/health
GET /v1/catalogs    — lists loaded catalog names
GET /v1/defaults
```

### 12b. Full server port
- [ ] Multi-catalog support — dict of `name → Catalog`
- [ ] Copy browser UI + ScanBucket client from `07_web_scanner/client/`
- [ ] SSL self-signed cert generation (LAN camera access)
- [ ] `collectorvision-server` CLI entry point (`pyproject.toml`)

### 12c. Packaging
- [ ] `pip install collectorvision[server]`
- [ ] Docker image; publish to GHCR on `v*` tags

### 12d. Hosted demo
- [ ] HF Space (`CollectorVision/demo`)

---

## 13. Mobile

### Strategy A — API-backed (ship now)
- [ ] Document REST API for mobile developers
- [ ] React Native client package (npm)
- [ ] Flutter client package (pub.dev)
- [ ] Swift / iOS — URLSession wrapper, Swift Package Manager
- [ ] Kotlin / Android — OkHttp wrapper, Maven

### Strategy B — On-device

#### B1. ONNX models ✅
- [x] Cornelius exported to ONNX (`cornelius.onnx`, 8.2 MB) and verified
- [x] Milo exported to ONNX (`milo.onnx`, 5.0 MB) and verified
- [ ] Upload to HF Hub alongside future `.pt` reference files

#### B2. Android
- [ ] ONNX Runtime for Android; Android Archive (AAR) on Maven Central
- [ ] Bundle phash16 catalog (~3 MB) for offline; milo1 catalog streamed on demand

#### B3. iOS
- [ ] CoreML conversion via `coremltools`; Swift package `CollectorVisionKit`

#### B4. On-device catalog considerations
- [ ] Catalog size tiers: phash16 ~3 MB (bundleable), milo1 ~54 MB (stream on first use)
- [ ] Consider flat binary format for faster mobile load vs NPZ

---

## 14. CollectorVision-Pipeline (separate project)

> Suggested repo: `github.com/HanClinto/CollectorVision-Pipeline`

### 14a. Data sources
- [ ] Scryfall — sync `default_cards.json`, download PNGs (~108k)
- [ ] Pokémon TCG API — sync all cards, download images
- [ ] Future: Yu-Gi-Oh, Flesh and Blood, Lorcana, Digimon, One Piece, DBS

### 14b. Catalog builder
- [ ] `pipeline/build_catalog.py` — writes `{algo}-{source}-{game}-{YYYY-MM}.npz`
  with keys: `embeddings`, `card_ids`, `source`, `mode`, `embedder_spec`

### 14c. Publishing
- [ ] `pipeline/upload_catalog.py` — upload NPZ to `HanClinto/milo` under `catalogs/`,
      update `manifest.json` at repo root

### 14d. Automation
- [ ] GitHub Actions monthly refresh
- [ ] Incremental mode (only re-embed changed images)

---

## Milestone summary

| Milestone | Status | Key items |
|---|---|---|
| **M0 — Code complete** | ✅ | `identify(*images)`, Cornelius + Milo wired, retrieval, Canny |
| **M1 — Weights finalized** | ✅ | `cornelius.onnx` + `milo.onnx` bundled; single-file, clean names |
| **M1.5 — Examples** | ✅ | `examples/identify_image.py`, `examples/server/` |
| **M2 — First catalog** | ✅ | `milo1-scryfall-mtg` built, uploaded to `HanClinto/milo`, `HFD` resolves it |
| **M3 — End-to-end works** | ✅ | `pip install -e .`, smoke test passes, `Catalog.for_game(Game.MTG)` confirmed |
| **M4 — Full catalog set** | ⬜ | Magic + Pokémon milo1 + phash16 catalogs live |
| **M5 — PyPI v0.1.0** | ⬜ | CI green, tests pass, published to PyPI |
| **M6 — Automated** | ⬜ | Dependabot, docs site, CHANGELOG |
| **M6p — Pipeline v1** | ⬜ | CollectorVision-Pipeline repo; first catalogs built and published |
| **M7 — Benchmark** | ⬜ | Public benchmark on HF, eval harness, results in README |
| **M8 — API server** | ⬜ | Full web_scanner port, Docker, HF Space demo |
| **M9 — Mobile (API)** | ⬜ | React Native + Flutter packages |
| **M10 — Mobile (on-device)** | ⬜ | Android AAR, iOS Swift package |
