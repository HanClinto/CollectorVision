# CollectorVision — Road to v0.1.0

Checklist for turning the scaffold into a shippable library.

---

## 1. Core library

### 1a. Pipeline API ✅
- [x] Explicit detect → dewarp → embed → search pipeline (no magic wrapper class)
- [x] `DetectionResult.dewarp(bgr)` → PIL Image
- [x] `Catalog.search(emb, top_k)` → `[(score, card_id), ...]`
- [x] `min_sharpness` gate on `detect()` — blurry/blank frames return `card_present=False`
- [x] Cosine retrieval in `collector_vision/retrieval.py`
- [x] `examples/identify_image.py` walks through all five steps including Scryfall lookup
- [x] `examples/custom_identifier.py` shows swapping in Canny detector or pHash embedder

### 1b. NeuralCornerDetector (Cornelius) ✅
- [x] ONNX-based inference via `onnxruntime` — no PyTorch at runtime
- [x] SimCC sharpness gate (mean peak of 8 softmax distributions) instead of
      unreliable presence logit
- [x] Bundled as `collector_vision/weights/cornelius.onnx` (8.2 MB, single file)

### 1c. NeuralEmbedder (Milo) ✅
- [x] ONNX-based inference via `onnxruntime`
- [x] 128-d L2-normalised embeddings from 448×448 input
- [x] Bundled as `collector_vision/weights/milo.onnx` (5.0 MB, single file)
- [ ] **Re-export with dynamic batch dimension** for throughput-oriented use cases
      (bulk eval, catalog building).  Currently exported at batch_size=1 so every
      image requires a separate `sess.run()` call — the Python loop overhead adds up.

      **How to re-export:**
      The source checkpoint is a PyTorch `EmbeddingNet` / `MultiTaskEmbeddingNet`
      in `ccg_card_id/04_build/mobilevit_xxs/models.py`.  Export with:
      ```python
      torch.onnx.export(
          model,
          dummy_input,                        # shape (1, 3, 448, 448)
          "milo.onnx",
          input_names=["input"],
          output_names=["embedding"],
          dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
          opset_version=17,
      )
      ```
      After export, verify that passing a (N, 3, 448, 448) batch produces (N, 128)
      output, then update `NeuralEmbedder.embed()` to stack the batch into a single
      `sess.run()` call instead of looping.  The `.onnx.data` sidecar file is normal
      for large models (weights stored externally) — both files must be kept together.

      **Same applies to Cornelius** (`collector_vision/weights/cornelius.onnx`),
      though batching the detector is less valuable since detection misses are
      decided per-frame and batching would complicate the sharpness gate logic.

### 1d. Retrieval helpers ✅
- [x] `collector_vision/retrieval.py` — `cosine_search()` (cosine similarity over L2-normalised vectors)

### 1f. Metadata lookup (future module)
The pipeline returns IDs only. A thin lookup helper is planned but not blocking v0.1.0:

- [ ] `sources/scryfall.py` — `get(scryfall_id) -> dict` via Scryfall REST API,
  with local SQLite cache
- [ ] `sources/tcgplayer.py` — `get(tcgplayer_id) -> dict` with price data

---

## 2. Model weights ✅

- [x] Corner detector (Cornelius) — `cornelius.onnx` (8.2 MB, single file)
- [x] Embedder (Milo) — `milo.onnx` (5.0 MB, single file)
- [x] Both bundled in `collector_vision/weights/`; `package_data` configured in `pyproject.toml`
- [x] Both uploaded to HF Hub (`HanClinto/cornelius`, `HanClinto/milo`) with model cards
- [x] Verify weights survive `python -m build` → wheel → fresh venv install

---

## 3. Catalog format

CollectorVision consumes catalog NPZ files — it does not build them.
Catalog construction lives in **CollectorVision-Pipeline** (section 14).

**Required NPZ keys:**

| Key | Shape / type | Description |
|---|---|---|
| `embeddings` | (N, D) float32 | Embedding matrix |
| `card_ids` | (N,) str | Primary key per row (e.g. Scryfall UUID) |
| `source` | scalar str | `"scryfall"`, `"tcgplayer"`, … |
| `embedder_spec` | scalar str | JSON spec for reconstructing the embedder |

Card names and metadata are not stored in the catalog — callers use the returned
ID to look up metadata via Scryfall API or a game-specific data source.

- [ ] Document the NPZ format fully in `collector_vision/catalog.py` module docstring
- [ ] `tests/test_catalog.py` — `Catalog.load()` round-trips a synthetic NPZ; missing
      optional keys handled gracefully; `_merge()` rejects incompatible specs
- [x] `Catalog.load("hf://HanClinto/milo/scryfall-mtg")` → `search()` confirmed end-to-end ✅

---

## 4. HuggingFace setup ✅

- [x] `HanClinto/milo` — model repo hosting Milo weights + catalogs (`catalogs/*.npz`)
- [x] `HanClinto/cornelius` — model repo hosting Cornelius weights
- [x] Model cards written for both (architecture, input spec, usage examples, AGPL-3.0)
- [x] `Catalog.load("hf://HanClinto/milo/scryfall-mtg")` confirmed end-to-end ✅

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
- [ ] `test_retrieval.py` — cosine search correctness + top-k ordering

### 7b. Integration tests
- [x] `examples/identify_image.py --smoke-test` — headless pipeline check (no real card needed)
- [ ] `tests/integration/test_pipeline.py`
  - Synthetic catalog NPZ + small test card image (checked in)
  - Full detect → dewarp → embed → search returns the correct card ID
  - Multi-frame score aggregation produces a better result than single-frame
  - Gated by `pytest -m integration` (requires bundled weights)

### 7c. Smoke test (post-install)
- [ ] `tests/smoke/test_install.py`
  - `import collector_vision as cvg` — no error
  - `cvg.__version__` is a string
  - `cvg.Game.MTG`, `cvg.Catalog`, `cvg.NeuralCornerDetector` accessible
  - `cvg.HFD` callable

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
- [ ] `eval/benchmark.py` — CLI; runs the full detect → dewarp → embed → search pipeline
  on each image; reports top-1/top-3 accuracy + latency; writes `results.csv`

### 11c. Published results
- [ ] Results table in README
- [ ] Embed in Milo model card on HF Hub
- [ ] HF Space — live demo (upload image → identified card)

---

## 12. API server

> A minimal version is already in `examples/server/server.py`.
> The production port below adds multi-catalog, browser UI, and Docker.

### 12a. API format
```
POST /identify
  body:     {"records": [{"_base64": "..."}]}
  response: {"records": [{"card_id": "...", "confidence": 0.94, ...}]}

GET /health
GET /catalogs    — lists loaded catalog names
```

### 12b. Full server port
- [ ] Multi-catalog support — dict of `name → Catalog`
- [ ] Copy browser UI + ScanBucket client from `07_web_scanner/client/`
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
- [x] Both uploaded to HF Hub

#### B2. Android
- [ ] ONNX Runtime for Android; Android Archive (AAR) on Maven Central
- [ ] Bundle small milo1 catalog subset for offline; full catalog streamed on demand

#### B3. iOS
- [ ] CoreML conversion via `coremltools`; Swift package `CollectorVisionKit`

#### B4. On-device catalog considerations
- [ ] Catalog size tiers: milo1 ~54 MB (stream on first use); consider flat binary format
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
  with keys: `embeddings`, `card_ids`, `source`, `embedder_spec`

### 14c. Publishing
- [ ] `pipeline/upload_catalog.py` — upload NPZ to `HanClinto/milo` under `catalogs/`,
      update `catalogs/manifest.json`

### 14d. Automation
- [ ] GitHub Actions monthly refresh
- [ ] Incremental mode (only re-embed changed images)

---

## Milestone summary

| Milestone | Status | Key items |
|---|---|---|
| **M0 — Code complete** | ✅ | Explicit pipeline API, Cornelius + Milo wired, Catalog.search() |
| **M1 — Weights finalized** | ✅ | `cornelius.onnx` + `milo.onnx` bundled and on HF Hub with model cards |
| **M1.5 — Examples** | ✅ | `examples/identify_image.py` (5-step walkthrough + smoke test), `examples/server/` |
| **M2 — First catalog** | ✅ | `milo1-scryfall-mtg` built, uploaded to `HanClinto/milo`, `hf://` URI confirmed |
| **M3 — End-to-end works** | ✅ | `pip install -e .`, smoke test passes, full pipeline verified |
| **M4 — Full catalog set** | ⬜ | Magic + Pokémon milo1 catalogs live |
| **M5 — PyPI v0.1.0** | ⬜ | CI green, tests pass, published to PyPI |
| **M6 — Automated** | ⬜ | Dependabot, docs site, CHANGELOG |
| **M6p — Pipeline v1** | ⬜ | CollectorVision-Pipeline repo; first catalogs built and published |
| **M7 — Benchmark** | ⬜ | Public benchmark on HF, eval harness, results in README |
| **M8 — API server** | ⬜ | Full web_scanner port, Docker, HF Space demo |
| **M9 — Mobile (API)** | ⬜ | React Native + Flutter packages |
| **M10 — Mobile (on-device)** | ⬜ | Android AAR, iOS Swift package |
