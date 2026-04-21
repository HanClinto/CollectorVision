# CollectorVision — Road to v0.1.0

Checklist for turning the scaffold into a shippable library.

---

## 1. Core library (stubs → real code)

### 1a. identify()
- [ ] Implement `identify()` in `collector_vision/identify.py`
  - Load image (path or ndarray)
  - Run detector (or accept pre-supplied corners / FULL_IMAGE_CORNERS)
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

## 3. Gallery builder

- [ ] **Create `galleries/build_gallery.py`** — generic builder script
  - Accepts: game, source, algo variant, snapshot date
  - Reads reference images from the data source
  - Embeds with the specified embedder (hash or neural)
  - Writes `{game}-{source}-{algo}-{YYYY-MM}.npz` with all required keys
  - Stores `embedder_spec` JSON inside the NPZ
- [ ] **Build Magic / Scryfall galleries**
  - `magic-scryfall-milo1-YYYY-MM.npz` (neural, ~108k cards, ~54 MB at 128-d float32)
  - `magic-scryfall-phash16-YYYY-MM.npz` (hash, ~108k cards, ~3.4 MB at 256-bit)
- [ ] **Build Pokémon / TCGplayer galleries**
  - `pokemon-tcgplayer-milo1-YYYY-MM.npz`
  - `pokemon-tcgplayer-phash16-YYYY-MM.npz`
- [ ] **Upload galleries to HuggingFace Datasets**
  - Org: `CollectorVision`
  - Dataset repo: `CollectorVision/galleries`
  - Also upload `manifest.json`
- [ ] **Update bundled manifest** in `collector_vision/manifest.py` (`_BUNDLED_MANIFEST`)
  to include the real filenames and set `version` to the snapshot date

---

## 4. HuggingFace setup

- [ ] Create HF organization `CollectorVision`
- [ ] Create HF Datasets repo `CollectorVision/galleries`
  - Set license to AGPL-3.0 (or CC-BY for data alone, TBD)
  - README explaining the gallery file format and embedder_spec
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
  - End-to-end: `identify(img, gallery=test_gallery)` returns the correct card
  - Run only when weights are present (`pytest -m integration`)

### 7c. Smoke test (post-install)
- [ ] `tests/smoke/test_install.py`
  - `import collector_vision as cvg` — imports without error
  - `cvg.__version__` is a string
  - `cvg.FULL_IMAGE_CORNERS` has correct shape
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

## Milestone summary

| Milestone | Key items |
|---|---|
| **M0 — Code complete** | identify() + batch, Canny detector, retrieval helper |
| **M1 — Weights finalized** | Corner + embedder checkpoints exported, bundled, mirrored to HF Hub |
| **M2 — First gallery** | magic-scryfall-phash16 built + uploaded + downloadable |
| **M3 — End-to-end works** | `pip install`, `Gallery.for_game("magic")`, `identify()` returns a card |
| **M4 — Full gallery set** | magic milo1, pokemon phash16 + milo1 all live |
| **M5 — PyPI v0.1.0** | CI green, tests pass, published to PyPI |
| **M6 — Automated** | Monthly gallery refresh CI, dependabot, docs site |
