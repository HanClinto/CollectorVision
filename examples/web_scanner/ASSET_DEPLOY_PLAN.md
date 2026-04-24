# Web Scanner Asset Deployment

## Problem

The GitHub Pages scanner must serve its catalog and model assets from the same
published site.

That means the browser needs local copies of:

- `assets/catalog/scryfall-mtg-embeddings.f16.bin`
- `assets/catalog/scryfall-mtg-card-ids.json`
- `assets/models/cornelius.onnx`
- `assets/models/milo.onnx`
- vendored browser runtime files

Today those generated files are checked into git and published directly. That is
simple, but it bloats `main` with large build artifacts.

## Goals

- keep the browser app fully static
- keep `main` focused on source, not generated blobs
- avoid HF fetches during normal Pages deploys
- only rebuild the asset bundle when the upstream catalog actually changes
- keep the deploy path easy to reason about

## Recommended Shape

Use two separate concerns:

1. `refresh-web-scanner-assets`
2. `deploy-pages`

The key rule is:

- the asset refresh workflow may talk to Hugging Face
- the normal Pages deploy workflow should only talk to GitHub

## Source Of Truth

Use a GitHub release asset as the durable artifact store.

Recommended release name:

- `web-scanner-assets`

Recommended uploaded files:

- `web-scanner-assets.tar.zst`
- `web-scanner-assets-metadata.json`

Why release assets:

- durable across workflow runs
- clean separation between source and generated output
- easy for Actions to download
- better fit than using git history as a blob store

## Metadata Contract

The bundle metadata should be explicit and human-readable.

Suggested shape:

```json
{
  "bundle_version": "2026-04-24",
  "catalog_key": "hf://HanClinto/milo/scryfall-mtg",
  "catalog_fingerprint": "sha256:...",
  "catalog_rows": 108354,
  "catalog_dims": 128,
  "models": {
    "cornelius": "sha256:...",
    "milo": "sha256:..."
  },
  "vendor": {
    "onnxruntime_web": "1.24.3",
    "opencv_js": "4.12.0-release.1"
  },
  "generated_at": "2026-04-24T16:22:00Z"
}
```

The important field is `catalog_fingerprint`.

That should change when the HF source catalog changes. It can be derived from:

- HF ETag
- HF commit SHA
- SHA256 of the downloaded `.npz`

Any of those are fine. The point is to compare upstream state cheaply and
deterministically.

## Workflow Design

### 1. Refresh Workflow

Trigger:

- `workflow_dispatch`
- optional schedule

Responsibilities:

- inspect the current HF catalog fingerprint
- inspect the published release metadata
- stop early if the fingerprint is unchanged
- otherwise:
  - download the HF catalog
  - run `scripts/export_web_scanner_assets.py`
  - package `assets/` and `vendor/`
  - upload the new tarball and metadata to the `web-scanner-assets` release

Important property:

- if HF has not changed, no rebuild happens

### 2. Pages Deploy Workflow

Trigger:

- pushes that affect the web scanner app
- `workflow_dispatch`

Responsibilities:

- checkout the repo
- download the latest prepared release asset from GitHub
- unpack the generated bundle into the Pages staging directory
- upload the final Pages artifact

Important property:

- normal app deploys do not touch HF
- deploy speed depends on GitHub only

## Pages Layout

The deployed Pages folder should still look like a normal static site:

- `index.html`
- `style.css`
- `app.js`
- `assets/**`
- `vendor/**`

The difference is only where those heavy generated files come from.

In source control, the checked-in app code stays small. In the deploy artifact,
the full runtime bundle is restored before publish.

## Local Development

Local development should stay simple.

Use:

```bash
./.venv/bin/python scripts/export_web_scanner_assets.py
cd examples/web_scanner
python -m http.server 8040
```

So:

- local dev still writes directly into `examples/web_scanner`
- CI later gains an `--out-dir` path for cleaner staging

## Suggested Repo Metadata

Add one small checked-in pointer file:

- `examples/web_scanner/assets.bundle.json`

Suggested contents:

```json
{
  "release": "web-scanner-assets",
  "catalog_key": "hf://HanClinto/milo/scryfall-mtg",
  "bundle_channel": "latest"
}
```

This file is not the generated asset manifest used by the browser.

It is only the repo-level deploy pointer that tells CI which prepared bundle it
expects to fetch.

## Why Not Just Use Actions Cache

Actions cache is useful as an optimization, but not as the source of truth.

If the cache evicts:

- deploys become slow again
- behavior becomes inconsistent

So the clean design is:

- release assets are the durable store
- Actions cache is optional icing on top

## Why Not Keep Assets In A Separate Branch

A dedicated branch is acceptable as a fallback, but it is not the preferred
design.

Branch pros:

- keeps `main` cleaner than checking blobs into source
- easy to inspect in the repo UI

Branch cons:

- still uses git as a generated artifact store
- accumulates noisy blob history
- weaker separation between source and published build output

Recommendation:

- prefer release assets
- use a dedicated branch only if you specifically want the files browseable in
  the repository UI

If a branch is used, name it something explicit like:

- `web-scanner-assets`

and treat it as generated output only.

## Rollout Plan

### Phase 1

- keep the current checked-in assets while the scanner is still changing quickly
- add the repo-level bundle pointer file
- teach the export script to emit into a staging directory

### Phase 2

- add the asset refresh workflow
- publish the first release-hosted asset bundle and metadata

### Phase 3

- update Pages deploy to fetch the prepared release bundle
- verify that ordinary UI deploys no longer touch HF

### Phase 4

- stop tracking large generated assets in `main`
- keep only source files and the small bundle pointer file in git

## Final Recommendation

Use GitHub release assets as the durable catalog/model bundle store.

Then:

- rebuild only when the HF catalog fingerprint changes
- deploy the app often without rebuilding the catalog
- keep `main` clean

That is the cleanest compromise between transparency, speed, and operational
simplicity.
