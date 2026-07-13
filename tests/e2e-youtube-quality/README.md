# YouTube E2E Quality Harness Plan

This folder is for deterministic, debuggable comparisons between CollectorVision scanner runs across Cornelius detector models, catalogs, and scan parameters.

The goal is not a pytest-style pass/fail test. It is a repeatable command-line harness that turns a YouTube video into sampled frames, runs one or more scanner configurations, and writes artifacts that make model quality easy to compare.

## Primary CLI

Script: `run_youtube_quality.py`

Example:

```bash
python tests/e2e-youtube-quality/run_youtube_quality.py \
  --video-id dQw4w9WgXcQ \
  --fps 2 \
  --run baseline:collector_vision/weights/cornelius.onnx:hf://HanClinto/milo/scryfall-mtg \
  --run candidate:/path/to/cornelius.onnx:/path/to/catalog.npz \
  --min-sharpness 0.02 \
  --presence-threshold 0.5 \
  --top-k 5 \
  --output-dir build/e2e-youtube-quality
```

### Inputs

- `--video-id`: YouTube video ID or URL.
- `--frames-dir`: existing image-frame directory for no-network smoke/debug runs.
- `--fps`: frame sampling rate, default `2`.
- `--run LABEL:CORNELIUS:CATALOG`: repeatable scanner definition.
- `--min-sharpness`: passed to `NeuralCornerDetector.detect()`.
- `--min-corner-quality`: minimum `0..1` geometric plausibility score for the detected corners, default `0`; the score is recorded for every detection, and positive values opt into early rejection before dewarp.
- `--presence-threshold`: passed to `NeuralCornerDetector(...)` for older Cornelius checkpoints without sharpness.
- `--min-score`: minimum best-match score for inclusion in `seen_cards.csv`, default `0.0`.
- `--top-k`: number of recognition hits logged per detected card.
- `--no-rot-invariant`: opt out of the default upright + 180-degree rotated recognition search.
- `--max-frames`: optional debug limiter for short smoke runs.
- `--force-frames`: re-extract frames even when cached `frames/frame_*.jpg` files already exist.
- `--discard-video`: optional flag to delete downloaded source media after frame extraction.
- `--save-crops`: optional flag to write dewarped card crops for visual inspection.
- `--no-viewer`: skip automatic `viewer.html` generation.

### Model And Catalog Resolution

The current Python detector constructor accepts local ONNX paths. The current catalog loader accepts local `.npz` paths and `hf://` catalog URIs. The harness should use those existing library conventions rather than inventing a second naming scheme.

Supported forms:

- Cornelius local absolute or relative path: `/path/to/cornelius.onnx`.
- Cornelius bundled default: use the installed `collector_vision.weights.CORNER_DETECTOR` path when omitted.
- Catalog local absolute or relative path: `/path/to/catalog.npz`.
- Catalog HuggingFace URI supported by `Catalog.load()`: `hf://user/repo/catalog-key`, for example `hf://HanClinto/milo/scryfall-mtg`.

Catalog HuggingFace caching should remain delegated to the library's existing `HFD.resolve()` path. No v1 HuggingFace model-file syntax is needed for Cornelius unless the library adds one later.

If a future detector model resolver is added, resolved remote model files can be cached under:

```text
~/.cache/collectorvision/e2e-youtube-quality/models/{sha-or-safe-name}/
```

This is separate from YouTube media caching. `yt-dlp` handles its own download/cache behavior; the harness keeps stable output paths and reuses extracted `frames/frame_*.jpg` files by default. Pass `--force-frames` to regenerate frames with `ffmpeg`.

Viewer card names are resolved at report-generation time from Scryfall bulk data, cached under `~/.cache/collectorvision/e2e-youtube-quality/scryfall/` or `$COLLECTORVISION_CACHE/e2e-youtube-quality/scryfall/`. The browser reads names from the generated HTML payload and does not query Scryfall when opened.

Corner quality is documented in `docs/corner_quality.md`. For comparison runs, prefer `--min-corner-quality 0` so the generated viewer can apply threshold changes interactively without rerunning detection.

## Processing Pipeline

1. Create an output directory for the video and parameterized run group.
2. Download or locate the source media with `yt-dlp`.
3. Split frames with `ffmpeg` using deterministic naming and timestamps.
4. Load each run's catalog with `Catalog.load()`.
5. For each run, instantiate `NeuralCornerDetector(cornelius, presence_threshold=...)`, use `catalog.embedder` for the catalog-compatible Milo embedder, run detect -> dewarp -> embed -> catalog.search() for each frame, and write frame-level JSONL plus optional overlays/crops.
6. Aggregate recognized cards into chronological CSV summaries per run and an optional comparison table across runs.

## Output Layout

```text
build/e2e-youtube-quality/
  {video-id}/
    frames/
      frame_000001.jpg
      frame_000002.jpg
    runs/
      baseline/
        run_config.json
        frames.jsonl
        seen_cards.csv
        overlays/
          frame_000001.jpg
        crops/
          frame_000001.jpg
      candidate/
        run_config.json
        frames.jsonl
        seen_cards.csv
        overlays/
    compare_seen_cards.csv
    manifest.json
```

## Frame Log Schema

`frames.jsonl` should contain one JSON object per frame per run.

Fields:

- `run_label`
- `video_id`
- `frame_index`
- `frame_path`
- `timestamp_sec`
- `card_present`
- `confidence`
- `sharpness`
- `presence`
- `corner_quality`: scalar geometric quality, acceptance flag, weakest failing reason, and diagnostic sub-metrics
- `corners_normalized`
- `corners_pixels`
- `crop_path`
- `orientation`: chosen recognition orientation, either `upright` or `rotated_180` when a card is recognized
- `orientation_results`: per-orientation top-k results for debugging upside-down recognition
- `top_k`: list of `{rank, card_id, score, oracle_id}` when available
- `best_card_id`
- `best_score`
- `duration_ms`: detector, dewarp, embed, search, total
- `error`: nullable diagnostic string

This JSONL file is the source of truth for debugging and viewer playback.

## Seen Cards CSV

`seen_cards.csv` should be one row per unique recognized card per run, ordered chronologically by first appearance in the video. V1 should assume normal duplicate prevention is enabled, so the final CSV reads like the ordered list of cards seen in the video.

Suggested columns:

- `run_label`
- `card_id`
- `oracle_id`
- `first_seen_frame`
- `first_seen_timestamp_sec`
- `last_seen_frame`
- `last_seen_timestamp_sec`
- `count_frames_best_match`
- `max_score`
- `mean_score`
- `best_orientation`
- `best_frame_path`
- `best_overlay_path`
- `best_crop_path`

`compare_seen_cards.csv` should outer-join these summaries across run labels so regressions are easy to spot.

## Viewer

Script: `viewer.py`

The viewer should be dependency-free static HTML generated from `manifest.json` plus `frames.jsonl`. It should be easy to archive alongside a run and open directly from disk.

Features:

- frame scrubber with timestamp and frame index;
- configurable sharpness threshold, default `0.02`, to treat low-sharpness detections as non-detections in the comparison view;
- configurable match score threshold to treat low-confidence recognition results as non-detections in the comparison view and detected-card summary;
- frame-by-frame comparison table with one row per video frame, comparing each run at the same frame index only;
- comparison mode toggle for exact card ID versus oracle/secondary ID;
- optional condensation of agreeing frame rows, keeping the first row of each agreeing group visible;
- previous/next disagreement navigation that skips over frames where runs agree;
- strongest agreement and strongest disagreement jumps for quick triage;
- bottom detected-card summary showing per-run chronological card lists, shared cards, run-only cards, and a side-by-side chronological comparison table;
- summary tabs for grouping by oracle/secondary ID, which ignores minor edition variations when available, or by exact card ID;
- side-by-side columns for each run;
- original frame with corner polygon overlay;
- dewarped crop preview;
- top-k recognition table with scores;
- visual highlight when two runs disagree on best card;
- links to raw frame, overlay, crop, and JSON record.

The CLI can generate overlays during the run with OpenCV drawing, so the viewer does not need canvas geometry in the first version.

Report regeneration is cheap when `frames.jsonl` and `seen_cards.csv` already exist: rerun `viewer.py <run-dir>` to rebuild the static HTML from cached raw results without running detection, recognition, `yt-dlp`, or `ffmpeg`.

## Determinism

- Pin frame extraction to explicit `ffmpeg -vf fps={fps}` output names.
- Persist `yt-dlp`, `ffmpeg`, Python, CollectorVision, model paths, model hashes, catalog source, catalog hash, and parameter values in `manifest.json`.
- Process frames in sorted filename order.
- Avoid random sampling in the default path.
- Record all command lines used for download and extraction.
- Keep v1 aligned to the current single-card Cornelius detector. Do not add multi-card schema or viewer behavior until the detector pipeline supports it explicitly.

## Suggested Implementation Steps

1. Add `run_youtube_quality.py` with one `--run`, local Cornelius path, `Catalog.load()` support, frame extraction, `frames.jsonl`, and chronological `seen_cards.csv`.
2. Add multi-run support and `compare_seen_cards.csv`.
3. Add model/catalog hash and cache metadata.
4. Add overlay/crop output flags.
5. Add static `viewer.py` generation from existing run artifacts.
6. Add a tiny smoke test mode using `--max-frames` against local fixture frames so the harness can be checked without network access.