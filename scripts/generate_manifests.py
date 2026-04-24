#!/usr/bin/env python3
"""Generate per-pipeline manifests for captured .json.gz frames.

For each capture, this script produces manifests for offline pipelines:

    tests/fixtures/captures/cv_TIMESTAMP.python.json
    tests/fixtures/captures/cv_TIMESTAMP.python.dewarp.png
    tests/fixtures/captures/cv_TIMESTAMP.js-webgpu.json
    tests/fixtures/captures/cv_TIMESTAMP.js-webgpu.dewarp.png

The ``js-webgpu`` manifest uses browser-reported corners from the capture
bundle (``orderedCorners`` field) and runs Python dewarp/embed on them.  It
represents what the browser pipeline produced at capture time, with the
embedding step applied consistently by Python.

The ``js-cpu`` manifest is written separately by ``tests/js/test_pipeline.mjs``
when you run ``npm test`` in that directory.

Manifest schema (shared across all pipelines)::

    {
      "pipeline":      "python",              # pipeline identifier
      "source":        "offline",             # "offline" | "live"
      "captureId":     "cv_TIMESTAMP",        # stem of the .json.gz file
      "cardPresent":   true,
      "sharpness":     0.046,
      "corners":       [[x, y], ...],         # TL, TR, BR, BL — normalised [0,1]
      "dewarpPng":     "cv_TIMESTAMP.python.dewarp.png",  # filename in same dir
      "embedding":     [0.12, ...],           # L2-normalised float32 vector
      "topMatchId":    "ace86fac-...",        # top-1 Scryfall card ID (null if no catalog)
      "topMatchScore": 0.987
    }

Usage::

    # regenerate manifests for all captures (with catalog for topMatchId)
    python scripts/generate_manifests.py

    # skip catalog lookup (topMatchId will be null; fast / no network)
    python scripts/generate_manifests.py --no-catalog

    # specific capture files only
    python scripts/generate_manifests.py tests/fixtures/captures/cv_*.json.gz

This script is also importable by ingest_bug_reports.py.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

CAPTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "captures"

# Dewarp output size — must match interfaces.py _DEWARP_W / _DEWARP_H
_DEWARP_W = 252
_DEWARP_H = 352

# Tolerance for the reproduction check (normalised units; same as JS test)
CORNER_TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _capture_id(path: Path) -> str:
    """Return the bare capture stem, e.g. ``cv_2026-04-24T16-43-41``."""
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    return path.stem


def _load_frame(bundle: dict):
    """Decode the ``framePng`` field to a BGR numpy array (cv2 format)."""
    import cv2

    png_bytes = base64.b64decode(bundle["framePng"])
    bgr = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return bgr


def _dewarp_pil(bgr, corners_xy):
    """Perspective-warp ``bgr`` to a 252×352 PIL RGB image.

    Parameters
    ----------
    bgr:
        Full-frame BGR uint8 numpy array.
    corners_xy:
        Four ``[x_norm, y_norm]`` pairs in TL, TR, BR, BL order, normalised
        to ``[0, 1]``.  Matches the format returned by the Python detector
        and stored in ``orderedCorners`` in the capture bundle.
    """
    import cv2
    from PIL import Image

    h, w = bgr.shape[:2]
    src = np.array(corners_xy, dtype=np.float32) * np.array([w, h], dtype=np.float32)
    dst = np.array(
        [[0, 0], [_DEWARP_W - 1, 0], [_DEWARP_W - 1, _DEWARP_H - 1], [0, _DEWARP_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bgr, M, (_DEWARP_W, _DEWARP_H))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def _empty_manifest(pipeline: str, source: str, capture_id: str) -> dict:
    return {
        "pipeline": pipeline,
        "source": source,
        "captureId": capture_id,
        "cardPresent": None,
        "sharpness": None,
        "corners": None,
        "dewarpPng": None,
        "embedding": None,
        "topMatchId": None,
        "topMatchScore": None,
    }


# ---------------------------------------------------------------------------
# Public manifest generators
# ---------------------------------------------------------------------------

def _extract_detector_input_png(bundle: dict, out_dir: Path, cap_id: str) -> Path | None:
    """Save the browser's detector input pixels as a .npy file for exact comparison.

    Reads ``detectorInputRgba`` from the bundle (raw RGBA Uint8ClampedArray,
    base64-encoded, no color-space metadata) and saves it as a numpy `.npy`
    file with shape ``(384, 384, 3)`` uint8 RGB (alpha stripped).

    Older bundles that pre-date this field are silently skipped.
    Returns the output path on success, or ``None``.
    """
    raw = bundle.get("detectorInputRgba")
    if not raw:
        return None
    rgba = np.frombuffer(base64.b64decode(raw), dtype=np.uint8).reshape(384, 384, 4)
    rgb = rgba[:, :, :3]  # drop alpha
    out_path = out_dir / f"{cap_id}.browser-detector-input.npy"
    np.save(str(out_path), rgb)
    print(f"    → wrote {out_path.name}  (browser 384×384 detector input, uint8 RGB)")
    return out_path


def generate_python_manifest(capture_path: Path, catalog=None) -> dict | None:
    """Run the Python detector pipeline on the capture frame.

    Writes ``<captureId>.python.json`` and (if a card is found)
    ``<captureId>.python.dewarp.png`` next to the capture bundle.

    Parameters
    ----------
    capture_path:
        Path to a ``.json.gz`` capture bundle.
    catalog:
        Optional pre-loaded ``Catalog`` instance.  When supplied, ``topMatchId``
        and ``topMatchScore`` are populated via nearest-neighbour search.

    Returns
    -------
    The manifest dict, or ``None`` if frame decoding failed.
    """
    try:
        import cv2  # noqa: PLC0415
        import collector_vision as cvg  # noqa: PLC0415
        from collector_vision.embedders import NeuralEmbedder  # noqa: PLC0415
    except ImportError as exc:
        print(f"    → cannot generate Python manifest (missing dep: {exc})")
        return None

    cap_id = _capture_id(capture_path)
    out_dir = capture_path.parent
    manifest_path = out_dir / f"{cap_id}.python.json"
    png_path = out_dir / f"{cap_id}.python.dewarp.png"

    with gzip.open(capture_path, "rb") as fh:
        bundle = json.load(fh)

    bgr = _load_frame(bundle)
    if bgr is None:
        print(f"    → cannot decode framePng in {capture_path.name}")
        return None

    detector = cvg.NeuralCornerDetector()
    result = detector.detect(bgr)

    # Save Python's 384×384 detector input for side-by-side comparison with
    # the browser's detectorInputRgba.
    det_input_path = out_dir / f"{cap_id}.python-detector-input.npy"
    _rgb = cv2.cvtColor(
        cv2.resize(bgr, (384, 384), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )
    np.save(str(det_input_path), _rgb)  # (384, 384, 3) uint8
    print(f"    → wrote {det_input_path.name}  (Python 384×384 detector input, uint8 RGB)")

    # Save the browser's 384×384 detector input from the capture bundle
    # (present only in bundles captured after the detectorInputRgba field was added).
    browser_input_path = _extract_detector_input_png(bundle, out_dir, cap_id)
    if browser_input_path is not None:
        diff = np.abs(_rgb.astype(np.int16) - np.load(str(browser_input_path)).astype(np.int16))
        print(f"       pixel diff vs browser: mean={diff.mean():.2f}  max={diff.max()}")

    manifest = _empty_manifest("python", "offline", cap_id)
    manifest["cardPresent"] = bool(result.card_present)
    manifest["sharpness"] = (
        float(result.sharpness) if result.sharpness is not None else None
    )
    manifest["corners"] = (
        [[float(x), float(y)] for x, y in result.corners]
        if result.corners is not None
        else None
    )

    if result.card_present and result.corners is not None:
        crop = result.dewarp(bgr)  # PIL RGB 252×352
        crop.save(str(png_path))
        manifest["dewarpPng"] = png_path.name

        embedder = NeuralEmbedder()
        emb = embedder.embed(crop)  # (128,) float32
        manifest["embedding"] = emb.tolist()

        if catalog is not None:
            hits = catalog.search(emb, top_k=1)
            if hits:
                score, card_id = hits[0]
                manifest["topMatchId"] = card_id
                manifest["topMatchScore"] = float(score)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    sharpness_str = (
        f"{result.sharpness:.4f}" if result.sharpness is not None else "n/a"
    )
    print(
        f"    → wrote {manifest_path.name}  "
        f"(present={result.card_present}, sharpness={sharpness_str})"
    )
    if png_path.exists():
        print(f"    → wrote {png_path.name}")
    return manifest


def extract_webgpu_manifest(capture_path: Path, catalog=None) -> dict | None:
    """Build a ``js-webgpu`` manifest from browser-reported corners in the bundle.

    Uses the ``orderedCorners`` field (written by the browser at capture time)
    as the corner coordinates, then applies Python dewarp + embed to produce
    a complete manifest.  The resulting manifest shows what the browser
    *actually* saw at capture time, using the same embedding pipeline as Python.

    Writes ``<captureId>.js-webgpu.json`` and (when dewarp succeeds)
    ``<captureId>.js-webgpu.dewarp.png`` next to the capture bundle.

    Returns ``None`` and prints a message if ``orderedCorners`` is absent.
    """
    try:
        from collector_vision.embedders import NeuralEmbedder  # noqa: PLC0415
    except ImportError as exc:
        print(f"    → cannot extract WebGPU manifest (missing dep: {exc})")
        return None

    cap_id = _capture_id(capture_path)
    out_dir = capture_path.parent
    manifest_path = out_dir / f"{cap_id}.js-webgpu.json"
    png_path = out_dir / f"{cap_id}.js-webgpu.dewarp.png"

    with gzip.open(capture_path, "rb") as fh:
        bundle = json.load(fh)

    ordered_corners = bundle.get("orderedCorners")
    if not ordered_corners:
        print(f"    → no orderedCorners in bundle; skipping js-webgpu manifest")
        return None

    corners_xy = [[float(c["x"]), float(c["y"])] for c in ordered_corners]

    manifest = _empty_manifest("js-webgpu", "live", cap_id)
    manifest["cardPresent"] = bundle.get("cardPresent")
    manifest["sharpness"] = bundle.get("sharpness")
    manifest["corners"] = corners_xy

    bgr = _load_frame(bundle)
    if bgr is not None:
        try:
            crop = _dewarp_pil(bgr, corners_xy)
            crop.save(str(png_path))
            manifest["dewarpPng"] = png_path.name

            embedder = NeuralEmbedder()
            emb = embedder.embed(crop)
            manifest["embedding"] = emb.tolist()

            if catalog is not None:
                hits = catalog.search(emb, top_k=1)
                if hits:
                    score, card_id = hits[0]
                    manifest["topMatchId"] = card_id
                    manifest["topMatchScore"] = float(score)
        except Exception as exc:
            print(f"    → dewarp/embed failed for js-webgpu manifest: {exc}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"    → wrote {manifest_path.name}")
    if png_path.exists():
        print(f"    → wrote {png_path.name}")
    return manifest


def verify_reproduction(capture_path: Path) -> bool:
    """Check that the browser corners and Python corners disagree enough to
    confirm that the reported bug is reproducible with this capture.

    Reads the ``js-webgpu`` and ``python`` manifests and compares their corner
    coordinates.  Returns ``True`` (reproduced) if the maximum per-coordinate
    delta exceeds ``CORNER_TOLERANCE``.

    If either manifest is missing, the check is skipped and ``True`` is
    returned (to avoid blocking ingestion when manifests couldn't be generated).
    """
    cap_id = _capture_id(capture_path)
    out_dir = capture_path.parent
    webgpu_path = out_dir / f"{cap_id}.js-webgpu.json"
    python_path = out_dir / f"{cap_id}.python.json"

    if not webgpu_path.exists() or not python_path.exists():
        print("    → missing manifests for reproduction check — skipping")
        return True

    webgpu_m = json.loads(webgpu_path.read_text())
    python_m = json.loads(python_path.read_text())

    wc = webgpu_m.get("corners")
    pc = python_m.get("corners")

    if not wc or not pc or len(wc) != len(pc):
        print("    → corners missing or corner-count mismatch — skipping reproduction check")
        return True

    sorted_wc = sorted(wc, key=lambda p: (p[0], p[1]))
    sorted_pc = sorted(pc, key=lambda p: (p[0], p[1]))
    deltas = [
        abs(a - b)
        for w_pt, p_pt in zip(sorted_wc, sorted_pc)
        for a, b in zip(w_pt, p_pt)
    ]
    max_delta = max(deltas) if deltas else 0.0

    with gzip.open(capture_path, "rb") as fh:
        bundle = json.load(fh)

    if max_delta > CORNER_TOLERANCE:
        print(
            f"    → REPRODUCED: browser corners differ from Python by "
            f"{max_delta:.3f} (> tolerance {CORNER_TOLERANCE})"
        )
        # Bonus check: does Python correctly identify the expected card?
        expected = bundle.get("expectedCardId")
        top_id = python_m.get("topMatchId")
        if expected and top_id:
            if top_id == expected:
                print(f"    → Python identifies card correctly ✓  ({expected})")
            else:
                print(
                    f"    ✗ Python got {top_id!r}, expected {expected!r} "
                    f"(topMatchId may be null if catalog was not loaded)"
                )
        return True

    print(
        f"\n  ✗ CANNOT REPRODUCE bug in {capture_path.name}\n"
        f"    Browser (js-webgpu) and Python corners agree "
        f"(max diff={max_delta:.3f} ≤ tolerance {CORNER_TOLERANCE}).\n"
        f"    This capture does not exhibit the reported browser discrepancy.\n"
        f"    Provide a capture from the affected device/browser, "
        f"or use --force to ingest anyway."
    )
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pipeline manifests for captured .json.gz frames."
    )
    parser.add_argument(
        "captures",
        nargs="*",
        type=Path,
        metavar="CAPTURE",
        help=(
            "paths to .json.gz files "
            "(default: all captures in tests/fixtures/captures/)"
        ),
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="skip catalog lookup — topMatchId will be null (fast / no network)",
    )
    args = parser.parse_args()

    paths: list[Path]
    if args.captures:
        paths = list(args.captures)
    else:
        if not CAPTURES_DIR.exists():
            print(f"No captures directory found: {CAPTURES_DIR}")
            return
        paths = sorted(CAPTURES_DIR.glob("*.json.gz"))

    if not paths:
        print("No .json.gz captures found.")
        return

    catalog = None
    if not args.no_catalog:
        try:
            import collector_vision as cvg  # noqa: PLC0415

            print("Loading catalog from HuggingFace…")
            catalog = cvg.Catalog.load("hf://HanClinto/milo/scryfall-mtg")
            print("Catalog loaded.\n")
        except Exception as exc:
            print(
                f"Warning: could not load catalog ({exc})\n"
                f"topMatchId will be null.  Re-run without --no-catalog to retry.\n"
            )

    for path in paths:
        print(f"\n[{path.name}]")
        generate_python_manifest(path, catalog=catalog)
        extract_webgpu_manifest(path, catalog=catalog)


if __name__ == "__main__":
    main()
