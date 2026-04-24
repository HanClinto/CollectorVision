"""Cross-pipeline consistency tests for captured frames.

For each capture in ``tests/fixtures/captures/``, this test discovers all
pipeline manifest files and verifies they are self-consistent.

What is asserted
----------------
* If the ``python`` manifest exists and ``expectedCardId`` is set in the
  capture bundle, the Python pipeline's ``topMatchId`` must match.
* If both ``python`` and ``js-cpu`` manifests exist, their corners must agree
  within ``CORNER_TOLERANCE`` and their embeddings must have cosine similarity
  above ``EMBEDDING_MIN_DOT``.

What is NOT asserted
--------------------
* Correctness of the ``js-webgpu`` pipeline.  That manifest is stored for
  human inspection only.  Bug captures will show a ``js-webgpu`` manifest
  that disagrees with offline pipelines by design — no assumption is made
  about it here.

Manifest format
---------------
Each ``<captureId>.<pipeline>.json`` file (written by ``generate_manifests.py``
or ``tests/js/test_pipeline.mjs``) contains::

    {
      "pipeline":      "python",
      "source":        "offline",      # "offline" | "live"
      "captureId":     "cv_TIMESTAMP",
      "cardPresent":   true,
      "sharpness":     0.046,
      "corners":       [[x, y], ...],  # TL, TR, BR, BL — normalised [0,1]
      "dewarpPng":     "cv_TIMESTAMP.python.dewarp.png",
      "embedding":     [0.12, ...],
      "topMatchId":    "ace86fac-...",
      "topMatchScore": 0.987
    }

Generating manifests
--------------------
Python and js-webgpu manifests::

    python scripts/ingest_bug_reports.py <issue-number>
    # or independently:
    python scripts/generate_manifests.py

JS-CPU manifest (also runs self-contained assertions)::

    cd tests/js && npm test
"""
from __future__ import annotations

import gzip
import json
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = ROOT / "tests" / "fixtures" / "captures"

# Tolerances — keep in sync with generate_manifests.py and test_pipeline.mjs
CORNER_TOLERANCE = 0.15   # maximum per-coordinate delta (normalised units)
EMBEDDING_MIN_DOT = 0.90  # minimum cosine similarity between offline pipelines
                           # (PIL bilinear vs canvas drawImage give ~0.93 baseline)

# Pipelines whose manifests participate in cross-pipeline consistency checks.
# "live" pipelines (js-webgpu) are excluded — their manifests are for
# reference only and may validly disagree with offline pipelines.
_OFFLINE_PIPELINES = {"python", "js-cpu"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_captures() -> list[Path]:
    if not CAPTURES_DIR.exists():
        return []
    return sorted(CAPTURES_DIR.glob("*.json.gz"))


def _capture_id(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    return path.stem


def _load_manifests(capture_path: Path) -> dict[str, dict]:
    """Return all available pipeline manifests keyed by pipeline name."""
    cap_id = _capture_id(capture_path)
    manifests: dict[str, dict] = {}
    for suffix_json in capture_path.parent.glob(f"{cap_id}.*.json"):
        # Skip anything that isn't a pipeline manifest (e.g. no double dot).
        parts = suffix_json.stem.split(".")  # ['cv_TIMESTAMP', 'python']
        if len(parts) != 2:
            continue
        pipeline = parts[1]
        try:
            manifests[pipeline] = json.loads(suffix_json.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return manifests


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class PipelineConsistencyTests(unittest.TestCase):
    """One test per .json.gz in tests/fixtures/captures/."""

    def _run_consistency(self, capture_path: Path) -> None:
        cap_id = _capture_id(capture_path)

        manifests = _load_manifests(capture_path)
        if not manifests:
            self.skipTest(
                f"No pipeline manifests found for {cap_id}.\n"
                f"  Run: python scripts/generate_manifests.py\n"
                f"  Then: cd tests/js && npm test"
            )

        offline = {
            p: m for p, m in manifests.items()
            if m.get("source") == "offline" and p in _OFFLINE_PIPELINES
        }

        # --- (1) Python topMatchId must equal expectedCardId -----------------
        with gzip.open(capture_path, "rb") as fh:
            bundle = json.load(fh)
        expected = bundle.get("expectedCardId") or None

        if expected:
            python_m = manifests.get("python")
            if python_m is None:
                warnings.warn(
                    f"{cap_id}: expectedCardId is set but no python manifest found.\n"
                    f"  Run: python scripts/generate_manifests.py --with-catalog",
                    stacklevel=2,
                )
            elif python_m.get("topMatchId") is None:
                warnings.warn(
                    f"{cap_id}: python manifest has no topMatchId "
                    f"(catalog may not have been loaded during manifest generation).\n"
                    f"  Run: python scripts/generate_manifests.py  (without --no-catalog)",
                    stacklevel=2,
                )
            else:
                self.assertEqual(
                    python_m["topMatchId"],
                    expected,
                    f"{cap_id}: Python topMatchId {python_m['topMatchId']!r} "
                    f"does not match expectedCardId {expected!r}",
                )

        # --- (2) Offline pipelines must agree on corners ---------------------
        if len(offline) < 2:
            return  # nothing to compare across pipelines

        pipeline_pairs = [
            (p1, p2)
            for i, p1 in enumerate(sorted(offline))
            for p2 in sorted(offline)[i + 1 :]
        ]

        for p1, p2 in pipeline_pairs:
            m1, m2 = offline[p1], offline[p2]

            c1 = m1.get("corners")
            c2 = m2.get("corners")
            if c1 and c2 and len(c1) == len(c2):
                sorted_c1 = sorted(c1, key=lambda p: (p[0], p[1]))
                sorted_c2 = sorted(c2, key=lambda p: (p[0], p[1]))
                deltas = [
                    abs(a - b)
                    for pt1, pt2 in zip(sorted_c1, sorted_c2)
                    for a, b in zip(pt1, pt2)
                ]
                max_delta = max(deltas)
                self.assertLess(
                    max_delta,
                    CORNER_TOLERANCE,
                    f"{cap_id}: {p1} vs {p2} corners differ by "
                    f"{max_delta:.4f} (tolerance {CORNER_TOLERANCE}).\n"
                    f"  This suggests a regression in the {p1} or {p2} "
                    f"preprocessing pipeline.",
                )

            # --- (3) Offline pipelines must agree on embeddings --------------
            e1 = m1.get("embedding")
            e2 = m2.get("embedding")
            if e1 and e2:
                arr1 = np.array(e1, dtype=np.float32)
                arr2 = np.array(e2, dtype=np.float32)
                dot = float(np.dot(arr1, arr2))
                self.assertGreater(
                    dot,
                    EMBEDDING_MIN_DOT,
                    f"{cap_id}: {p1} vs {p2} embedding cosine similarity "
                    f"{dot:.4f} < minimum {EMBEDDING_MIN_DOT}.\n"
                    f"  The {p1} and {p2} preprocessing pipelines have diverged.",
                )


# ---------------------------------------------------------------------------
# Dynamic test generation — one test method per capture
# ---------------------------------------------------------------------------

def _make_test(path: Path):
    def test_method(self: PipelineConsistencyTests) -> None:
        self._run_consistency(path)

    cap_id = _capture_id(path)
    safe = cap_id.replace("-", "_").replace(":", "_").replace("T", "T", 1)
    test_method.__name__ = f"test_{safe}"
    test_method.__doc__ = f"Cross-pipeline consistency for capture: {path.name}"
    return test_method


for _p in _find_captures():
    _t = _make_test(_p)
    setattr(PipelineConsistencyTests, _t.__name__, _t)
