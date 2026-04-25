"""Cross-pipeline embedding consistency tests.

Verifies that the Python (onnxruntime/CPU) and JS-WASM (ort-web/CPU) pipelines
produce cosine-similar embeddings for the same captured input frame.

Fixtures live in tests/fixtures/captures/ as pre-computed JSON files containing
128-d embeddings from each pipeline.  No network access required.

Not all captures in that directory are reference captures — some were recorded
during the WebGPU bug investigation when the JS pipeline was in a broken state.
Only captures listed in _REFERENCE_CAPTURES are tested here.
"""

import json
import unittest
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures" / "captures"

# Capture IDs where both Python-ONNX and JS-WASM were known to be working
# correctly, paired with the minimum acceptable cosine similarity.
# Measured values: 16-43-41 → 0.9313, 19-17-23 → 0.9821.
_REFERENCE_CAPTURES = [
    ("cv_2026-04-24T16-43-41", 0.80),
    ("cv_2026-04-24T19-17-23", 0.80),
]

_MIN_NORM = 0.99  # L2-normalised embeddings should have unit norm


class CrossPipelineConsistencyTests(unittest.TestCase):
    """Embedding agreement between the Python and JS-WASM pipelines."""

    def _load(self, capture_id: str, pipeline: str) -> np.ndarray:
        path = FIXTURES / f"{capture_id}.{pipeline}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.array(data["embedding"], dtype=np.float32)

    def _check(self, capture_id: str, min_sim: float) -> None:
        py_emb = self._load(capture_id, "python")
        cpu_emb = self._load(capture_id, "js-cpu")

        # Both embeddings should be unit-norm (L2-normalised)
        self.assertGreater(
            float(np.linalg.norm(py_emb)),
            _MIN_NORM,
            f"{capture_id}: python embedding is not unit-norm",
        )
        self.assertGreater(
            float(np.linalg.norm(cpu_emb)),
            _MIN_NORM,
            f"{capture_id}: js-cpu embedding is not unit-norm",
        )

        sim = float(py_emb @ cpu_emb)
        self.assertGreater(
            sim,
            min_sim,
            f"{capture_id}: python↔js-cpu cosine similarity {sim:.4f} < {min_sim} "
            f"— possible embedding model or preprocessing regression",
        )

    def test_cv_2026_04_24T16_43_41(self) -> None:
        self._check("cv_2026-04-24T16-43-41", 0.80)

    def test_cv_2026_04_24T19_17_23(self) -> None:
        self._check("cv_2026-04-24T19-17-23", 0.80)


if __name__ == "__main__":
    unittest.main()
