import unittest

import numpy as np

from collector_vision.catalog import Catalog
from collector_vision.retrieval import cosine_search


class RetrievalTests(unittest.TestCase):
    def test_cosine_search_rejects_dimension_mismatch(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "incompatible dimensions"):
            cosine_search(query, embeddings, top_k=1)

    def test_catalog_for_games_requires_at_least_one_game(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one game"):
            Catalog.for_games()


if __name__ == "__main__":
    unittest.main()
