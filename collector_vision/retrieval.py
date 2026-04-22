"""Nearest-neighbour retrieval helpers.

Returns a list of (score, index) tuples sorted by descending score
(higher = better match), truncated to top_k entries.

Score is cosine similarity — dot product of L2-normalised vectors, range [-1, 1].
"""
from __future__ import annotations

import numpy as np


def cosine_search(
    query_vec: np.ndarray,
    catalog_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[tuple[float, int]]:
    """Top-k cosine similarity search.

    Parameters
    ----------
    query_vec:
        (D,) float32, L2-normalised.
    catalog_embeddings:
        (N, D) float32, L2-normalised rows.
    top_k:
        Number of results to return.

    Returns
    -------
    List of (score, index) sorted by descending score.
    """
    scores: np.ndarray = catalog_embeddings @ query_vec  # (N,)
    n = len(scores)

    if top_k >= n:
        idxs = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        idxs = part[np.argsort(scores[part])[::-1]]

    return [(float(scores[i]), int(i)) for i in idxs]
