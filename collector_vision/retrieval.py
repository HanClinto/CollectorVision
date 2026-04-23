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
    if query_vec.ndim != 1:
        raise ValueError(f"query_vec must be 1-D, got shape {query_vec.shape}")
    if catalog_embeddings.ndim != 2:
        raise ValueError(
            f"catalog_embeddings must be 2-D, got shape {catalog_embeddings.shape}"
        )
    if catalog_embeddings.shape[1] != query_vec.shape[0]:
        raise ValueError(
            "query_vec and catalog_embeddings have incompatible dimensions: "
            f"{query_vec.shape[0]} vs {catalog_embeddings.shape[1]}"
        )

    scores: np.ndarray = catalog_embeddings @ query_vec  # (N,)
    n = len(scores)
    if n == 0:
        return []

    if top_k >= n:
        idxs = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        idxs = part[np.argsort(scores[part])[::-1]]

    return [(float(scores[i]), int(i)) for i in idxs]
