"""Nearest-neighbour retrieval helpers.

Both functions return a list of (score, index) tuples sorted by descending
score (higher = better match), truncated to top_k entries.

For cosine search the score is the cosine similarity (dot product of
L2-normalised vectors, range [-1, 1]).

For Hamming search the score is normalised Hamming similarity
(1 - hamming_distance / n_bits, range [0, 1]).
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


def hamming_search(
    query_bits: np.ndarray,
    catalog_bits: np.ndarray,
    top_k: int = 5,
) -> list[tuple[float, int]]:
    """Top-k Hamming similarity search.

    Parameters
    ----------
    query_bits:
        (B,) uint8 — packed bit vector.
    catalog_bits:
        (N, B) uint8 — packed bit vectors, one row per catalog card.
    top_k:
        Number of results to return.

    Returns
    -------
    List of (score, index) sorted by descending score (similarity, not distance).
    """
    n_bits = query_bits.shape[0] * 8

    xor = np.bitwise_xor(catalog_bits, query_bits[np.newaxis, :])  # (N, B)
    distances = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)  # (N,)
    scores = 1.0 - distances / n_bits  # (N,) normalised similarity

    n = len(scores)
    if top_k >= n:
        idxs = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -top_k)[-top_k:]
        idxs = part[np.argsort(scores[part])[::-1]]

    return [(float(scores[i]), int(i)) for i in idxs]
