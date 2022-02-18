"""Evalify metrics module used for calculating the evaluation metrics.

Optimized calculations using einstein sum. Embeddings array and norm arrays are indexed with every
split and calculations happens over large data chunks very quickly.
"""
from typing import Iterable

import numpy as np


def get_norms(X):
    return np.sqrt(np.einsum("ij,ij->i", X, X, optimize=True))


def cosine_similarity(embs, ix, iy, norms):
    return np.einsum("ij,ij->i", embs[ix], embs[iy], optimize=True) / (
        norms[ix] * norms[iy]
    )


def euclidean_distance(embs, ix, iy, norms=None):

    if norms is None:
        X = embs[ix] - embs[iy]
    else:
        X = embs[ix] / norms[ix].reshape(-1, 1) - embs[iy] / norms[iy].reshape(-1, 1)
    return get_norms(X)


metrics_caller = {
    "cosine_similarity": cosine_similarity,
    "euclidean_distance": euclidean_distance,
    "euclidean_distance_l2": euclidean_distance,
}

METRICS_NEED_NORM = ["cosine_similarity", "euclidean_distance_l2"]
DISTANCE_TO_SIMILARITY = {
    "euclidean_distance": lambda x: 1 / (1 + x),
    "euclidean_distance_l2": lambda x: 1 - x,
}
