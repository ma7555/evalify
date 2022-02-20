"""Evalify metrics module used for calculating the evaluation metrics.

Optimized calculations using einstein sum. Embeddings array and norm arrays are indexed with every
split and calculations happens over large data chunks very quickly.
"""
import numpy as np


def get_norms(X):
    return np.sqrt(np.einsum("ij,ij->i", X, X, optimize=True))


def cosine_similarity(embs, ix, iy, norms, return_distance=False, **kwargs):
    similarity = np.einsum("ij,ij->i", embs[ix], embs[iy], optimize=True) / (
        norms[ix] * norms[iy]
    )
    if return_distance:
        return 1 - similarity
    return similarity


def euclidean_distance(embs, ix, iy, **kwargs):
    X = embs[ix] - embs[iy]
    return get_norms(X)


def euclidean_distance_l2(embs, ix, iy, norms, **kwargs):
    X = embs[ix] / norms[ix].reshape(-1, 1) - embs[iy] / norms[iy].reshape(-1, 1)
    return get_norms(X)


def minkowski_distance(embs, ix, iy, p, **kwargs):
    return np.linalg.norm(embs[ix] - embs[iy], ord=p, axis=1)


metrics_caller = {
    "cosine_similarity": cosine_similarity,
    "cosine_distance": lambda embs, ix, iy, norms, **kwargs: cosine_similarity(
        embs, ix, iy, norms, return_distance=True
    ),
    "euclidean_distance": euclidean_distance,
    "euclidean_distance_l2": euclidean_distance_l2,
    "minkowski_distance": minkowski_distance,
    "manhattan_distance": lambda embs, ix, iy, **kwargs: minkowski_distance(
        embs, ix, iy, p=1
    ),
    "chebyshev_distance": lambda embs, ix, iy, **kwargs: minkowski_distance(
        embs, ix, iy, p=np.inf
    ),
}

METRICS_NEED_NORM = ["cosine_similarity", "cosine_distance", "euclidean_distance_l2"]
METRICS_NEED_ORDER = ["minkowski_distance"]
DISTANCE_TO_SIMILARITY = {
    "cosine_distance": lambda x: 1 - x,
    "euclidean_distance": lambda x: 1 / (1 + x),
    "euclidean_distance_l2": lambda x: 1 - x,
    "minkowski_distance": lambda x: 1 / (1 + x),
    "manhattan_distance": lambda x: 1 / (1 + x),
    "chebyshev_distance": lambda x: 1 / (1 + x),
}
