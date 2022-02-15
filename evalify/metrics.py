"""Metrics module."""
import numpy as np


def cosine_similarity(embs, norms, ix, iy):
    return np.einsum("ij,ij->i", embs[ix], embs[iy]) / (norms[ix] * norms[iy])


def euclidean_distance(embs, ix, iy):
    return np.linalg.norm(embs[ix] - embs[iy], axis=1)


metrics_caller = {
    "cosine_similarity": cosine_similarity,
    "euclidean_distance": euclidean_distance,
}
