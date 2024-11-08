"""Evalify metrics module used for calculating the evaluation metrics.

Optimized calculations using einstein sum. Embeddings array and norm arrays are indexed
with every
split and calculations happens over large data chunks very quickly.
"""

import numpy as np


def _inner1d(A, B):
    """Calculate the inner product between two arrays of vectors.

    Args:
        A (numpy.ndarray): 2D array of shape (n_samples, n_features)
        B (numpy.ndarray): 2D array of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: 1D array of shape (n_samples,) where each element is the inner
        product of the corresponding rows in A and B

    """
    return np.einsum("ij,ij->i", A, B, optimize="optimal")


def cosine_similarity(embs, ix, iy, norms, return_distance=False, **kwargs):
    """Calculate the cosine similarity between two arrays of vectors.

    Args:
        embs (numpy.ndarray): 2D array of shape (n_samples, n_features)
        ix (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the first array
        iy (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the second array
        norms (numpy.ndarray): 1D array of shape (n_samples,) containing the L2 norm
        of each row in X
        return_distance (bool): Whether to return the cosine distance instead of the
        cosine similarity. Defaults to False.

    Returns:
        numpy.ndarray: 1D array of shape (n_samples,) where each element is the cosine
        similarity (or cosine distance) of the corresponding rows in X.

    """
    similarity = _inner1d(embs[ix], embs[iy]) / (norms[ix] * norms[iy])
    return 1 - similarity if return_distance else similarity


def euclidean_distance_l2(embs, ix, iy, norms, **kwargs):
    """Calculate the L2-normalized Euclidean distance between two arrays of vectors.

    Args:
        embs (numpy.ndarray): 2D array of shape (n_samples, n_features).
        ix (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the first array.
        iy (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the second array.
        norms (numpy.ndarray): 1D array of shape (n_samples,) containing the L2 norm
        of each row in embs.

    Returns:
        numpy.ndarray: 1D array of shape (n_samples,) where each element is the
        L2-normalized Euclidean distance of the corresponding rows in embs.

    """
    X = embs[ix] / norms[ix].reshape(-1, 1) - embs[iy] / norms[iy].reshape(-1, 1)
    return np.linalg.norm(X, axis=1)


def minkowski_distance(embs, ix, iy, p, **kwargs):
    """Calculate the element-wise Minkowski or Manhattan or Chebyshev distance.

    Args:
        embs (numpy.ndarray): 2D array of shape (n_samples, n_features)
        ix (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the first array
        iy (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the second array
        p (int): The order of the norm of the difference.

    Returns:
        numpy.ndarray: 1D array of shape (n_samples,) where each element is the
        Minkowski distance of the corresponding rows in embs.

    """
    return np.linalg.norm(embs[ix] - embs[iy], ord=p, axis=1)


def pearson_similarity(embs, ix, iy, **kwargs):
    """Calculate the Pearson correlation coefficient between two arrays of vectors.

    Args:
        embs (numpy.ndarray): 2D array of shape (n_samples, n_features)
        ix (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the first array
        iy (numpy.ndarray): 1D array of shape (n_samples,) containing the indices of
        the second array

    Returns:
        numpy.ndarray: 1D array of shape (n_samples,) where each element is the Pearson
        correlation coefficient
        of the corresponding rows in embs.

    """
    A = embs[ix]
    B = embs[iy]
    A_mA = A - np.expand_dims(A.mean(axis=1), -1)
    B_mB = B - np.expand_dims(B.mean(axis=1), -1)
    ssA = np.expand_dims((A_mA**2).sum(axis=1), -1)
    ssB = np.expand_dims((B_mB**2).sum(axis=1), -1)
    return _inner1d(A_mA, B_mB) / np.sqrt(_inner1d(ssA, ssB))


metrics_caller = {
    "cosine_similarity": cosine_similarity,
    "pearson_similarity": pearson_similarity,
    "cosine_distance": lambda embs, ix, iy, norms, **kwargs: cosine_similarity(
        embs,
        ix,
        iy,
        norms,
        return_distance=True,
    ),
    "euclidean_distance": lambda embs, ix, iy, **kwargs: minkowski_distance(
        embs,
        ix,
        iy,
        p=2,
    ),
    "euclidean_distance_l2": euclidean_distance_l2,
    "minkowski_distance": minkowski_distance,
    "manhattan_distance": lambda embs, ix, iy, **kwargs: minkowski_distance(
        embs,
        ix,
        iy,
        p=1,
    ),
    "chebyshev_distance": lambda embs, ix, iy, **kwargs: minkowski_distance(
        embs,
        ix,
        iy,
        p=np.inf,
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

REVERSE_DISTANCE_TO_SIMILARITY = {
    "cosine_distance": lambda x: 1 - x,
    "euclidean_distance": lambda x: (1 / x) - 1,
    "euclidean_distance_l2": lambda x: 1 - x,
    "minkowski_distance": lambda x: (1 / x) - 1,
    "manhattan_distance": lambda x: (1 / x) - 1,
    "chebyshev_distance": lambda x: (1 / x) - 1,
}
