"""Evalify utils module contains various utilites serving other modules."""
import numpy as np
import psutil

GB_TO_BYTE = 1024**3


def _validate_vectors(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if X.ndim != 2:
        raise ValueError("Embeddings vector should be 2-D.")
    if y.ndim != 1:
        raise ValueError("Target vector should be 1-D.")
    return X, y


def _calc_available_memory():
    """Calculate available memory in system"""
    mem = psutil.virtual_memory()
    return mem[1]


def _keep_to_max_rows(embs, available_mem):
    """Calculate maximum rows to fetch per split without going out of memory.

    We need 3 big arrays to be held in memory (A, B, A*B)
    """
    if available_mem > 2 * GB_TO_BYTE:
        max_total_rows = np.floor(available_mem - GB_TO_BYTE / (embs[0].nbytes))
        return int(max_total_rows / 3)
    else:
        max_total_rows = np.floor(available_mem / (embs[0].nbytes))
        return int(max_total_rows / 5)


def calculate_best_split_size(X, experiment_size):
    """Calculate best number of splits."""
    available_mem = _calc_available_memory()
    max_rows = _keep_to_max_rows(X, available_mem)
    return int(experiment_size / max_rows) + 1
