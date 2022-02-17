import numpy as np
import psutil


def _calc_available_memory():
    mem = psutil.virtual_memory()
    return mem[1]


def _keep_to_max_rows(embs, available_mem):
    """Calculate maximum rows to fetch per split without going out of memory.

    We need 3 big arrays to be held in memory (A, B, A*B)
    """
    if available_mem > 2e9:
        max_total_rows = np.floor(available_mem - 1e9 / (embs[0].nbytes))
        return int(max_total_rows / 3)
    else:
        max_total_rows = np.floor(available_mem / (embs[0].nbytes))
        return int(max_total_rows / 5)


def calculate_best_split_size(X, experiment_size):
    """Calculate best number of splits."""
    available_mem = _calc_available_memory()
    max_rows = _keep_to_max_rows(X, available_mem)
    return int(experiment_size / max_rows) + 1
