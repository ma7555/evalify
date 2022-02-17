import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import confusion_matrix, roc_curve


def _calc_available_memory():
    mem = psutil.virtual_memory()
    available_mem = mem[1]
    return available_mem


def _keep_to_max_rows(embs, available_mem):
    """Calculate maximum rows to fetch per split without going out of memory.

    We need 3 big arrays to be held in memory (A, B, A*B)
    """
    if available_mem > 2e9:
        max_total_rows = np.floor(available_mem - 1e9 / (embs[0].nbytes))
        max_rows_per_side = int(max_total_rows / 3)
    else:
        max_total_rows = np.floor(available_mem / (embs[0].nbytes))
        max_rows_per_side = int(max_total_rows / 5)

    return max_rows_per_side


def calculate_best_split_size(X, experiment_size):
    """Calculate best number of splits."""
    available_mem = _calc_available_memory()
    max_rows = _keep_to_max_rows(X, available_mem)
    nsplits = int(experiment_size / max_rows)
    # add a split to avoid 0 splits
    nsplits += 1
    return nsplits
