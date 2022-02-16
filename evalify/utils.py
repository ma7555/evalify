import numpy as np
import psutil


def calc_available_memory():
    mem = psutil.virtual_memory()
    available_mem = mem[1]
    return available_mem


def keep_to_max_rows(embs, available_mem):
    """
    Calculate maximum rows to fetch per split without going out of memory.
    We will need 3 big arrays to be held in memory (A, B, A*B)
    """
    if available_mem > 2e9:
        max_total_rows = np.floor(available_mem - 1e9 / (embs[0].nbytes))
        max_rows_per_side = int(max_total_rows / 3)
    else:
        max_total_rows = np.floor(available_mem / (embs[0].nbytes))
        max_rows_per_side = int(max_total_rows / 5)

    return max_rows_per_side
