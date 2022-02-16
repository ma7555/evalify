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


def find_optimal_cutoff(target, predicted):
    """Find the optimal cutoff point
    Args:
        target: Matrix with dependent or target data, where rows are observations
        predicted: Matrix with predicted data, where rows are observations

    Returns:
        float: optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t["threshold"].item()


def evaluate_at_threshold(one_to_one_df: pd.DataFrame, threshold: float, metric):
    """Evaluate performance at specific threshold
    Args:
        one_to_one_df: Experiment dataframe.
        threshold: cut-off threshold.

    Returns:
        dict: containing all evaluation metrics.
    """
    pred = one_to_one_df[metric].apply(lambda x: 1 if x > threshold else 0)
    cm = confusion_matrix(one_to_one_df["target"], pred)
    tn, fp, fn, tp = cm.ravel()
    TPR = tp / (tp + fn)  # recall / true positive rate
    TNR = tn / (tn + fp)  # true negative rate
    PPV = tp / (tp + fp)  # precision / positive predicted value
    NPV = tn / (tn + fn)  # negative predictive value
    FPR = fp / (fp + tn)  # false positive rate
    FNR = 1 - TPR  # false negative rate
    FDR = 1 - PPV  # false discovery rate
    FOR = 1 - NPV  # false omission rate
    LRp = TPR / FPR  # positive likelihood ratio (LR+)
    LRn = FNR / TNR  # negative likelihood ratio (LR+)

    evaluation = {
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "FOR": FOR,
        "LR+": LRp,
        "LR-": LRn,
    }

    return evaluation
