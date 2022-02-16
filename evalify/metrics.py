"""Metrics module."""
import numpy as np
from sklearn.metrics import auc,, roc_curve
from collections.abc import Iterable


def cosine_similarity(embs, norms, ix, iy):
    return np.einsum("ij,ij->i", embs[ix], embs[iy]) / (norms[ix] * norms[iy])


def euclidean_distance(embs, ix, iy):
    return np.linalg.norm(embs[ix] - embs[iy], axis=1)


def roc_auc(one_to_one_df, metrics=None):
    roc_dict = {}
    available_metrics = np.intersect1d(
        one_to_one_df.columns, metrics_caller.keys(), assume_unique=True
    )

    if len(available_metrics) < 1:
        raise ValueError(
            "Passed dataframe does not contain any calculated metrics yet. "
            "Please run create_experiment first"
        )

    if metrics is None:
        metrics = available_metrics

    elif isinstance(metrics, str):
        metrics = [metrics]

    if isinstance(metrics, Iterable):
        for metric in metrics:
            if metric in available_metrics:
                fpr, tpr, thresholds = roc_curve(
                    one_to_one_df["target"], one_to_one_df[metric]
                )
                roc_dict[metric] = auc(fpr, tpr)
            else:
                raise ValueError(
                    f"All `metrics` must be in {available_metrics} "
                    f"Received: metrics={metrics}"
                )
    else:
        raise ValueError(
            f"`metrics` argument must be either a list or a str "
            f"Received: metrics={type(metrics)}"
        )
    return roc_dict


metrics_caller = {
    "cosine_similarity": cosine_similarity,
    "euclidean_distance": euclidean_distance,
}
