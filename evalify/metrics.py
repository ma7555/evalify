"""Metrics module."""
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import auc, roc_curve


def _get_norms(X):
    return np.sqrt(np.einsum("ij,ij->i", X, X))


def cosine_similarity(embs, ix, iy, norms):
    return np.einsum("ij,ij->i", embs[ix], embs[iy]) / (norms[ix] * norms[iy])


def euclidean_distance(embs, ix, iy, norms=None):

    if norms is None:
        return np.linalg.norm(embs[ix] - embs[iy], axis=1)
    else:
        return np.linalg.norm(
            embs[ix] / norms[ix].reshape(-1, 1) - embs[iy] / norms[iy].reshape(-1, 1),
            axis=1,
        )


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
    "euclidean_distance_l2": euclidean_distance,
}
