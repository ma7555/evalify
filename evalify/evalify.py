"""Main module."""
import itertools

import numpy as np
import pandas as pd
from metrics import metrics_caller
from utils import calc_available_memory, keep_to_max_rows


def calculate_best_split_size(X, experiment_size):
    available_mem = calc_available_memory()
    max_rows = keep_to_max_rows(X, available_mem)
    nsplits = int(experiment_size / max_rows)
    # add a split to avoid 0 splits
    nsplits += 1
    return nsplits


def create_experiment(
    X: np.ndarray,
    y: np.ndarray,
    metric="cosine_similarity",
    same_class_samples="full",
    different_class_samples="minimal",
    nsplits="best",
    shuffle=False,
    return_embeddings=False,
):

    if same_class_samples != "full" and not isinstance(same_class_samples, int):
        raise ValueError(
            '`same_class_samples` argument must be one of "full" or an integer '
            f"Received: same_class_samples={same_class_samples}"
        )
    if different_class_samples not in ("full", "minimal") and not isinstance(
        different_class_samples, int
    ):
        raise ValueError(
            '`different_class_samples` argument must be one of "full", "minimal" '
            "or an integer "
            f"Received: different_class_samples={different_class_samples}."
        )
    if nsplits != "best" and not isinstance(nsplits, int):
        raise ValueError(
            '`nsplits` argument must be either "best" or of type integer '
            f"Received: nsplits={nsplits} with type {type(nsplits)}."
        )
    if metric not in metrics_caller:
        raise ValueError(
            f"`metric` argument must be one of {tuple(metrics_caller.keys())} "
            f"Received: metric={metric}"
        )

    all_targets = np.unique(y)
    all_pairs = list()
    metric_fn = metrics_caller.get(metric)
    for target in all_targets:
        same_ixs = np.argwhere(y == target).ravel()
        same_pairs = itertools.combinations(same_ixs, 2)
        same_pairs = [(a, b, target, target, 1) for a, b in same_pairs]
        different_ixs = np.argwhere(y != target).ravel()

        df = pd.DataFrame(data={"ix": different_ixs, "target": y[different_ixs]})

        if different_class_samples == "minimal":
            df = df.sample(frac=1).drop_duplicates(subset=["target"])

        different_ixs = df.ix.to_numpy()
        different_pairs = itertools.product([np.random.choice(same_ixs)], different_ixs)
        different_pairs = [(a, b, target, y[b], 1) for a, b in different_pairs]
        all_pairs += same_pairs + different_pairs

    one_to_one_df = pd.DataFrame(
        data=all_pairs, columns=["img_a", "img_b", "target_a", "target_b", "target"]
    )
    if shuffle:
        one_to_one_df = one_to_one_df.frac(1)
    if nsplits == "best":
        nsplits = calculate_best_split_size(X, len(one_to_one_df))

    Xs = np.array_split(one_to_one_df.img_a.to_numpy(), nsplits)
    ys = np.array_split(one_to_one_df.img_b.to_numpy(), nsplits)

    if metric == "cosine_similarity":
        norms = np.linalg.norm(X, axis=1)
        one_to_one_df[metric] = np.hstack(
            [metric_fn(X, norms, ix, iy) for (ix, iy) in zip(Xs, ys)]
        )
    else:
        one_to_one_df[metric] = np.hstack(
            [metric_fn(X, ix, iy) for (ix, iy) in zip(Xs, ys)]
        )

    if return_embeddings:
        one_to_one_df["img_a"] = X[one_to_one_df.img_a.to_numpy()].tolist()
        one_to_one_df["img_b"] = X[one_to_one_df.img_b.to_numpy()].tolist()

    return one_to_one_df
