"""Evalify main module used for creating the verification experiments.

Creates experiments with embedding pairs to compare for face verification tasks
including positive pairs, negative pairs and metrics calculations using a very
optimized einstein sum. Many operations are dispatched to canonical BLAS, cuBLAS,
or other specialized routines. Extremely large arrays are split into smaller batches,
every batch would consume the roughly the maximum available memory.

  Typical usage example:


  ```
  experiment = Experiment()
  experiment.run(X, y)
  ```
  """
import itertools
import sys
from collections import OrderedDict
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from evalify.metrics import (
    DISTANCE_TO_SIMILARITY,
    METRICS_NEED_NORM,
    METRICS_NEED_ORDER,
    get_norms,
    metrics_caller,
)
from evalify.utils import _validate_vectors, calculate_best_split_size

T_str_int = Union[str, int]


class Experiment:
    def __init__(self) -> None:
        self.experiment_sucess = False
        self.cached_predicted_as_similarity = {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Union[str, Sequence[str]] = "cosine_similarity",
        same_class_samples: Union[str, int, Sequence[T_str_int]] = "full",
        different_class_samples: T_str_int = "minimal",
        nsplits: T_str_int = "best",
        shuffle: bool = False,
        seed: int = None,
        return_embeddings: bool = False,
        p: int = 3,
    ):
        """Runs an experiment for face verification

        Args:
            X: Embeddings array
            y: Targets for X as integers
            metrics:
                - 'cosine_similarity'
                - 'cosine_distance'
                - 'euclidean_distance'
                - 'euclidean_distance_l2'
                - 'minkowski_distance'
                - 'manhattan_distance'
                - 'chebyshev_distance'
                - list/tuple containing more than one of them.
            same_class_samples:
                - 'full': Samples all possible images within each class to create all
                    all possible positive pairs.
                -  int: Samples specific number of images for every class to create
                    nC2 pairs where n is passed integer.
            different_class_samples:
                - 'full': Samples one image from every class with all possible pairs
                    of different classes. This can grow exponentially as the number
                    of images increase. (N, M) = (1, "full")
                - 'minimal': Samples one image from every class with one image of
                    all other classes. (N, M) = (1, 1). (Default)
                - int: Samples one image from every class with provided number of
                    images of every other class.
                - tuple or list: (N, M) Samples N images from every class with M images of
                    every other class.
            nsplits:
                - 'best': Let the program decide based on available memory such that every
                    split will fit into the available memory. (Default)
                - int: Manually decide the number of splits.
            shuffle: Whether to shuffle the returned experiment dataframe. Default: False.
            return_embeddings: Whether to return the embeddings instead of indexes.
                Default: False
            p:
                The order of the norm of the difference :math:`{\\|u-v\\|}_p`.
                :math:`0 < p < 1`, Only valid with minkowski_distance as a metric.
                Default = 3

        Returns:
            pandas.DataFrame: A DataFrame representing the experiment results.

        Raises:
            ValueError: An error occurred with the provided arguments.

        Notes:
            same_class_samples:
                If the provided number is greater than the achievable for the class,
                the maximum possible combinations are used.
            different_class_samples:
                If the provided number is greater than the achievable for the class,
                the maximum possible combinations are used. (N, M) can also be
                ('full', 'full') but this will calculate all possible combinations
                between all posibile negative samples. If the dataset is not small
                this will probably result in an extremely large array!.
        """
        if isinstance(metrics, str):
            metrics = (metrics,)

        self._validate_args(
            metrics, same_class_samples, different_class_samples, nsplits, p
        )
        X, y = _validate_vectors(X, y)
        all_targets = np.unique(y)
        all_pairs = []
        metric_fns = list(map(metrics_caller.get, metrics))
        rng = np.random.default_rng(seed)
        already_added = set()
        for target in all_targets:
            same_ixs_full = np.argwhere(y == target).ravel()
            if isinstance(same_class_samples, int):
                same_class_samples = min(len(same_ixs_full), same_class_samples)
                same_ixs = rng.choice(same_ixs_full, same_class_samples)
            elif same_class_samples == "full":
                same_ixs = same_ixs_full
            same_pairs = itertools.combinations(same_ixs, 2)
            same_pairs_expanded = [(a, b, target, target, 1) for a, b in same_pairs]

            different_ixs = np.argwhere(y != target).ravel()
            diff_df = pd.DataFrame(
                data={"ix": different_ixs, "target": y[different_ixs]}
            )

            diff_df = diff_df.sample(frac=1, random_state=seed)
            if different_class_samples in ["full", "minimal"] or isinstance(
                different_class_samples, int
            ):
                N = 1
                if different_class_samples == "minimal":
                    diff_df = diff_df.drop_duplicates(subset=["target"])
            else:
                N, M = different_class_samples
                N = len(same_ixs_full) if N == "full" else min(N, len(same_ixs_full))
                if M != "full":
                    diff_df = (
                        diff_df.groupby("target").apply(lambda x: x[:M]).droplevel(0)
                    )

            different_ixs = diff_df.ix.to_numpy()

            different_pairs = itertools.product(
                rng.choice(same_ixs_full, N, replace=False), different_ixs
            )
            different_pairs_expanded = []
            for a, b in different_pairs:
                if (a, b) not in already_added:
                    different_pairs_expanded.append((a, b, target, y[b], 0))
                    already_added.update(((a, b), (b, a)))
            all_pairs += same_pairs_expanded + different_pairs_expanded

        self.df = pd.DataFrame(
            data=all_pairs, columns=["img_a", "img_b", "target_a", "target_b", "target"]
        )
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed)
        if nsplits == "best":
            nsplits = calculate_best_split_size(X, len(self.df))

        Xs = np.array_split(self.df.img_a.to_numpy(), nsplits)
        ys = np.array_split(self.df.img_b.to_numpy(), nsplits)

        kwargs = {}
        if any(metric in METRICS_NEED_NORM for metric in metrics):
            kwargs["norms"] = get_norms(X)
        if any(metric in METRICS_NEED_ORDER for metric in metrics):
            kwargs["p"] = p

        for metric, metric_fn in zip(metrics, metric_fns):
            self.df[metric] = np.hstack(
                [metric_fn(X, ix, iy, **kwargs) for (ix, iy) in zip(Xs, ys)]
            )

        if return_embeddings:
            self.df["img_a"] = X[self.df.img_a.to_numpy()].tolist()
            self.df["img_b"] = X[self.df.img_b.to_numpy()].tolist()

        self.experiment_sucess = True
        self.metrics = metrics
        return self.df

    def _validate_args(
        self, metrics, same_class_samples, different_class_samples, nsplits, p
    ):
        if same_class_samples != "full" and not isinstance(same_class_samples, int):
            raise ValueError(
                "`same_class_samples` argument must be one of 'full' or an integer "
                f"Received: same_class_samples={same_class_samples}"
            )

        if different_class_samples not in ("full", "minimal"):
            if not isinstance(different_class_samples, (int, list, tuple)):
                raise ValueError(
                    "`different_class_samples` argument must be one of 'full', 'minimal', "
                    "an integer, a list or tuple of integers or keyword 'full'."
                    f"Received: different_class_samples={different_class_samples}."
                )
            elif isinstance(different_class_samples, (list, tuple)):
                if (
                    not (
                        all(
                            isinstance(i, int) or i == "full"
                            for i in different_class_samples
                        )
                    )
                    or (len(different_class_samples)) != 2
                ):
                    raise ValueError(
                        "When passing `different_class_samples` as a tuple or list, "
                        "elements must be exactly two of integer type or keyword 'full' "
                        "(N, M). "
                        f"Received: different_class_samples={different_class_samples}."
                    )

        if nsplits != "best" and not isinstance(nsplits, int):
            raise ValueError(
                '`nsplits` argument must be either "best" or of type integer '
                f"Received: nsplits={nsplits} with type {type(nsplits)}."
            )

        if any(metric not in metrics_caller for metric in metrics):
            raise ValueError(
                f"`metric` argument must be one of {tuple(metrics_caller.keys())} "
                f"Received: metric={metrics}"
            )

        if p < 1:
            raise ValueError(f"`p` must be at least 1. Received: p={p}")

    def find_optimal_cutoff(self):
        """Find the optimal cutoff point
        Returns:
            float: optimal cutoff value

        """
        self.check_experiment_run()
        self.optimal_cutoff = {}
        for metric in self.metrics:
            fpr, tpr, threshold = roc_curve(self.df["target"], self.df[metric])
            i = np.arange(len(tpr))
            roc = pd.DataFrame(
                {
                    "tf": pd.Series(tpr - (1 - fpr), index=i),
                    "threshold": pd.Series(threshold, index=i),
                }
            )
            roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
            self.optimal_cutoff[metric] = roc_t["threshold"].item()
        return self.optimal_cutoff

    def find_thr_at_fpr(self, fpr):
        raise NotImplementedError

    def get_binary_prediction(self, metric, threshold):
        return (
            self.df[metric].apply(lambda x: 1 if x < threshold else 0)
            if metric in DISTANCE_TO_SIMILARITY
            else self.df[metric].apply(lambda x: 1 if x > threshold else 0)
        )

    def evaluate_at_threshold(self, threshold: float, metric: str):
        """Evaluate performance at specific threshold
        Args:
            threshold: cut-off threshold.
            metric: metric to use.

        Returns:
            dict: containing all evaluation metrics.
        """
        self.metrics_evaluation = {}
        self.check_experiment_run(metric)
        for metric in self.metrics:
            predicted = self.get_binary_prediction(metric, threshold)
            cm = confusion_matrix(self.df["target"], predicted)
            tn, fp, fn, tp = cm.ravel()
            TPR = tp / (tp + fn)  # recall / true positive rate
            TNR = tn / (tn + fp)  # true negative rate
            PPV = tp / (tp + fp)  # precision / positive predicted value
            NPV = tn / (tn + fn)  # negative predictive value
            FPR = fp / (fp + tn)  # false positive rate
            FNR = 1 - TPR  # false negative rate
            FDR = 1 - PPV  # false discovery rate
            FOR = 1 - NPV  # false omission rate
            # LRp = TPR / FPR  # positive likelihood ratio (LR+)
            # LRn = FNR / TNR  # negative likelihood ratio (LR+)

            evaluation = {
                "TPR": TPR,
                "TNR": TNR,
                "PPV": PPV,
                "NPV": NPV,
                "FPR": FPR,
                "FNR": FNR,
                "FDR": FDR,
                "FOR": FOR,
                # "LR+": LRp,
                # "LR-": LRn,
            }

            # self.metrics_evaluation[metric] = evaluation

        return evaluation

    def check_experiment_run(self, metric=None):
        caller = sys._getframe().f_back.f_code.co_name
        if not self.experiment_sucess:
            raise NotImplementedError(
                f"`{caller}` function can only be run after running "
                "`run_experiment`."
            )
        if metric is not None and metric not in self.metrics:
            raise ValueError(
                f"`{caller}` function was can only be called with `metric` from "
                f"{self.metrics} which were used while running the experiment"
            )

    def get_roc_auc(self):
        self.check_experiment_run()
        self.roc_auc = {}
        for metric in self.metrics:
            predicted = self.predicted_as_similarity(metric)
            fpr, tpr, thresholds = roc_curve(self.df["target"], predicted)
            self.roc_auc[metric] = auc(fpr, tpr)
        self.roc_auc = OrderedDict(
            sorted(self.roc_auc.items(), key=lambda x: x[1], reverse=True)
        )
        return self.roc_auc

    def predicted_as_similarity(self, metric):
        predicted = self.df[metric]
        if metric in DISTANCE_TO_SIMILARITY:
            predicted = (
                self.cached_predicted_as_similarity[metric]
                if metric in self.cached_predicted_as_similarity
                else DISTANCE_TO_SIMILARITY.get(metric)(predicted)
            )
        return predicted
