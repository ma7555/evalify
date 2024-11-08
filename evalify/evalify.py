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
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from evalify.metrics import (
    DISTANCE_TO_SIMILARITY,
    METRICS_NEED_NORM,
    METRICS_NEED_ORDER,
    REVERSE_DISTANCE_TO_SIMILARITY,
    metrics_caller,
)
from evalify.utils import _validate_vectors, calculate_best_batch_size

StrOrInt = Union[str, int]
StrIntSequence = Union[str, int, Sequence[Union[str, int]]]


class Experiment:
    """Defines an experiment for evalifying.

    Args:
        metrics: The list of metrics to use. Can be one or more of the following:
            `cosine_similarity`, `pearson_similarity`, `cosine_distance`,
            `euclidean_distance`, `euclidean_distance_l2`, `minkowski_distance`,
            `manhattan_distance` and `chebyshev_distance`
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
        seed: Optional random seed for reproducibility.


    Notes:
        - `same_class_samples`:
            If the provided number is greater than the achievable for the class,
            the maximum possible combinations are used.
        - `different_class_samples`:
            If the provided number is greater than the achievable for the class,
            the maximum possible combinations are used. (N, M) can also be
            ('full', 'full') but this will calculate all possible combinations
            between all posibile negative samples. If the dataset is not small
            this will probably result in an extremely large array!.

    """

    def __init__(
        self,
        metrics: Union[str, Sequence[str]] = "cosine_similarity",
        same_class_samples: StrOrInt = "full",
        different_class_samples: StrIntSequence = "minimal",
        seed: Optional[int] = None,
    ) -> None:
        self.experiment_success = False
        self.cached_predicted_as_similarity = {}
        self.metrics = (metrics,) if isinstance(metrics, str) else metrics
        self.same_class_samples = same_class_samples
        self.different_class_samples = different_class_samples
        self.seed = seed

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)

    @staticmethod
    def _validate_args(
        metrics: Sequence[str],
        same_class_samples: StrOrInt,
        different_class_samples: StrIntSequence,
        batch_size: Optional[StrOrInt],
        p,
    ) -> None:
        """Validates passed arguments to Experiment.run() method."""
        if same_class_samples != "full" and not isinstance(same_class_samples, int):
            msg = (
                "`same_class_samples` argument must be one of 'full' or an integer "
                f"Received: same_class_samples={same_class_samples}"
            )
            raise ValueError(
                msg,
            )

        if different_class_samples not in ("full", "minimal"):
            if not isinstance(different_class_samples, (int, list, tuple)):
                msg = (
                    "`different_class_samples` argument must be one of 'full', "
                    "'minimal', an integer, a list or tuple of integers or keyword "
                    "'full'."
                    f"Received: different_class_samples={different_class_samples}."
                )
                raise ValueError(
                    msg,
                )
            if isinstance(different_class_samples, (list, tuple)) and (
                not (
                    all(
                        isinstance(i, int) or i == "full"
                        for i in different_class_samples
                    )
                )
                or (len(different_class_samples)) != 2
            ):
                msg = (
                    "When passing `different_class_samples` as a tuple or list, "
                    "elements must be exactly two of integer type or keyword 'full' "
                    "(N, M). "
                    f"Received: different_class_samples={different_class_samples}."
                )
                raise ValueError(
                    msg,
                )

        if (
            batch_size != "best"
            and not isinstance(batch_size, int)
            and batch_size is not None
        ):
            msg = (
                '`batch_size` argument must be either "best" or of type integer '
                f"Received: batch_size={batch_size} with type {type(batch_size)}."
            )
            raise ValueError(
                msg,
            )

        if any(metric not in metrics_caller for metric in metrics):
            msg = (
                f"`metric` argument must be one of {tuple(metrics_caller.keys())} "
                f"Received: metric={metrics}"
            )
            raise ValueError(
                msg,
            )

        if p < 1:
            msg = f"`p` must be an int and at least 1. Received: p={p}"
            raise ValueError(msg)

    def _get_pairs(
        self,
        y,
        same_class_samples,
        different_class_samples,
        target,
    ) -> List[Tuple]:
        """Generates experiment pairs."""
        same_ixs_full = np.argwhere(y == target).ravel()
        if isinstance(same_class_samples, int):
            same_class_samples = min(len(same_ixs_full), same_class_samples)
            same_ixs = self.rng.choice(same_ixs_full, same_class_samples)
        elif same_class_samples == "full":
            same_ixs = same_ixs_full
        same_pairs = itertools.combinations(same_ixs, 2)
        same_pairs = [(a, b, target, target, 1) for a, b in same_pairs]

        different_ixs = np.argwhere(y != target).ravel()
        diff_df = pd.DataFrame(
            data={"sample_idx": different_ixs, "target": y[different_ixs]},
        )

        diff_df = diff_df.sample(frac=1, random_state=self.seed)
        if different_class_samples in ["full", "minimal"] or isinstance(
            different_class_samples,
            int,
        ):
            N = 1
            if different_class_samples == "minimal":
                diff_df = diff_df.drop_duplicates(subset=["target"])
        else:
            N, M = different_class_samples
            N = len(same_ixs_full) if N == "full" else min(N, len(same_ixs_full))
            if M != "full":
                diff_df = (
                    diff_df.groupby("target")
                    .apply(lambda x: x[:M], include_groups=False)
                    .droplevel(0)
                )

        different_ixs = diff_df.sample_idx.to_numpy()

        different_pairs = itertools.product(
            self.rng.choice(same_ixs_full, N, replace=False),
            different_ixs,
        )
        different_pairs = [(a, b, target, y[b], 0) for a, b in different_pairs if a < b]

        return same_pairs + different_pairs

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: Optional[StrOrInt] = "best",
        shuffle: bool = False,
        return_embeddings: bool = False,
        p: int = 3,
    ) -> pd.DataFrame:
        """Runs an experiment for face verification
        Args:
            X: Embeddings array
            y: Targets for X as integers
            batch_size:
                - 'best': Let the program decide based on available memory such that
                    every batch will fit into the available memory. (Default)
                - int: Manually decide the batch_size.
                - None: No batching. All experiment and intermediate results must fit
                    entirely into memory or a MemoryError will be raised.
            shuffle: Shuffle the returned experiment dataframe. Default: False.
            return_embeddings: Whether to return the embeddings instead of indexes.
                Default: False
            p:
                The order of the norm of the difference. Should be `p >= 1`, Only valid
                with minkowski_distance as a metric. Default = 3.

        Returns:
            pandas.DataFrame: A DataFrame representing the experiment results.

        Raises:
            ValueError: An error occurred with the provided arguments.

        """
        self._validate_args(
            self.metrics,
            self.same_class_samples,
            self.different_class_samples,
            batch_size,
            p,
        )
        X, y = _validate_vectors(X, y)
        all_targets = np.unique(y)
        all_pairs = []
        metric_fns = list(map(metrics_caller.get, self.metrics))
        self.rng = np.random.default_rng(self.seed)
        for target in all_targets:
            all_pairs += self._get_pairs(
                y,
                self.same_class_samples,
                self.different_class_samples,
                target,
            )

        self.df = pd.DataFrame(
            data=all_pairs,
            columns=["emb_a", "emb_b", "target_a", "target_b", "target"],
        )
        experiment_size = len(self.df)
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed)
        if batch_size == "best":
            batch_size = calculate_best_batch_size(X)
        elif batch_size is None:
            batch_size = experiment_size
        kwargs = {}
        if any(metric in METRICS_NEED_NORM for metric in self.metrics):
            kwargs["norms"] = np.linalg.norm(X, axis=1)
        if any(metric in METRICS_NEED_ORDER for metric in self.metrics):
            kwargs["p"] = p

        emb_a = self.df.emb_a.to_numpy()
        emb_b = self.df.emb_b.to_numpy()

        emb_a_s = np.array_split(emb_a, np.ceil(experiment_size / batch_size))
        emb_b_s = np.array_split(emb_b, np.ceil(experiment_size / batch_size))

        for metric, metric_fn in zip(self.metrics, metric_fns):
            self.df[metric] = np.hstack(
                [metric_fn(X, i, j, **kwargs) for i, j in zip(emb_a_s, emb_b_s)],
            )
        if return_embeddings:
            self.df["emb_a"] = X[emb_a].tolist()
            self.df["emb_b"] = X[emb_b].tolist()

        self.experiment_success = True
        return self.df

    def find_optimal_cutoff(self) -> dict:
        """Finds the optimal cutoff threshold for each metric based on the ROC curve.

        This function calculates the optimal threshold for each metric by finding the
        point on the Receiver Operating Characteristic (ROC) curve where the difference
        between the True Positive Rate (TPR) and the False Positive Rate (FPR) is
        minimized.

        Returns:
            dict: A dictionary with metrics as keys and their corresponding optimal
            threshold as values.
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
                },
            )
            roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
            self.optimal_cutoff[metric] = roc_t["threshold"].item()
        return self.optimal_cutoff

    def threshold_at_fpr(self, fpr: float) -> dict:
        """Find the threshold at a specified False Positive Rate (FPR) for each metric.

        The function calculates the threshold at the specified FPR for each metric
        by using the Receiver Operating Characteristic (ROC) curve. If the desired
        FPR is 0 or 1, or no exact match is found, the closest thresholds are used.

        Args:
            fpr (float): Desired False Positive Rate. Must be between 0 and 1.

        Returns:
            dict: A dictionary where keys are the metrics and values are dictionaries
            containing FPR, TPR, and threshold at the specified FPR.

        Raises:
            ValueError: If the provided `fpr` is not between 0 and 1.
        """

        self.check_experiment_run()
        if not 0 <= fpr <= 1:
            msg = "`fpr` must be between 0 and 1. " f"Received wanted_fpr={fpr}"
            raise ValueError(
                msg,
            )
        threshold_at_fpr = {}
        for metric in self.metrics:
            predicted = self.predicted_as_similarity(metric)
            FPR, TPR, thresholds = roc_curve(
                self.df["target"],
                predicted,
                drop_intermediate=False,
            )
            df_fpr_tpr = pd.DataFrame({"FPR": FPR, "TPR": TPR, "threshold": thresholds})
            ix_left = np.searchsorted(df_fpr_tpr["FPR"], fpr, side="left")
            ix_right = np.searchsorted(df_fpr_tpr["FPR"], fpr, side="right")

            if fpr == 0:
                best = df_fpr_tpr.iloc[ix_right]
            elif fpr == 1 or ix_left == ix_right:
                best = df_fpr_tpr.iloc[ix_left]
            else:
                best = (
                    df_fpr_tpr.iloc[ix_left]
                    if abs(df_fpr_tpr.iloc[ix_left].FPR - fpr)
                    < abs(df_fpr_tpr.iloc[ix_right].FPR - fpr)
                    else df_fpr_tpr.iloc[ix_right]
                )
            best = best.to_dict()
            if metric in REVERSE_DISTANCE_TO_SIMILARITY:
                best["threshold"] = REVERSE_DISTANCE_TO_SIMILARITY.get(metric)(
                    best["threshold"],
                )
            threshold_at_fpr[metric] = best
        return threshold_at_fpr

    def get_binary_prediction(self, metric: str, threshold: float) -> pd.Series:
        """Binary classification prediction based on the given metric and threshold.

        Args:
            metric: Metric name for the desired prediction.
            threshold: Cut off threshold.

        Returns:
            pd.Series: Binary predictions.

        """
        return (
            self.df[metric].apply(lambda x: 1 if x < threshold else 0)
            if metric in DISTANCE_TO_SIMILARITY
            else self.df[metric].apply(lambda x: 1 if x > threshold else 0)
        )

    def evaluate_at_threshold(self, threshold: float, metric: str) -> dict:
        """Evaluate performance at specific threshold
        Args:
            threshold: Cut-off threshold.
            metric: Metric to use.

        Returns:
            dict: A dict ontaining all evaluation metrics.

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
            F1 = 2 * (PPV * TPR) / (PPV + TPR)

            evaluation = {
                "TPR": TPR,
                "TNR": TNR,
                "PPV": PPV,
                "NPV": NPV,
                "FPR": FPR,
                "FNR": FNR,
                "FDR": FDR,
                "FOR": FOR,
                "F1": F1,
            }

        return evaluation

    def check_experiment_run(self, metric: Optional[str] = None) -> bool:
        caller = sys._getframe().f_back.f_code.co_name
        if not self.experiment_success:
            msg = (
                f"`{caller}` function can only be run after running "
                "`run_experiment`."
            )
            raise NotImplementedError(
                msg,
            )
        if metric is not None and metric not in self.metrics:
            msg = (
                f"`{caller}` function can only be called with `metric` from "
                f"{self.metrics} which were used while running the experiment"
            )
            raise ValueError(
                msg,
            )
        return True

    def roc_auc(self) -> OrderedDict:
        """Find ROC AUC for all the metrics used.

        Returns:
            OrderedDict: An OrderedDict with AUC for all metrics.

        """
        self.check_experiment_run()
        self.roc_auc = {}
        for metric in self.metrics:
            predicted = self.predicted_as_similarity(metric)
            fpr, tpr, thresholds = roc_curve(
                self.df["target"],
                predicted,
                drop_intermediate=False,
            )
            self.roc_auc[metric] = auc(fpr, tpr).item()
        self.roc_auc = OrderedDict(
            sorted(self.roc_auc.items(), key=lambda x: x[1], reverse=True),
        )
        return self.roc_auc

    def predicted_as_similarity(self, metric: str) -> pd.Series:
        """Convert distance metrics to a similarity measure.

        Args:
            metric: distance metric to convert to similarity. If a similarity metric is
                passed, It gets returned unchanged.

        Returns:
            pd.Series: Converted distance to similarity.

        """
        predicted = self.df[metric]
        if metric in DISTANCE_TO_SIMILARITY:
            predicted = (
                self.cached_predicted_as_similarity[metric]
                if metric in self.cached_predicted_as_similarity
                else DISTANCE_TO_SIMILARITY.get(metric)(predicted)
            )
            self.cached_predicted_as_similarity[metric] = predicted
        return predicted

    def eer(self) -> OrderedDict:
        """Calculates the Equal Error Rate (EER) for each metric.

        Returns:
            OrderedDict: A dictionary containing the EER value and threshold for each
            metric.
                The metrics are sorted in ascending order based on the EER values.
                Example: {'metric1': {'EER': 0.123, 'threshold': 0.456},
                        ...}

        """
        self.check_experiment_run()
        self.eer = {}
        for metric in self.metrics:
            predicted = self.predicted_as_similarity(metric)
            actual = self.df["target"]

            fpr, tpr, thresholds = roc_curve(
                actual,
                predicted,
                pos_label=1,
                drop_intermediate=False,
            )
            fnr = 1 - tpr
            eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))].item()
            eer_1 = fpr[np.nanargmin(np.absolute(fnr - fpr))].item()
            eer_2 = fnr[np.nanargmin(np.absolute(fnr - fpr))].item()
            if metric in REVERSE_DISTANCE_TO_SIMILARITY:
                eer_threshold = REVERSE_DISTANCE_TO_SIMILARITY.get(metric)(
                    eer_threshold,
                )

            self.eer[metric] = {"EER": (eer_1 + eer_2) / 2, "threshold": eer_threshold}
        self.eer = OrderedDict(
            sorted(self.eer.items(), key=lambda x: x[1]["EER"], reverse=False),
        )

        return self.eer

    def tar_at_far(self, far_values: List[float]) -> OrderedDict:
        """Calculates TAR at specified FAR values for each metric.

        Args:
            far_values (List[float]): A list of False Accept Rates (FAR) to get TAR
                values for.

        Returns:
            OrderedDict: A dictionary with keys as metrics and values as dictionaries
            of FAR:TAR pairs.

        Raises:
            ValueError: If any FAR in far_values is not between 0 and 1.
        """
        if isinstance(far_values, (float, int)):
            far_values = [float(far_values)]

        if not all(0 <= far <= 1 for far in far_values):
            raise ValueError("All FAR values must be between 0 and 1.")

        self.check_experiment_run()
        tar_at_far_results = {}

        for metric in self.metrics:
            predicted = self.predicted_as_similarity(metric)
            fpr, tpr, _ = roc_curve(self.df["target"], predicted, pos_label=1)

            tar_values = {}
            for far in far_values:
                idx = np.searchsorted(fpr, far, side="right") - 1
                idx = max(0, min(idx, len(fpr) - 1))  # Ensure idx is within bounds
                tar_values[far] = tpr[idx].item()

            tar_at_far_results[metric] = tar_values

        self.tar_at_far_results = OrderedDict(
            sorted(tar_at_far_results.items(), key=lambda x: list(x[1].keys())[0])
        )

        return self.tar_at_far_results
