# tests/test_experiment_real_data_small.py

import os
import pathlib
import unittest
from collections import OrderedDict

import numpy as np

from evalify import Experiment


class TestExperimentRealDataSmall(unittest.TestCase):
    """Tests for Experiment class using a subset of the LFW dataset"""

    def setUp(self):
        """Set up test fixtures."""
        # Path to LFW.npz, assuming it's in the tests/data/ directory
        self.lfw_npz = os.path.join(pathlib.Path(__file__).parent, "data", "LFW.npz")
        if not os.path.exists(self.lfw_npz):
            self.fail(f"LFW.npz not found at {self.lfw_npz}")

        X_y_array = np.load(self.lfw_npz)
        self.X = X_y_array["X"][:1000]
        self.y = X_y_array["y"][:1000]

        self.metrics = [
            "cosine_similarity",
            "pearson_similarity",
            "euclidean_distance_l2",
        ]

        self.experiment = Experiment(
            metrics=self.metrics,
            same_class_samples="full",
            different_class_samples=("full", "full"),
            seed=555,  # To ensure reproducibility
        )

        # Run the experiment once during setup to reuse the results in multiple tests
        self.df = self.experiment.run(self.X, self.y)

    def test_number_of_samples(self):
        """Test that the number of generated samples matches the expected count."""
        expected_num_samples = 499500
        actual_num_samples = len(self.df)
        self.assertEqual(
            actual_num_samples,
            expected_num_samples,
            f"Expected {expected_num_samples} samples, got {actual_num_samples}.",
        )

    def test_roc_auc(self):
        """Test that ROC AUC values match the expected results."""
        expected_roc_auc = OrderedDict(
            {
                "euclidean_distance_l2": 0.9998640116393942,
                "cosine_similarity": 0.9998640114481793,
                "pearson_similarity": 0.999858162377461,
            }
        )

        actual_roc_auc = self.experiment.roc_auc()

        self.assertEqual(
            len(actual_roc_auc),
            len(self.metrics),
            f"Expected ROC AUC for {len(self.metrics)} metrics, got "
            f"{len(actual_roc_auc)}.",
        )

        for metric in self.metrics:
            self.assertIn(
                metric, actual_roc_auc, f"ROC AUC for metric '{metric}' not found."
            )
            self.assertAlmostEqual(
                actual_roc_auc[metric],
                expected_roc_auc[metric],
                places=6,
                msg=f"ROC AUC for metric '{metric}' does not match.",
            )

    def test_threshold_at_fpr(self):
        """Test that thresholds at a specified FPR match expected values."""
        far = 0.01
        expected_threshold_at_fpr = {
            "cosine_similarity": {
                "FPR": 0.010001841326240518,
                "TPR": 0.9973539973539973,
                "threshold": 0.37717896699905396,
            },
            "pearson_similarity": {
                "FPR": 0.010001841326240518,
                "TPR": 0.9973539973539973,
                "threshold": 0.37802454829216003,
            },
            "euclidean_distance_l2": {
                "FPR": 0.010001841326240518,
                "TPR": 0.9973539973539973,
                "threshold": 1.1160835027694702,
            },
        }

        actual_threshold_at_fpr = self.experiment.threshold_at_fpr(far)

        self.assertEqual(
            len(actual_threshold_at_fpr),
            len(self.metrics),
            f"Expected Threshold @ FPR for {len(self.metrics)} metrics, got "
            f"{len(actual_threshold_at_fpr)}.",
        )

        for metric in self.metrics:
            self.assertIn(
                metric,
                actual_threshold_at_fpr,
                f"Threshold @ FPR for metric '{metric}' not found.",
            )
            expected = expected_threshold_at_fpr[metric]
            actual = actual_threshold_at_fpr[metric]

            self.assertAlmostEqual(
                actual["FPR"],
                expected["FPR"],
                places=6,
                msg=f"FPR for metric '{metric}' does not match.",
            )
            self.assertAlmostEqual(
                actual["TPR"],
                expected["TPR"],
                places=6,
                msg=f"TPR for metric '{metric}' does not match.",
            )
            self.assertAlmostEqual(
                actual["threshold"],
                expected["threshold"],
                places=6,
                msg=f"Threshold for metric '{metric}' at FAR={far} does not match.",
            )

    def test_eer(self):
        """Test that EER values and thresholds match the expected results."""
        expected_eer = OrderedDict(
            {
                "cosine_similarity": {
                    "EER": 0.004724863226023654,
                    "threshold": 0.4244731664657593,
                },
                "euclidean_distance_l2": {
                    "EER": 0.004724863226023654,
                    "threshold": 1.0728718042373657,
                },
                "pearson_similarity": {
                    "EER": 0.004914464785693375,
                    "threshold": 0.4228288531303406,
                },
            }
        )

        actual_eer = self.experiment.eer()

        self.assertEqual(
            len(actual_eer),
            len(self.metrics),
            f"Expected EER for {len(self.metrics)} metrics, got {len(actual_eer)}.",
        )

        for metric in self.metrics:
            self.assertIn(metric, actual_eer, f"EER for metric '{metric}' not found.")
            expected = expected_eer[metric]
            actual = actual_eer[metric]

            self.assertAlmostEqual(
                actual["EER"],
                expected["EER"],
                places=6,
                msg=f"EER for metric '{metric}' does not match.",
            )
            self.assertAlmostEqual(
                actual["threshold"],
                expected["threshold"],
                places=6,
                msg=f"Threshold for EER of metric '{metric}' does not match.",
            )

    def test_tar_at_far(self):
        """Test the tar_at_far method with specific FAR values."""
        # Define FAR values to test
        far_values = [0.01, 0.001]

        # Define expected TAR values based on the recent experiment
        expected_tar_at_far = OrderedDict(
            {
                "cosine_similarity": {
                    0.01: 0.9973539973539973,
                    0.001: 0.9795879795879796,
                },
                "pearson_similarity": {
                    0.01: 0.9973539973539973,
                    0.001: 0.9793989793989794,
                },
                "euclidean_distance_l2": {
                    0.01: 0.9973539973539973,
                    0.001: 0.9795879795879796,
                },
            }
        )

        # Call tar_at_far with the FAR values
        actual_tar_at_far = self.experiment.tar_at_far(far_values)

        # Assert the returned TAR@FAR matches expected values
        self.assertEqual(
            len(actual_tar_at_far),
            len(self.metrics),
            f"Expected TAR@FAR for {len(self.metrics)} metrics, got "
            f"{len(actual_tar_at_far)}.",
        )

        for metric in self.metrics:
            self.assertIn(
                metric, actual_tar_at_far, f"TAR@FAR for metric '{metric}' not found."
            )

            for far in far_values:
                self.assertIn(
                    far,
                    actual_tar_at_far[metric],
                    f"TAR@FAR for metric '{metric}' at FAR={far} not found.",
                )

                expected_tar = expected_tar_at_far[metric][far]
                actual_tar = actual_tar_at_far[metric][far]

                self.assertAlmostEqual(
                    actual_tar,
                    expected_tar,
                    places=6,
                    msg=f"TAR@FAR for metric '{metric}' at FAR={far} does not match.",
                )


# if __name__ == '__main__':
#     unittest.main()
