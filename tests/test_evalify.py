#!/usr/bin/env python

"""Tests for `evalify` package."""
import unittest

import numpy as np
from scipy.special import comb

from evalify import Experiment
from evalify.metrics import metrics_caller


class TestEvalify(unittest.TestCase):
    """Tests for `evalify` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        rng = np.random.default_rng(555)
        self.nphotos = 500
        self.emb_size = 8
        self.nclasses = 10
        self.embs = rng.random((self.nphotos, self.emb_size), dtype=np.float32)
        self.targets = rng.integers(self.nclasses, size=self.nphotos)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_run_euclidean_distance(self):
        """Test run with euclidean_distance"""
        experiment = Experiment()
        df = experiment.run(self.embs, self.targets, metrics="euclidean_distance")
        df_l2 = experiment.run(self.embs, self.targets, metrics="euclidean_distance_l2")
        self.assertGreater(df.euclidean_distance.max(), 0)
        self.assertGreater(df_l2.euclidean_distance_l2.max(), 0)

    def test_run_cosine_similarity(self):
        """Test run with cosine_similarity"""
        experiment = Experiment()
        df = experiment.run(self.embs, self.targets)
        self.assertLessEqual(df.cosine_similarity.max(), 1)

    def test_run_all_metrics_separated(self):
        experiment = Experiment()
        for metric in metrics_caller.keys():
            df = experiment.run(self.embs, self.targets, metrics=metric)
            self.assertTrue(metric in df.columns)

    def test_run_all_metrics_combined(self):
        experiment = Experiment()
        metrics = set(metrics_caller.keys())
        df = experiment.run(self.embs, self.targets, metrics=metrics)
        self.assertTrue(metrics.issubset(df.columns))

    def test_run_full_class_samples(self):
        """Test run with return_embeddings"""
        experiment = Experiment()
        df = experiment.run(
            self.embs,
            self.targets,
            same_class_samples="full",
            different_class_samples=("full", "full"),
        )
        self.assertEqual(len(df), comb(self.nphotos, 2))

    def test_run_custom_class_samples(self):
        """Test run with custom same_class_samples and
        different_class_samples
        """
        experiment = Experiment()
        N, M = (2, 5)
        same_class_samples = 3
        df = experiment.run(
            self.embs,
            self.targets,
            same_class_samples=2,
            different_class_samples=(N, M),
        )

        self.assertLessEqual(
            len(df),
            (comb(same_class_samples, 2) * self.nclasses)
            + (self.nclasses * (self.nclasses - 1)) * M * N,
        )

    def test_run_shuffle(self):
        """Test run with shuffle"""
        experiment = Experiment()
        df1 = experiment.run(self.embs, self.targets, shuffle=True, seed=555)
        df2 = experiment.run(self.embs, self.targets, shuffle=True, seed=555)
        self.assertEqual(len(df1), len(df2))
        self.assertEqual(sum(df1.index), sum(df2.index))
        self.assertTrue(all(ix in df2.index for ix in df1.index))

    def test_run_return_embeddings(self):
        """Test run with return_embeddings"""
        experiment = Experiment()
        df = experiment.run(self.embs, self.targets, return_embeddings=True)
        self.assertLessEqual(len(df.at[0, "img_a"]), self.emb_size)

    def test_run_evaluate_at_threshold(self):
        """Test run with evaluate_at_threshold"""
        experiment = Experiment()
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment.run(
            self.embs,
            self.targets,
            metrics=metrics,
        )
        evaluations = experiment.evaluate_at_threshold(0.5, "cosine_similarity")
        # self.assertEqual(len(evaluations), len(metrics))
        self.assertEqual(len(evaluations), 9)

    def test_run_find_optimal_cutoff(self):
        """Test run with find_optimal_cutoff"""
        experiment = Experiment()
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment.run(
            self.embs,
            self.targets,
            metrics=metrics,
        )
        evaluations = experiment.find_optimal_cutoff()
        # self.assertEqual(len(evaluations), len(metrics))
        self.assertEqual(len(evaluations), len(metrics))
        self.assertTrue(all(evaluation in metrics for evaluation in evaluations))

    def test_run_get_roc_auc(self):
        """Test run with get_roc_auc"""
        experiment = Experiment()
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment.run(
            self.embs,
            self.targets,
            metrics=metrics,
        )
        roc_auc = experiment.get_roc_auc()
        # self.assertEqual(len(evaluations), len(metrics))
        self.assertEqual(len(roc_auc), len(metrics))
        self.assertTrue(all(auc in metrics for auc in roc_auc))

    def test_run_predicted_as_similarity(self):
        """Test run with predicted_as_similarity"""
        experiment = Experiment()
        df = experiment.run(
            self.embs, self.targets, metrics=["cosine_similarity", "cosine_distance"]
        )
        result = experiment.predicted_as_similarity("cosine_similarity")
        result_2 = experiment.predicted_as_similarity("cosine_distance")
        self.assertTrue(np.allclose(result, result_2))

    def test_run_find_threshold_at_fpr(self):
        """Test run with find_threshold_at_fpr"""
        experiment = Experiment()
        metric = "cosine_similarity"
        df = experiment.run(
            self.embs,
            self.targets,
            metrics=metric,
            different_class_samples=("full", "full"),
        )
        fpr_d01 = experiment.find_threshold_at_fpr(0.1)
        fpr_d1 = experiment.find_threshold_at_fpr(1)
        fpr_d0 = experiment.find_threshold_at_fpr(0)
        self.assertEqual(len(fpr_d01[metric]), 3)
        self.assertAlmostEqual(fpr_d01[metric]["Threshold"], 0.8939142, 3)
        self.assertAlmostEqual(fpr_d0[metric]["Threshold"], 0.9953355, 3)
        self.assertAlmostEqual(fpr_d1[metric]["Threshold"], 0.2060538, 3)

    def test__call__(self):
        """Test run with __call__"""
        experiment = Experiment()
        result = experiment.run(self.embs, self.targets, seed=555)
        result_2 = experiment(self.embs, self.targets, seed=555)
        self.assertTrue(np.array_equal(result.to_numpy(), result_2.to_numpy()))

    def test_run_errors(self):
        """Test run errors"""
        with self.assertRaisesRegex(
            ValueError,
            "`same_class_samples` argument must be one of 'full' or an integer ",
        ):
            experiment = Experiment()
            _ = experiment.run(self.embs, self.targets, same_class_samples=54.4)

        with self.assertRaisesRegex(
            ValueError,
            "`different_class_samples` argument must be one of 'full', 'minimal', ",
        ):
            experiment = Experiment()
            _ = experiment.run(self.embs, self.targets, different_class_samples="all")

        with self.assertRaisesRegex(
            ValueError,
            "When passing `different_class_samples` as a tuple or list. ",
        ):
            experiment = Experiment()
            _ = experiment.run(
                self.embs, self.targets, different_class_samples=(1, 2, 3)
            )

        with self.assertRaisesRegex(
            ValueError, '`batch_size` argument must be either "best" or of type integer'
        ):
            experiment = Experiment()
            _ = experiment.run(self.embs, self.targets, batch_size="all")

        with self.assertRaisesRegex(ValueError, "`metric` argument must be one of "):
            experiment = Experiment()
            _ = experiment.run(self.embs, self.targets, metrics="dot_prod")

        with self.assertRaisesRegex(ValueError, "`p` must be at least 1. Received: p="):
            experiment = Experiment()
            _ = experiment.run(self.embs, self.targets, p=0.1)

        with self.assertRaisesRegex(
            NotImplementedError,
            "`evaluate_at_threshold` function can only be run after running "
            "`run_experiment`.",
        ):
            experiment = Experiment()
            _ = experiment.evaluate_at_threshold(0.5, "euclidean_distance")

        with self.assertRaisesRegex(
            ValueError,
            "`evaluate_at_threshold` function was can only be called with `metric` from ",
        ):
            experiment = Experiment()
            experiment.run(self.embs, self.targets, metrics="euclidean_distance")
            _ = experiment.evaluate_at_threshold(0.5, "cosine_similarity")

        with self.assertRaisesRegex(
            ValueError, "`fpr` must be between 0 and 1. Received wanted_fpr="
        ):
            experiment = Experiment()
            experiment.run(self.embs, self.targets, metrics="euclidean_distance")
            _ = experiment.find_threshold_at_fpr(-1.1)
