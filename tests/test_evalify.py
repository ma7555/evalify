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

    def test_run_euclidean_distance(self):
        """Test run with euclidean_distance"""
        experiment = Experiment(metrics="euclidean_distance")
        df = experiment.run(self.embs, self.targets)
        experiment = Experiment(metrics="euclidean_distance_l2")
        df_l2 = experiment.run(self.embs, self.targets)
        self.assertGreater(df.euclidean_distance.max(), 0)
        self.assertGreater(df_l2.euclidean_distance_l2.max(), 0)

    def test_run_cosine_similarity(self):
        """Test run with cosine_similarity"""
        experiment = Experiment(metrics="cosine_similarity")
        df = experiment.run(self.embs, self.targets)
        self.assertLessEqual(df.cosine_similarity.max(), 1)

    def test_run_all_metrics_separated(self):
        for metric in metrics_caller.keys():
            experiment = Experiment(metrics=metric)
            df = experiment.run(self.embs, self.targets)
            self.assertTrue(metric in df.columns)

    def test_run_all_metrics_combined(self):
        metrics = set(metrics_caller.keys())
        experiment = Experiment(metrics=metrics)
        df = experiment.run(self.embs, self.targets)
        self.assertTrue(metrics.issubset(df.columns))

    def test_run_full_class_samples(self):
        """Test run with return_embeddings"""
        experiment = Experiment(
            same_class_samples="full",
            different_class_samples=("full", "full"),
        )
        df = experiment.run(
            self.embs,
            self.targets,
        )
        self.assertEqual(len(df), comb(self.nphotos, 2))

    def test_run_custom_class_samples(self):
        """Test run with custom same_class_samples and different_class_samples"""
        N, M = (2, 5)
        experiment = Experiment(same_class_samples=2, different_class_samples=(N, M))
        same_class_samples = 3
        df = experiment.run(
            self.embs,
            self.targets,
        )

        self.assertLessEqual(
            len(df),
            (comb(same_class_samples, 2) * self.nclasses)
            + (self.nclasses * (self.nclasses - 1)) * M * N,
        )

    def test_run_shuffle(self):
        """Test run with shuffle"""
        experiment = Experiment(seed=555)
        df1 = experiment.run(self.embs, self.targets, shuffle=True)
        df2 = experiment.run(self.embs, self.targets, shuffle=True)
        self.assertEqual(len(df1), len(df2))
        self.assertEqual(sum(df1.index), sum(df2.index))
        self.assertTrue(all(ix in df2.index for ix in df1.index))

    def test_run_no_batch_size(self):
        """Test run with no batch_size"""
        experiment = Experiment(
            same_class_samples=2,
            different_class_samples=(1, 1),
            seed=555,
        )
        experiment.run(self.embs, self.targets, batch_size=None)
        self.assertTrue(experiment.check_experiment_run())

    def test_run_return_embeddings(self):
        """Test run with return_embeddings"""
        experiment = Experiment()
        df = experiment.run(self.embs, self.targets, return_embeddings=True)
        self.assertLessEqual(len(df.at[0, "emb_a"]), self.emb_size)

    def test_run_evaluate_at_threshold(self):
        """Test run with evaluate_at_threshold"""
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment = Experiment(metrics=metrics)
        experiment.run(
            self.embs,
            self.targets,
        )
        evaluations = experiment.evaluate_at_threshold(0.5, "cosine_similarity")
        # self.assertEqual(len(evaluations), len(metrics))
        self.assertEqual(len(evaluations), 9)

    def test_run_find_optimal_cutoff(self):
        """Test run with find_optimal_cutoff"""
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment = Experiment(metrics=metrics)
        experiment.run(
            self.embs,
            self.targets,
        )
        evaluations = experiment.find_optimal_cutoff()
        self.assertEqual(len(evaluations), len(metrics))
        self.assertTrue(all(evaluation in metrics for evaluation in evaluations))

    def test_run_get_roc_auc(self):
        """Test run with get_roc_auc"""
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment = Experiment(metrics=metrics)
        experiment.run(
            self.embs,
            self.targets,
        )
        roc_auc = experiment.roc_auc()
        # self.assertEqual(len(evaluations), len(metrics))
        self.assertEqual(len(roc_auc), len(metrics))
        self.assertTrue(all(auc in metrics for auc in roc_auc))

    def test_run_predicted_as_similarity(self):
        """Test run with predicted_as_similarity"""
        experiment = Experiment(metrics=["cosine_similarity", "cosine_distance"])
        experiment.run(
            self.embs,
            self.targets,
        )
        result = experiment.predicted_as_similarity("cosine_similarity")
        result_2 = experiment.predicted_as_similarity("cosine_distance")
        self.assertTrue(np.allclose(result, result_2))

    def test_run_find_threshold_at_fpr(self):
        """Test run with find_threshold_at_fpr"""
        metric = "cosine_similarity"
        experiment = Experiment(
            metrics=metric,
            different_class_samples=("full", "full"),
        )
        experiment.run(
            self.embs,
            self.targets,
        )
        fpr_d01 = experiment.threshold_at_fpr(0.1)
        fpr_d1 = experiment.threshold_at_fpr(1)
        fpr_d0 = experiment.threshold_at_fpr(0)
        self.assertEqual(len(fpr_d01[metric]), 3)
        self.assertAlmostEqual(fpr_d01[metric]["threshold"], 0.8939142, 3)
        self.assertAlmostEqual(fpr_d0[metric]["threshold"], 0.9953355, 3)
        self.assertAlmostEqual(fpr_d1[metric]["threshold"], 0.2060538, 3)

    def test_run_calculate_eer(self):
        """Test run with calculate_eer"""
        metric = "cosine_similarity"
        experiment = Experiment(
            metrics=metric,
            different_class_samples=("full", "full"),
        )
        experiment.run(
            self.embs,
            self.targets,
        )
        eer = experiment.eer()
        self.assertTrue("EER" in eer[metric])

    def test__call__(self):
        """Test run with __call__"""
        experiment = Experiment(seed=555)
        result = experiment.run(self.embs, self.targets)
        result_2 = experiment(self.embs, self.targets)
        self.assertTrue(np.array_equal(result.to_numpy(), result_2.to_numpy()))

    def test_run_errors(self):
        """Test run errors"""
        with self.assertRaisesRegex(
            ValueError,
            "`same_class_samples` argument must be one of 'full' or an integer ",
        ):
            experiment = Experiment(same_class_samples=54.4)
            experiment.run(self.embs, self.targets)

        with self.assertRaisesRegex(
            ValueError,
            "`different_class_samples` argument must be one of 'full', 'minimal'",
        ):
            experiment = Experiment(different_class_samples="all")
            experiment.run(self.embs, self.targets)

        with self.assertRaisesRegex(
            ValueError,
            "When passing `different_class_samples` as a tuple or list. ",
        ):
            experiment = Experiment(different_class_samples=(1, 2, 3))
            experiment.run(
                self.embs,
                self.targets,
            )

        with self.assertRaisesRegex(
            ValueError,
            '`batch_size` argument must be either "best" or of type integer',
        ):
            experiment = Experiment()
            experiment.run(self.embs, self.targets, batch_size="all")

        with self.assertRaisesRegex(ValueError, "`metric` argument must be one of "):
            experiment = Experiment(metrics="dot_prod")
            experiment.run(self.embs, self.targets)

        with self.assertRaisesRegex(
            ValueError,
            "`p` must be an int and at least 1. Received: p=",
        ):
            experiment = Experiment()
            experiment.run(self.embs, self.targets, p=0.1)

        with self.assertRaisesRegex(
            NotImplementedError,
            "`evaluate_at_threshold` function can only be run after running "
            "`run_experiment`.",
        ):
            experiment = Experiment()
            experiment.evaluate_at_threshold(0.5, "euclidean_distance")

        with self.assertRaisesRegex(
            ValueError,
            "`evaluate_at_threshold` function can only be called with `metric` from ",
        ):
            experiment = Experiment(metrics="euclidean_distance")
            experiment.run(self.embs, self.targets)
            experiment.evaluate_at_threshold(0.5, "cosine_similarity")

        with self.assertRaisesRegex(
            ValueError,
            "`fpr` must be between 0 and 1. Received wanted_fpr=",
        ):
            experiment = Experiment(metrics="euclidean_distance")
            experiment.run(self.embs, self.targets)
            experiment.threshold_at_fpr(-1.1)
