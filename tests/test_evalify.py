#!/usr/bin/env python

"""Tests for `evalify` package."""
import os
import pathlib
import sys
from scipy.special import comb

sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent, "evalify"))
import unittest

import numpy as np

from evalify import evalify


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
        experiment = evalify.Experiment()
        df = experiment.run(self.embs, self.targets, metrics="euclidean_distance")
        df_l2 = experiment.run(self.embs, self.targets, metrics="euclidean_distance_l2")
        self.assertGreater(df.euclidean_distance.max(), 0)
        self.assertGreater(df_l2.euclidean_distance_l2.max(), 0)

    def test_run_cosine_similarity(self):
        """Test run with cosine_similarity"""
        experiment = evalify.Experiment()
        df = experiment.run(self.embs, self.targets, metrics="cosine_similarity")
        self.assertLessEqual(df.cosine_similarity.max(), 1)

    def test_run_full_class_samples(self):
        """Test run with return_embeddings"""
        experiment = evalify.Experiment()
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
        experiment = evalify.Experiment()
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

    def test_run_return_embeddings(self):
        """Test run with return_embeddings"""
        experiment = evalify.Experiment()
        df = experiment.run(self.embs, self.targets, return_embeddings=True)
        self.assertLessEqual(len(df.at[0, "img_a"]), self.emb_size)

    def test_run_evaluate_at_threshold(self):
        """Test run with evaluate_at_threshold"""
        experiment = evalify.Experiment()
        metrics = ["cosine_similarity", "euclidean_distance_l2"]
        experiment.run(
            self.embs,
            self.targets,
            metrics=metrics,
        )
        evaluations = experiment.evaluate_at_threshold(0.5)
        self.assertEqual(len(evaluations), len(metrics))
        self.assertEqual(len(evaluations[metrics[0]]), 10)

    def test_run_errors(self):
        """Test run errors"""
        with self.assertRaisesRegex(
            ValueError,
            "`same_class_samples` argument must be one of 'full' or an integer ",
        ):
            experiment = evalify.Experiment()
            _ = experiment.run(self.embs, self.targets, same_class_samples=54.4)

        with self.assertRaisesRegex(
            ValueError,
            "`different_class_samples` argument must be one of 'full', 'minimal', ",
        ):
            experiment = evalify.Experiment()
            _ = experiment.run(self.embs, self.targets, different_class_samples="all")

        with self.assertRaisesRegex(
            ValueError, '`nsplits` argument must be either "best" or of type integer'
        ):
            experiment = evalify.Experiment()
            _ = experiment.run(self.embs, self.targets, nsplits="all")

        with self.assertRaisesRegex(ValueError, "`metric` argument must be one of "):
            experiment = evalify.Experiment()
            _ = experiment.run(self.embs, self.targets, metrics="dot_prod")
