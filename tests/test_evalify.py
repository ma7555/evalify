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

    def test_create_experiment_euclidean_distance(self):
        """Test create_experiment with euclidean_distance"""
        df = evalify.create_experiment(
            self.embs, self.targets, metric="euclidean_distance"
        )
        df_l2 = evalify.create_experiment(
            self.embs, self.targets, metric="euclidean_distance_l2"
        )
        self.assertGreater(df.euclidean_distance.max(), 0)
        self.assertGreater(df_l2.euclidean_distance_l2.max(), 0)

    def test_create_experiment_cosine_similarity(self):
        """Test create_experiment with cosine_similarity"""
        df = evalify.create_experiment(
            self.embs, self.targets, metric="cosine_similarity"
        )
        self.assertLessEqual(df.cosine_similarity.max(), 1)

    def test_create_experiment_full_class_samples(self):
        """Test create_experiment with return_embeddings"""
        df = evalify.create_experiment(
            self.embs,
            self.targets,
            same_class_samples="full",
            different_class_samples=("full", "full"),
        )
        self.assertEqual(len(df), comb(self.nphotos, 2))

    def test_create_experiment_custom_class_samples(self):
        """Test create_experiment with custom same_class_samples and
        different_class_samples
        """
        N, M = (2, 5)
        same_class_samples = 3
        df = evalify.create_experiment(
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

    def test_create_experiment_return_embeddings(self):
        """Test create_experiment with return_embeddings"""
        df = evalify.create_experiment(self.embs, self.targets, return_embeddings=True)
        self.assertLessEqual(len(df.at[0, "img_a"]), self.emb_size)

    def test_create_experiment_errors(self):
        """Test create_experiment errors"""
        with self.assertRaisesRegex(
            ValueError,
            "`same_class_samples` argument must be one of 'full' or an integer ",
        ):
            _ = evalify.create_experiment(
                self.embs, self.targets, same_class_samples=54.4
            )

        with self.assertRaisesRegex(
            ValueError,
            "`different_class_samples` argument must be one of 'full', 'minimal', ",
        ):
            _ = evalify.create_experiment(
                self.embs, self.targets, different_class_samples="all"
            )

        with self.assertRaisesRegex(
            ValueError, '`nsplits` argument must be either "best" or of type integer'
        ):
            _ = evalify.create_experiment(self.embs, self.targets, nsplits="all")

        with self.assertRaisesRegex(ValueError, "`metric` argument must be one of "):
            _ = evalify.create_experiment(self.embs, self.targets, metric="dot_prod")
