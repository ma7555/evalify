#!/usr/bin/env python

"""Tests for `evalify` package."""
import os
import pathlib
import sys

sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent, "evalify"))
import unittest

import numpy as np

from evalify import metrics


class TestEvalify(unittest.TestCase):
    """Tests for `evalify` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        self.embs = np.arange(12).reshape(4, 3)
        self.ix_for_metrics = range(4)
        self.norms = np.linalg.norm(self.embs, axis=1)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_cosine_similarity(self):
        """Test cosine_similarity"""
        result = metrics.cosine_similarity(
            self.embs, self.ix_for_metrics, self.ix_for_metrics, self.norms
        )
        self.assertEqual(result.shape, (len(self.ix_for_metrics),))
        self.assertAlmostEqual(result.sum(), len(self.ix_for_metrics))

    def test_euclidean_distance(self):
        """Test euclidean_distance"""
        result = metrics.euclidean_distance(
            self.embs, self.ix_for_metrics, self.ix_for_metrics
        )
        self.assertEqual(result.shape, (len(self.ix_for_metrics),))
        self.assertAlmostEqual(result.sum(), 0)

    def test_euclidean_distance_l2(self):
        """Test euclidean_distance"""
        result = metrics.euclidean_distance(
            self.embs, self.ix_for_metrics, self.ix_for_metrics, self.norms
        )
        self.assertEqual(result.shape, (len(self.ix_for_metrics),))
        self.assertAlmostEqual(result.sum(), 0)
