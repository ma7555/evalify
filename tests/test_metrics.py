#!/usr/bin/env python

"""Tests for `evalify` package."""
import unittest

import numpy as np
from scipy.spatial import distance

from evalify import metrics


class TestMetrics(unittest.TestCase):
    """Tests for `evalify` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        rng = np.random.default_rng(555)
        self.nphotos = 500
        self.emb_size = 8
        self.slice_size = 100
        self.embs = rng.random((self.nphotos, self.emb_size), dtype=np.float32)
        self.norms = np.linalg.norm(self.embs, axis=1)
        self.ix = rng.integers(self.nphotos, size=self.slice_size)
        self.iy = rng.integers(self.nphotos, size=self.slice_size)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_cosine_similarity(self):
        """Test cosine_similarity"""
        result = metrics.cosine_similarity(self.embs, self.ix, self.iy, self.norms)
        result_2 = 1 - np.array(
            [
                distance.cosine(self.embs[ix], self.embs[iy])
                for (ix, iy) in zip(self.ix, self.iy)
            ]
        )
        self.assertEqual(result.shape, (self.slice_size,))
        self.assertTrue(np.allclose(result, result_2))

    def test_euclidean_distance(self):
        """Test euclidean_distance"""
        result = metrics.metrics_caller.get("euclidean_distance")(
            self.embs, self.ix, self.iy
        )
        result_2 = np.array(
            [
                distance.euclidean(self.embs[ix], self.embs[iy])
                for (ix, iy) in zip(self.ix, self.iy)
            ]
        )
        self.assertEqual(result.shape, (self.slice_size,))
        self.assertTrue(np.allclose(result, result_2))

    def test_euclidean_distance_l2(self):
        """Test euclidean_distance"""
        result = metrics.metrics_caller.get("euclidean_distance_l2")(
            self.embs, self.ix, self.iy, self.norms
        )
        result_2 = np.array(
            [
                distance.euclidean(
                    self.embs[ix] / np.sqrt(np.sum(self.embs[ix] ** 2)),
                    self.embs[iy] / np.sqrt(np.sum(self.embs[iy] ** 2)),
                )
                for (ix, iy) in zip(self.ix, self.iy)
            ]
        )

        self.assertEqual(result.shape, (len(self.ix),))
        self.assertTrue(np.allclose(result, result_2))

    def test_minkowski_distance_distance(self):
        """Test euclidean_distance"""
        result = metrics.metrics_caller.get("minkowski_distance")(
            self.embs, self.ix, self.iy, p=3
        )
        result_2 = np.array(
            [
                distance.minkowski(self.embs[ix], self.embs[iy], p=3)
                for (ix, iy) in zip(self.ix, self.iy)
            ]
        )
        self.assertEqual(result.shape, (self.slice_size,))
        self.assertTrue(np.allclose(result, result_2))

    def test_manhattan_distance_distance(self):
        """Test euclidean_distance"""
        result = metrics.metrics_caller.get("manhattan_distance")(
            self.embs, self.ix, self.iy
        )
        result_2 = np.array(
            [
                distance.cityblock(self.embs[ix], self.embs[iy])
                for (ix, iy) in zip(self.ix, self.iy)
            ]
        )
        self.assertEqual(result.shape, (self.slice_size,))
        self.assertTrue(np.allclose(result, result_2))

    def test_chebyshev_distance_distance(self):
        """Test euclidean_distance"""
        result = metrics.metrics_caller.get("chebyshev_distance")(
            self.embs, self.ix, self.iy
        )
        result_2 = np.array(
            [
                distance.chebyshev(self.embs[ix], self.embs[iy])
                for (ix, iy) in zip(self.ix, self.iy)
            ]
        )
        self.assertEqual(result.shape, (self.slice_size,))
        self.assertTrue(np.allclose(result, result_2))

    def test_get_norm(self):
        """Test get_norm"""
        self.assertTrue(np.allclose(self.norms, metrics.get_norms(self.embs)))
