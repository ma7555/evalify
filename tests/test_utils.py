#!/usr/bin/env python

"""Tests for `evalify` package."""
import unittest

import numpy as np
from scipy.spatial import distance

from evalify import utils


class TestUtils(unittest.TestCase):
    """Tests for `evalify` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.rng = np.random.default_rng(555)
        self.nphotos = 100
        self.emb_size = 8
        self.nclasses = 10
        self.embs = self.rng.random((self.nphotos, self.emb_size), dtype=np.float32)
        self.targets = self.rng.integers(self.nclasses, size=self.nphotos)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_validate_vectors(self):
        """Test cosine_similarity"""
        embs = self.embs.tolist()
        targets = self.targets.tolist()
        X, y = utils._validate_vectors(embs, targets)
        self.assertEqual(X.shape, (self.nphotos, self.emb_size))
        self.assertEqual(y.shape, (self.nphotos,))

    def test_keep_to_max_rows(self):
        """Test cosine_similarity"""
        rows = utils._keep_to_max_rows(self.embs, 4 * utils.GB_TO_BYTE)
        self.assertEqual(rows, 1420470954)

    def test_run_errors(self):
        """Test run errors"""
        with self.assertRaisesRegex(ValueError, "Embeddings vector should be 2-D."):
            _ = utils._validate_vectors(
                X=self.rng.random(5), y=self.rng.integers(10, size=5)
            )
        with self.assertRaisesRegex(ValueError, "Target vector should be 1-D."):
            _ = utils._validate_vectors(
                X=self.rng.random((5, 5)), y=self.rng.integers(10, size=(5, 2))
            )
