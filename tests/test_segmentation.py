"""Tests for segmentation utilities."""
import numpy as np
import pytest
from src.segmentation.consensus import compute_icc, dice_score


class TestDiceScore:
    def test_identical_masks(self):
        m = np.array([1, 1, 0, 0, 1], dtype=bool)
        assert dice_score(m, m) == 1.0

    def test_no_overlap(self):
        a = np.array([1, 1, 0, 0], dtype=bool)
        b = np.array([0, 0, 1, 1], dtype=bool)
        assert dice_score(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([1, 1, 1, 0], dtype=bool)
        b = np.array([0, 1, 1, 1], dtype=bool)
        score = dice_score(a, b)
        assert 0.0 < score < 1.0

    def test_empty_masks_returns_one(self):
        a = np.zeros(5, dtype=bool)
        b = np.zeros(5, dtype=bool)
        assert dice_score(a, b) == 1.0


class TestICC:
    def test_perfect_agreement(self):
        n, f = 10, 5
        a = np.random.randn(n, f)
        icc = compute_icc(a, a)
        np.testing.assert_allclose(icc, np.ones(f), atol=0.01)

    def test_shape(self):
        a = np.random.randn(20, 10)
        b = np.random.randn(20, 10)
        icc = compute_icc(a, b)
        assert icc.shape == (10,)

    def test_icc_range(self):
        a = np.random.randn(20, 5)
        b = np.random.randn(20, 5)
        icc = compute_icc(a, b)
        # ICC can be negative in degenerate cases, but should be bounded
        assert icc.max() <= 1.0 + 1e-6
