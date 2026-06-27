"""Tests for evaluation metrics and statistical tests."""
import numpy as np
import pytest
from src.evaluation.metrics import compute_metrics, bootstrap_ci, build_calibration_data
from src.evaluation.statistical_tests import delong_test, build_ablation_table


def _perfect_preds():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    return y_true, y_prob


def _random_preds(n=50, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_prob = rng.uniform(0, 1, n)
    return y_true, y_prob


class TestComputeMetrics:
    def test_perfect_classifier(self):
        y_true, y_prob = _perfect_preds()
        m = compute_metrics(y_true, y_prob)
        assert m["auc"] == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0, abs=1e-6)
        assert m["specificity"] == pytest.approx(1.0, abs=1e-6)

    def test_metric_keys_present(self):
        y_true, y_prob = _random_preds()
        m = compute_metrics(y_true, y_prob)
        for key in ("auc", "accuracy", "sensitivity", "specificity", "f1", "brier_score"):
            assert key in m

    def test_metric_ranges(self):
        y_true, y_prob = _random_preds()
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["auc"] <= 1.0
        assert 0.0 <= m["sensitivity"] <= 1.0
        assert 0.0 <= m["specificity"] <= 1.0


class TestBootstrapCI:
    def test_ci_contains_true_auc(self):
        from sklearn.metrics import roc_auc_score
        y_true, y_prob = _random_preds(n=100)
        true_auc = roc_auc_score(y_true, y_prob)
        lo, hi = bootstrap_ci(y_true, y_prob, n_iterations=500)
        assert lo <= true_auc <= hi

    def test_ci_order(self):
        y_true, y_prob = _random_preds(n=100)
        lo, hi = bootstrap_ci(y_true, y_prob, n_iterations=200)
        assert lo < hi


class TestDeLong:
    def test_identical_models_p_one(self):
        y_true, y_prob = _random_preds(n=100)
        z, p, delta = delong_test(y_true, y_prob, y_prob)
        assert abs(z) < 1e-8
        assert p >= 0.99

    def test_perfect_vs_random_significant(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_prob_perfect = y_true.astype(float)
        y_prob_random = rng.uniform(0, 1, n)
        _, p, _ = delong_test(y_true, y_prob_perfect, y_prob_random)
        assert p < 0.05

    def test_returns_three_values(self):
        y_true, y_prob = _random_preds(n=50)
        result = delong_test(y_true, y_prob, y_prob * 0.9 + 0.05)
        assert len(result) == 3


class TestCalibration:
    def test_returns_dataframe(self):
        import pandas as pd
        y_true, y_prob = _random_preds(n=100)
        df = build_calibration_data(y_true, y_prob)
        assert isinstance(df, pd.DataFrame)
        assert "mean_predicted" in df.columns
        assert "fraction_positive" in df.columns
