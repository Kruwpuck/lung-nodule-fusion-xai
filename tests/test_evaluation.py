"""Tests for evaluation metrics and statistical tests."""
import numpy as np
import pytest
from src.evaluation.metrics import compute_metrics, bootstrap_ci, build_calibration_data
from src.evaluation.statistical_tests import (
    delong_test, build_ablation_table, holm_correction,
    brown_forsythe_test, levene_test, pairwise_variance_tests, factorial_anova,
)


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


class TestHolmCorrection:
    def test_textbook_example(self):
        # Standard Holm-Bonferroni worked example, n=4:
        # sorted p = [0.01, 0.02, 0.03, 0.05] -> multipliers [4,3,2,1]
        # raw products = [0.04, 0.06, 0.06, 0.05] -> cummax = [0.04, 0.06, 0.06, 0.06]
        adj = holm_correction([0.01, 0.02, 0.03, 0.05])
        assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06])

    def test_monotonic_nondecreasing_in_sorted_order(self):
        raw = [0.2, 0.001, 0.04, 0.03, 0.5]
        adj = holm_correction(raw)
        order = np.argsort(raw)
        sorted_adj = np.array(adj)[order]
        assert np.all(np.diff(sorted_adj) >= -1e-12)

    def test_capped_at_one(self):
        adj = holm_correction([0.9, 0.95, 0.99])
        assert all(a <= 1.0 for a in adj)


class TestVarianceTests:
    def test_brown_forsythe_detects_unequal_variance(self):
        rng = np.random.default_rng(0)
        low_var = rng.normal(0.9, 0.01, 30)
        high_var = rng.normal(0.9, 0.15, 30)
        stat, p = brown_forsythe_test(low_var, high_var)
        assert p < 0.05

    def test_brown_forsythe_no_diff_for_equal_variance(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.9, 0.02, 200)
        b = rng.normal(0.85, 0.02, 200)
        stat, p = brown_forsythe_test(a, b)
        assert p > 0.05

    def test_brown_forsythe_matches_scipy_median_centered(self):
        from scipy import stats as sstats
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, 40)
        b = rng.normal(0, 2, 40)
        stat, p = brown_forsythe_test(a, b)
        ref_stat, ref_p = sstats.levene(a, b, center="median")
        assert stat == pytest.approx(ref_stat)
        assert p == pytest.approx(ref_p)

    def test_levene_matches_scipy_mean_centered(self):
        from scipy import stats as sstats
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 40)
        b = rng.normal(0, 2, 40)
        stat, p = levene_test(a, b)
        ref_stat, ref_p = sstats.levene(a, b, center="mean")
        assert stat == pytest.approx(ref_stat)
        assert p == pytest.approx(ref_p)

    def test_pairwise_variance_tests_holm_correction_applied(self):
        import pandas as pd
        rng = np.random.default_rng(4)
        rows = []
        for model, scale in [("a", 0.01), ("b", 0.15), ("c", 0.01)]:
            for v in rng.normal(0.9, scale, 30):
                rows.append({"model": model, "best_score": v})
        df = pd.DataFrame(rows)
        result = pairwise_variance_tests(df, "model", "best_score", test="brown-forsythe")
        assert len(result) == 3  # C(3,2) pairs
        assert "p_holm" in result.columns
        assert (result["p_holm"] >= result["p_value"]).all()
        # a vs b (very different variance) should be the smallest, most significant pair
        ab_row = result[(result["group_a"] == "a") & (result["group_b"] == "b")].iloc[0]
        assert ab_row["significant_holm_05"]

    def test_pairwise_variance_tests_null_when_all_equal(self):
        import pandas as pd
        rng = np.random.default_rng(5)
        rows = []
        for model in ["a", "b", "c"]:
            for v in rng.normal(0.9, 0.02, 100):
                rows.append({"model": model, "best_score": v})
        df = pd.DataFrame(rows)
        result = pairwise_variance_tests(df, "model", "best_score", test="brown-forsythe")
        assert not result["significant_holm_05"].any()


class TestFactorialAnova:
    def test_dominant_factor_gets_most_variance(self):
        import pandas as pd
        rng = np.random.default_rng(6)
        rows = []
        # Factor "big" moves the mean by 10, factor "small" by 0.1; tiny noise.
        for big in [0, 1]:
            for small in [0, 1]:
                for _ in range(20):
                    y = 10 * big + 0.1 * small + rng.normal(0, 0.01)
                    rows.append({"big": big, "small": small, "fold": _ % 5, "y": y})
        df = pd.DataFrame(rows)
        table = factorial_anova(df, ["big", "small"], "y")
        eta = table.set_index("term")["eta_sq"]
        assert eta["big"] > 0.9
        assert eta["small"] < eta["big"]

    def test_eta_squared_sums_to_one(self):
        import pandas as pd
        rng = np.random.default_rng(7)
        rows = []
        for big in [0, 1]:
            for small in [0, 1]:
                for _ in range(15):
                    y = 10 * big + 0.1 * small + rng.normal(0, 0.05)
                    rows.append({"big": big, "small": small, "y": y})
        df = pd.DataFrame(rows)
        table = factorial_anova(df, ["big", "small"], "y")
        assert table["eta_sq"].sum() == pytest.approx(1.0, abs=1e-8)
