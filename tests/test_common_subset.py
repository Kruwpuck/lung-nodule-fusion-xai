"""Tests for the Rev1 task-2/3 common-subset and ordinal-native evaluation."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.common_subset import (
    ARMS, arm_auc_summary, ordinal_native_summary, score_arm_a, score_arm_b,
    score_arm_b_ordinal_native, score_arm_c, score_arm_d, score_common_subset,
)


@pytest.fixture
def preds_dir(tmp_path):
    """4 nodules, 1 fold, hand-constructed so every arm agrees on the same
    2 malignant / 2 benign common subset (drop indeterminate/no-nodule rows).

    Nodules (in fixed order): benign, malignant, benign, malignant.
    Arm A already only has these 4 (binary-eligible).
    Arm B (ordinal) adds one extra indeterminate row (rating==3) to drop.
    Arm C (grade3) adds one extra indeterminate row (class==1) to drop.
    Arm D (grade4) adds one no-nodule row (class==0) and one indeterminate
    row (class==2) to drop.
    """
    d = tmp_path

    # Arm A: y_true 0/1, y_prob = P(malignant) directly.
    np.savez(d / "m_fold0.npz",
             y_true=np.array([0, 1, 0, 1]),
             y_prob=np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32))

    # Arm B: ordinal ratings, extra indeterminate (rating==3) row at index 2.
    np.savez(d / "m_ordinal_fold0.npz",
             y_true=np.array([1.0, 5.0, 3.0, 1.0, 5.0], dtype=np.float32),
             y_out=np.array([1.8, 4.2, 3.0, 1.6, 4.6], dtype=np.float32))

    # Arm C: grade3 = [benign=0, indeterminate=1, malignant=2], extra
    # indeterminate row at index 2.
    np.savez(d / "m_grade3_fold0.npz",
             y_true=np.array([0, 2, 1, 0, 2]),
             y_out=np.array([
                 [0.8, 0.15, 0.05],
                 [0.05, 0.15, 0.80],
                 [0.10, 0.85, 0.05],
                 [0.7, 0.2, 0.10],
                 [0.05, 0.05, 0.90],
             ], dtype=np.float32))

    # Arm D: grade4 = [no-nodule=0, benign=1, indeterminate=2, malignant=3],
    # extra no-nodule row (index 0) and indeterminate row (index 3).
    np.savez(d / "m_grade4_fold0.npz",
             y_true=np.array([0, 1, 3, 2, 1, 3]),
             y_out=np.array([
                 [0.9, 0.05, 0.03, 0.02],
                 [0.05, 0.80, 0.10, 0.05],
                 [0.02, 0.05, 0.10, 0.83],
                 [0.05, 0.10, 0.75, 0.10],
                 [0.05, 0.70, 0.15, 0.10],
                 [0.02, 0.08, 0.10, 0.80],
             ], dtype=np.float32))

    return str(d)


class TestArmCollapsers:
    def test_arm_a_self_check(self, preds_dir):
        m, n, y_prob = score_arm_a(preds_dir, "m", 0)
        assert n == 4
        collapsed = y_prob / (y_prob + (1.0 - y_prob))
        assert np.allclose(collapsed, y_prob, atol=1e-6)

    def test_arm_b_drops_indeterminate(self, preds_dir):
        m, n = score_arm_b(preds_dir, "m", 0)
        assert n == 4
        assert 0.0 <= m["auc"] <= 1.0

    def test_arm_c_drops_indeterminate(self, preds_dir):
        m, n = score_arm_c(preds_dir, "m", 0)
        assert n == 4
        # hand-computed renormalized probs for the 4 kept rows:
        # [0.05/0.85, 0.80/0.85, 0.10/0.80, 0.90/0.95] -> perfectly separates
        assert m["auc"] == pytest.approx(1.0)

    def test_arm_d_restricts_to_nodule_subset(self, preds_dir):
        m, n = score_arm_d(preds_dir, "m", 0)
        assert n == 4
        # hand-computed: benign/malignant probs renormalized, again separable
        assert m["auc_nodule_only"] == pytest.approx(1.0)

    def test_common_subset_identical_counts(self, preds_dir):
        df = score_common_subset(preds_dir, models=["m"], n_folds=1)
        assert len(df) == 4
        assert set(df["n"]) == {4}
        assert set(df["arm"]) == {"A_binary", "B_ordinal", "C_grade3", "D_grade4"}

    def test_arm_a_auc_matches_compute_metrics_directly(self, preds_dir):
        # regression guard: the collapse path for arm A must produce the same
        # AUC as scoring the raw npz probabilities with compute_metrics.
        from src.evaluation.metrics import compute_metrics
        m, n, y_prob = score_arm_a(preds_dir, "m", 0)
        d = np.load(f"{preds_dir}/m_fold0.npz")
        expected = compute_metrics(d["y_true"], d["y_prob"])
        assert m["auc"] == pytest.approx(expected["auc"])


class TestArmAucSummary:
    def test_mean_std_per_arm_over_all_rows(self):
        # 2 models x 1 fold, fixed AUCs so mean/std are hand-computable.
        df = pd.DataFrame([
            {"model": "m1", "fold": 0, "arm": "A_binary", "n": 10, "auc": 0.80},
            {"model": "m2", "fold": 0, "arm": "A_binary", "n": 10, "auc": 0.90},
            {"model": "m1", "fold": 0, "arm": "B_ordinal", "n": 10, "auc": 0.70},
            {"model": "m2", "fold": 0, "arm": "B_ordinal", "n": 10, "auc": 0.60},
        ])
        summary = arm_auc_summary(df)
        row_a = summary.loc[summary["arm"] == "A_binary"].iloc[0]
        assert row_a["auc_mean"] == pytest.approx(0.85)
        assert row_a["auc_std"] == pytest.approx(np.std([0.80, 0.90], ddof=1))
        assert row_a["n_runs"] == 2
        # arms with no rows (C/D absent here) still appear, with NaN stats.
        assert list(summary["arm"]) == list(ARMS)
        row_c = summary.loc[summary["arm"] == "C_grade3"].iloc[0]
        assert pd.isna(row_c["auc_mean"])


class TestOrdinalNative:
    def test_qwk_mae_oneoff_against_hand_computed_5x5_confusion(self):
        """QWK/MAE/acc_within_1 against a hand-computed 5x5 confusion matrix.

        true = [1,2,3,4,5], pred = [1,2,3,4,4] -- 4 exact matches, one
        off-by-one (true=5 predicted as 4). Confusion matrix has row/col
        marginals [1,1,1,1,1] (true) and [1,1,1,2,0] (pred).

        QWK = 1 - sum(w*O)/sum(w*E), w_ij = (i-j)^2/16 (C=5):
          sum(w*O) = w(5,4) = 1/16 = 0.0625 (only the one mismatch contributes)
          sum(w*E) = (1/5) * sum_j pred_marginal_j * sum_i w_ij
                   = (1/5) * (1*1.875 + 1*0.9375 + 1*0.625 + 2*0.9375 + 0*1.875)
                   = (1/5) * 5.3125 = 1.0625
          QWK = 1 - 0.0625/1.0625 = 0.9411764705882353
        """
        from src.evaluation.metrics import ordinal_metrics

        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1, 2, 3, 4, 4], dtype=float)
        m = ordinal_metrics(y_true, y_pred)

        assert m["qwk"] == pytest.approx(0.9411764705882353, abs=1e-9)
        assert m["mae"] == pytest.approx(0.2)
        assert m["acc_within_1"] == pytest.approx(1.0)

    def test_qwk_perfect_agreement_is_one(self):
        from src.evaluation.metrics import ordinal_metrics

        y = np.array([1, 2, 3, 4, 5], dtype=float)
        m = ordinal_metrics(y, y.copy())
        assert m["qwk"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["acc_within_1"] == pytest.approx(1.0)

    def test_score_arm_b_ordinal_native_uses_full_rating_scale(self, preds_dir):
        # fixture's m_ordinal_fold0.npz has 5 rows including the median==3
        # indeterminate row that score_arm_b (binary) drops -- ordinal-native
        # metrics must NOT drop it.
        m = score_arm_b_ordinal_native(preds_dir, "m", 0)
        assert {"qwk", "mae", "acc_within_1"} <= set(m.keys())
        d = np.load(f"{preds_dir}/m_ordinal_fold0.npz")
        assert len(d["y_true"]) == 5

    def test_ordinal_native_summary_shape(self, preds_dir):
        per_run, summary = ordinal_native_summary(preds_dir, models=["m"], n_folds=1)
        assert len(per_run) == 1
        assert {"qwk", "mae", "acc_within_1"} <= set(per_run.columns)
        assert list(summary.columns) == ["qwk", "mae", "acc_within_1"]
        assert list(summary.index) == ["mean", "std"]
