"""Tests for the Rev1 task-4 arm-comparison statistics (DeLong/Friedman/Nemenyi)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.arm_comparison import (
    delong_by_model, friedman_omnibus, nemenyi_posthoc,
)
from src.evaluation.common_subset import ARMS


@pytest.fixture
def preds_dir(tmp_path):
    """1 model, 2 folds, arm A vs arm B on identical cases per fold so paired
    DeLong has a deterministic, hand-checkable input; arms C/D duplicate arm
    A's npz files (also identical cases) so every arm pair is well-defined."""
    d = tmp_path
    rng = np.random.default_rng(0)

    for fold in range(2):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob_a = np.array([0.1, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.8]) + fold * 0.0
        np.savez(d / f"m_fold{fold}.npz", y_true=y_true, y_prob=y_prob_a)

        # ordinal: rating = 1 + 4*prob so derive_binary's linear back-map
        # reproduces the same probabilities exactly, and no rating==3 rows.
        rating_true = np.where(y_true == 1, 5.0, 1.0)
        rating_pred = 1.0 + 4.0 * y_prob_a
        np.savez(d / f"m_ordinal_fold{fold}.npz", y_true=rating_true, y_out=rating_pred)

        # grade3: benign=0/malignant=2, no indeterminate rows, renormalized
        # probs equal to y_prob_a by construction.
        y_true3 = np.where(y_true == 1, 2, 0)
        p_mal = y_prob_a
        y_out3 = np.stack([1 - p_mal, np.zeros_like(p_mal), p_mal], axis=1)
        np.savez(d / f"m_grade3_fold{fold}.npz", y_true=y_true3, y_out=y_out3)

        # grade4: benign=1/malignant=3, no no-nodule/indeterminate rows.
        y_true4 = np.where(y_true == 1, 3, 1)
        y_out4 = np.stack([np.zeros_like(p_mal), 1 - p_mal,
                            np.zeros_like(p_mal), p_mal], axis=1)
        np.savez(d / f"m_grade4_fold{fold}.npz", y_true=y_true4, y_out=y_out4)

    return str(d)


class TestDeLongByModel:
    def test_identical_arms_give_p_one(self, preds_dir):
        # arms A, B, C, D collapse to the exact same y_true/y_prob here, so
        # every pairwise DeLong z must be ~0 and p ~1 (no self-difference).
        df, skipped = delong_by_model(preds_dir, models=["m"], n_folds=2)
        assert skipped == []
        assert len(df) == 6  # C(4,2) arm pairs
        assert np.allclose(df["delong_z"], 0.0, atol=1e-8)
        assert np.allclose(df["delong_p"], 1.0, atol=1e-8)
        assert np.allclose(df["delta_auc"], 0.0, atol=1e-8)

    def test_pooled_n_is_folds_times_cases(self, preds_dir):
        df, _ = delong_by_model(preds_dir, models=["m"], n_folds=2)
        assert set(df["n"]) == {16}  # 8 cases x 2 folds


class TestFriedmanNemenyi:
    def test_friedman_matches_scipy_reference(self):
        # 3 blocks x 4 arms, hand-built so arm ranking is deterministic:
        # D always best, A always worst -> non-trivial chi2.
        rows = []
        auc_by_block = [
            {"A_binary": 0.70, "B_ordinal": 0.75, "C_grade3": 0.80, "D_grade4": 0.90},
            {"A_binary": 0.60, "B_ordinal": 0.65, "C_grade3": 0.72, "D_grade4": 0.85},
            {"A_binary": 0.68, "B_ordinal": 0.70, "C_grade3": 0.77, "D_grade4": 0.88},
        ]
        for i, block in enumerate(auc_by_block):
            for arm, auc in block.items():
                rows.append({"model": "m", "fold": i, "arm": arm, "auc": auc})
        df = pd.DataFrame(rows)

        chi2, p, ranks_df = friedman_omnibus(df)

        expected_chi2, expected_p = stats.friedmanchisquare(
            *[[b[a] for b in auc_by_block] for a in ARMS]
        )
        assert chi2 == pytest.approx(expected_chi2)
        assert p == pytest.approx(expected_p)

        # D_grade4 always ranks 1st (best), A_binary always ranks 4th (worst).
        rank_map = dict(zip(ranks_df["arm"], ranks_df["mean_rank"]))
        assert rank_map["D_grade4"] == pytest.approx(1.0)
        assert rank_map["A_binary"] == pytest.approx(4.0)

    def test_nemenyi_boundary_matches_demsar_critical_value(self):
        # Hand-checked boundary: k=4 arms, N=30 blocks, alpha=0.05 Nemenyi
        # critical difference from Demsar (2006) table is q_alpha=2.569, i.e.
        # CD = 2.569 * sqrt(4*5/(6*30)). A rank gap exactly at CD must give
        # p ~ 0.05.
        k, n_blocks = 4, 30
        q_alpha_demsar = 2.569
        cd = q_alpha_demsar * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

        ranks_df = pd.DataFrame({
            "arm": ["A_binary", "B_ordinal", "C_grade3", "D_grade4"],
            "mean_rank": [1.0, 1.0 + cd / 3, 1.0 + 2 * cd / 3, 1.0 + cd],
        })
        nemenyi_df = nemenyi_posthoc(ranks_df, n_blocks=n_blocks)
        row = nemenyi_df[(nemenyi_df["arm_a"] == "A_binary") &
                          (nemenyi_df["arm_b"] == "D_grade4")]
        assert row["p_value"].iloc[0] == pytest.approx(0.05, abs=1e-3)

    def test_nemenyi_zero_gap_gives_p_one(self):
        ranks_df = pd.DataFrame({
            "arm": ["A_binary", "B_ordinal", "C_grade3", "D_grade4"],
            "mean_rank": [2.5, 2.5, 2.5, 2.5],
        })
        nemenyi_df = nemenyi_posthoc(ranks_df, n_blocks=10)
        assert np.allclose(nemenyi_df["p_value"], 1.0)
