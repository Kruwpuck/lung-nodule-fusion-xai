"""Tests for data loading and labeling logic."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


class MockAnnotation:
    def __init__(self, malignancy: int):
        self.malignancy = malignancy


class TestMedianLabel:
    def _call(self, ratings, exclude=3):
        from src.data_loading.lidc_loader import _median_label
        anns = [MockAnnotation(r) for r in ratings]
        return _median_label(anns, exclude_score=exclude)

    def test_malignant_all_high(self):
        assert self._call([4, 5, 4, 5]) == 1

    def test_benign_all_low(self):
        assert self._call([1, 2, 1, 2]) == 0

    def test_ambiguous_returns_none(self):
        assert self._call([3, 3, 3, 3]) is None

    def test_mixed_majority_malignant(self):
        assert self._call([4, 4, 2, 5]) == 1

    def test_mixed_majority_benign(self):
        assert self._call([1, 2, 2, 1]) == 0

    def test_median_exactly_3_excluded(self):
        assert self._call([2, 3, 4, 3]) is None


class TestKFoldSplits:
    def _make_df(self, n_patients=20, n_nodules_per_patient=2):
        rows = []
        for i in range(n_patients):
            for j in range(n_nodules_per_patient):
                rows.append({
                    "patient_id": f"LIDC-{i:04d}",
                    "nodule_idx": j,
                    "label": i % 2,  # alternating
                })
        return pd.DataFrame(rows)

    def test_no_patient_leakage(self):
        from src.data_loading.lidc_loader import add_kfold_splits
        df = self._make_df()
        df = add_kfold_splits(df, n_folds=5, seed=42)

        for pid, group in df.groupby("patient_id"):
            folds = group["fold"].unique()
            assert len(folds) == 1, f"Patient {pid} split across folds"

    def test_all_folds_assigned(self):
        from src.data_loading.lidc_loader import add_kfold_splits
        df = self._make_df(n_patients=50)
        df = add_kfold_splits(df, n_folds=5, seed=42)
        assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}

    def test_no_negative_folds(self):
        from src.data_loading.lidc_loader import add_kfold_splits
        df = self._make_df()
        df = add_kfold_splits(df, n_folds=5, seed=42)
        assert (df["fold"] >= 0).all()
