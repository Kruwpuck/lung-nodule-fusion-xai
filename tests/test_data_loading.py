"""Tests for data loading and labeling logic."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


class MockAnnotation:
    def __init__(self, malignancy: int):
        self.malignancy = malignancy


class TestMalignancyTargets:
    """Median==3 is kept as indeterminate (label -1), never dropped."""

    def _call(self, ratings):
        from src.data_loading.lidc_loader import _malignancy_targets
        return _malignancy_targets([MockAnnotation(r) for r in ratings])

    def test_malignant_all_high(self):
        t = self._call([4, 5, 4, 5])
        assert (t["median_rating"], t["label"], t["grade3"], t["grade4"]) == (4.5, 1, 2, 3)

    def test_benign_all_low(self):
        t = self._call([1, 2, 1, 2])
        assert (t["median_rating"], t["label"], t["grade3"], t["grade4"]) == (1.5, 0, 0, 1)

    def test_ambiguous_is_indeterminate(self):
        t = self._call([3, 3, 3, 3])
        assert (t["median_rating"], t["label"], t["grade3"], t["grade4"]) == (3.0, -1, 1, 2)

    def test_mixed_majority_malignant(self):
        t = self._call([4, 4, 2, 5])
        assert (t["median_rating"], t["label"], t["grade3"], t["grade4"]) == (4.0, 1, 2, 3)

    def test_mixed_majority_benign(self):
        t = self._call([1, 2, 2, 1])
        assert (t["median_rating"], t["label"], t["grade3"], t["grade4"]) == (1.5, 0, 0, 1)

    def test_median_exactly_3_is_indeterminate(self):
        t = self._call([2, 3, 4, 3])
        assert (t["median_rating"], t["label"], t["grade3"], t["grade4"]) == (3.0, -1, 1, 2)

    def test_annotation_spread_reported(self):
        t = self._call([4, 5, 4, 5])
        assert t["n_annotations"] == 4
        assert t["rating_std"] == pytest.approx(0.5)


class TestKFoldSplits:
    def _make_df(self, n_patients=20, n_nodules_per_patient=2):
        rows = []
        for i in range(n_patients):
            for j in range(n_nodules_per_patient):
                rows.append({
                    "patient_id": f"LIDC-{i:04d}",
                    "nodule_idx": j,
                    "label": i % 2,                  # alternating
                    "grade3": 0 if i % 2 == 0 else 2,  # stratification key
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

    def test_frozen_patients_keep_their_original_fold(self, tmp_path):
        from src.data_loading.lidc_loader import add_kfold_splits
        old = self._make_df(n_patients=20)
        old = add_kfold_splits(old, n_folds=5, seed=42)
        old_path = tmp_path / "labels.csv"
        old.to_csv(old_path, index=False)
        frozen = dict(old.drop_duplicates("patient_id")[["patient_id", "fold"]].values)

        # same 20 patients plus 20 new ones
        grown = self._make_df(n_patients=40)
        grown = add_kfold_splits(grown, n_folds=5, seed=7, freeze_from=str(old_path))

        for pid, group in grown.groupby("patient_id"):
            if pid in frozen:
                assert group["fold"].unique().tolist() == [frozen[pid]]

    def test_new_patients_get_assigned_alongside_frozen_ones(self, tmp_path):
        from src.data_loading.lidc_loader import add_kfold_splits
        old = self._make_df(n_patients=20)
        old = add_kfold_splits(old, n_folds=5, seed=42)
        old_path = tmp_path / "labels.csv"
        old.to_csv(old_path, index=False)

        grown = self._make_df(n_patients=40)
        grown = add_kfold_splits(grown, n_folds=5, seed=7, freeze_from=str(old_path))

        assert len(grown) == len(self._make_df(n_patients=40))
        assert (grown["fold"] >= 0).all()
        for pid, group in grown.groupby("patient_id"):
            assert len(group["fold"].unique()) == 1, f"Patient {pid} split across folds"
