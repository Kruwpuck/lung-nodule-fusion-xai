"""Tests for XAI utilities."""
import numpy as np
import pytest
from src.xai.shap_utils import get_top_shap_features, spatial_cross_check
from src.xai.gradcam_utils import cam_in_nodule_fraction


class TestCAMInNodule:
    def test_full_overlap(self):
        cam = np.ones((10, 10))
        mask = np.ones((10, 10), dtype=bool)
        assert cam_in_nodule_fraction(cam, mask) == 1.0

    def test_no_overlap(self):
        cam = np.zeros((10, 10))
        cam[0, 0] = 1.0  # only one pixel active
        mask = np.zeros((10, 10), dtype=bool)
        # high activation (>0.5) at (0,0), mask all zeros
        result = cam_in_nodule_fraction(cam, mask, threshold=0.5)
        assert result == 0.0

    def test_empty_cam_returns_zero(self):
        cam = np.zeros((10, 10))
        mask = np.ones((10, 10), dtype=bool)
        result = cam_in_nodule_fraction(cam, mask, threshold=0.5)
        assert result == 0.0


class TestTopSHAPFeatures:
    def test_returns_top_n(self):
        import pandas as pd
        shap_vals = np.random.randn(20, 30)
        names = [f"feat_{i}" for i in range(30)]
        df = get_top_shap_features(shap_vals, names, n_top=5)
        assert len(df) == 5

    def test_sorted_descending(self):
        shap_vals = np.random.randn(20, 10)
        names = [f"feat_{i}" for i in range(10)]
        df = get_top_shap_features(shap_vals, names, n_top=10)
        assert (df["mean_abs_shap"].diff().dropna() <= 0).all()


class TestSpatialCrossCheck:
    def _make_inputs(self, n=5, size=32):
        cam_maps = [np.random.rand(size, size) for _ in range(n)]
        masks = [np.ones((size, size), dtype=bool) for _ in range(n)]
        shap_vals = np.random.randn(n, 20)
        feature_names = [f"original_glcm_feat_{i}" for i in range(20)]
        return shap_vals, feature_names, cam_maps, masks

    def test_returns_dict_keys(self):
        shap_vals, names, cams, masks = self._make_inputs()
        result = spatial_cross_check(shap_vals, names, cams, masks)
        for k in ("top_features", "texture_dominated", "mean_cam_in_nodule", "spurious_flag"):
            assert k in result

    def test_texture_dominated_true_for_glcm_features(self):
        shap_vals, names, cams, masks = self._make_inputs()
        result = spatial_cross_check(shap_vals, names, cams, masks)
        assert result["texture_dominated"] is True
