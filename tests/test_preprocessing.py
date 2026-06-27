"""Tests for CT preprocessing utilities."""
import numpy as np
import pytest
from src.preprocessing.ct_preprocessing import (
    clip_and_normalize_hu,
    extract_patch_3d,
    extract_2_5d_patch,
)


class TestHUNormalization:
    def test_output_range_zero_one(self):
        vol = np.array([-1000.0, 0.0, 400.0])
        out = clip_and_normalize_hu(vol)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_clipping(self):
        vol = np.array([-2000.0, 1000.0])
        out = clip_and_normalize_hu(vol)
        assert out.min() == 0.0 and out.max() == 1.0

    def test_output_shape_preserved(self):
        vol = np.random.randn(10, 10, 10).astype(np.float32)
        out = clip_and_normalize_hu(vol)
        assert out.shape == vol.shape


class TestPatchExtraction:
    def test_3d_patch_shape(self):
        vol = np.zeros((128, 128, 128))
        patch = extract_patch_3d(vol, (64, 64, 64), patch_size=(32, 32, 32))
        assert patch.shape == (32, 32, 32)

    def test_2_5d_patch_shape(self):
        vol = np.zeros((64, 128, 128))
        patch = extract_2_5d_patch(vol, 32, 64, 64, patch_xy=64, n_slices=3)
        assert patch.shape == (3, 64, 64)

    def test_boundary_centroid_no_error(self):
        vol = np.zeros((64, 128, 128))
        # centroid near edge
        patch = extract_patch_3d(vol, (0, 0, 0), patch_size=(32, 32, 32))
        assert patch.shape == (32, 32, 32)
