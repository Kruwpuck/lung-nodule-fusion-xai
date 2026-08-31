"""Tests for the geometry and ranking helpers behind stage 11a/11b.

Only the parts that are easy to get silently wrong are covered: a rotation that is never
undone, a crop that never excludes the emptied border, and a top-k/bottom-k selection that
overlaps. Each of those produces plausible-looking numbers rather than a crash, so an
assert is the only thing that catches them.
"""
import numpy as np
import pytest


class TestValidCrop:
    def test_three_degrees_on_a_64px_patch_leaves_60(self):
        from src.stage_11b_ssim_stability import _valid_crop
        assert _valid_crop(64, 3.0) == 60

    def test_zero_rotation_keeps_the_whole_patch(self):
        from src.stage_11b_ssim_stability import _valid_crop
        assert _valid_crop(64, 0.0) == 64

    def test_sign_of_the_angle_does_not_matter(self):
        from src.stage_11b_ssim_stability import _valid_crop
        assert _valid_crop(64, 3.0) == _valid_crop(64, -3.0)

    def test_larger_tilt_crops_harder(self):
        from src.stage_11b_ssim_stability import _valid_crop
        assert _valid_crop(64, 10.0) < _valid_crop(64, 3.0)


class TestCentreCrop:
    def test_takes_the_middle_not_a_corner(self):
        from src.stage_11b_ssim_stability import _centre_crop
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        out = _centre_crop(arr, 60)
        assert out.shape == (60, 60)
        assert out[0, 0] == arr[2, 2]


class TestRotationIsUndone:
    """The reason `_rotate_map(-theta)` exists at all.

    A map computed on a tilted input lives in the tilted frame. Comparing it directly
    against the untilted map measures the tilt, which would report every model as
    unstable. This test fails if the back-rotation is ever dropped.
    """

    def _blob(self):
        yy, xx = np.mgrid[0:64, 0:64]
        # Off-centre so a rotation actually moves it; a centred symmetric blob would be
        # nearly rotation-invariant and the test would pass even with the bug present.
        return np.exp(-(((yy - 26) ** 2 + (xx - 38) ** 2) / 40.0)).astype(np.float32)

    def test_back_rotation_recovers_far_more_similarity_than_leaving_it_tilted(self):
        from src.stage_11b_ssim_stability import _centre_crop, _rotate_map, _ssim

        base = self._blob()
        tilted = _rotate_map(base, 3.0)
        side = 60
        base_c = _centre_crop(base, side)
        undone = _ssim(base_c, _centre_crop(_rotate_map(tilted, -3.0), side))
        left_tilted = _ssim(base_c, _centre_crop(tilted, side))

        assert undone > left_tilted
        assert undone > 0.95, f"roundtrip rusak: SSIM {undone:.4f}"

    def test_ssim_of_a_map_with_itself_is_one(self):
        from src.stage_11b_ssim_stability import _ssim
        base = self._blob()
        assert _ssim(base, base) == pytest.approx(1.0)


class TestQuantileMask:
    def _cam(self):
        return np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)

    def test_selects_exactly_k_fraction(self):
        from src.stage_11a_faithfulness import _quantile_mask
        mask = _quantile_mask(self._cam(), 0.20, most_important=True)
        assert mask.sum() == round(0.20 * 64 * 64)

    def test_important_and_unimportant_never_overlap(self):
        from src.stage_11a_faithfulness import _quantile_mask
        cam = self._cam()
        top = _quantile_mask(cam, 0.20, most_important=True)
        bottom = _quantile_mask(cam, 0.20, most_important=False)
        assert not (top & bottom).any()

    def test_important_holds_the_high_values(self):
        from src.stage_11a_faithfulness import _quantile_mask
        cam = self._cam()
        top = _quantile_mask(cam, 0.20, most_important=True)
        bottom = _quantile_mask(cam, 0.20, most_important=False)
        assert cam[top].min() > cam[bottom].max()

    def test_a_flat_map_still_yields_k_pixels(self):
        """Rank-based, not value-based: a degenerate map must not return an empty mask,
        or PGI would silently become zero instead of reporting an uninformative map."""
        from src.stage_11a_faithfulness import _quantile_mask
        flat = np.full((64, 64), 0.5, dtype=np.float32)
        assert _quantile_mask(flat, 0.20, most_important=True).sum() == round(0.20 * 64 * 64)
