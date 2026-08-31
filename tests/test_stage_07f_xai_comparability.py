"""Tests for the Rev1 task 8 cross-model XAI comparability grid (stage_07f)."""
import torch
import torch.nn as nn

from src.stage_07f_xai_comparability import (
    BACKBONES,
    CAM_12_CSV,
    _caption,
    _target_layer_spatial_size,
)


def test_backbone_rows_match_rev1_spec():
    # High-pointing pair, then the zero-pointing group -- exact set from task 8.
    assert set(BACKBONES) == {
        "densenet121", "convnext_tiny", "mobilenetv3_small", "vit_base", "googlenet",
    }


def test_caption_states_method_and_spatial_size():
    caption = _caption({"densenet121": (8, 8), "vit_base": None})
    assert "layercam" in caption.lower()
    assert "8x8" in caption
    assert "vit_base=n/a" in caption
    assert "not evidence of a worse classifier" in caption
    # Assert the caption names the table it quotes rather than a specific figure. The
    # pointing values are read from that CSV at call time; pinning a digit here is what
    # let the caption drift away from the resolver in the first place.
    assert CAM_12_CSV in caption


class _DummySpatialModel(nn.Module):
    """Stack whose stages separate the canonical rule from the retired band rule."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),  # 32 -> 16
            nn.Conv2d(4, 4, 3, stride=2, padding=1),  # 16 -> 8
            nn.Conv2d(4, 4, 3, stride=2, padding=1),  # 8 -> 4
        )

    def forward(self, x):
        return self.features(x)


def test_target_layer_spatial_size_is_the_deepest_spatial_stage():
    # 4x4, not 8x8. Commit f81ba0d replaced the band rule -- deepest module whose feature
    # height lies in [7, 10], which would stop at the middle conv -- with the canonical
    # rule, the deepest module that still has spatial extent. The caption has to report
    # the site `compute_gradcam` actually draws from, so this expects the final conv.
    model = _DummySpatialModel()
    dummy = torch.zeros(1, 3, 32, 32)
    size = _target_layer_spatial_size(model, backbone_internal="densenet121", dummy=dummy)
    assert size == (4, 4)
