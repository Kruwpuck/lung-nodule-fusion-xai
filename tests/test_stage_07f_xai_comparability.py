"""Tests for the Rev1 task 8 cross-model XAI comparability grid (stage_07f)."""
import torch
import torch.nn as nn

from src.stage_07f_xai_comparability import (
    BACKBONES,
    POINTING_ACC,
    _caption,
    _target_layer_spatial_size,
)


def test_backbone_rows_match_rev1_spec():
    # High-pointing pair, then the zero-pointing group -- exact set from task 8.
    assert set(BACKBONES) == {
        "densenet121", "convnext_tiny", "mobilenetv3_small", "vit_base", "googlenet",
    }
    assert POINTING_ACC["densenet121"] == POINTING_ACC["convnext_tiny"] == 0.7167
    assert POINTING_ACC["mobilenetv3_small"] == POINTING_ACC["vit_base"] == POINTING_ACC["googlenet"] == 0.0


def test_caption_states_method_and_spatial_size():
    caption = _caption({"densenet121": (8, 8), "vit_base": None})
    assert "layercam" in caption.lower()
    assert "8x8" in caption
    assert "0.7167" in caption
    assert "not evidence of a worse classifier" in caption


class _DummySpatialModel(nn.Module):
    """Deep-enough stack that _auto_target_layer can find an 8x8-ish map."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),  # 32 -> 16
            nn.Conv2d(4, 4, 3, stride=2, padding=1),  # 16 -> 8
            nn.Conv2d(4, 4, 3, stride=2, padding=1),  # 8 -> 4
        )

    def forward(self, x):
        return self.features(x)


def test_target_layer_spatial_size_finds_8x8_stage():
    model = _DummySpatialModel()
    dummy = torch.zeros(1, 3, 32, 32)
    size = _target_layer_spatial_size(model, backbone_internal="densenet121", dummy=dummy)
    assert size == (8, 8)
