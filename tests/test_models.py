"""Tests for backbone and fusion model architectures."""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skip model tests")
from src.models.backbones import BackboneClassifier
from src.models.fusion_net import FusionNet


BACKBONES_2_5D = ["mobilenet_v3_large", "efficientnet_b0", "resnet50", "densenet121"]


class TestBackboneClassifier:
    @pytest.mark.parametrize("name", BACKBONES_2_5D[:2])  # fast: only 2 in CI
    def test_forward_2_5d(self, name):
        model = BackboneClassifier(name, n_input_channels=3, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 2)

    @pytest.mark.parametrize("name", BACKBONES_2_5D[:2])
    def test_embedding_shape(self, name):
        model = BackboneClassifier(name, n_input_channels=3, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        emb = model.get_embedding(x)
        assert emb.ndim == 2
        assert emb.shape[0] == 2


class TestFusionNet:
    def test_forward(self):
        model = FusionNet(
            n_radiomic=20,
            backbone_name="mobilenet_v3_large",
            n_input_channels=3,
            pretrained=False,
        )
        img = torch.randn(2, 3, 64, 64)
        rad = torch.randn(2, 20)
        out = model(img, rad)
        assert out.shape == (2, 2)

    def test_cnn_embedding_shape(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                          emb_dim=256, pretrained=False)
        img = torch.randn(4, 3, 64, 64)
        emb = model.get_cnn_embedding(img)
        assert emb.shape == (4, 256)

    def test_radiomic_embedding_shape(self):
        model = FusionNet(n_radiomic=15, rad_dim=64, backbone_name="mobilenet_v3_large",
                          pretrained=False)
        rad = torch.randn(4, 15)
        emb = model.get_radiomic_embedding(rad)
        assert emb.shape == (4, 64)
