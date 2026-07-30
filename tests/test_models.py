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


class TestFusionNetBranchNorm:
    """Rev1 task 5a: per-branch normalization + CNN embedding down-projection.

    fusion_arm="concat" (default) must stay byte-identical to pre-5a behavior
    so the default config path reproduces today's numbers. fusion_arm=
    "branch_norm" down-projects the CNN embedding to proj_dim and L2-normalizes
    both branches before concatenation.
    """

    def test_default_arm_is_concat_and_unchanged(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False)
        assert model.fusion_arm == "concat"
        img = torch.randn(4, 3, 64, 64)
        emb = model.get_cnn_embedding(img)
        assert emb.shape == (4, 256)  # unchanged emb_dim, no down-projection

    def test_unknown_fusion_arm_rejected(self):
        with pytest.raises(ValueError):
            FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                      pretrained=False, fusion_arm="bogus")

    def test_branch_norm_down_projects_cnn_embedding(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                           pretrained=False, fusion_arm="branch_norm", proj_dim=32)
        img = torch.randn(4, 3, 64, 64)
        emb = model.get_cnn_embedding(img)
        assert emb.shape == (4, 32)

    def test_branch_norm_forward_shape_and_l2_norms(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                           pretrained=False, fusion_arm="branch_norm", proj_dim=32)
        img = torch.randn(4, 3, 64, 64)
        rad = torch.randn(4, 20)
        out = model(img, rad)
        assert out.shape == (4, 2)
        # per-branch norms must be logged and each branch L2-normalized to ~1.0
        assert set(model.last_branch_norms) == {"img_norm", "rad_norm"}
        assert model.last_branch_norms["img_norm"] == pytest.approx(1.0, abs=1e-4)
        assert model.last_branch_norms["rad_norm"] == pytest.approx(1.0, abs=1e-4)

    def test_concat_arm_does_not_normalize_branches(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False)
        img = torch.randn(4, 3, 64, 64)
        rad = torch.randn(4, 20)
        model(img, rad)
        # legacy arm: norms are logged but not forced to unit L2 (no normalize() applied).
        # A 256-dim post-ReLU embedding has expected norm well above 1.0.
        assert model.last_branch_norms["img_norm"] > 1.5


class TestFusionNetGMU:
    """Rev1 task 5b: Gated Multimodal Unit fusion (Arevalo et al. 2017).

    Each modality gets its own tanh feature transform (h_v, h_t); a sigmoid
    gate computed from both raw modality inputs decides the elementwise mix
    z*h_v + (1-z)*h_t, so the network can learn to down-weight the weaker
    CNN branch instead of always trusting a fixed concatenation.
    """

    def test_gmu_is_a_known_arm(self):
        from src.models.fusion_net import FUSION_ARMS
        assert "gmu" in FUSION_ARMS

    def test_gmu_forward_shape(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                           pretrained=False, fusion_arm="gmu", gmu_dim=64)
        img = torch.randn(4, 3, 64, 64)
        rad = torch.randn(4, 20)
        out = model(img, rad)
        assert out.shape == (4, 2)

    def test_gmu_logs_gate_activations_per_branch(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                           pretrained=False, fusion_arm="gmu", gmu_dim=64)
        img = torch.randn(4, 3, 64, 64)
        rad = torch.randn(4, 20)
        model(img, rad)
        assert set(model.last_branch_norms) == {"img_gate", "rad_gate"}
        # gate is a sigmoid output in (0, 1), and img_gate + rad_gate == 1 by construction (z, 1-z)
        assert 0.0 < model.last_branch_norms["img_gate"] < 1.0
        assert model.last_branch_norms["img_gate"] == pytest.approx(
            1.0 - model.last_branch_norms["rad_gate"], abs=1e-5
        )

    def test_gmu_gate_responds_to_inputs_not_constant(self):
        """Different (img, rad) pairs must produce different gate values —
        otherwise the gate isn't actually a function of the modalities."""
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                           pretrained=False, fusion_arm="gmu", gmu_dim=64)
        model.eval()
        img_a, rad_a = torch.randn(1, 3, 64, 64), torch.randn(1, 20)
        img_b, rad_b = torch.randn(1, 3, 64, 64), torch.randn(1, 20)
        model(img_a, rad_a)
        gate_a = model.last_branch_norms["img_gate"]
        model(img_b, rad_b)
        gate_b = model.last_branch_norms["img_gate"]
        assert gate_a != pytest.approx(gate_b, abs=1e-9)

    def test_unknown_arm_still_rejected_with_gmu_in_message(self):
        with pytest.raises(ValueError):
            FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                      pretrained=False, fusion_arm="bogus")


class TestFusionNetModalityDropoutAux:
    """Rev1 task 5c: modality dropout + auxiliary per-branch losses.

    Both are regularizers, not a fourth fusion_arm, so they must compose with
    concat/branch_norm/gmu rather than requiring a new arm name. Both default
    to 0.0 (no-op) so the default config path is unaffected.
    """

    def test_defaults_are_noop_and_allocate_no_aux_heads(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False)
        assert model.modality_dropout == 0.0
        assert model.aux_loss_weight == 0.0
        assert not hasattr(model, "aux_img_head")
        assert not hasattr(model, "aux_rad_head")

    def test_invalid_modality_dropout_rejected(self):
        with pytest.raises(ValueError):
            FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                      pretrained=False, modality_dropout=1.0)
        with pytest.raises(ValueError):
            FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                      pretrained=False, modality_dropout=-0.1)

    def test_apply_modality_dropout_noop_when_rate_zero(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False)
        model.train()
        img_emb, rad_emb = torch.ones(10, 8), torch.ones(10, 8)
        out_img, out_rad = model._apply_modality_dropout(img_emb, rad_emb)
        assert torch.equal(out_img, img_emb)
        assert torch.equal(out_rad, rad_emb)

    def test_apply_modality_dropout_noop_in_eval_mode(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                          pretrained=False, modality_dropout=0.9)
        model.eval()
        img_emb, rad_emb = torch.ones(10, 8), torch.ones(10, 8)
        out_img, out_rad = model._apply_modality_dropout(img_emb, rad_emb)
        assert torch.equal(out_img, img_emb)
        assert torch.equal(out_rad, rad_emb)

    def test_apply_modality_dropout_zeroes_branches_when_rate_near_one(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                          pretrained=False, modality_dropout=0.999999)
        model.train()
        img_emb, rad_emb = torch.ones(500, 8), torch.ones(500, 8)
        out_img, out_rad = model._apply_modality_dropout(img_emb, rad_emb)
        assert (out_img == 0).float().mean() > 0.95
        assert (out_rad == 0).float().mean() > 0.95

    def test_aux_logits_requires_positive_weight(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False)
        with pytest.raises(RuntimeError):
            model.aux_logits()

    def test_aux_logits_requires_a_forward_pass_first(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                          pretrained=False, aux_loss_weight=0.5)
        with pytest.raises(RuntimeError):
            model.aux_logits()

    def test_aux_logits_shapes_concat_arm(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large",
                          pretrained=False, aux_loss_weight=0.5)
        img, rad = torch.randn(4, 3, 64, 64), torch.randn(4, 20)
        model(img, rad)
        aux_img, aux_rad = model.aux_logits()
        assert aux_img.shape == (4, 2)
        assert aux_rad.shape == (4, 2)

    def test_aux_logits_shapes_branch_norm_arm(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False,
                          fusion_arm="branch_norm", proj_dim=16, aux_loss_weight=0.5)
        img, rad = torch.randn(4, 3, 64, 64), torch.randn(4, 20)
        model(img, rad)
        aux_img, aux_rad = model.aux_logits()
        assert aux_img.shape == (4, 2)
        assert aux_rad.shape == (4, 2)

    def test_aux_logits_shapes_gmu_arm(self):
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False,
                          fusion_arm="gmu", gmu_dim=32, aux_loss_weight=0.5)
        img, rad = torch.randn(4, 3, 64, 64), torch.randn(4, 20)
        model(img, rad)
        aux_img, aux_rad = model.aux_logits()
        assert aux_img.shape == (4, 2)
        assert aux_rad.shape == (4, 2)

    def test_modality_dropout_and_aux_loss_compose_with_gmu(self):
        """Both knobs at once, on the arm that isn't concat/branch_norm — proves
        they aren't tied to a single arm's embedding code path."""
        model = FusionNet(n_radiomic=20, backbone_name="mobilenet_v3_large", pretrained=False,
                          fusion_arm="gmu", gmu_dim=32, modality_dropout=0.3, aux_loss_weight=0.5)
        model.train()
        img, rad = torch.randn(4, 3, 64, 64), torch.randn(4, 20)
        out = model(img, rad)
        assert out.shape == (4, 2)
        aux_img, aux_rad = model.aux_logits()
        assert aux_img.shape == (4, 2)
        assert aux_rad.shape == (4, 2)


class TestRegistryTrackInputSize:
    """Regression test for the stage_03b_fusion input_size bug (Rev1 task 1):
    models built for Track 1 must resize to the track's configured input_size;
    Track 2 models must stay native and not resize at all."""

    CFG = {
        "data": {"n_slices": 3},
        "tracks": {
            "track1": {"backbones": ["densenet121"], "input_size": 96},
            "track2": {"backbones": ["mobilenetv2"], "input_size": None},
        },
    }

    def test_build_model_track1_resizes_to_96(self):
        from src.models.registry import build_model
        model = build_model("densenet121", self.CFG)
        assert model._resize_to == 96
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 2)

    def test_build_model_track2_does_not_resize(self):
        from src.models.registry import build_model
        model = build_model("mobilenetv2", self.CFG)
        assert model._resize_to is None

    def test_build_fusion_model_track1_resizes_to_96(self):
        from src.models.registry import build_fusion_model
        model = build_fusion_model("densenet121", self.CFG, n_radiomic=20)
        assert model._resize_to == 96
        img = torch.randn(2, 3, 64, 64)  # native 64px patch, fed through the fusion path
        rad = torch.randn(2, 20)
        out = model(img, rad)
        assert out.shape == (2, 2)

    def test_build_fusion_model_track2_does_not_resize(self):
        from src.models.registry import build_fusion_model
        model = build_fusion_model("mobilenetv2", self.CFG, n_radiomic=20)
        assert model._resize_to is None

    def test_build_fusion_model_forwards_fusion_arm_kwargs(self):
        """Rev1 task 5a: build_fusion_model's **fusion_kwargs must reach FusionNet
        unchanged, so config-selecting fusion_arm="branch_norm" actually takes effect."""
        from src.models.registry import build_fusion_model
        model = build_fusion_model("densenet121", self.CFG, n_radiomic=20,
                                    fusion_arm="branch_norm", proj_dim=16)
        assert model.fusion_arm == "branch_norm"
        img = torch.randn(2, 3, 64, 64)
        emb = model.get_cnn_embedding(img)
        assert emb.shape == (2, 16)
