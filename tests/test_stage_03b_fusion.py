"""Tests for the config-selectable fusion arm plumbing (Rev1 task 5a).

Only exercises the pure helpers `_fusion_arms_list` / `_arm_row_name` — no
torch/data involved, so these run without a GPU or checkpoints.
"""
from src.stage_03b_fusion import _fusion_arms_list, _arm_row_name, _regularizer_suffix


class TestFusionArmsList:
    def test_default_config_is_concat_only(self):
        """No `track1_fusion.fusion_arms` key -> today's numbers reproduce unchanged."""
        assert _fusion_arms_list({}) == ["concat"]
        assert _fusion_arms_list({"track1_fusion": {}}) == ["concat"]

    def test_explicit_config_can_add_branch_norm(self):
        cfg = {"track1_fusion": {"fusion_arms": ["concat", "branch_norm"]}}
        assert _fusion_arms_list(cfg) == ["concat", "branch_norm"]

    def test_explicit_config_can_add_gmu(self):
        """Rev1 task 5b: gmu goes through the same fusion_arms mechanism as 5a."""
        cfg = {"track1_fusion": {"fusion_arms": ["concat", "gmu"]}}
        assert _fusion_arms_list(cfg) == ["concat", "gmu"]


class TestArmRowName:
    def test_concat_keeps_legacy_row_name(self):
        assert _arm_row_name("concat") == "fusion_intermediate"

    def test_other_arms_get_suffixed_row_name(self):
        assert _arm_row_name("branch_norm") == "fusion_intermediate_branch_norm"

    def test_gmu_gets_suffixed_row_name(self):
        assert _arm_row_name("gmu") == "fusion_intermediate_gmu"


class TestRegularizerSuffix:
    """Rev1 task 5c: modality dropout / aux losses compose with any arm via a
    row-name/checkpoint-subdir suffix, reusing 5a/5b's naming mechanism rather
    than adding a 4th fusion_arm value."""

    def test_no_suffix_by_default(self):
        assert _regularizer_suffix({}) == ""
        assert _regularizer_suffix({"track1_fusion": {}}) == ""

    def test_moddrop_suffix(self):
        cfg = {"track1_fusion": {"modality_dropout_rate": 0.2}}
        assert _regularizer_suffix(cfg) == "_moddrop"

    def test_auxloss_suffix(self):
        cfg = {"track1_fusion": {"aux_loss_weight": 0.3}}
        assert _regularizer_suffix(cfg) == "_auxloss"

    def test_both_suffixes_compose(self):
        cfg = {"track1_fusion": {"modality_dropout_rate": 0.2, "aux_loss_weight": 0.3}}
        assert _regularizer_suffix(cfg) == "_moddrop_auxloss"


class TestArmRowNameWithRegularizers:
    def test_default_cfg_none_unchanged(self):
        assert _arm_row_name("concat") == "fusion_intermediate"
        assert _arm_row_name("branch_norm") == "fusion_intermediate_branch_norm"

    def test_regularizers_compose_with_any_arm(self):
        cfg = {"track1_fusion": {"modality_dropout_rate": 0.2}}
        assert _arm_row_name("concat", cfg) == "fusion_intermediate_moddrop"
        assert _arm_row_name("branch_norm", cfg) == "fusion_intermediate_branch_norm_moddrop"
        assert _arm_row_name("gmu", cfg) == "fusion_intermediate_gmu_moddrop"
