"""Tests for run_id / hparam-resolution helpers used by the Track 2 sweep
(Rev1 task 7: SGD-specific learning-rate sweep).

Only exercises the pure helpers `build_run_id` / `_resolve_hparams` -- no
torch/data involved, so these run without a GPU or checkpoints.
"""
from src.stage_03_train import build_run_id, _resolve_hparams


class TestResolveHparams:
    def test_defaults_from_cfg_when_none_passed(self):
        cfg = {"train": {"lr": 1e-4, "weight_decay": 1e-4}}
        wd, lr, is_override = _resolve_hparams(cfg, None, None)
        assert wd == 1e-4
        assert lr == 1e-4
        assert is_override is False

    def test_explicit_lr_equal_to_cfg_is_not_an_override(self):
        cfg = {"train": {"lr": 1e-4, "weight_decay": 1e-4}}
        _, lr, is_override = _resolve_hparams(cfg, None, 1e-4)
        assert lr == 1e-4
        assert is_override is False

    def test_explicit_lr_different_from_cfg_is_an_override(self):
        cfg = {"train": {"lr": 1e-4, "weight_decay": 1e-4}}
        _, lr, is_override = _resolve_hparams(cfg, None, 1e-2)
        assert lr == 1e-2
        assert is_override is True

    def test_explicit_weight_decay_passthrough(self):
        cfg = {"train": {"lr": 1e-4, "weight_decay": 1e-4}}
        wd, _, _ = _resolve_hparams(cfg, 1e-3, None)
        assert wd == 1e-3


class TestBuildRunId:
    """Rev1 task 7: adding the lr sweep must not change any of the 215
    already-completed run_ids, and each new SGD-lr-sweep run must get its own
    id so a relaunch of stage_03c_sweep never restarts a finished run."""

    def test_default_lr_run_id_unchanged(self):
        """This is the format the 215 existing runs.csv rows were written under."""
        run_id = build_run_id("vgg16", "binary", 2, "adamw", 1e-4, 1e-4, cfg_lr=1e-4)
        assert run_id == "vgg16_binary_f2_adamw_wd0.0001"

    def test_existing_sgd_wd_sweep_run_id_unchanged(self):
        run_id = build_run_id("vgg16", "binary", 2, "sgd", 1e-3, 1e-4, cfg_lr=1e-4)
        assert run_id == "vgg16_binary_f2_sgd_wd0.001"
        assert "_lr" not in run_id

    def test_sgd_lr_sweep_run_id_gets_lr_suffix(self):
        run_id = build_run_id("vgg16", "binary", 2, "sgd", 1e-4, 1e-2, cfg_lr=1e-4)
        assert run_id == "vgg16_binary_f2_sgd_wd0.0001_lr0.01"

    def test_three_sgd_lr_sweep_values_are_pairwise_distinct(self):
        ids = {
            build_run_id("vgg16", "binary", 0, "sgd", 1e-4, lr, cfg_lr=1e-4)
            for lr in (1e-3, 1e-2, 1e-1)
        }
        assert len(ids) == 3

    def test_sgd_lr_sweep_run_id_distinct_from_default_lr_sgd_run(self):
        default_id = build_run_id("vgg16", "binary", 0, "sgd", 1e-4, 1e-4, cfg_lr=1e-4)
        swept_id = build_run_id("vgg16", "binary", 0, "sgd", 1e-4, 1e-3, cfg_lr=1e-4)
        assert default_id != swept_id

    def test_sgd_lr_sweep_run_id_distinct_across_folds_and_models(self):
        a = build_run_id("vgg16", "binary", 0, "sgd", 1e-4, 1e-2, cfg_lr=1e-4)
        b = build_run_id("vgg16", "binary", 1, "sgd", 1e-4, 1e-2, cfg_lr=1e-4)
        c = build_run_id("resnet50", "binary", 0, "sgd", 1e-4, 1e-2, cfg_lr=1e-4)
        assert len({a, b, c}) == 3
