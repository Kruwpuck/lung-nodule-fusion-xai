"""Tests for the Track 2 sweep runner's resumability contract (Rev1 task 7:
SGD-specific learning-rate sweep).

`_completed_run_ids` is exercised against a real runs.csv-shaped file so the
"skip only status==completed" contract is verified on disk, not just in
memory. The full `run()` loop (which imports torch via stage_03_train) is
exercised with `train_run` monkeypatched, so no GPU/data is needed.
"""
import csv

from src.stage_03c_sweep import _completed_run_ids


def _write_runs_csv(path, rows):
    fieldnames = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestCompletedRunIdsSkipsOnlyCompleted:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _completed_run_ids(str(tmp_path / "runs.csv")) == set()

    def test_only_completed_status_rows_are_returned(self, tmp_path):
        p = tmp_path / "runs.csv"
        _write_runs_csv(str(p), [
            {"run_id": "vgg16_binary_f0_sgd_wd0.0001_lr0.01", "status": "completed"},
            {"run_id": "vgg16_binary_f1_sgd_wd0.0001_lr0.01", "status": "failed"},
            {"run_id": "vgg16_binary_f2_sgd_wd0.0001_lr0.01", "status": "running"},
        ])
        done = _completed_run_ids(str(p))
        assert done == {"vgg16_binary_f0_sgd_wd0.0001_lr0.01"}

    def test_215_pre_existing_default_lr_rows_stay_completed_and_unaffected(self, tmp_path):
        """Simulates the remote machine's runs.csv: old rows use the pre-task-7
        run_id format (no _lr suffix). A relaunch must keep treating them as done."""
        p = tmp_path / "runs.csv"
        old_rows = [
            {"run_id": f"vgg16_binary_f{f}_sgd_wd0.0001", "status": "completed"}
            for f in range(5)
        ]
        _write_runs_csv(str(p), old_rows)
        done = _completed_run_ids(str(p))
        assert done == {r["run_id"] for r in old_rows}
        # None of the new lr-sweep run_ids alias an old one.
        new_ids = {f"vgg16_binary_f{f}_sgd_wd0.0001_lr0.01" for f in range(5)}
        assert done.isdisjoint(new_ids)


class TestSweepRunLoopSkipsCompletedSgdLrRuns:
    """Drives stage_03c_sweep.run() end to end (train_run monkeypatched) to
    confirm a relaunch with a partially-populated runs.csv only re-runs the
    SGD-lr-sweep combos that are not yet status=completed."""

    def _cfg(self):
        return {
            "track2_sweep": {
                "enabled": True,
                "optimizers": [],
                "weight_decays": {},
                "sgd_lr_sweep": [1e-3, 1e-2, 1e-1],
            },
            "tracks": {"track2": {"backbones": ["vgg16"]}},
            "train": {"lr": 1e-4, "weight_decay": 1e-4},
            "data": {"n_folds": 2},
            "paths": {"logs": None},  # set per-test
        }

    def test_relaunch_skips_already_completed_sgd_lr_runs(self, tmp_path, monkeypatch):
        import src.stage_03c_sweep as sweep_mod

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        runs_csv = logs_dir / "runs.csv"
        # fold 0 at lr=1e-3 already completed; everything else still pending.
        _write_runs_csv(str(runs_csv), [
            {"run_id": "vgg16_binary_f0_sgd_wd0.0001_lr0.001", "status": "completed"},
        ])

        cfg = self._cfg()
        cfg["paths"]["logs"] = str(logs_dir)

        called_run_ids = []

        def fake_train_run(cfg, model_name, fold, task, optimizer_name, weight_decay, lr=None):
            from src.stage_03_train import build_run_id
            cfg_lr = cfg["train"]["lr"]
            called_run_ids.append(build_run_id(model_name, task, fold, optimizer_name, weight_decay, lr, cfg_lr))

        monkeypatch.setattr("src.stage_03_train.run", fake_train_run)

        sweep_mod.run(cfg, task="binary")

        # 2 folds x 3 lrs = 6 combos total, 1 already completed -> 5 executed.
        assert len(called_run_ids) == 5
        assert "vgg16_binary_f0_sgd_wd0.0001_lr0.001" not in called_run_ids
        assert "vgg16_binary_f1_sgd_wd0.0001_lr0.001" in called_run_ids
        assert "vgg16_binary_f0_sgd_wd0.0001_lr0.01" in called_run_ids
        assert "vgg16_binary_f0_sgd_wd0.0001_lr0.1" in called_run_ids
