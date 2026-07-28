"""Stage 03c: Track 2 hyperparameter sweep runner (3 optimizers x 3 weight decays
x Track 2 backbones x fold), resumable via runs.csv status like other stages'
`cached()` pattern.

Skips (model, fold, optimizer, weight_decay) combos already marked
status=completed in artifacts/logs/runs.csv, so a crashed/interrupted sweep can
be re-launched with the same command and only pick up remaining runs.
"""
import argparse
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _completed_run_ids(runs_csv_path: str) -> set:
    if not os.path.exists(runs_csv_path):
        return set()
    import pandas as pd
    df = pd.read_csv(runs_csv_path)
    if df.empty or "run_id" not in df.columns or "status" not in df.columns:
        return set()
    return set(df.loc[df["status"] == "completed", "run_id"])


def run(cfg: dict, task: str = "binary") -> None:
    from src.stage_03_train import run as train_run

    sweep_cfg = cfg.get("track2_sweep", {})
    if not sweep_cfg.get("enabled", False):
        print("[SKIP] track2_sweep.enabled is false")
        return

    backbones = cfg.get("tracks", {}).get("track2", {}).get("backbones", [])
    optimizers = sweep_cfg.get("optimizers", [])
    weight_decays = sweep_cfg.get("weight_decays", {})
    n_folds = cfg["data"].get("n_folds", 5)
    runs_csv_path = os.path.join(cfg["paths"]["logs"], "runs.csv")

    total = 0
    skipped = 0
    for model_name in backbones:
        for optimizer_name in optimizers:
            for wd in weight_decays.get(optimizer_name, []):
                for fold in range(n_folds):
                    total += 1
                    done = _completed_run_ids(runs_csv_path)
                    run_id = f"{model_name}_{task}_f{fold}_{optimizer_name}_wd{wd:g}"
                    if run_id in done:
                        skipped += 1
                        logger.info("[SKIP] %s already completed", run_id)
                        continue
                    logger.info("[RUN] %s", run_id)
                    try:
                        train_run(cfg, model_name, fold, task, optimizer_name, wd)
                    except Exception:
                        logger.exception("[RUN FAILED] %s -- continuing with next combo", run_id)

    print(f"[DONE] sweep: {total} combos, {skipped} skipped (already completed)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.task)


if __name__ == "__main__":
    main()
