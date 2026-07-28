"""Single entrypoint: train every Track 1 backbone (all folds, default optimizer)
then run the full Track 2 hyperparameter sweep. Resumable — reruns skip whatever
stage_03_train / stage_03c_sweep already find completed (maybe_resume / runs.csv).
"""
import argparse
import logging

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict, task: str = "binary") -> None:
    from src.stage_03_train import run as train_run
    from src.stage_03c_sweep import run as sweep_run

    track1_backbones = cfg.get("tracks", {}).get("track1", {}).get("backbones", [])
    n_folds = cfg["data"].get("n_folds", 5)

    for model_name in track1_backbones:
        for fold in range(n_folds):
            logger.info("[TRACK1] %s fold%d", model_name, fold)
            train_run(cfg, model_name, fold, task)

    logger.info("[TRACK2] starting sweep")
    sweep_run(cfg, task)

    print("[DONE] all Track1 backbones + Track2 sweep")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.task)


if __name__ == "__main__":
    main()
