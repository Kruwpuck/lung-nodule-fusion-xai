"""Stage 02: labels.csv fold column -> folds.json (resumable)."""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    from src.utils.io import cached
    out = cfg["paths"]["splits"]
    if cached(out) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out}")
        return

    import json
    import pandas as pd

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    folds = {str(f): df[df["fold"] == f].index.tolist() for f in sorted(df["fold"].unique())}
    with open(out, "w") as fp:
        json.dump(folds, fp)
    print(f"[DONE] {out}  ({len(folds)} folds)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
