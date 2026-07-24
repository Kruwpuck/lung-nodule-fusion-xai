"""Stage 01: patches -> radiomics features parquet (resumable)."""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    from src.utils.io import cached
    out = cfg["paths"]["features"]
    incremental = cfg["radiomics"].get("incremental", True)
    force_rerun = cfg.get("force_rerun", False)
    if cached(out) and not force_rerun and not incremental:
        print(f"[SKIP] {out}")
        return

    import pandas as pd
    from src.radiomics.extraction import extract_dataset_features

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    features_df = extract_dataset_features(
        df,
        params_yaml=cfg["radiomics"]["params_yaml"],
        output_parquet=out,
        force_rebuild=force_rerun,
        incremental=incremental,
    )
    features_df.to_parquet(out, index=False)
    print(f"[DONE] {out}  ({len(features_df)} rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
