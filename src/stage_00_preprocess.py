"""Stage 00: DICOM -> patches + labels.csv (resumable)."""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    from src.utils.io import cached
    out = os.path.join(cfg["paths"]["interim"], "labels.csv")
    if cached(out) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out}")
        return

    from src.data_loading.lidc_loader import load_and_split
    os.makedirs(cfg["paths"]["interim"], exist_ok=True)

    df = load_and_split(
        lidc_path=cfg["paths"]["raw"],
        interim_path=cfg["paths"]["interim"],
        n_folds=cfg["data"]["n_folds"],
    )
    df.to_csv(out, index=False)
    print(f"[DONE] {out}  ({len(df)} nodules)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
