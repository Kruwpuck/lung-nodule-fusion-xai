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

    force_rerun = cfg.get("force_rerun", False)
    df = load_and_split(
        lidc_path=cfg["paths"]["raw"],
        interim_path=cfg["paths"]["interim"],
        n_folds=cfg["data"]["n_folds"],
        force_rebuild=force_rerun,
        include_indeterminate=cfg["data"].get("include_indeterminate", True),
        skip_existing=cfg["data"].get("skip_existing", True),
        freeze_from=cfg["data"].get("freeze_from"),
    )
    df.to_csv(out, index=False)
    n_indet = int((df["label"] == -1).sum())
    print(f"[DONE] {out}  ({len(df)} nodules, {n_indet} indeterminate)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
