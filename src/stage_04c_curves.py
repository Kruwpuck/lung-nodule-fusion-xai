"""Stage 04c: training-curve figures from the per-epoch CSV logs.

Reads every per-epoch log through src.utils.log_io.load_epoch_logs (which
normalizes the legacy `{model}_fold{N}.csv` files alongside the new
`epochs/{run_id}.csv` ones), then draws the curves the report needs:

    figures/curves_track1.png   loss + val AUC per Track 1 backbone (mean over folds)
    figures/curves_track2.png   same for Track 2 backbones
    figures/curves_overfit.png  train_loss vs val_loss, new-format runs only
                                (legacy logs never recorded val_loss)

Curves are averaged over folds with a +/-1 std band, since a 5-fold overlay of
raw per-fold lines is unreadable at 7 backbones.
"""
import argparse
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mean_band(ax, sub, ycol: str, label: str) -> bool:
    """Plot mean-over-folds of ycol vs epoch with a +/-1 std band. False if no data."""
    import numpy as np

    s = sub.dropna(subset=[ycol])
    if s.empty:
        return False
    g = s.groupby("epoch")[ycol]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    ax.plot(mean.index, mean.to_numpy(), label=label, linewidth=1.4)
    ax.fill_between(mean.index, (mean - std).to_numpy(), (mean + std).to_numpy(), alpha=0.15)
    return True


def _plot_track(df, backbones: list[str], title: str, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["model"].isin(backbones)]
    if sub.empty:
        logger.warning("no epoch logs for %s — skipping %s", backbones, out_png)
        return

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in sorted(sub["model"].unique()):
        m = sub[sub["model"] == model]
        _mean_band(ax_loss, m, "train_loss", model)
        _mean_band(ax_auc, m, "val_score", model)

    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("train loss"); ax_loss.grid(alpha=0.3)
    ax_auc.set_xlabel("epoch"); ax_auc.set_ylabel("val AUC"); ax_auc.grid(alpha=0.3)
    ax_auc.legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_png)


def _plot_overfit(df, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df.dropna(subset=["val_loss"])
    if sub.empty:
        logger.warning("no run recorded val_loss — skipping %s", out_png)
        return

    models = sorted(sub["model"].unique())
    ncol = min(4, len(models))
    nrow = (len(models) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, model in zip(axes.flat, models):
        m = sub[sub["model"] == model]
        _mean_band(ax, m, "train_loss", "train")
        _mean_band(ax, m, "val_loss", "val")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    axes[0][0].set_ylabel("loss")
    fig.suptitle("Train vs validation loss (mean over folds)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_png)


def run(cfg: dict, task: str = "binary") -> None:
    from src.utils.log_io import load_epoch_logs

    logs_dir = cfg["paths"]["logs"]
    fig_dir = os.path.join(cfg["paths"]["results"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = load_epoch_logs(logs_dir)
    if df.empty:
        print(f"[SKIP] no epoch logs in {logs_dir}")
        return
    df = df[df["task"] == task]
    if df.empty:
        print(f"[SKIP] no epoch logs for task={task}")
        return

    tracks = cfg.get("tracks", {})
    track1 = tracks.get("track1", {}).get("backbones", [])
    track2 = tracks.get("track2", {}).get("backbones", [])

    _plot_track(df, track1, "Track 1 backbones — training curves (mean +/- 1 std over folds)",
                os.path.join(fig_dir, "curves_track1.png"))
    _plot_track(df, track2, "Track 2 backbones — training curves (mean +/- 1 std over folds)",
                os.path.join(fig_dir, "curves_track2.png"))
    _plot_overfit(df, os.path.join(fig_dir, "curves_overfit.png"))

    print(f"[DONE] curves from {len(df)} epoch rows / {df['run_id'].nunique()} runs -> {fig_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.task)


if __name__ == "__main__":
    main()
