"""Stage 04b: Track 2 hyperparameter-stability analysis from runs.csv.

Laporan Rev1 §3.1 asks for AUC variance, coefficient of variation and a 95% CI
per (model, optimizer, weight_decay) cell across folds, plus the marginal views
per optimizer and per model. Everything is derived from the run-level table
written by stage_03_train, so this stage needs no checkpoints and no GPU — it
can be run while the sweep is still going to inspect partial results.

Outputs (under cfg["paths"]["results"]):
    track2_stability.csv          one row per (model, optimizer, weight_decay)
    track2_stability_by_opt.csv   marginal over weight decay + fold
    track2_stability_by_model.csv marginal over optimizer + weight decay + fold
    figures/track2_auc_by_optimizer.png
    figures/track2_auc_heatmap.png
"""
import argparse
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CELL_KEYS = ["model", "optimizer", "weight_decay"]


def _agg(df, keys: list[str], score_col: str = "best_score"):
    """mean/std/CV/CI95 of score_col over the rows in each group."""
    import numpy as np
    import pandas as pd

    rows = []
    for key_vals, g in df.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        scores = g[score_col].dropna().to_numpy(dtype=float)
        n = len(scores)
        if n == 0:
            continue
        mean = float(scores.mean())
        # ddof=1 is the sample std the report asks for; undefined for a single run
        std = float(scores.std(ddof=1)) if n > 1 else float("nan")
        # 95% CI of the mean via the normal approximation (n=5 folds, matches
        # how the per-fold CV results are reported elsewhere in the pipeline)
        halfwidth = 1.96 * std / np.sqrt(n) if n > 1 else float("nan")
        rows.append({
            **dict(zip(keys, key_vals)),
            "n_runs": n,
            "auc_mean": round(mean, 6),
            "auc_std": round(std, 6) if n > 1 else None,
            "auc_var": round(std ** 2, 8) if n > 1 else None,
            "auc_cv": round(std / mean, 6) if n > 1 and mean else None,
            "ci95_low": round(mean - halfwidth, 6) if n > 1 else None,
            "ci95_high": round(mean + halfwidth, 6) if n > 1 else None,
            "auc_min": round(float(scores.min()), 6),
            "auc_max": round(float(scores.max()), 6),
        })
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _plot_by_optimizer(df, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    optimizers = sorted(df["optimizer"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        data = [sub.loc[sub["optimizer"] == o, "best_score"].dropna() for o in optimizers]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(optimizers) + 1), optimizers)
        ax.set_title(model)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("val AUC (best epoch)")
    fig.suptitle("Track 2 — AUC spread per optimizer (all weight decays x folds)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_heatmap(cells, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = cells.pivot_table(
        index="model", columns=["optimizer", "weight_decay"], values="auc_mean"
    )
    fig, ax = plt.subplots(figsize=(1.1 * pivot.shape[1] + 3, 0.7 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xticks(
        range(len(pivot.columns)),
        [f"{o}\nwd={w:g}" for o, w in pivot.columns],
        fontsize=8,
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy(dtype=float)[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean val AUC")
    ax.set_title("Track 2 — mean AUC per (model, optimizer, weight decay)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _variance_and_anova(df, results_dir: str) -> None:
    """Task 6: replace CV-only stability claims with real tests.

    - Brown-Forsythe (primary, median-centered) + Levene pairwise across
      backbones on AUC, Holm-corrected across the pairwise family.
    - Factorial ANOVA over backbone x optimizer x weight_decay x fold,
      reporting eta-squared per factor so "optimizer >> weight_decay" is a
      variance-explained claim, not a raw-delta claim.
    """
    from src.evaluation.statistical_tests import factorial_anova, pairwise_variance_tests

    bf = pairwise_variance_tests(df, "model", "best_score", test="brown-forsythe")
    lv = pairwise_variance_tests(df, "model", "best_score", test="levene")
    bf.to_csv(os.path.join(results_dir, "track2_variance_brown_forsythe.csv"), index=False)
    lv.to_csv(os.path.join(results_dir, "track2_variance_levene.csv"), index=False)

    anova = factorial_anova(df, ["model", "optimizer", "weight_decay", "fold"], "best_score")
    anova.to_csv(os.path.join(results_dir, "track2_anova_eta_sq.csv"), index=False)

    logger.info("wrote track2_variance_brown_forsythe.csv, track2_variance_levene.csv, "
                "track2_anova_eta_sq.csv to %s", results_dir)

    if not bf.empty:
        n_sig = int(bf["significant_holm_05"].sum())
        print(f"[STABILITY] Brown-Forsythe pairwise (Holm-corrected): "
              f"{n_sig}/{len(bf)} pairs significant at p<0.05")
        if n_sig == 0:
            print("[STABILITY] No pairwise variance difference survives Holm correction "
                  "-> drop any 'most stable model' claim; only descriptive CV is supported.")

    if not anova.empty:
        eta = anova.set_index("term")["eta_sq"]
        print("[STABILITY] ANOVA eta-squared per term:")
        for term, val in eta.items():
            print(f"    {term}: {val:.4f}")


def run(cfg: dict, task: str = "binary") -> None:
    from src.utils.log_io import load_runs

    logs_dir = cfg["paths"]["logs"]
    results_dir = cfg["paths"]["results"]
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    runs = load_runs(logs_dir)
    if runs.empty:
        print(f"[SKIP] no runs.csv in {logs_dir}")
        return

    track2_backbones = cfg.get("tracks", {}).get("track2", {}).get("backbones", [])
    df = runs[
        (runs["status"] == "completed")
        & (runs["task"] == task)
        & (runs["model"].isin(track2_backbones))
    ].copy()
    if df.empty:
        print(f"[SKIP] no completed track2 runs for task={task}")
        return

    n_failed = int((runs["status"] == "failed").sum())
    logger.info("track2 completed runs: %d (failed elsewhere: %d)", len(df), n_failed)

    cells = _agg(df, _CELL_KEYS)
    by_opt = _agg(df, ["model", "optimizer"])
    by_model = _agg(df, ["model"])

    cells.to_csv(os.path.join(results_dir, "track2_stability.csv"), index=False)
    by_opt.to_csv(os.path.join(results_dir, "track2_stability_by_opt.csv"), index=False)
    by_model.to_csv(os.path.join(results_dir, "track2_stability_by_model.csv"), index=False)

    _plot_by_optimizer(df, os.path.join(fig_dir, "track2_auc_by_optimizer.png"))
    _plot_heatmap(cells, os.path.join(fig_dir, "track2_auc_heatmap.png"))
    _variance_and_anova(df, results_dir)

    best = cells.loc[cells["auc_mean"].idxmax()]
    print(f"[DONE] {len(df)} runs -> {len(cells)} cells")
    print(f"       best cell: {best['model']} / {best['optimizer']} / wd={best['weight_decay']:g} "
          f"AUC={best['auc_mean']:.4f} +/- {best['auc_std']}")
    print(f"       wrote track2_stability*.csv + figures/ to {results_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.task)


if __name__ == "__main__":
    main()
