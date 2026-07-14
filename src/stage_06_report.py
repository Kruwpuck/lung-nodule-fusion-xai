"""Stage 06: generate all figures + tables from artifacts (no retraining needed)."""
import argparse
import glob
import logging
import os
import yaml

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # will fail at plot time with a clear error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pooled_preds(preds_dir: str, model_name: str, n_folds: int = 5):
    """Concatenate per-fold npz files. Valid because folds are disjoint."""
    ys, ps = [], []
    for fold in range(n_folds):
        fp = os.path.join(preds_dir, f"{model_name}_fold{fold}.npz")
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        ys.append(d["y_true"]); ps.append(d["y_prob"])
    if not ys:
        return None, None
    return np.concatenate(ys), np.concatenate(ps)


def _old_buggy_extract(volume: np.ndarray, n_slices: int = 3, patch_size: int = 64):
    """Replicate old (Y,X,Z) axis bug for before/after QC plot."""
    # Old code treated last axis (Z, ~2-20 slices) as Width → mostly black
    Z, H, W = volume.shape  # volume is already (Z,Y,X) after fix
    # Simulate old bug: treat dim[2] as W and crop in wrong plane
    cz = Z // 2; half = n_slices // 2; p = patch_size
    cy, cx = H // 2, W // 2
    y0, y1 = max(0, cy - p // 2), min(H, cy + p // 2)
    x0, x1 = max(0, cx - p // 2), min(W, cx + p // 2)
    slices = []
    for offset in range(-half, half + 1):
        z = max(0, min(Z - 1, cz + offset))
        sl = np.zeros((p, p), dtype=np.float32)
        crop = volume[z, y0:y1, x0:x1]
        sl[:crop.shape[0], :crop.shape[1]] = crop
        slices.append(sl)
    return np.stack(slices, axis=0)


def _correct_extract(volume: np.ndarray, n_slices: int = 3):
    Z, H, W = volume.shape
    cz = Z // 2; half = n_slices // 2
    slices = []
    for offset in range(-half, half + 1):
        z = max(0, min(Z - 1, cz + offset))
        slices.append(volume[z])
    return np.stack(slices, axis=0)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _find_qc_pair(broken_dir: str, fixed_dir: str):
    """Find a nodule patch present in both the pre-fix backup and the fresh run."""
    for patient_dir in sorted(glob.glob(os.path.join(broken_dir, "*"))):
        patient_id = os.path.basename(patient_dir)
        for nodule_dir in sorted(glob.glob(os.path.join(patient_dir, "nodule_*"))):
            nodule_id = os.path.basename(nodule_dir)
            broken_patch = os.path.join(nodule_dir, "patch.npy")
            fixed_patch = os.path.join(fixed_dir, patient_id, nodule_id, "patch.npy")
            if os.path.exists(broken_patch) and os.path.exists(fixed_patch):
                return broken_patch, fixed_patch
    return None, None


def _plot_patch_qc(labels_df: pd.DataFrame, out_dir: str, interim_dir: str) -> None:
    broken_dir = "artifacts/patches_broken"
    if not os.path.isdir(broken_dir):
        logger.warning("No artifacts/patches_broken found, falling back to simulated-bug QC panel")
        sample = labels_df.iloc[0]
        vol = np.load(sample["patch_path"]).astype(np.float32)
        old_patch = _old_buggy_extract(vol)
        new_patch = _correct_extract(vol)
    else:
        broken_path, fixed_path = _find_qc_pair(broken_dir, interim_dir)
        if broken_path is None:
            logger.warning("No matching nodule in both patches_broken and %s, skipping patch_qc.png", interim_dir)
            return
        broken_vol = np.load(broken_path).astype(np.float32)  # old (Y,X,Z)-shaped, variable size
        old_patch = _old_buggy_extract(broken_vol)
        fixed_vol = np.load(fixed_path).astype(np.float32)  # new (Z,Y,X), fixed size
        new_patch = _correct_extract(fixed_vol)

    mid_old = old_patch.shape[0] // 2
    mid_new = new_patch.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(old_patch[mid_old], cmap="gray"); axes[0].set_title("Before fix (axis-order bug)")
    axes[1].imshow(new_patch[mid_new], cmap="gray"); axes[1].set_title("After fix (physical window, centered)")
    for ax in axes: ax.axis("off")
    plt.suptitle("Patch QC — before vs after preprocessing fix", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "patch_qc.png"), dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("patch_qc.png saved")


def _plot_convergence(logs_dir: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for csv_path in sorted(os.listdir(logs_dir)):
        if not csv_path.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(logs_dir, csv_path))
        if "val_auc" not in df.columns:
            continue
        label = csv_path.replace(".csv", "")
        ax.plot(df["epoch"], df["val_auc"], alpha=0.6, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val AUC")
    ax.set_title("Convergence curves — all models × folds")
    ax.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=120)
    plt.close()
    logger.info("convergence.png saved")


def _plot_auc_boxplot(summary: pd.DataFrame, out_dir: str) -> None:
    models = summary["model"].unique()
    data = [summary[summary["model"] == m]["auc"].values for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=models, vert=True)
    ax.set_ylabel("AUC"); ax.set_title("AUC distribution per model (5-fold CV)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "auc_boxplot.png"), dpi=120)
    plt.close()
    logger.info("auc_boxplot.png saved")


def _plot_roc_curves(preds_dir: str, summary: pd.DataFrame, out_dir: str, n_folds: int = 5) -> None:
    from sklearn.metrics import roc_curve, auc as sk_auc
    mean_fpr = np.linspace(0, 1, 100)
    models = summary["model"].unique()

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in models:
        tprs = []
        for fold in range(n_folds):
            fp = os.path.join(preds_dir, f"{model_name}_fold{fold}.npz")
            if not os.path.exists(fp):
                continue
            d = np.load(fp)
            fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        if not tprs:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs, axis=0)
        mean_auc = sk_auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={mean_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.1)

    ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC curves (mean ± std, 5-fold)"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=120)
    plt.close()
    logger.info("roc_curves.png saved")


def _plot_calibration(preds_dir: str, summary: pd.DataFrame, out_dir: str, n_folds: int = 5) -> None:
    try:
        from src.evaluation.metrics import build_calibration_data
    except ImportError:
        from sklearn.calibration import calibration_curve
        def build_calibration_data(y_true, y_prob, n_bins=10):
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
            return mean_pred, frac_pos

    models = summary["model"].unique()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],"k--", label="Perfect")
    for model_name in models:
        y_true, y_prob = _pooled_preds(preds_dir, model_name, n_folds)
        if y_true is None:
            continue
        cal = build_calibration_data(y_true, y_prob)
        mp, fp = cal["mean_predicted"], cal["fraction_positive"]
        ax.plot(mp, fp, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration curves"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration.png"), dpi=120)
    plt.close()
    logger.info("calibration.png saved")


def _plot_confusion_matrices(summary: pd.DataFrame, out_dir: str) -> None:
    models = summary["model"].unique()
    n = len(models)
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = np.array(axes).flatten()

    needed = {"tp", "tn", "fp", "fn"}
    if not needed.issubset(summary.columns):
        logger.warning("summary.csv missing tp/tn/fp/fn columns — skip confusion matrices")
        plt.close()
        return

    for i, model_name in enumerate(models):
        sub = summary[summary["model"] == model_name]
        tp = sub["tp"].sum(); tn = sub["tn"].sum()
        fp = sub["fp"].sum(); fn = sub["fn"].sum()
        cm = np.array([[tn, fp], [fn, tp]])
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"])
        ax.set_yticklabels(["True Neg", "True Pos"])
        ax.set_title(model_name, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Confusion matrices (pooled 5 folds)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("confusion_matrices.png saved")


def _build_delong_matrix(preds_dir: str, summary: pd.DataFrame, out_dir: str, n_folds: int = 5) -> None:
    from src.evaluation.statistical_tests import delong_test
    models = list(summary["model"].unique())
    n = len(models)
    pvals = np.full((n, n), np.nan)

    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i == j:
                pvals[i, j] = 1.0
                continue
            ya, pa = _pooled_preds(preds_dir, ma, n_folds)
            yb, pb = _pooled_preds(preds_dir, mb, n_folds)
            if ya is None or yb is None:
                continue
            try:
                _, p, _ = delong_test(ya, pa, pb)
                pvals[i, j] = float(p)
            except Exception as e:
                logger.warning("DeLong %s vs %s: %s", ma, mb, e)

    df_p = pd.DataFrame(pvals, index=models, columns=models)
    df_p.to_csv(os.path.join(out_dir, "..", "delong_matrix.csv"))
    logger.info("delong_matrix.csv saved")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pvals, vmin=0, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("DeLong pairwise p-values (AUC comparison)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "delong_matrix.png"), dpi=120)
    plt.close()
    logger.info("delong_matrix.png saved")


def _build_efficiency_table(summary: pd.DataFrame, out_dir: str) -> None:
    from src.evaluation.efficiency import build_efficiency_table, plot_params_vs_auc, plot_flops_vs_auc

    agg = (
        summary.groupby("model")
        .agg(
            params_M=("params_M", "first"),
            gflops=("gflops", "first"),
            latency_ms=("latency_ms", "first"),
            auc=("auc", "mean"),
            auc_ci_low=("auc", lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),
            auc_ci_high=("auc", lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x))),
        )
        .reset_index()
    )
    agg["auc_per_M_params"] = agg["auc"] / agg["params_M"].replace(0, np.nan)

    table_csv = os.path.join(out_dir, "..", "efficiency_table.csv")
    agg.to_csv(table_csv, index=False)
    logger.info("efficiency_table.csv saved")

    try:
        from tabulate import tabulate
        md = tabulate(agg.round(4).to_dict("records"), headers="keys", tablefmt="pipe")
        with open(os.path.join(out_dir, "..", "efficiency_table.md"), "w") as f:
            f.write(md)
        logger.info("efficiency_table.md saved")
    except ImportError:
        logger.warning("tabulate not installed — skip .md table")

    plot_params_vs_auc(agg, out_png=os.path.join(out_dir, "params_vs_auc.png"))
    plot_flops_vs_auc(agg, out_png=os.path.join(out_dir, "flops_vs_auc.png"))


# ---------------------------------------------------------------------------
# Main stage
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    from src.utils.io import cached

    results_dir = cfg["paths"]["results"]
    out_dir = os.path.join(results_dir, "figures")
    sentinel = os.path.join(out_dir, "done.txt")

    if cached(sentinel) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {sentinel}")
        return

    summary_csv = os.path.join(results_dir, "summary.csv")
    preds_dir   = os.path.join(results_dir, "preds")
    logs_dir    = cfg["paths"]["logs"]
    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")

    if not os.path.exists(summary_csv):
        logger.error("summary.csv not found — run stage_04 first")
        return

    os.makedirs(out_dir, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    n_folds = cfg["data"].get("n_folds", 5)

    # 1. Patch QC (before/after)
    if os.path.exists(labels_path):
        try:
            labels_df = pd.read_csv(labels_path)
            _plot_patch_qc(labels_df, out_dir, cfg["paths"]["interim"])
        except Exception as e:
            logger.warning("patch_qc failed: %s", e)

    # 2. Convergence curves
    if os.path.exists(logs_dir):
        try:
            _plot_convergence(logs_dir, out_dir)
        except Exception as e:
            logger.warning("convergence plot failed: %s", e)

    # 3. AUC boxplot
    try:
        _plot_auc_boxplot(summary, out_dir)
    except Exception as e:
        logger.warning("auc_boxplot failed: %s", e)

    # 4. ROC curves
    if os.path.exists(preds_dir):
        try:
            _plot_roc_curves(preds_dir, summary, out_dir, n_folds)
        except Exception as e:
            logger.warning("roc_curves failed: %s", e)

        # 5. Calibration
        try:
            _plot_calibration(preds_dir, summary, out_dir, n_folds)
        except Exception as e:
            logger.warning("calibration failed: %s", e)

        # 6. DeLong matrix
        try:
            _build_delong_matrix(preds_dir, summary, out_dir, n_folds)
        except Exception as e:
            logger.warning("delong_matrix failed: %s", e)

    # 7. Confusion matrices
    try:
        _plot_confusion_matrices(summary, out_dir)
    except Exception as e:
        logger.warning("confusion_matrices failed: %s", e)

    # 8. Efficiency table + scatter plots
    try:
        _build_efficiency_table(summary, out_dir)
    except Exception as e:
        logger.warning("efficiency_table failed: %s", e)

    with open(sentinel, "w") as f:
        f.write("done")
    print(f"[DONE] {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
