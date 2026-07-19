"""Stage 05b: SHAP explanation for the arm-A radiomics branch (Track 1, XAI level 1).

The fusion ablation (stage_03b) found radiomics-only beats all 3 fusion variants
(DeLong non-significant / significantly worse) — so this is SHAP on the actual
headline model, not a fusion side-branch. Uses fold 0's train split to fit the
per-fold-selected feature set + XGBoost model (same recipe as stage_03b), then
computes TreeSHAP on the val split.

Outputs:
  artifacts/results/xai_track1/shap_beeswarm.png   — global feature importance
  artifacts/results/xai_track1/shap_waterfall_*.png — 2 local explanations (TP malignant, TP benign)
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict, fold: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    from src.stage_03b_fusion import _load_merged, _select_fold_features
    from src.fusion.early_fusion import train_early_fusion_xgboost

    out_dir = os.path.join(cfg["paths"]["results"], "xai_track1")
    sentinel = os.path.join(out_dir, "done.txt")
    from src.utils.io import cached
    if cached(sentinel) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {sentinel}")
        return
    os.makedirs(out_dir, exist_ok=True)

    merged, feat_cols = _load_merged(cfg)
    train_df = merged[merged["fold"] != fold].reset_index(drop=True)
    val_df = merged[merged["fold"] == fold].reset_index(drop=True)

    X_train_sel, X_val_sel, selected = _select_fold_features(train_df, val_df, feat_cols)
    logger.info("fold %d: %d radiomic features selected: %s", fold, len(selected), selected)

    clf = train_early_fusion_xgboost(X_train_sel, train_df["label"].values, cfg.get("xgboost", {}))

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_val_sel)

    # --- global beeswarm ---
    plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, X_val_sel, feature_names=selected, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("shap_beeswarm.png saved")

    # --- local waterfalls: one TP malignant, one TP benign (highest-confidence) ---
    y_val = val_df["label"].values
    prob = clf.predict_proba(X_val_sel)[:, 1]
    tp_malignant = np.where((y_val == 1) & (prob > 0.5))[0]
    tp_benign = np.where((y_val == 0) & (prob < 0.5))[0]

    picks = []
    if len(tp_malignant):
        picks.append(("malignant", tp_malignant[np.argmax(prob[tp_malignant])]))
    if len(tp_benign):
        picks.append(("benign", tp_benign[np.argmin(prob[tp_benign])]))

    base_value = explainer.expected_value
    for label_name, idx in picks:
        exp = shap.Explanation(
            values=shap_values[idx], base_values=base_value,
            data=X_val_sel[idx], feature_names=selected,
        )
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.tight_layout()
        fname = f"shap_waterfall_{label_name}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("%s saved (nodule idx=%d, prob=%.3f)", fname, idx, prob[idx])

    pd.DataFrame({"feature": selected, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False) \
        .to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)

    with open(sentinel, "w") as f:
        f.write("done")
    print(f"[DONE] {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.fold)


if __name__ == "__main__":
    main()
