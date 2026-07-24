"""Stage 03b: Track 1 fusion ablation (arm A binary only).

Compares 3 arms on the exact same fold split + nodule subset:
  1. CNN-only    — re-inference with the existing arm A checkpoint
                   (src.models.registry backbone, cfg["track1_fusion"]["backbone"])
  2. Radiomics-only — XGBoost on per-fold-selected radiomic features (mRMR -> LASSO,
                   fit on the training fold only; no ICC filter, no perturbed-mask
                   extraction available)
  3. Fusion      — intermediate (joint) fusion: CNN embedding + radiomic vector -> dense

All 3 arms share the identical nodule subset per fold: labels.csv joined with
radiomics.parquet on (patient_id, nodule_idx), restricted to arm A binary rows
(label != -1). Nodules whose (patient_id, nodule_idx) key is ambiguous across
multiple scans (radiomics.parquet has no scan_id) are dropped from all 3 arms.

Decision rule (fixed before results were inspected): fusion is reported as the
headline only if its DeLong p-value against the better of {CNN-only, radiomics-only}
is < 0.05. Otherwise the best single-modality arm is reported as headline and this
is stated plainly — a radiomics-only or CNN-only win is a valid finding, not a
fusion failure.
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_merged(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    rad = pd.read_parquet(cfg["paths"]["features"])

    key = ["patient_id", "nodule_idx"]
    dup_labels = df.duplicated(subset=key, keep=False)
    dup_rad = rad.duplicated(subset=key, keep=False)
    n_drop = int(dup_labels.sum())
    if n_drop:
        logger.warning(
            "Dropping %d nodules with an ambiguous (patient_id, nodule_idx) key "
            "(radiomics.parquet has no scan_id to disambiguate multi-scan patients)",
            n_drop,
        )
    df = df[~dup_labels]
    rad = rad[~dup_rad]

    feat_cols = [c for c in rad.columns if c not in ("patient_id", "nodule_idx", "label", "fold")]
    merged = df.merge(rad[key + feat_cols], on=key, how="inner")
    merged = merged[merged["label"] != -1].reset_index(drop=True)
    return merged, feat_cols


def _select_fold_features(train_df: pd.DataFrame, val_df: pd.DataFrame, feat_cols: list[str],
                           mrmr_n: int = 50) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from src.radiomics.feature_selection import mrmr_select, lasso_select

    X_train_raw = train_df[feat_cols].values
    y_train = train_df["label"].values

    mrmr_cols = mrmr_select(X_train_raw, y_train, feat_cols, n_select=mrmr_n)
    X_mrmr_train = train_df[mrmr_cols].values
    selected, scaler, lasso = lasso_select(X_mrmr_train, y_train, mrmr_cols)
    if not selected:
        logger.warning("LASSO zeroed out all features — falling back to top-10 mRMR features")
        selected = mrmr_cols[:10]

    idx = [mrmr_cols.index(c) for c in selected]
    X_train_sel = scaler.transform(train_df[mrmr_cols].values)[:, idx]
    X_val_sel = scaler.transform(val_df[mrmr_cols].values)[:, idx]
    return X_train_sel, X_val_sel, selected


def _cnn_only_preds(model_name: str, backbone_internal: str, cfg: dict, fold: int,
                     df_fold: pd.DataFrame, device) -> np.ndarray:
    """Re-run the existing arm A checkpoint on df_fold's own patches (fusion-eligible subset)."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.backbones import BackboneClassifier
    from src.training.dataset import NoduleDataset2_5D

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)

    ckpt = os.path.join(cfg["paths"]["checkpoints"], model_name, f"fold{fold}_best.pt")
    model = BackboneClassifier(backbone_internal, n_input_channels=n_slices, n_classes=2, pretrained=True).to(device)
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
    model.eval()

    ds = NoduleDataset2_5D(df_fold, patch_size=patch_xy, n_slices=n_slices, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs.append(torch.softmax(model(imgs), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def _train_fusion_fold(model_name: str, backbone_internal: str, cfg: dict, fold: int,
                        train_df: pd.DataFrame, val_df: pd.DataFrame,
                        X_train_sel: np.ndarray, X_val_sel: np.ndarray, device) -> np.ndarray:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.models.fusion_net import FusionNet
    from src.fusion.intermediate_fusion import RadiomicDataset, train_fusion_epoch, eval_fusion
    from src.training.dataset import NoduleDataset2_5D
    from src.training.trainer import EarlyStopping

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)
    epochs = cfg["train"].get("epochs", 50)
    patience = cfg["train"].get("early_stopping_patience", 10)
    lr = cfg["train"].get("lr", 1e-4)
    weight_decay = cfg["train"].get("weight_decay", 1e-4)

    img_train_ds = NoduleDataset2_5D(train_df, patch_size=patch_xy, n_slices=n_slices, augment=True)
    img_val_ds = NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False)
    train_ds = RadiomicDataset(img_train_ds, X_train_sel)
    val_ds = RadiomicDataset(img_val_ds, X_val_sel)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FusionNet(
        n_radiomic=X_train_sel.shape[1], backbone_name=backbone_internal,
        n_input_channels=n_slices, n_classes=2, pretrained=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopper = EarlyStopping(patience=patience, mode="max")
    amp_scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    ckpt_dir = os.path.join(cfg["paths"]["checkpoints"], f"{model_name}_fusion_intermediate")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_pt = os.path.join(ckpt_dir, f"fold{fold}_best.pt")

    from sklearn.metrics import roc_auc_score
    best_auc = 0.0
    for epoch in range(epochs):
        train_fusion_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        y_true, y_prob = eval_fusion(model, val_loader, device)
        auc = roc_auc_score(y_true, y_prob)
        scheduler.step()
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_pt)
        if early_stopper.step(auc):
            logger.info("[fusion fold %d] early stop at epoch %d (best AUC %.4f)", fold, epoch, best_auc)
            break

    model.load_state_dict(torch.load(best_pt, weights_only=True, map_location=device))
    _, y_prob = eval_fusion(model, val_loader, device)
    return y_prob


def run(cfg: dict) -> None:
    import torch
    from src.models.registry import _NAME_MAP
    from src.fusion.early_fusion import extract_cnn_embeddings, build_early_fusion_features, train_early_fusion_xgboost
    from src.fusion.late_fusion import average_fusion, stacking_fusion
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.statistical_tests import delong_test
    from src.models.backbones import BackboneClassifier
    from src.training.dataset import NoduleDataset2_5D
    from torch.utils.data import DataLoader

    results_dir = cfg["paths"]["results"]
    fusion_dir = os.path.join(results_dir, "fusion")
    os.makedirs(fusion_dir, exist_ok=True)
    out_csv = os.path.join(fusion_dir, "ablation_summary.csv")

    from src.utils.io import cached
    if cached(out_csv) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out_csv}")
        return

    if not cfg.get("track1_fusion", {}).get("enabled", False):
        logger.info("track1_fusion.enabled is False in config — skipping")
        return

    merged, feat_cols = _load_merged(cfg)
    n_folds = cfg["data"].get("n_folds", 5)
    model_name = cfg["track1_fusion"].get("backbone", "mobilenetv3_small")
    backbone_internal = _NAME_MAP.get(model_name, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    pooled = {"cnn": [], "radiomics": [], "fusion_intermediate": [], "fusion_early": [], "fusion_late": [], "y_true": []}

    for fold in range(n_folds):
        train_df = merged[merged["fold"] != fold].reset_index(drop=True)
        val_df = merged[merged["fold"] == fold].reset_index(drop=True)
        y_val = val_df["label"].values

        X_train_sel, X_val_sel, selected = _select_fold_features(train_df, val_df, feat_cols)
        logger.info("[fold %d] n_train=%d n_val=%d n_radiomic_selected=%d",
                    fold, len(train_df), len(val_df), len(selected))

        # --- Arm 1: CNN-only ---
        cnn_prob = _cnn_only_preds(model_name, backbone_internal, cfg, fold, val_df, device)
        m_cnn = compute_metrics(y_val, cnn_prob)

        # --- Arm 2: radiomics-only (XGBoost) ---
        xgb_params = cfg.get("xgboost", {})
        clf = train_early_fusion_xgboost(X_train_sel, train_df["label"].values, xgb_params)
        rad_prob = clf.predict_proba(X_val_sel)[:, 1]
        m_rad = compute_metrics(y_val, rad_prob)

        # --- Arm 3a: intermediate fusion (default, end-to-end) ---
        fusion_prob = _train_fusion_fold(model_name, backbone_internal, cfg, fold,
                                          train_df, val_df, X_train_sel, X_val_sel, device)
        m_fus = compute_metrics(y_val, fusion_prob)

        # --- Arm 3b: late fusion (cheap, reuses cnn_prob + rad_prob — no retrain) ---
        late_prob = average_fusion(cnn_prob, rad_prob, weight_cnn=0.5)
        m_late = compute_metrics(y_val, late_prob)

        # --- Arm 3c: early fusion (CNN embedding + radiomics -> XGBoost, cheap given trained CNN) ---
        n_slices = cfg["data"].get("n_slices", 3)
        patch_xy = cfg["data"].get("patch_xy", 64)
        batch_size = cfg["train"].get("batch_size", 16)
        ckpt = os.path.join(cfg["paths"]["checkpoints"], model_name, f"fold{fold}_best.pt")
        cnn_model = BackboneClassifier(backbone_internal, n_input_channels=n_slices, n_classes=2, pretrained=True).to(device)
        state = torch.load(ckpt, weights_only=True, map_location="cpu")
        cnn_model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
        train_loader = DataLoader(NoduleDataset2_5D(train_df, patch_size=patch_xy, n_slices=n_slices, augment=False),
                                   batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False),
                                 batch_size=batch_size, shuffle=False, num_workers=0)
        emb_train = extract_cnn_embeddings(cnn_model, train_loader, device)
        emb_val = extract_cnn_embeddings(cnn_model, val_loader, device)
        X_early_train = build_early_fusion_features(emb_train, X_train_sel)
        X_early_val = build_early_fusion_features(emb_val, X_val_sel)
        clf_early = train_early_fusion_xgboost(X_early_train, train_df["label"].values, xgb_params)
        early_prob = clf_early.predict_proba(X_early_val)[:, 1]
        m_early = compute_metrics(y_val, early_prob)

        for arm_name, m in [("cnn_only", m_cnn), ("radiomics_only", m_rad),
                             ("fusion_intermediate", m_fus), ("fusion_late", m_late),
                             ("fusion_early", m_early)]:
            rows.append({"arm": arm_name, "fold": fold, "n_val": len(val_df), **m})

        pooled["y_true"].append(y_val)
        pooled["cnn"].append(cnn_prob)
        pooled["radiomics"].append(rad_prob)
        pooled["fusion_intermediate"].append(fusion_prob)
        pooled["fusion_late"].append(late_prob)
        pooled["fusion_early"].append(early_prob)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    print(f"[DONE] {out_csv}  ({len(rows)} rows)")

    # --- pooled DeLong across arms + decision rule ---
    y_true_all = np.concatenate(pooled["y_true"])
    delong_rows = []
    arm_probs = {k: np.concatenate(v) for k, v in pooled.items() if k != "y_true"}
    from sklearn.metrics import roc_auc_score
    aucs = {k: roc_auc_score(y_true_all, p) for k, p in arm_probs.items()}

    best_single = max(("cnn", "radiomics"), key=lambda k: aucs[k])
    for fusion_arm in ("fusion_intermediate", "fusion_early", "fusion_late"):
        _, p_val, _ = delong_test(y_true_all, arm_probs[fusion_arm], arm_probs[best_single])
        delong_rows.append({
            "fusion_arm": fusion_arm, "fusion_auc": aucs[fusion_arm],
            "best_single_arm": best_single, "best_single_auc": aucs[best_single],
            "delong_p": p_val,
            "fusion_significantly_better": bool(p_val < 0.05 and aucs[fusion_arm] > aucs[best_single]),
        })
    delong_df = pd.DataFrame(delong_rows)
    delong_csv = os.path.join(fusion_dir, "delong_fusion.csv")
    delong_df.to_csv(delong_csv, index=False)
    print(f"[DONE] {delong_csv}")
    for r in delong_rows:
        headline = r["fusion_arm"] if r["fusion_significantly_better"] else r["best_single_arm"]
        logger.info("%s AUC=%.4f vs %s AUC=%.4f (p=%.4f) -> headline: %s",
                    r["fusion_arm"], r["fusion_auc"], r["best_single_arm"], r["best_single_auc"],
                    r["delong_p"], headline)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
