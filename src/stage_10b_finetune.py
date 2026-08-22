"""Stage 10b (run03 F-1): two-stage fine-tuning under nested cross-validation.

One cell = one (backbone, fold, unfreeze) combination. Stage 1 trains the head
with the backbone frozen; stage 2 reopens the top `unfreeze` percent of the
backbone's layer groups with a discriminative learning rate. BatchNorm stays in
eval mode throughout, re-applied inside the epoch loop (src/training/finetune.py).

Two things this stage does differently from `stage_03_train`, both deliberate:

1. **Nested CV.** `stage_03_train` selects `fold{f}_best.pt` by AUC on the very
   fold it then reports (:184, :268, :294), so every published `cnn_only` number
   carries a checkpoint-selection advantage. run02 measured it at 0.0071 to
   0.0240 AUC (artifacts/results/run02/t0_checkpoint_sensitivity.csv) -- the same
   size as the effect this run is looking for. Here the epoch is chosen on a
   patient-level inner split carved from the outer training fold, and the outer
   fold is scored exactly once, at the end. The split parameters match the fusion
   branch (`stage_03b_fusion.py:187`) so the two branches stay comparable.

2. **Versioned outputs.** Checkpoints go to `checkpoints/run03/{backbone}_uf{pct}/`
   and predictions to `results/run03/preds/`. Nothing under `checkpoints/{backbone}/`
   or `results/preds/` is touched. This is the direct consequence of the
   densenet121 incident: predictions were overwritten in place by a retrain, the
   downstream metrics were never re-derived, and one manuscript paragraph ended up
   describing a model that no longer existed.

Both AUCs are recorded per cell. `outer_auc_merged` is scored on the
labels-intersect-radiomics subset, which is what `ablation_summary.csv` and
`radiomics_only` use; `outer_auc_full` is scored on every binary row, which is
what `summary_binary.csv` uses. Writing both means no later comparison can
quietly mix two different denominators.

Usage:
    python -m src.stage_10b_finetune --backbone convnext_tiny --fold 0 --unfreeze 10

handoff/PREREG_run03.md fixes the four unfreeze cells {0, 10, 20, 100}, the epoch
caps and the decision rules. Do not add a fifth.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import subprocess
import time

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_ID = "2026-08-22-run03"
OUT_DIR = os.path.join("artifacts", "results", "run03")
SUMMARY_CSV = os.path.join(OUT_DIR, "finetune_cnn_only.csv")

# Locked by handoff/GOAL4.md and handoff/PREREG_run03.md. Do not extend.
BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
UNFREEZE_PCT = [0, 10, 20, 100]

# Pre-registered in handoff/PREREG_run03.md section 3.
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 50
PATIENCE = 10
LR_DECAY_FACTOR = 2.6
INNER_VAL_FRACTION = 0.15


def _commit_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _iso_mtime(path: str) -> str:
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")


def _split_indices(train_df, fold: int):
    """Patient-level inner split, identical to the fusion branch's.

    `stage_03b_fusion.py:187` uses GroupShuffleSplit(n_splits=1, test_size=0.15,
    random_state=fold). Matching it exactly means a CNN cell and a fusion cell on
    the same outer fold hold out the same kind of split, so comparing the two
    branches is not confounded by a protocol difference.
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRACTION, random_state=fold)
    return next(gss.split(train_df, groups=train_df["patient_id"]))


def _score(model, df, cfg, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Score `model` on `df`. Returns (auc, y_true, y_prob) in df row order."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from src.training.dataset import NoduleDataset2_5D

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)

    ds = NoduleDataset2_5D(df, patch_size=patch_xy, n_slices=n_slices, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            logits = model(imgs.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            trues.append(targets.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)
    return float(roc_auc_score(y_true, y_prob)), y_true, y_prob


def _train_stage(model, loader, inner_val_df, cfg, device, param_groups, epochs,
                  best_pt, best_auc, stage_name, csv_log, row_id):
    """Run one stage. Returns (epochs_ran, best_auc, improved_in_this_stage).

    The best epoch is tracked across BOTH stages against a single `best_auc`, so
    a 0%-unfreeze cell whose head-only stage already peaked keeps those weights
    instead of being forced to accept a worse stage-2 epoch.
    """
    import torch
    import torch.nn as nn
    from src.training.finetune import apply_bn_eval
    from src.training.trainer import EarlyStopping, save_ckpt

    optimizer = torch.optim.AdamW(param_groups,
                                   weight_decay=cfg["train"].get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    stopper = EarlyStopping(patience=PATIENCE, mode="max")
    stopper.best = best_auc
    amp_scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    improved = False
    epochs_ran = 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        # Must be inside the loop: model.train() above just put every BatchNorm
        # back into training mode. Calling this once before the loop would be a
        # silent no-op from epoch 1 onward.
        n_bn = apply_bn_eval(model)

        total_loss = 0.0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            if amp_scaler:
                with torch.amp.autocast(device.type):
                    loss = criterion(model(imgs), targets)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss = criterion(model(imgs), targets)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
        train_loss = total_loss / len(loader.dataset)

        inner_auc, _, _ = _score(model, inner_val_df, cfg, device)
        lr_epoch = scheduler.get_last_lr()[0]
        scheduler.step()
        epochs_ran = epoch + 1

        is_best = inner_auc > best_auc
        if is_best:
            best_auc = inner_auc
            improved = True
            save_ckpt(best_pt, model, optimizer, epoch, best_auc)

        csv_log.log({**row_id, "stage": stage_name, "epoch": epoch,
                     "lr_epoch": lr_epoch, "n_bn_eval": n_bn,
                     "train_loss": round(train_loss, 6),
                     "inner_val_auc": round(inner_auc, 6),
                     "is_best": is_best,
                     "epoch_time_sec": round(time.time() - t0, 2),
                     "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})

        if stopper.step(inner_auc):
            logger.info("[%s] early stop at epoch %d (best inner-val AUC %.4f)",
                        stage_name, epoch, best_auc)
            break

    return epochs_ran, best_auc, improved


def run(cfg: dict, backbone: str, fold: int, unfreeze_pct: int) -> None:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.models.registry import build_model
    from src.stage_03_train import _filter_for_task
    from src.stage_03b_fusion import _load_merged
    from src.training.dataset import NoduleDataset2_5D
    from src.training.finetune import (build_param_groups, count_trainable,
                                        freeze_all, unfreeze_top_modules)
    from src.utils.logger import CSVLogger, append_row
    from src.utils.seed import fix_seed
    from src.utils.tracks import track_input_size

    if backbone not in BACKBONES:
        raise ValueError(f"{backbone!r} is not one of the locked backbones {BACKBONES}")
    if unfreeze_pct not in UNFREEZE_PCT:
        raise ValueError(f"unfreeze {unfreeze_pct} is not pre-registered; allowed: {UNFREEZE_PCT}")

    fix_seed(cfg.get("seed", 42))

    cell = f"{backbone}_uf{unfreeze_pct}"
    ckpt_dir = os.path.join(cfg["paths"]["checkpoints"], "run03", cell)
    best_pt = os.path.join(ckpt_dir, f"fold{fold}_best.pt")
    preds_dir = os.path.join(OUT_DIR, "preds")
    preds_path = os.path.join(preds_dir, f"{cell}_fold{fold}.npz")

    # ponytail: resume is per cell, not per epoch. A two-stage run would have to
    # persist which stage and which optimizer state it died in; skipping a
    # finished cell costs at most one interrupted cell of GPU time and keeps the
    # restart path trivial. Delete the npz to force a rerun.
    if os.path.exists(preds_path) and os.path.exists(best_pt):
        print(f"[SKIP] {cell} fold{fold} already done ({preds_path})")
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)
    head_lr = cfg["train"].get("lr", 1e-4)
    input_size = track_input_size(cfg, backbone)

    labels = pd.read_csv(os.path.join(cfg["paths"]["interim"], "labels.csv"))
    full = _filter_for_task(labels, "binary")
    train_df = full[full["fold"] != fold].reset_index(drop=True)
    outer_full_df = full[full["fold"] == fold].reset_index(drop=True)

    merged, _feat_cols = _load_merged(cfg)
    outer_merged_df = merged[merged["fold"] == fold].reset_index(drop=True)

    inner_train_idx, inner_val_idx = _split_indices(train_df, fold)
    inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
    inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)

    # drop_last mirrors stage_03b_fusion.py:205 -- an inner split can leave a
    # size-1 final batch, which BatchNorm rejects in training mode.
    train_loader = DataLoader(
        NoduleDataset2_5D(inner_train_df, patch_size=patch_xy, n_slices=n_slices, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    logger.info("[%s fold %d] inner_train=%d inner_val=%d outer_merged=%d outer_full=%d",
                cell, fold, len(inner_train_df), len(inner_val_df),
                len(outer_merged_df), len(outer_full_df))

    model = build_model(backbone, cfg).to(device)

    epoch_fields = ["run_id", "backbone", "unfreeze_pct", "fold", "stage", "epoch",
                    "lr_epoch", "n_bn_eval", "train_loss", "inner_val_auc", "is_best",
                    "epoch_time_sec", "timestamp"]
    epoch_log = os.path.join(cfg["paths"]["logs"], "epochs", "run03", f"{cell}_fold{fold}.csv")
    csv_log = CSVLogger(epoch_log, epoch_fields)
    row_id = {"run_id": RUN_ID, "backbone": backbone, "unfreeze_pct": unfreeze_pct, "fold": fold}

    t_start = time.time()
    try:
        # --- Stage 1: head only ---
        freeze_all(model)
        unfreeze_top_modules(model, 0.0)
        s1_epochs, best_auc, _s1_improved = _train_stage(
            model, train_loader, inner_val_df, cfg, device,
            build_param_groups(model, head_lr, LR_DECAY_FACTOR),
            STAGE1_EPOCHS, best_pt, 0.0, "head_only", csv_log, row_id)

        # --- Stage 2: staged unfreeze ---
        # The 100% cell runs the same two-stage protocol and the same frozen
        # BatchNorm; only the unfreeze breadth differs. That is what makes it a
        # clean control for the nested-CV protocol penalty rather than a second
        # variable moving at the same time.
        freeze_all(model)
        n_open, n_child = unfreeze_top_modules(model, unfreeze_pct / 100.0)
        n_trainable, n_total = count_trainable(model)
        logger.info("[%s fold %d] stage 2 opens %d/%d child modules, %d/%d parameters (%.1f%%)",
                    cell, fold, n_open, n_child, n_trainable, n_total,
                    100.0 * n_trainable / n_total)

        s2_epochs, best_auc, s2_improved = _train_stage(
            model, train_loader, inner_val_df, cfg, device,
            build_param_groups(model, head_lr, LR_DECAY_FACTOR),
            STAGE2_EPOCHS, best_pt, best_auc, f"unfreeze_{unfreeze_pct}", csv_log, row_id)
    finally:
        csv_log.close()

    best_stage = f"unfreeze_{unfreeze_pct}" if s2_improved else "head_only"

    # --- outer fold, scored exactly once, with the selected weights ---
    state = torch.load(best_pt, weights_only=True, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)

    auc_merged, y_true_m, y_prob_m = _score(model, outer_merged_df, cfg, device)
    auc_full, _, _ = _score(model, outer_full_df, cfg, device)

    np.savez(preds_path,
             y_true=y_true_m, y_prob=y_prob_m,
             patient_id=outer_merged_df["patient_id"].values.astype(str),
             nodule_idx=outer_merged_df["nodule_idx"].values)

    append_row(SUMMARY_CSV, {
        "run_id": RUN_ID, "commit_sha": _commit_sha(), "input_size": input_size,
        "checkpoint_mtime": _iso_mtime(best_pt),
        "backbone": backbone, "unfreeze_pct": unfreeze_pct, "fold": fold,
        "n_child_open": n_open, "n_child_total": n_child,
        "n_trainable": n_trainable, "n_total": n_total,
        "inner_val_auc": round(best_auc, 6),
        "outer_auc_merged": round(auc_merged, 6), "outer_auc_full": round(auc_full, 6),
        "n_val_merged": len(outer_merged_df), "n_val_full": len(outer_full_df),
        "stage1_epochs_ran": s1_epochs, "stage2_epochs_ran": s2_epochs,
        "best_stage": best_stage,
        "total_time_sec": round(time.time() - t_start, 1),
    })

    print(f"[DONE] {cell} fold{fold}  inner_val={best_auc:.4f}  "
          f"outer_merged={auc_merged:.4f}  outer_full={auc_full:.4f}  best_stage={best_stage}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--backbone", required=True, choices=BACKBONES)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--unfreeze", type=int, required=True, choices=UNFREEZE_PCT,
                    help="percent of the backbone's top layer GROUPS to reopen "
                         "(not percent of weights -- see PREREG_run03.md section 2a)")
    args = p.parse_args()
    run(yaml.safe_load(open(args.config)), args.backbone, args.fold, args.unfreeze)


if __name__ == "__main__":
    main()
