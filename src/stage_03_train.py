"""Stage 03: train 1 model x 1 fold with checkpoint resume + CSV logging.

Supports 3 tasks (arm A binary, arm B ordinal, arm D grade4):
  --task binary   label column "label"          (drop label==-1: negatives + median==3)
  --task ordinal  label column "median_rating"   (drop NaN: negatives only; keeps median==3)
  --task grade4   label column "grade4"          (all rows: negatives=0, benign=1, indet=2, malignant=3)
"""
import argparse
import logging
import os
import time
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CFG = {
    "binary":  {"target_col": "label",         "target_dtype": "long",  "n_classes": 2},
    "ordinal": {"target_col": "median_rating",  "target_dtype": "float", "n_classes": 1},
    "grade3":  {"target_col": "grade3",         "target_dtype": "long",  "n_classes": 3},
    "grade4":  {"target_col": "grade4",         "target_dtype": "long",  "n_classes": 4},
}


def _filter_for_task(df, task: str):
    if task == "binary":
        return df[df["label"] != -1].reset_index(drop=True)
    if task == "ordinal":
        return df[df["median_rating"].notna()].reset_index(drop=True)
    if task == "grade3":
        return df[df["grade3"] != -1].reset_index(drop=True)
    if task == "grade4":
        return df.reset_index(drop=True)
    raise ValueError(f"Unknown task: {task!r}")


def _evaluate(model, loader, device, task: str):
    import torch
    import numpy as np
    model.eval()
    all_out, all_true = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            if task == "ordinal":
                pred = logits.squeeze(-1).cpu().numpy()
            else:
                pred = torch.softmax(logits, dim=1).cpu().numpy()
            all_out.append(pred)
            all_true.append(targets.numpy())
    return np.concatenate(all_true), np.concatenate(all_out)


def _score(y_true, y_out, task: str) -> float:
    """Return a scalar metric to maximize (higher = better) for ckpt/early-stop."""
    from sklearn.metrics import roc_auc_score, cohen_kappa_score
    import numpy as np
    if task == "binary":
        return float(roc_auc_score(y_true, y_out[:, 1]))
    if task in ("grade4", "grade3"):
        return float(roc_auc_score(y_true, y_out, multi_class="ovr", average="macro"))
    if task == "ordinal":
        y_pred_round = np.clip(np.round(y_out), 1, 5)
        return float(cohen_kappa_score(y_true.round().astype(int), y_pred_round.astype(int), weights="quadratic"))
    raise ValueError(task)


def run(cfg: dict, model_name: str, fold: int, task: str = "binary") -> None:
    from src.utils.seed import fix_seed
    from src.models.registry import build_model
    from src.training.trainer import EarlyStopping, save_ckpt, maybe_resume
    from src.utils.logger import CSVLogger

    fix_seed(cfg.get("seed", 42))

    tcfg = _TASK_CFG[task]
    ckpt_dir = os.path.join(cfg["paths"]["checkpoints"], f"{model_name}_{task}")
    last_pt = os.path.join(ckpt_dir, f"fold{fold}_last.pt")
    best_pt = os.path.join(ckpt_dir, f"fold{fold}_best.pt")
    log_path = os.path.join(cfg["paths"]["logs"], f"{model_name}_{task}_fold{fold}.csv")

    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.training.dataset import NoduleDataset2_5D

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    df = _filter_for_task(df, task)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)

    ds_kwargs = dict(target_col=tcfg["target_col"], target_dtype=tcfg["target_dtype"])
    train_ds = NoduleDataset2_5D(train_df, patch_size=patch_xy, n_slices=n_slices, augment=True, **ds_kwargs)
    val_ds = NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, cfg, task=task).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    criterion = nn.SmoothL1Loss() if task == "ordinal" else nn.CrossEntropyLoss()
    epochs = cfg["train"].get("epochs", 50)
    ckpt_every = cfg["train"].get("checkpoint_every", 5)
    patience = cfg["train"].get("early_stopping_patience", 10)
    mixed_precision = cfg["train"].get("mixed_precision", True)

    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, best_score = maybe_resume(last_pt, model, optimizer)

    if start_epoch >= epochs:
        print(f"[SKIP] {model_name}[{task}] fold{fold} already complete (epoch {start_epoch})")
        return

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(start_epoch):
        scheduler.step()

    amp_scaler = torch.amp.GradScaler() if mixed_precision and device.type == "cuda" else None
    early_stopper = EarlyStopping(patience=patience, mode="max")
    early_stopper.best = best_score

    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    csv_log = CSVLogger(log_path, ["epoch", "train_loss", "val_score", "time_sec"])

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            if amp_scaler:
                with torch.amp.autocast(device.type):
                    out = model(imgs)
                    out = out.squeeze(-1) if task == "ordinal" else out
                    loss = criterion(out, targets)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                out = model(imgs)
                out = out.squeeze(-1) if task == "ordinal" else out
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
        train_loss = total_loss / len(train_loader.dataset)

        y_true, y_out = _evaluate(model, val_loader, device, task)
        score = _score(y_true, y_out, task)
        scheduler.step()

        csv_log.log({"epoch": epoch, "train_loss": round(train_loss, 6),
                     "val_score": round(score, 6), "time_sec": round(time.time() - t0, 2)})

        if score > best_score:
            best_score = score
            save_ckpt(best_pt, model, optimizer, epoch, best_score)

        if (epoch + 1) % ckpt_every == 0:
            save_ckpt(last_pt, model, optimizer, epoch, best_score)

        if early_stopper.step(score):
            logger.info("Early stop epoch %d (fold %d, task %s)", epoch, fold, task)
            save_ckpt(last_pt, model, optimizer, epoch, best_score)
            break

    csv_log.close()
    print(f"[DONE] {model_name}[{task}] fold{fold}  best_score={best_score:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.model, args.fold, args.task)


if __name__ == "__main__":
    main()
