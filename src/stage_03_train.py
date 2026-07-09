"""Stage 03: train 1 model x 1 fold with checkpoint resume + CSV logging."""
import argparse
import logging
import os
import time
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict, model_name: str, fold: int) -> None:
    from src.utils.io import cached
    from src.utils.seed import fix_seed
    from src.models.registry import build_model
    from src.training.trainer import (
        train_one_epoch, evaluate, EarlyStopping, save_ckpt, maybe_resume
    )
    from src.utils.logger import CSVLogger

    fix_seed(cfg.get("seed", 42))

    ckpt_dir = os.path.join(cfg["paths"]["checkpoints"], model_name)
    last_pt = os.path.join(ckpt_dir, f"fold{fold}_last.pt")
    best_pt = os.path.join(ckpt_dir, f"fold{fold}_best.pt")
    log_path = os.path.join(cfg["paths"]["logs"], f"{model_name}_fold{fold}.csv")

    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.training.dataset import NoduleDataset2_5D

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)

    train_ds = NoduleDataset2_5D(train_df, patch_size=patch_xy, n_slices=n_slices, augment=True)
    val_ds = NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = cfg["train"].get("epochs", 50)
    ckpt_every = cfg["train"].get("checkpoint_every", 5)
    patience = cfg["train"].get("early_stopping_patience", 10)
    mixed_precision = cfg["train"].get("mixed_precision", True)

    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, best_auc = maybe_resume(last_pt, model, optimizer)

    if start_epoch >= epochs:
        print(f"[SKIP] {model_name} fold{fold} already complete (epoch {start_epoch})")
        return

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(start_epoch):
        scheduler.step()

    amp_scaler = torch.amp.GradScaler() if mixed_precision and device.type == "cuda" else None
    early_stopper = EarlyStopping(patience=patience, mode="max")
    early_stopper.best = best_auc

    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    csv_log = CSVLogger(log_path, ["epoch", "train_loss", "val_auc", "time_sec"])

    from sklearn.metrics import roc_auc_score
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        y_true, y_prob = evaluate(model, val_loader, device)
        auc = roc_auc_score(y_true, y_prob)
        scheduler.step()

        csv_log.log({"epoch": epoch, "train_loss": round(train_loss, 6),
                     "val_auc": round(auc, 6), "time_sec": round(time.time() - t0, 2)})

        if auc > best_auc:
            best_auc = auc
            save_ckpt(best_pt, model, optimizer, epoch, best_auc)

        if (epoch + 1) % ckpt_every == 0:
            save_ckpt(last_pt, model, optimizer, epoch, best_auc)

        if early_stopper.step(auc):
            logger.info("Early stop epoch %d (fold %d)", epoch, fold)
            save_ckpt(last_pt, model, optimizer, epoch, best_auc)
            break

    csv_log.close()
    print(f"[DONE] {model_name} fold{fold}  best_auc={best_auc:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.model, args.fold)


if __name__ == "__main__":
    main()
