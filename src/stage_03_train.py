"""Stage 03: train 1 model x 1 fold with checkpoint resume + CSV logging.

Supports 3 tasks (arm A binary, arm B ordinal, arm D grade4):
  --task binary   label column "label"          (drop label==-1: negatives + median==3)
  --task ordinal  label column "median_rating"   (drop NaN: negatives only; keeps median==3)
  --task grade4   label column "grade4"          (all rows: negatives=0, benign=1, indet=2, malignant=3)

Optimizer/weight-decay sweep (Track 2 stability study): --optimizer / --weight-decay.
Default combo (adamw + cfg weight_decay) keeps the pre-sweep checkpoint path so the
120 existing checkpoints keep resuming; non-default combos get an isolated checkpoint
dir. Every run (default or not) writes a self-describing per-epoch CSV under
artifacts/logs/epochs/{run_id}.csv and one row to artifacts/logs/runs.csv.
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


def _build_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float = 0.9):
    import torch
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name!r}")


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


def _evaluate(model, loader, device, task: str, criterion=None):
    """Run model over loader. Returns (y_true, y_out, val_loss).

    val_loss is None if criterion is not passed (back-compat for callers that
    only need predictions, e.g. stage_04_evaluate.py).
    """
    import torch
    import numpy as np
    model.eval()
    all_out, all_true = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            if criterion is not None:
                targets_dev = targets.to(device)
                out_for_loss = logits.squeeze(-1) if task == "ordinal" else logits
                loss = criterion(out_for_loss, targets_dev)
                total_loss += loss.item() * len(targets)
                n += len(targets)
            if task == "ordinal":
                pred = logits.squeeze(-1).cpu().numpy()
            else:
                pred = torch.softmax(logits, dim=1).cpu().numpy()
            all_out.append(pred)
            all_true.append(targets.numpy())
    val_loss = (total_loss / n) if criterion is not None and n else None
    return np.concatenate(all_true), np.concatenate(all_out), val_loss


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


def run(
    cfg: dict,
    model_name: str,
    fold: int,
    task: str = "binary",
    optimizer_name: str = "adamw",
    weight_decay: float | None = None,
) -> None:
    from src.utils.seed import fix_seed
    from src.models.registry import build_model
    from src.training.trainer import EarlyStopping, save_ckpt, maybe_resume
    from src.utils.logger import CSVLogger, append_row
    from src.utils.tracks import resolve_track
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.efficiency import count_params, measure_flops

    fix_seed(cfg.get("seed", 42))

    tcfg = _TASK_CFG[task]
    cfg_weight_decay = cfg["train"].get("weight_decay", 1e-4)
    if weight_decay is None:
        weight_decay = cfg_weight_decay
    lr = cfg["train"].get("lr", 1e-4)
    momentum = cfg.get("track2_sweep", {}).get("momentum", 0.9)

    is_default_combo = (optimizer_name == "adamw" and abs(weight_decay - cfg_weight_decay) < 1e-12)
    ckpt_subdir = model_name if task == "binary" else f"{model_name}_{task}"
    if not is_default_combo:
        ckpt_subdir = f"{ckpt_subdir}_{optimizer_name}_wd{weight_decay:g}"
    ckpt_dir = os.path.join(cfg["paths"]["checkpoints"], ckpt_subdir)
    last_pt = os.path.join(ckpt_dir, f"fold{fold}_last.pt")
    best_pt = os.path.join(ckpt_dir, f"fold{fold}_best.pt")

    run_id = f"{model_name}_{task}_f{fold}_{optimizer_name}_wd{weight_decay:g}"
    epoch_log_path = os.path.join(cfg["paths"]["logs"], "epochs", f"{run_id}.csv")
    runs_csv_path = os.path.join(cfg["paths"]["logs"], "runs.csv")
    track = resolve_track(cfg, model_name) or "legacy"

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, cfg, task=task).to(device)
    optimizer = _build_optimizer(optimizer_name, model.parameters(), lr, weight_decay, momentum)
    params_M = round(count_params(model) / 1e6, 3)
    gflops, _ = measure_flops(model, input_res=(n_slices, patch_xy, patch_xy))

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

    os.makedirs(os.path.dirname(epoch_log_path), exist_ok=True)
    epoch_fields = [
        "run_id", "track", "model", "task", "fold", "optimizer", "weight_decay", "lr_initial",
        "epoch", "lr_epoch", "train_loss", "val_loss", "val_score", "val_auc", "val_acc",
        "val_sensitivity", "val_specificity", "epoch_time_sec", "timestamp", "is_best",
    ]
    csv_log = CSVLogger(epoch_log_path, epoch_fields)

    import datetime
    run_started_at = datetime.datetime.now().isoformat(timespec="seconds")
    run_t0 = time.time()
    best_epoch = start_epoch
    early_stopped = False
    epochs_ran = start_epoch

    try:
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

            y_true, y_out, val_loss = _evaluate(model, val_loader, device, task, criterion=criterion)
            score = _score(y_true, y_out, task)
            lr_epoch = scheduler.get_last_lr()[0]
            scheduler.step()

            is_best = score > best_score
            if task == "binary":
                m = compute_metrics(y_true, y_out[:, 1])
                val_auc, val_acc = m["auc"], m["accuracy"]
                val_sens, val_spec = m["sensitivity"], m["specificity"]
            else:
                val_auc = val_acc = val_sens = val_spec = float("nan")

            csv_log.log({
                "run_id": run_id, "track": track, "model": model_name, "task": task, "fold": fold,
                "optimizer": optimizer_name, "weight_decay": weight_decay, "lr_initial": lr,
                "epoch": epoch, "lr_epoch": lr_epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else "",
                "val_score": round(score, 6),
                "val_auc": val_auc, "val_acc": val_acc,
                "val_sensitivity": val_sens, "val_specificity": val_spec,
                "epoch_time_sec": round(time.time() - t0, 2),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "is_best": is_best,
            })

            epochs_ran = epoch + 1
            if is_best:
                best_score = score
                best_epoch = epoch
                save_ckpt(best_pt, model, optimizer, epoch, best_score)

            if (epoch + 1) % ckpt_every == 0:
                save_ckpt(last_pt, model, optimizer, epoch, best_score)

            if early_stopper.step(score):
                logger.info("Early stop epoch %d (fold %d, task %s)", epoch, fold, task)
                save_ckpt(last_pt, model, optimizer, epoch, best_score)
                early_stopped = True
                break

        run_status = "completed"
    except Exception:
        run_status = "failed"
        raise
    finally:
        csv_log.close()
        latency_ms = None
        try:
            from src.evaluation.efficiency import measure_latency
            latency_ms = measure_latency(model.cpu(), input_res=(n_slices, patch_xy, patch_xy))
        except Exception:
            pass
        append_row(runs_csv_path, {
            "run_id": run_id, "track": track, "model": model_name, "task": task, "fold": fold,
            "optimizer": optimizer_name, "weight_decay": weight_decay, "lr": lr, "batch_size": batch_size,
            "epochs_planned": epochs, "epochs_ran": epochs_ran, "early_stopped": early_stopped,
            "best_epoch": best_epoch, "best_score": round(best_score, 6),
            "params_M": params_M, "gflops": gflops, "latency_ms": latency_ms,
            "n_train": len(train_df), "n_val": len(val_df), "seed": cfg.get("seed", 42),
            "device": str(device), "total_time_sec": round(time.time() - run_t0, 2),
            "started_at": run_started_at,
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": run_status,
        })

    print(f"[DONE] {model_name}[{task}] fold{fold}  best_score={best_score:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    p.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--weight-decay", type=float, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.model, args.fold, args.task, args.optimizer, args.weight_decay)


if __name__ == "__main__":
    main()
