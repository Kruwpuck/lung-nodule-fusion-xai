"""Unified loader for training logs — normalizes old-format per-model CSVs
(`artifacts/logs/{model}[_{task}]_fold{N}.csv`, header `val_auc`/`val_score`)
and new-format self-describing CSVs (`artifacts/logs/epochs/{run_id}.csv`)
into one long DataFrame, so plotting doesn't need to branch on file age.
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

_OLD_NAME_RE = re.compile(
    r"^(?P<model>.+?)(?:_(?P<task>ordinal|grade3|grade4))?_fold(?P<fold>\d+)\.csv$"
)

_EPOCH_SCHEMA_COLS = [
    "run_id", "track", "model", "task", "fold", "optimizer", "weight_decay",
    "lr_initial", "epoch", "lr_epoch", "train_loss", "val_loss", "val_score",
    "val_auc", "val_acc", "val_sensitivity", "val_specificity",
    "epoch_time_sec", "timestamp", "is_best",
]


def _load_old_epoch_csv(path: str) -> pd.DataFrame:
    fname = os.path.basename(path)
    m = _OLD_NAME_RE.match(fname)
    if not m:
        return pd.DataFrame()

    model = m.group("model")
    task = m.group("task") or "binary"
    fold = int(m.group("fold"))

    df = pd.read_csv(path)
    if "val_auc" in df.columns and "val_score" not in df.columns:
        df = df.rename(columns={"val_auc": "val_score"})
    if "time_sec" in df.columns:
        df = df.rename(columns={"time_sec": "epoch_time_sec"})

    df["model"] = model
    df["task"] = task
    df["fold"] = fold
    df["run_id"] = f"{model}_{task}_f{fold}_adamw_wdlegacy"
    df["track"] = "legacy"
    df["optimizer"] = "adamw"
    df["weight_decay"] = None
    df["lr_initial"] = None
    df["lr_epoch"] = None
    df["val_auc"] = df.get("val_score")
    df["val_acc"] = None
    df["val_sensitivity"] = None
    df["val_specificity"] = None
    df["timestamp"] = None
    df["is_best"] = None

    for col in _EPOCH_SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_EPOCH_SCHEMA_COLS]


def load_epoch_logs(logs_dir: str = "artifacts/logs") -> pd.DataFrame:
    """Load all per-epoch logs (old + new format) into one normalized DataFrame."""
    frames = []

    for path in sorted(glob.glob(os.path.join(logs_dir, "epochs", "*.csv"))):
        df = pd.read_csv(path)
        for col in _EPOCH_SCHEMA_COLS:
            if col not in df.columns:
                df[col] = None
        frames.append(df[_EPOCH_SCHEMA_COLS])

    for path in sorted(glob.glob(os.path.join(logs_dir, "*_fold*.csv"))):
        df = _load_old_epoch_csv(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_EPOCH_SCHEMA_COLS)
    return pd.concat(frames, ignore_index=True)


def load_runs(logs_dir: str = "artifacts/logs") -> pd.DataFrame:
    """Load the run-level `runs.csv` table (new format only — no legacy equivalent)."""
    path = os.path.join(logs_dir, "runs.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
