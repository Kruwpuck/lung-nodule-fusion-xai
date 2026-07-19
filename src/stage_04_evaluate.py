"""Stage 04: load best checkpoints -> metrics -> summary.csv.

Task-aware: --task binary|ordinal|grade4, matches checkpoint dirs
written by stage_03_train.py (checkpoints/{model}_{task}/foldN_best.pt).
"""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict, task: str = "binary") -> None:
    import pickle
    import pandas as pd
    import numpy as np
    import torch
    from src.models.registry import build_model
    from src.training.dataset import NoduleDataset2_5D
    from src.evaluation.metrics import compute_metrics, ordinal_metrics, grade4_metrics, grade4_nodule_only_metrics
    from src.evaluation.efficiency import count_params, measure_flops, measure_latency
    from src.stage_03_train import _TASK_CFG, _filter_for_task, _evaluate
    from torch.utils.data import DataLoader

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    df = _filter_for_task(df, task)

    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"summary_{task}.csv")

    from src.utils.io import cached
    if cached(out) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out}")
        return

    preds_dir = os.path.join(results_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_models = (cfg["models"].get("lightweight", []) +
                  cfg["models"].get("heavyweight", []))
    n_folds = cfg["data"].get("n_folds", 5)
    rows = []
    tcfg = _TASK_CFG[task]

    for model_name in all_models:
        model = build_model(model_name, cfg, task=task)
        params_M = round(count_params(model) / 1e6, 3)
        gflops, _ = measure_flops(model, input_res=(n_slices, patch_xy, patch_xy))
        latency_ms = measure_latency(model, input_res=(n_slices, patch_xy, patch_xy))
        model = model.to(device)

        for fold in range(n_folds):
            best_pt = os.path.join(cfg["paths"]["checkpoints"], f"{model_name}_{task}", f"fold{fold}_best.pt")
            if not os.path.exists(best_pt):
                logger.warning("Missing checkpoint: %s", best_pt)
                continue

            try:
                state = torch.load(best_pt, weights_only=True, map_location="cpu")
            except (TypeError, pickle.UnpicklingError):
                state = torch.load(best_pt, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            else:
                model.load_state_dict(state)

            val_df = df[df["fold"] == fold].reset_index(drop=True)
            val_ds = NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False,
                                        target_col=tcfg["target_col"], target_dtype=tcfg["target_dtype"])
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            y_true, y_out = _evaluate(model, val_loader, device, task)
            np.savez(
                os.path.join(preds_dir, f"{model_name}_{task}_fold{fold}.npz"),
                y_true=y_true, y_out=y_out,
            )
            if task == "binary":
                m = compute_metrics(y_true, y_out[:, 1])
            elif task in ("grade4", "grade3"):
                m = grade4_metrics(y_true, y_out)
                if task == "grade4":
                    m.update(grade4_nodule_only_metrics(y_true, y_out))
            else:
                m = ordinal_metrics(y_true, y_out)

            rows.append({
                "model": model_name,
                "task": task,
                "fold": fold,
                "params_M": params_M,
                "gflops": gflops,
                "latency_ms": latency_ms,
                **m,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out, index=False)
    print(f"[DONE] {out}  ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--task", default="binary", choices=["binary", "ordinal", "grade3", "grade4"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.task)


if __name__ == "__main__":
    main()
