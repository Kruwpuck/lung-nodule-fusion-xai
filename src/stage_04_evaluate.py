"""Stage 04: load best checkpoints -> metrics -> summary.csv."""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    import pandas as pd
    import torch
    from src.models.registry import build_model
    from src.training.trainer import evaluate
    from src.training.dataset import NoduleDataset2_5D
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.efficiency import count_params, measure_flops, measure_latency
    from torch.utils.data import DataLoader

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "summary.csv")

    from src.utils.io import cached
    if cached(out) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out}")
        return

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    batch_size = cfg["train"].get("batch_size", 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_models = (cfg["models"].get("lightweight", []) +
                  cfg["models"].get("heavyweight", []))
    n_folds = cfg["data"].get("n_folds", 5)
    rows = []

    for model_name in all_models:
        model = build_model(model_name, cfg).to(device)
        params_M = round(count_params(model) / 1e6, 3)
        gflops, _ = measure_flops(model, input_res=(n_slices, patch_xy, patch_xy))
        latency_ms = measure_latency(model, input_res=(n_slices, patch_xy, patch_xy))
        model = model.to(device)  # measure_latency moves model to CPU in-place

        for fold in range(n_folds):
            best_pt = os.path.join(cfg["paths"]["checkpoints"], model_name, f"fold{fold}_best.pt")
            if not os.path.exists(best_pt):
                logger.warning("Missing checkpoint: %s", best_pt)
                continue

            try:
                state = torch.load(best_pt, weights_only=True, map_location="cpu")
            except (TypeError, __import__("pickle").UnpicklingError):
                state = torch.load(best_pt, weights_only=False, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            else:
                model.load_state_dict(state)

            val_df = df[df["fold"] == fold].reset_index(drop=True)
            val_ds = NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            y_true, y_prob = evaluate(model, val_loader, device)
            m = compute_metrics(y_true, y_prob)
            rows.append({
                "model": model_name,
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
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
