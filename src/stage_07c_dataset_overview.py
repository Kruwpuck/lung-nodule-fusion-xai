"""Stage 07c: dataset overview figure -- representative patches per class
(VISUALIZATION_FIGURES_PLAN.md, Bagian 2). No model/GPU needed.

For each class (benign / malignant / indeterminate / no-nodule), picks 3
typical examples: mask centered (offset <= 6px), diameter close to the
in-class median (avoids outliers), sorted by |diameter - median|.

Output: artifacts/results/figures/dataset_overview.png
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MAX_CENTER_OFFSET_PX = 6.0
_N_PER_CLASS = 3
_CLASS_ORDER = [(1, "benign"), (3, "malignant"), (2, "indeterminate"), (0, "no_nodule")]


def _mask_center_offset(mask_path: str) -> tuple[float, bool]:
    mask_vol = np.load(mask_path)
    mid = mask_vol.shape[0] // 2
    mask2d = mask_vol[mid].astype(bool)
    if not mask2d.any():
        return float("inf"), False
    H, W = mask2d.shape
    ys, xs = np.nonzero(mask2d)
    cy, cx = ys.mean(), xs.mean()
    return float(np.hypot(cy - H / 2, cx - W / 2)), True


def _load_ct_center_slice(patch_path: str, hu_min=-1000.0, hu_max=400.0) -> np.ndarray:
    vol = np.load(patch_path).astype(np.float32)
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    return vol[vol.shape[0] // 2]


def _center_slice_is_valid(patch_path: str) -> bool:
    """Reject degenerate (blank / zero-padded) center slices."""
    return bool(_load_ct_center_slice(patch_path).std() > 0.02)


def run(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(cfg["paths"]["results"], "figures")
    out_path = os.path.join(out_dir, "dataset_overview.png")
    from src.utils.io import cached
    if cached(out_path) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)

    rad_path = cfg["paths"]["features"]
    rad = pd.read_parquet(rad_path, columns=["patient_id", "nodule_idx", "original_shape_Maximum3DDiameter"])
    df = df.merge(rad, on=["patient_id", "nodule_idx"], how="left")

    picks_by_class: dict[str, list[dict]] = {}
    for cls_val, cls_name in _CLASS_ORDER:
        sub = df[df["grade4"] == cls_val].copy()
        if cls_name != "no_nodule":
            centering = sub["mask_path"].map(_mask_center_offset)
            sub["offset_px"] = [c[0] for c in centering]
            sub["mask_nonempty"] = [c[1] for c in centering]
            sub = sub[sub["mask_nonempty"] & (sub["offset_px"] <= _MAX_CENTER_OFFSET_PX)]
            med = sub["original_shape_Maximum3DDiameter"].median()
            sub["dist_to_median"] = (sub["original_shape_Maximum3DDiameter"] - med).abs()
            sub = sub.sort_values("dist_to_median")
        # drop degenerate (blank/zero-padded) patches before taking top-N
        valid_rows = []
        for _, r in sub.iterrows():
            if _center_slice_is_valid(r["patch_path"]):
                valid_rows.append(r)
            if len(valid_rows) >= _N_PER_CLASS:
                break
        picks_by_class[cls_name] = [r.to_dict() for r in valid_rows]
        logger.info("%s: %d examples picked", cls_name, len(picks_by_class[cls_name]))

    n_rows = len(_CLASS_ORDER)
    fig, axes = plt.subplots(n_rows, _N_PER_CLASS, figsize=(3.2 * _N_PER_CLASS, 3.2 * n_rows))

    for r, (cls_val, cls_name) in enumerate(_CLASS_ORDER):
        rows = picks_by_class[cls_name]
        for col in range(_N_PER_CLASS):
            ax = axes[r, col]
            if col >= len(rows):
                ax.axis("off")
                continue
            row = rows[col]
            ct = _load_ct_center_slice(row["patch_path"])
            ax.imshow(ct, cmap="gray")
            if cls_name != "no_nodule":
                mask_vol = np.load(row["mask_path"])
                mask2d = mask_vol[mask_vol.shape[0] // 2].astype(bool)
                ax.contour(mask2d, colors="lime", linewidths=1.2)
                caption = f"median_rating={row['median_rating']:.1f}\ndiam={row['original_shape_Maximum3DDiameter']:.1f}mm"
            else:
                caption = "no-nodule (hard neg.)"
            ax.set_title(caption, fontsize=9)
            ax.axis("off")
        axes[r, 0].text(-0.25, 0.5, cls_name, transform=axes[r, 0].transAxes,
                         fontsize=13, fontweight="bold", rotation=90,
                         va="center", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
