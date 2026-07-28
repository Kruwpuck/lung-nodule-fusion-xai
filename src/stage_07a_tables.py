"""Stage 07a: dataset distribution tables for the paper (VISUALIZATION_FIGURES_PLAN.md,
Bagian 3). No GPU/checkpoints needed -- reads labels.csv (+ radiomics.parquet for nodule
size) only.

Outputs (artifacts/results/tables/):
  table_3_1_class_distribution.csv/.md   -- class counts per arm (A/B/C/D)
  table_3_2_fold_distribution.csv/.md    -- class counts per fold, per arm (stratification check)
  table_3_3_dataset_characteristics.csv/.md -- patients/scans/nodules, slice thickness, nodule size
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_GRADE4_NAMES = {0: "no_nodule", 1: "benign", 2: "indeterminate", 3: "malignant"}
_GRADE3_NAMES = {0: "benign", 1: "indeterminate", 2: "malignant"}
_LABEL_NAMES = {0: "benign", 1: "malignant"}


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def _write(df: pd.DataFrame, out_dir: str, name: str) -> None:
    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write(_df_to_markdown(df))
    logger.info("%s.csv/.md written (%d rows)", name, len(df))


def _class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls_val, cls_name in _LABEL_NAMES.items():
        rows.append({"class": cls_name, "arm_A_binary": int((df["label"] == cls_val).sum())})
    rows.append({"class": "indeterminate", "arm_A_binary": "dropped"})
    arm_a = pd.DataFrame(rows).set_index("class")

    arm_b = pd.Series({"n_included (all nodules w/ rating)": int(df["median_rating"].notna().sum())})

    rows_c = []
    for cls_val, cls_name in _GRADE3_NAMES.items():
        rows_c.append({"class": cls_name, "arm_C_grade3": int((df["grade3"] == cls_val).sum())})
    rows_c.append({"class": "no_nodule", "arm_C_grade3": "excluded"})
    arm_c = pd.DataFrame(rows_c).set_index("class")

    rows_d = []
    for cls_val, cls_name in _GRADE4_NAMES.items():
        rows_d.append({"class": cls_name, "arm_D_grade4": int((df["grade4"] == cls_val).sum())})
    arm_d = pd.DataFrame(rows_d).set_index("class")

    merged = arm_a.join(arm_c, how="outer").join(arm_d, how="outer")
    merged = merged.reindex(["benign", "malignant", "indeterminate", "no_nodule"])
    merged = merged.reset_index().rename(columns={"index": "class"})
    merged.loc[len(merged)] = ["TOTAL", int(df["label"].isin([0, 1]).sum()),
                                int((df["grade3"] != -1).sum()), int(len(df))]

    rating_counts = df["median_rating"].round().value_counts(dropna=True).sort_index()
    rating_row = {"class": "rating_1/2/3/4/5 (arm B, rounded)"}
    rating_row["arm_B_ordinal"] = " / ".join(
        f"{int(k)}:{int(v)}" for k, v in rating_counts.items()
    )
    merged = pd.concat([merged, pd.DataFrame([rating_row])], ignore_index=True)
    return merged


def _fold_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        row = {"fold": fold, "n_total": len(sub)}
        for cls_val, cls_name in _GRADE4_NAMES.items():
            row[f"grade4_{cls_name}"] = int((sub["grade4"] == cls_val).sum())
        for cls_val, cls_name in _LABEL_NAMES.items():
            row[f"binary_{cls_name}"] = int((sub["label"] == cls_val).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_characteristics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = [
        {"metric": "n_patients", "value": int(df["patient_id"].nunique())},
        {"metric": "n_scans", "value": int(df["scan_id"].nunique())},
        {"metric": "n_nodule_rows_total", "value": int(len(df))},
        {"metric": "n_nodule_rows (class_kind=nodule)", "value": int((df["class_kind"] == "nodule").sum())},
        {"metric": "n_no_nodule_hard_negatives", "value": int((df["class_kind"] == "no_nodule_hard_negative").sum())},
        {"metric": "slice_thickness_mm_min", "value": round(float(df["spacing_z"].min()), 3)},
        {"metric": "slice_thickness_mm_median", "value": round(float(df["spacing_z"].median()), 3)},
        {"metric": "slice_thickness_mm_max", "value": round(float(df["spacing_z"].max()), 3)},
        {"metric": "n_annotations_per_nodule_median", "value": float(df["n_annotations"].median())},
        {"metric": "label_aggregation_rule", "value": "median of radiologist malignancy ratings (1-5)"},
        {"metric": "arm_A_exclude_rule", "value": "median_rating == 3 (indeterminate) dropped"},
        {"metric": "no_nodule_source", "value": "LUNA16 class-0 candidates (hard negatives)"},
    ]

    rad_path = cfg["paths"]["features"]
    try:
        rad = pd.read_parquet(rad_path, columns=["patient_id", "nodule_idx", "original_shape_Maximum3DDiameter"])
        merged = df.merge(rad, on=["patient_id", "nodule_idx"], how="inner")
        d = merged["original_shape_Maximum3DDiameter"]
        rows.append({"metric": "nodule_diameter_mm_median (3D max, radiomics)", "value": round(float(d.median()), 2)})
        rows.append({"metric": "nodule_diameter_mm_range (3D max, radiomics)",
                     "value": f"{d.min():.2f}-{d.max():.2f}"})
    except Exception as e:
        logger.warning("radiomics diameter merge failed: %s", e)

    return pd.DataFrame(rows)


def run(cfg: dict) -> None:
    out_dir = os.path.join(cfg["paths"]["results"], "tables")
    sentinel = os.path.join(out_dir, "done.txt")
    from src.utils.io import cached
    if cached(sentinel) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {sentinel}")
        return
    os.makedirs(out_dir, exist_ok=True)

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)

    _write(_class_distribution(df), out_dir, "table_3_1_class_distribution")
    _write(_fold_distribution(df), out_dir, "table_3_2_fold_distribution")
    _write(_dataset_characteristics(df, cfg), out_dir, "table_3_3_dataset_characteristics")

    with open(sentinel, "w") as f:
        f.write("done")
    print(f"[DONE] {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
