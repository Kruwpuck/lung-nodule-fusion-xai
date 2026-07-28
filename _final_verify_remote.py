
import pandas as pd, numpy as np

labels = pd.read_csv("artifacts/patches/labels.csv")
feats = pd.read_parquet("artifacts/features/radiomics.parquet")

print("### labels.csv rows:", len(labels))
print("### labels.csv columns:", list(labels.columns))
print("### radiomics.parquet rows:", len(feats))

print("### class_kind breakdown:", labels["class_kind"].value_counts().to_dict() if "class_kind" in labels.columns else "N/A")
print("### label breakdown:", labels["label"].value_counts(dropna=False).to_dict())
print("### grade3 breakdown:", labels["grade3"].value_counts(dropna=False).to_dict())
print("### grade4 breakdown:", labels["grade4"].value_counts(dropna=False).to_dict())
print("### unique patients:", labels["patient_id"].nunique())
print("### fold breakdown:", labels["fold"].value_counts(dropna=False).to_dict())

# expected NaN: median_rating/rating_std NaN only for negatives (grade4==0 & label==-1 & n_annotations==0)
neg_mask = labels["class_kind"] == "no_nodule_hard_negative" if "class_kind" in labels.columns else labels["nodule_idx"] < 0
pos = labels[~neg_mask]
neg = labels[neg_mask]
print("### positives:", len(pos), " negatives:", len(neg))
print("### unexpected NaN in positives median_rating:", pos["median_rating"].isna().sum())
print("### unexpected NaN in negatives median_rating (expected all NaN):", neg["median_rating"].isna().sum(), "/", len(neg))

# key match between labels and radiomics parquet
lab_keys = set(zip(labels["patient_id"].astype(str), labels["nodule_idx"].astype(int)))
if "patient_id" in feats.columns and "nodule_idx" in feats.columns:
    feat_keys = set(zip(feats["patient_id"].astype(str), feats["nodule_idx"].astype(int)))
    print("### labels keys:", len(lab_keys), " radiomics keys:", len(feat_keys))
    print("### labels-not-in-radiomics:", len(lab_keys - feat_keys))
    print("### radiomics-not-in-labels:", len(feat_keys - lab_keys))
else:
    print("### radiomics columns:", list(feats.columns)[:10])

# nan check across radiomics feature columns (excluding id cols)
id_cols = {"patient_id", "nodule_idx", "scan_id"}
feat_cols = [c for c in feats.columns if c not in id_cols]
nan_counts = feats[feat_cols].isna().sum()
bad_cols = nan_counts[nan_counts > 0]
print("### radiomics feature columns with any NaN:", len(bad_cols), "of", len(feat_cols))
if len(bad_cols):
    print(bad_cols.head(10))
