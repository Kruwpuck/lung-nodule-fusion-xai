
import pandas as pd, numpy as np, random

new = pd.read_csv("data/processed/labels.csv")
old = pd.read_csv("data/processed/labels_binary_v1.csv")

print("### row counts: new=%d old=%d" % (len(new), len(old)))
print("### new columns:", list(new.columns))

# GATE 3: shape check on 20 random new patches (not in old)
old_keys = set(zip(old["patient_id"], old["nodule_idx"])) if "nodule_idx" in old.columns else set()
new_keys = list(zip(new["patient_id"], new["nodule_idx"]))
fresh = [k for k in new_keys if k not in old_keys]
print("### fresh (new) nodule count:", len(fresh))
random.seed(0)
sample = random.sample(fresh, min(20, len(fresh)))
bad = 0
for pid, nidx in sample:
    row = new[(new.patient_id == pid) & (new.nodule_idx == nidx)].iloc[0]
    patch = np.load(row["patch_path"])
    mask = np.load(row["mask_path"])
    ok = patch.shape == (16, 64, 64) and mask.shape == (16, 64, 64) and mask.sum() > 0
    if not ok:
        bad += 1
        print("  BAD", pid, nidx, patch.shape, mask.shape, mask.sum())
print("### GATE3 shape check: %d/%d bad" % (bad, len(sample)))

# GATE 4: fold freeze — old patients must keep same fold
old_fold = dict(old.drop_duplicates("patient_id")[["patient_id","fold"]].values)
new_fold = dict(new.drop_duplicates("patient_id")[["patient_id","fold"]].values)
mismatches = []
for pid, f in old_fold.items():
    if pid in new_fold and new_fold[pid] != f:
        mismatches.append((pid, f, new_fold[pid]))
print("### GATE4 fold freeze: %d old patients checked, %d mismatches" % (len(old_fold), len(mismatches)))
for m in mismatches[:10]:
    print("  MISMATCH", m)

# class balance
print("### label breakdown:", new["label"].value_counts().to_dict())
print("### grade3 breakdown:", new["grade3"].value_counts().to_dict())
print("### unique patients new:", new["patient_id"].nunique(), " old:", old["patient_id"].nunique())
