"""Audit: why 1695 (label-filter-pass) nodules -> only 1391 in labels.csv.
Read-only. Does not touch data/processed/labels.csv or artifacts/patches.
"""
import os
import sys
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus

sys.path.insert(0, os.getcwd())

RAW = "./data/raw/LIDC-IDRI"

old = pd.read_csv("data/processed/labels.csv")
print("### current labels.csv rows:", len(old))
print("    columns:", list(old.columns))

scans = pl.query(pl.Scan).all()
print("### pylidc scans:", len(scans))

disk_pids = set(os.listdir(RAW)) if os.path.isdir(RAW) else set()
print("### DICOM folders on disk:", len(disk_pids))

reasons = {"disk_missing": 0, "volume_load_fail": 0, "consensus_fail": 0,
           "empty_mask_pre": 0, "crop_fail": 0, "empty_mask_post": 0, "ok": 0}
label_filtered_out = 0
total_nodule_groups_min_ann = 0
ok_keys = set()
nonthree_keys = set()

for _i, scan in enumerate(scans):
    if _i % 100 == 0:
        print(f"... progress {_i}/{len(scans)}", flush=True)
    pid = scan.patient_id
    if pid not in disk_pids:
        # can't even attempt — count nodule groups lost this way
        try:
            groups = scan.cluster_annotations()
        except Exception:
            groups = []
        for anns in groups:
            if len(anns) < 1:
                continue
            reasons["disk_missing"] += 1
        continue

    # to_volume() skipped on purpose: the 304-shortfall question is about which
    # median!=3 groups yield a valid consensus mask, which needs no pixel volume.
    # Loading full volumes for 1018 scans was slow + produced spurious failures.
    try:
        groups = scan.cluster_annotations()
    except Exception:
        groups = []

    for nidx, anns in enumerate(groups):
        if len(anns) < 1:
            continue
        ratings = [a.malignancy for a in anns]
        med = float(np.median(ratings))
        if med == 3:
            label_filtered_out += 1
            continue  # old scheme excludes this — not part of 1695

        total_nodule_groups_min_ann += 1
        nonthree_keys.add((pid, nidx))

        try:
            cmask, cbbox, _ = consensus(anns, clevel=0.5, pad=0)
        except Exception as e:
            reasons["consensus_fail"] += 1
            if reasons["consensus_fail"] <= 5:
                print("  consensus_fail example:", repr(e), flush=True)
            continue

        if cmask.sum() == 0:
            reasons["empty_mask_pre"] += 1
            continue

        reasons["ok"] += 1
        ok_keys.add((pid, nidx))

print()
print("### median!=3 nodule groups (= should-be 1695 count):", total_nodule_groups_min_ann)
print("### breakdown of losses within that set:")
for k, v in reasons.items():
    if k != "disk_missing":
        print(f"    {k}: {v}")
print("### disk_missing nodule-groups (8 scans not on disk, any median):", reasons["disk_missing"])
print()
print("### label_filtered_out (median==3, expected loss, NOT part of the 304):", label_filtered_out)
print()
print("### SUMMARY: should-be(1695) - ok({}) = {}".format(reasons["ok"], total_nodule_groups_min_ann - reasons["ok"]))
print("### actual labels.csv rows: {}  | gap vs 'ok': {}".format(len(old), reasons["ok"] - len(old)))

missing_scan_ids = set(s.patient_id for s in scans) - disk_pids
print()
print("### scan patient_ids in DB but missing from disk:", len(missing_scan_ids))
for m in sorted(missing_scan_ids):
    print("   ", m)

# ---- cross-check against labels.csv by (patient_id, nodule_idx) ----
print()
print("### CROSS-CHECK vs labels.csv (patient_id, nodule_idx)")
if "nodule_idx" in old.columns:
    label_keys = set(zip(old["patient_id"].astype(str), old["nodule_idx"].astype(int)))
    ok_norm = {(str(p), int(n)) for (p, n) in ok_keys}
    in_ok_not_labels = ok_norm - label_keys   # derivable but never made it to csv = the real "missing"
    in_labels_not_ok = label_keys - ok_norm    # in csv but audit didn't derive (idx-scheme mismatch?)
    print("    derivable-ok groups:", len(ok_norm))
    print("    labels.csv keys:", len(label_keys))
    print("    OK-but-absent-from-labels (candidate 'missing'):", len(in_ok_not_labels))
    print("    labels-but-not-in-OK (idx mismatch/other):", len(in_labels_not_ok))
    for k in sorted(in_ok_not_labels)[:20]:
        print("      missing:", k)
else:
    print("    labels.csv has no nodule_idx column, skipping key diff")
