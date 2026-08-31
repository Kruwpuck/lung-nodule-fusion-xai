"""F-3 -- ensemble the three Track 1 backbones inside each outer fold.

Ensembling across folds is not available: every fold holds out a different set
of patients, so there is no case on which two fold models both produce an
out-of-fold prediction. What is available is the other axis. Within one outer
fold the three backbones score the same held-out patients, so their
probabilities can be averaged without any case ever being scored by a model
that saw it in training.

Averaging probabilities positionally is only correct if the three arrays list
the same nodules in the same order. By construction they do -- the same
`merged` frame, the same fold filter, the same `reset_index(drop=True)`, and a
DataLoader with `shuffle=False`. That argument is exactly the kind that stays
true until someone changes one of the four things, and a misalignment would not
raise: it would silently average one patient's CNN probability onto another
patient's label and quietly lower every AUC by a few points. So the ordering is
asserted against `patient_id`, `nodule_idx` and `fold`, and the evidence is
printed before a single probability is averaged.

Both checkpoint regimes are measured, matching the F-2 decision that run03 is a
sensitivity analysis and not a replacement:

  * `selected_published` -- run02 probabilities, the checkpoints behind the
    published Track 1 tables, selected on the outer fold they are scored on.
  * `honest_nested_cv` -- run03 probabilities, the 100% unfreeze cell, whose
    epoch was selected on an inner split. `f2_sensitivity.md` section 6 bounds
    what these numbers are: a lower bound on the honest CNN branch, not an
    estimate of it, because four protocol differences move together.

Two arms are ensembled: `cnn_only`, the branch the ensemble is meant to
improve, and `fusion_late`, the arm the manuscript reports. The remaining
fusion arms retrain on top of the CNN branch and have no stored probabilities
to average.

Output:
    artifacts/results/run03/ensemble.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.evaluation.statistical_tests import delong_test
from src.stage_08b_run02_xai import _commit_sha
from src.utils.tracks import track_input_size

RUN_ID = "2026-08-22-run03"
CONFIG_PATH = os.path.join("configs", "config.yaml")

BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]

# (condition, probs dir, checkpoint kind). run03 stores no unselected final
# epoch -- stage 2 ends on early stopping, so `last` has no counterpart there.
REGIMES = [
    ("selected_published", os.path.join("artifacts", "results", "run02", "probs"), "best"),
    ("honest_nested_cv", os.path.join("artifacts", "results", "run03", "probs"), "best"),
]

# arm name -> npz key template
ARMS = {"cnn_only": "cnn_{k}", "fusion_late": "late_{k}"}

OUT_CSV = os.path.join("artifacts", "results", "run03", "ensemble.csv")


def _probs_path(probs_dir: str, backbone: str) -> str:
    return os.path.join(probs_dir, f"{backbone}.npz")


def _probs(probs_dir: str, backbone: str) -> dict:
    path = _probs_path(probs_dir, backbone)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} belum ada. Jalankan: python -m src.stage_08a_run02_probs "
            f"--out-dir {probs_dir}")
    return dict(np.load(path, allow_pickle=True))


def _mtime(path: str) -> str:
    """Timestamp of an input file, so a stale row is detectable rather than plausible.

    Recorded per backbone rather than aggregated: the three npz files are written by
    separate runs, and a `max()` over them would hide exactly the mismatch this column
    exists to surface. Same format as `stage_09d_cam_12._mtime`.
    """
    if not os.path.exists(path):
        return ""
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def _order_key(d: dict, mask: np.ndarray) -> list[str]:
    """The case ordering as plain strings, so the comparison is exact."""
    return [f"{p}#{int(n)}" for p, n in
            zip(np.asarray(d["patient_id"], dtype=str)[mask], d["nodule_idx"][mask])]


def verify_alignment(loaded: dict[str, dict], n_folds: int = 5,
                     label: str = "") -> None:
    """Assert the three backbones list the same cases in the same order.

    Prints the evidence per fold rather than only the verdict: a check whose
    output is the single word OK is indistinguishable from a check that never
    ran.
    """
    ref_name = BACKBONES[0]
    ref = loaded[ref_name]

    print(f"\n=== bukti urutan kasus  [{label}] ===")
    header = (f"{'fold':<5} {'n':>5}  {'backbone':<14} {'sha1(urutan)':<12} "
              f"{'kasus pertama':<22} {'kasus terakhir':<22} label")
    print(header)
    print("-" * len(header))

    for fold in range(n_folds):
        ref_mask = ref["fold"] == fold
        ref_keys = _order_key(ref, ref_mask)
        for backbone in BACKBONES:
            d = loaded[backbone]
            mask = d["fold"] == fold
            keys = _order_key(d, mask)

            assert len(keys) == len(ref_keys), (
                f"fold {fold}: {backbone} punya {len(keys)} kasus, "
                f"{ref_name} punya {len(ref_keys)}")
            assert keys == ref_keys, (
                f"fold {fold}: urutan kasus {backbone} berbeda dari {ref_name}. "
                f"Rerata probabilitas akan salah alamat. Beda pertama pada indeks "
                f"{next(i for i, (a, b) in enumerate(zip(keys, ref_keys)) if a != b)}")
            assert np.array_equal(d["y_true"][mask], ref["y_true"][ref_mask]), (
                f"fold {fold}: label {backbone} berbeda dari {ref_name} "
                f"meski urutan kasusnya sama")

            digest = hashlib.sha1("|".join(keys).encode()).hexdigest()[:10]
            lab = f"{int(d['y_true'][mask].sum())}/{int(mask.sum())} positif"
            print(f"{fold:<5} {len(keys):>5}  {backbone:<14} {digest:<12} "
                  f"{keys[0]:<22} {keys[-1]:<22} {lab}")
        print()

    total = sum(int((ref["fold"] == f).sum()) for f in range(n_folds))
    print(f"[OK] urutan kasus identik pada {len(BACKBONES)} backbone x {n_folds} "
          f"fold, {total} nodul. Rerata probabilitas boleh jalan.\n")


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def compute(n_folds: int = 5) -> pd.DataFrame:
    sha = _commit_sha()
    rows: list[dict] = []

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sizes = {b: track_input_size(cfg, b) for b in BACKBONES}
    # One scalar column only if the three genuinely share an input size, which they do
    # as Track 1 backbones. Writing one of three differing values under a singular name
    # is the kind of provenance that reads as complete and is not.
    assert len(set(sizes.values())) == 1, f"input_size tidak seragam: {sizes}"
    input_size = next(iter(sizes.values()))

    for condition, probs_dir, kind in REGIMES:
        loaded = {b: _probs(probs_dir, b) for b in BACKBONES}
        verify_alignment(loaded, n_folds, label=condition)
        provenance = {"input_size": input_size}
        provenance.update({f"probs_mtime_{b}": _mtime(_probs_path(probs_dir, b))
                           for b in BACKBONES})

        ref = loaded[BACKBONES[0]]
        y, fold = ref["y_true"], ref["fold"]

        # radiomics_only differs slightly between backbones because
        # mutual_info_classif runs without random_state; the strongest of the
        # three is the honest comparator, and all three are reported.
        rad = {b: loaded[b]["rad"] for b in BACKBONES}
        rad_best = max(rad, key=lambda b: _auc(y, rad[b]))

        for arm, template in ARMS.items():
            key = template.format(k=kind)
            single = {b: loaded[b][key] for b in BACKBONES}
            if any(np.isnan(v).any() for v in single.values()):
                raise ValueError(f"{condition}/{arm}: ada NaN di kunci {key}")
            ens = np.mean([single[b] for b in BACKBONES], axis=0)

            best_single = max(single, key=lambda b: _auc(y, single[b]))

            scopes = [(f"fold{f}", fold == f) for f in range(n_folds)]
            scopes.append(("pooled", np.ones_like(fold, dtype=bool)))
            for scope, mask in scopes:
                row = {
                    "run_id": RUN_ID,
                    "commit_sha": sha,
                    **provenance,
                    "condition": condition,
                    "arm": arm,
                    "scope": scope,
                    "n": int(mask.sum()),
                    "auc_ensemble": _auc(y[mask], ens[mask]),
                    "auc_radiomics_best": _auc(y[mask], rad[rad_best][mask]),
                    "best_single": best_single,
                    "radiomics_best": rad_best,
                }
                for b in BACKBONES:
                    row[f"auc_{b}"] = _auc(y[mask], single[b][mask])

                # DeLong on the pooled arrays only. A single fold holds ~270
                # nodules; the published tests are pooled, and mixing the two
                # scales in one column would invite reading a fold p as if it
                # carried the same power.
                if scope == "pooled":
                    _, p_single, d_single = delong_test(y, ens, single[best_single])
                    _, p_rad, d_rad = delong_test(y, ens, rad[rad_best])
                else:
                    p_single = d_single = p_rad = d_rad = float("nan")
                row.update({
                    "delta_vs_best_single": d_single, "p_vs_best_single": p_single,
                    "delta_vs_radiomics": d_rad, "p_vs_radiomics": p_rad,
                })
                rows.append(row)

    return pd.DataFrame(rows)


def _check(df: pd.DataFrame, n_folds: int = 5) -> None:
    expected = len(REGIMES) * len(ARMS) * (n_folds + 1)
    assert len(df) == expected, f"harus {expected} baris, dapat {len(df)}"
    auc_cols = ["auc_ensemble", "auc_radiomics_best"] + [f"auc_{b}" for b in BACKBONES]
    assert df[auc_cols].to_numpy().min() > 0.5, "ada AUC di bawah 0.5 -- cek penyejajaran"
    pooled = df[df["scope"] == "pooled"]
    assert pooled["p_vs_best_single"].notna().all(), "baris pooled kehilangan p DeLong"
    assert df[df["scope"] != "pooled"]["p_vs_best_single"].isna().all(), \
        "baris per fold tidak boleh membawa p DeLong"
    prov = ["input_size"] + [f"probs_mtime_{b}" for b in BACKBONES]
    missing = [c for c in prov if c not in df.columns]
    assert not missing, f"kolom provenance hilang: {missing}"
    assert (df[prov].astype(str) != "").all().all(), "ada kolom provenance kosong"


def self_check() -> None:
    """Prove the alignment guard fires. A guard that never fires is decoration."""
    n = 60
    rng = np.random.default_rng(0)
    base = {
        "y_true": np.repeat([0, 1], n // 2),
        "fold": np.tile(np.arange(5), n // 5),
        "patient_id": np.array([f"LIDC-IDRI-{i:04d}" for i in range(n)]),
        "nodule_idx": np.arange(n),
        "rad": rng.random(n),
    }
    loaded = {b: dict(base) for b in BACKBONES}
    verify_alignment(loaded, label="self-check, sejajar")

    # Swap two cases inside one fold of one backbone: same set, wrong order.
    bad = {k: v.copy() for k, v in base.items()}
    idx = np.flatnonzero(bad["fold"] == 0)[:2]
    bad["patient_id"][[idx[0], idx[1]]] = bad["patient_id"][[idx[1], idx[0]]]
    bad["nodule_idx"][[idx[0], idx[1]]] = bad["nodule_idx"][[idx[1], idx[0]]]
    loaded[BACKBONES[1]] = bad
    try:
        verify_alignment(loaded, label="self-check, sengaja tertukar")
    except AssertionError as e:
        print(f"[OK] penjaga menyala seperti seharusnya: {str(e).splitlines()[0]}")
    else:
        raise AssertionError("urutan tertukar lolos penjaga -- penjaganya tidak bekerja")

    print("[OK] self-check lolos")


def run(force: bool = False) -> None:
    if os.path.exists(OUT_CSV) and not force:
        print(f"[LEWAT] {OUT_CSV}")
        return

    df = compute()
    _check(df)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[SELESAI] {OUT_CSV}  ({len(df)} baris, commit {_commit_sha()})")

    for condition, _dir, _kind in REGIMES:
        print(f"\n--- {condition} ---")
        for arm in ARMS:
            g = df[(df["condition"] == condition) & (df["arm"] == arm)]
            per_fold = " ".join(f"{r['auc_ensemble']:.4f}"
                                for _, r in g[g["scope"] != "pooled"].iterrows())
            p = g[g["scope"] == "pooled"].iloc[0]
            print(arm)
            print(f"  AUC ensemble per fold : {per_fold}")
            print(f"  AUC ensemble pooled   : {p['auc_ensemble']:.4f}")
            print("  AUC backbone tunggal  : " + "  ".join(
                f"{b} {p['auc_' + b]:.4f}" for b in BACKBONES))
            print(f"  vs {p['best_single']} (terbaik) : "
                  f"delta {p['delta_vs_best_single']:+.4f}  p {p['p_vs_best_single']:.4g}")
            print(f"  vs radiomics_only ({p['radiomics_best']}, "
                  f"AUC {p['auc_radiomics_best']:.4f}) : "
                  f"delta {p['delta_vs_radiomics']:+.4f}  p {p['p_vs_radiomics']:.4g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="tulis ulang keluaran yang sudah ada")
    p.add_argument("--self-check", action="store_true", help="jalankan assert saja, tanpa menulis")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    run(force=args.force)


if __name__ == "__main__":
    main()
