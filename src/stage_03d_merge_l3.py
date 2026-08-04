"""Gabungkan hasil Langkah 3 ke CSV ablasi utama, lalu periksa keenam gate.

Kenapa tahap ini ada. `stage_03b_fusion` memanggil `summary.to_csv(out_csv)` yang
MENIMPA `ablation_summary.csv`, bukan menambah. Menjalankan dua kondisi Langkah 3
langsung terhadap `artifacts/results` akan menghapus 175 baris hasil Langkah 1,
termasuk seluruh baris `radiomics_only` yang justru jadi kontrol negatif gate G-2.
Karena itu kedua kondisi menulis ke direktori hasil sendiri, dan tahap ini yang
menyalin 60 baris arm baru ke CSV utama.

Dua mode:

    python -m src.stage_03d_merge_l3            # gabung lalu periksa
    python -m src.stage_03d_merge_l3 --check    # periksa saja, nol tulisan

Penggabungan bersifat menambah dan menolak duplikat. Baris `radiomics_only`,
`cnn_only`, `fusion_early`, `fusion_late` dan `fusion_intermediate` yang lama tidak
pernah disentuh.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

RUN_ID = "2026-08-04-run01"
MAIN_DIR = os.path.join("artifacts", "results", "fusion")

# Kondisi Langkah 3: nama -> direktori hasil, harus cocok dengan paths.results
# di configs/config_l3_plain.yaml dan configs/config_l3_reg.yaml.
CONDITIONS = {
    "plain": os.path.join("artifacts", "results", "l3_plain", "fusion"),
    "reg": os.path.join("artifacts", "results", "l3_reg", "fusion"),
}

BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
INPUT_SIZE = 96
N_FOLDS = 5

# Nama arm yang benar-benar ditulis kode adalah fusion_intermediate_branch_norm dan
# seterusnya, bukan branch_norm polos, jadi pencocokan pakai substring.
NEW_ARM_KEYS = ("branch_norm", "gmu")

# Ambang gate. Dikunci oleh handoff/GOAL.md, jangan diubah tanpa keputusan manusia.
G1_EXPECTED_ROWS = 60
G2_TARGET = 0.9318
G2_TOL = 0.0036
G3_EXPECTED_FS = "mutual_info_classif"

PROV_COLS = ["run_id", "input_size", "commit_sha", "condition"]


def _is_new_arm(arm) -> bool:
    return any(k in str(arm) for k in NEW_ARM_KEYS)


def _new_arm_mask(col: pd.Series) -> pd.Series:
    """Mask boolean untuk baris arm rebalanced.

    `astype(bool)` wajib. Pada kolom nol baris, `map` menghasilkan Series bertipe
    object, dan pandas memperlakukan Series non-boolean sebagai pemilihan KOLOM,
    bukan penyaringan baris, sehingga hasilnya frame tanpa kolom sama sekali.
    """
    return col.map(_is_new_arm).astype(bool)


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _read(path: str):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # Run yang terputus saat menulis meninggalkan berkas nol byte. Perlakukan
        # sama dengan belum ada, jangan menghantam penelusuran tumpukan.
        return None


def merge() -> None:
    """Salin baris arm baru dari tiap direktori kondisi ke CSV utama."""
    sha = _commit_sha()
    for name, arm_col in (("ablation_summary.csv", "arm"),
                          ("delong_fusion.csv", "fusion_arm")):
        main_path = os.path.join(MAIN_DIR, name)
        main = _read(main_path)
        if main is None:
            print(f"[SKIP] {main_path} belum ada")
            continue

        added = []
        for cond, src_dir in CONDITIONS.items():
            src = _read(os.path.join(src_dir, name))
            if src is None:
                print(f"[TUNGGU] kondisi {cond}: {os.path.join(src_dir, name)} belum ada")
                continue

            rows = src[_new_arm_mask(src[arm_col])].copy()
            if rows.empty:
                print(f"[TUNGGU] kondisi {cond}: nol baris arm baru di {name}")
                continue

            rows["run_id"] = RUN_ID
            rows["input_size"] = INPUT_SIZE
            rows["commit_sha"] = sha
            rows["condition"] = cond

            # Menambah, bukan menimpa. Baris yang sudah masuk dilewati, sehingga
            # menjalankan tahap ini dua kali tidak menggandakan apa pun.
            if "condition" in main.columns:
                already = set(main.loc[main["condition"] == cond, arm_col].unique())
                rows = rows[~rows[arm_col].isin(already)]
            if rows.empty:
                print(f"[LEWAT] kondisi {cond}: baris {name} sudah ada di CSV utama")
                continue

            added.append(rows)
            print(f"[TAMBAH] kondisi {cond}: {len(rows)} baris ke {name}")

        if added:
            pd.concat([main] + added, ignore_index=True).to_csv(main_path, index=False)
            print(f"[DONE] {main_path}")


def _gate(ok: bool, label: str, detail: str) -> bool:
    print(f"  {'LULUS' if ok else 'GAGAL'}  {label}: {detail}")
    return ok


def check() -> bool:
    """Periksa keenam gate handoff/GOAL.md. Cetak buktinya, jangan cuma putusannya."""
    abl = _read(os.path.join(MAIN_DIR, "ablation_summary.csv"))
    dl = _read(os.path.join(MAIN_DIR, "delong_fusion.csv"))
    if abl is None:
        print("ablation_summary.csv utama tidak ada")
        return False

    new = abl[_new_arm_mask(abl["arm"])]
    n = len(new)
    results = []

    # G-1 kelengkapan baris
    results.append(_gate(
        n == G1_EXPECTED_ROWS, "G-1 kelengkapan baris",
        f"{n} baris arm baru, diharapkan {G1_EXPECTED_ROWS} "
        f"(3 backbone x {N_FOLDS} fold x 2 arm x 2 kondisi)",
    ))

    # G-2 kontrol negatif, diperiksa pada CSV utama dan pada kedua run baru.
    # Yang di CSV utama seharusnya tidak bergerak sama sekali karena merge hanya
    # menambah baris arm baru; yang di run baru adalah kontrol yang sebenarnya.
    lo, hi = G2_TARGET - G2_TOL, G2_TARGET + G2_TOL
    main_mean = abl.loc[abl["arm"] == "radiomics_only", "auc"].mean()
    ok2 = bool(lo <= main_mean <= hi)
    print(f"  {'LULUS' if ok2 else 'GAGAL'}  G-2 kontrol negatif utama: "
          f"radiomics_only rerata {main_mean:.6f}, pita [{lo:.4f}, {hi:.4f}]")
    for cond, src_dir in CONDITIONS.items():
        src = _read(os.path.join(src_dir, "ablation_summary.csv"))
        if src is None:
            ok2 = False
            print(f"  GAGAL  G-2 kontrol negatif {cond}: hasil run belum ada")
            continue
        m = src.loc[src["arm"] == "radiomics_only", "auc"].mean()
        hit = bool(lo <= m <= hi)
        ok2 = ok2 and hit
        print(f"  {'LULUS' if hit else 'GAGAL'}  G-2 kontrol negatif {cond}: "
              f"radiomics_only rerata {m:.6f}, pita [{lo:.4f}, {hi:.4f}]")
    results.append(ok2)

    # G-3 fs_method
    vals = list(new["fs_method"].dropna().unique()) if n else []
    results.append(_gate(
        n > 0 and int(new["fs_method"].isna().sum()) == 0 and vals == [G3_EXPECTED_FS],
        "G-3 fs_method",
        f"{int(new['fs_method'].isna().sum()) if n else '-'} kosong, nilai unik {vals}",
    ))

    # G-4 kondisi benar-benar berbeda. AUC identik pada backbone dan fold yang sama
    # berarti arm atau regularizer tidak benar-benar terpasang.
    dupes = int(new.groupby(["backbone", "fold"])["auc"]
                   .apply(lambda s: len(s) - s.nunique()).sum()) if n else -1
    results.append(_gate(
        dupes == 0, "G-4 kondisi berbeda",
        f"{dupes} pasang AUC identik pada backbone dan fold yang sama" if dupes >= 0
        else "nol baris arm baru untuk diperiksa",
    ))

    # G-5 DeLong
    if dl is None or "fusion_arm" not in dl.columns:
        results.append(_gate(False, "G-5 DeLong", "delong_fusion.csv tidak ada"))
    else:
        dnew = dl[_new_arm_mask(dl["fusion_arm"])]
        expected = len(BACKBONES) * len(NEW_ARM_KEYS) * len(CONDITIONS)
        better = dnew[dnew["fusion_auc"] > dnew["best_single_auc"]]
        sig_better = better[better["delong_p"] < 0.05]
        worse = dnew[dnew["fusion_auc"] <= dnew["best_single_auc"]]
        sig_worse = worse[worse["delong_p"] < 0.05]
        tie = len(dnew) - len(sig_better) - len(sig_worse)
        results.append(_gate(
            len(dnew) == expected and int(dnew["delong_p"].isna().sum()) == 0,
            "G-5 DeLong",
            f"{len(dnew)} uji, diharapkan {expected}. "
            f"signifikan lebih baik {len(sig_better)}, seri {tie}, "
            f"signifikan lebih buruk {len(sig_worse)}. "
            f"pembanding {sorted(dnew['best_single_arm'].unique())}",
        ))

    # G-6 provenance
    missing = [c for c in PROV_COLS if c not in abl.columns]
    blank = int(new[PROV_COLS].isna().sum().sum()) if n and not missing else -1
    results.append(_gate(
        not missing and blank == 0, "G-6 provenance",
        f"kolom hilang {missing}, sel kosong pada baris baru {blank}",
    ))

    ok = all(results)
    print()
    print("Keenam gate lulus." if ok else f"{sum(results)}/{len(results)} gate lulus.")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                    help="periksa gate saja, jangan menulis apa pun")
    args = p.parse_args()

    if not args.check:
        merge()
        print()
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
