"""Uji ekuivalensi (TOST) fusion_late lawan radiomics_only pada margin AUC 0.02.

Manuskrip Track 1 selama ini menyatakan kesetaraan dengan radiomics lewat DeLong
yang gagal menolak H0. Itu tidak sah: tidak ada bukti perbedaan bukan bukti tidak
ada perbedaan, dan reviewer akan menyerang kalimat itu lebih dulu. Yang dibutuhkan
klaim kesetaraan adalah uji yang hipotesis nolnya justru "keduanya berbeda jauh",
lalu ditolak. Itulah TOST.

Margin ditetapkan 0.02 AUC atas dasar klinis, SEBELUM analisis ini dijalankan:
selisih sebesar itu tidak mengubah keputusan tindak lanjut nodul, dan berada di
bawah goyangan yang lazim muncul dari pilihan sampel dan label pada LIDC. Margin
TIDAK dipilih karena interval kebetulan muat di dalamnya -- memilih margin dengan
cara itu membuat ujinya sekadar formalitas.

Cakupannya sengaja mencakup dua perbandingan, bukan satu:

  * `fusion_late` vs `radiomics_only` -- pertanyaan sebenarnya. Di sini ekuivalensi
    yang dicari.
  * `fusion_late` vs `cnn_only` -- kontrol negatif. DeLong sudah menyatakannya beda
    jauh (p < 1e-9), jadi TOST wajib menolak ekuivalensi di keenam barisnya. Kalau
    ada satu saja yang lolos, ujinya yang salah, bukan datanya.

Sumbernya artifacts/results/run02/probs/*.npz, probabilitas yang sudah dihitung
run02. Nol pelatihan ulang, nol GPU, nol eksperimen baru. Varians yang dipakai
persis varians DeLong yang sama (src/evaluation/statistical_tests.auc_diff_variance),
dan --self-check membuktikan itu dengan menurunkan ulang p DeLong dari sisi TOST.

Keluaran:
  artifacts/results/run02/tost_run02.csv
  artifacts/results/run02/fig16_tost_equivalence.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.evaluation.statistical_tests import delong_test, tost_auc
from src.stage_08b_run02_xai import OUT_DIR, RUN_ID as SOURCE_RUN_ID, _commit_sha

# Uji ini dijalankan belakangan dan memakai ulang probabilitas run02. Dua run id
# yang berbeda, dan mencampurnya akan membuat provenance-nya bohong.
RUN_ID = "2026-08-19-run03"

PROBS_DIR = os.path.join(OUT_DIR, "probs")
DELONG_CSV = os.path.join(OUT_DIR, "delong_run02.csv")
OUT_CSV = os.path.join(OUT_DIR, "tost_run02.csv")
OUT_PNG = os.path.join(OUT_DIR, "fig16_tost_equivalence.png")

MARGIN = 0.02
ALPHA = 0.05

# Urutan bawah-ke-atas pada sumbu y, sama dengan Fig 15 supaya kedua figure bisa
# dibaca berdampingan tanpa memetakan ulang barisnya.
BACKBONES = ["densenet121", "convnext_tiny", "densenet201"]
LABELS = {
    "densenet201": "DenseNet201  (model utama)",
    "convnext_tiny": "ConvNeXt-Tiny",
    "densenet121": "DenseNet121",
}
# (nama perbandingan, label, arm A, arm B, kunci npz A, kunci npz B)
COMPARISONS = [
    ("fusion_late_vs_radiomics_only", "fusion_late  vs  radiomics_only",
     "fusion_late", "radiomics_only", "late_{k}", "rad"),
    ("fusion_late_vs_cnn_only", "fusion_late  vs  cnn_only",
     "fusion_late", "cnn_only", "late_{k}", "cnn_{k}"),
]
CKPT_KINDS = ["best", "last"]

C_OK = "#2b6cb0"    # ekuivalen dalam margin
C_BAD = "#c53030"   # tidak ekuivalen
C_BAND = "#68d391"  # pita margin


def _probs(backbone: str, probs_dir: str = PROBS_DIR) -> dict:
    path = os.path.join(probs_dir, f"{backbone}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} belum ada. Jalankan: python -m src.stage_08a_run02_probs")
    return dict(np.load(path, allow_pickle=True))


def compute(margin: float = MARGIN, alpha: float = ALPHA,
            probs_dir: str = PROBS_DIR, kinds: list[str] | None = None,
            delong_csv: str | None = DELONG_CSV,
            run_id: str = RUN_ID, source_run_id: str = SOURCE_RUN_ID) -> pd.DataFrame:
    """Satu baris per (backbone, rezim checkpoint, perbandingan) -- 12 baris.

    The five keyword arguments exist for run03 F-2, which asks the same
    equivalence question of the nested-CV probabilities. Their defaults
    reproduce the published run02 table exactly. run03 saves no unselected
    final epoch, so it passes ``kinds=["best"]`` and gets 6 rows, and it
    passes ``delong_csv=None`` so the DeLong column is recomputed through the
    same variance path rather than looked up in run02's table.
    """
    kinds = kinds or CKPT_KINDS
    delong = (pd.read_csv(delong_csv)
              if delong_csv and os.path.exists(delong_csv) else None)
    sha = _commit_sha()
    rows = []

    for backbone in BACKBONES:
        d = _probs(backbone, probs_dir)
        y = d["y_true"]
        for comp, _label, arm_a, arm_b, key_a, key_b in COMPARISONS:
            for k in kinds:
                a = d[key_a.format(k=k)]
                b = d[key_b.format(k=k)]
                r = tost_auc(y, a, b, margin=margin, alpha=alpha)

                # p DeLong ikut dibawa supaya kedua uji terbaca berdampingan.
                # Diambil dari CSV run02 kalau ada, dihitung ulang kalau tidak --
                # keduanya memakai jalur varians yang sama, jadi nilainya sama.
                if delong is not None:
                    hit = delong[(delong["backbone"] == backbone)
                                 & (delong["ckpt_kind"] == k)
                                 & (delong["comparison"] == comp)]
                    p_delong = float(hit["delong_p"].iloc[0]) if len(hit) else float("nan")
                else:
                    _, p_delong, _ = delong_test(y, a, b)

                rows.append({
                    "run_id": run_id,
                    "source_run_id": source_run_id,
                    "commit_sha": sha,
                    "backbone": backbone,
                    "ckpt_kind": k,
                    "comparison": comp,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "auc_a": r["auc_a"],
                    "auc_b": r["auc_b"],
                    "delta": r["delta"],
                    "se": r["se"],
                    "margin": r["margin"],
                    "ci90_low": r["ci90_low"],
                    "ci90_high": r["ci90_high"],
                    "ci95_low": r["ci95_low"],
                    "ci95_high": r["ci95_high"],
                    "p_lower": r["p_lower"],
                    "p_upper": r["p_upper"],
                    "p_tost": r["p_tost"],
                    "equivalent": r["equivalent"],
                    "noninferior": r["noninferior"],
                    "delong_p": p_delong,
                })
    return pd.DataFrame(rows)


def _check(df: pd.DataFrame, n_expected: int = 12) -> None:
    """Gerbang murah supaya CSV cacat tidak pernah terbit.

    Kegagalan senyap di proyek ini (sec 8.7 laporan) selalu berbentuk keluaran yang
    sah bentuknya tapi kosong atau tak lengkap isinya.
    """
    assert len(df) == n_expected, \
        f"harus {n_expected} baris (3 backbone x rezim x 2 perbandingan), dapat {len(df)}"
    assert df["margin"].eq(MARGIN).all(), "ada baris dengan margin berbeda"
    assert df["se"].gt(0).all(), "ada se nol -- prediktor identik atau kelas terlalu sedikit"
    assert df["p_tost"].between(0, 1).all(), "ada p_tost di luar [0, 1]"
    # Verdict dan interval adalah pernyataan yang sama persis; kalau berbeda,
    # salah satunya salah hitung.
    inside = (df["ci90_low"] > -MARGIN) & (df["ci90_high"] < MARGIN)
    assert inside.eq(df["equivalent"]).all(), \
        "verdict equivalent tidak cocok dengan CI 90 persen"


def self_check() -> None:
    """Empat assert, tanpa framework. tests/ tidak boleh disentuh sesi ini."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 300)
    p = np.clip(y * 0.35 + rng.normal(0.3, 0.15, size=y.size), 0.001, 0.999)

    # 1. Prediktor identik: delta nol, ekuivalen pada margin positif berapa pun.
    r = tost_auc(y, p, p, margin=0.02)
    assert r["delta"] == 0.0 and r["equivalent"], f"prediktor identik harus ekuivalen: {r}"

    # 2. Prediktor yang jelas berbeda: tidak ekuivalen pada 0.02.
    junk = rng.random(y.size)
    r = tost_auc(y, p, junk, margin=0.02)
    assert not r["equivalent"], f"prediktor acak tidak boleh dinyatakan setara: {r}"

    # 3. p_tost adalah yang terbesar dari dua uji satu arah.
    assert abs(r["p_tost"] - max(r["p_lower"], r["p_upper"])) < 1e-15

    # 4. Pemeriksaan silang terpenting, pada data nyata: di margin 0 kedua uji satu
    #    arah runtuh jadi uji dua arah yang sama dengan DeLong. Kalau angkanya
    #    meleset, TOST memakai varians yang berbeda dan seluruh tabel tidak sah.
    if os.path.isdir(PROBS_DIR):
        n = 0
        for backbone in BACKBONES:
            d = _probs(backbone)
            yt = d["y_true"]
            for comp, _l, _a, _b, key_a, key_b in COMPARISONS:
                for k in CKPT_KINDS:
                    a, b = d[key_a.format(k=k)], d[key_b.format(k=k)]
                    z = tost_auc(yt, a, b, margin=0.0)
                    recovered = 2 * min(z["p_lower"], z["p_upper"])
                    _, p_delong, _ = delong_test(yt, a, b)
                    assert abs(recovered - p_delong) < 1e-12, (
                        f"{backbone}/{k}/{comp}: TOST menurunkan p={recovered:.6e}, "
                        f"DeLong p={p_delong:.6e} -- varians tidak sama")
                    n += 1
        assert n == 12, f"pemeriksaan silang harus menutupi 12 baris, dapat {n}"
        print(f"[OK] pemeriksaan silang varians lolos untuk {n} baris")
    else:
        print(f"[LEWAT] {PROBS_DIR} tidak ada, pemeriksaan silang data nyata dilewati")
    print("[OK] self-check lolos")


def _plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    yticks, ylabels, y = [], [], 0.0

    ax.axvspan(-MARGIN, MARGIN, color=C_BAND, alpha=0.18, zorder=0)
    ax.axvline(0, color="#a0aec0", linewidth=1.0, zorder=1)
    for edge in (-MARGIN, MARGIN):
        ax.axvline(edge, color="#38a169", linestyle="--", linewidth=1.2, zorder=1)

    for ci, (comp, comp_label, _a, _b, _ka, _kb) in enumerate(COMPARISONS):
        if ci:
            y += 1.1
        for bb in BACKBONES:
            for k in CKPT_KINDS:
                row = df[(df["comparison"] == comp) & (df["backbone"] == bb)
                         & (df["ckpt_kind"] == k)].iloc[0]
                color = C_OK if row["equivalent"] else C_BAD
                ax.plot([row["ci90_low"], row["ci90_high"]], [y, y],
                        color=color, linewidth=2.0, solid_capstyle="butt", zorder=3)
                ax.scatter(row["delta"], y, s=54,
                           marker="o" if k == "best" else "D",
                           color=color, edgecolors="white", linewidths=0.8, zorder=4)
                yticks.append(y)
                ylabels.append(f"{LABELS[bb]}  —  {k}")
                y += 1.0
        # judul blok di atas bloknya, bukan di tengah, supaya tidak menimpa interval
        ax.text(-0.083, y - 0.55, comp_label, fontsize=10, weight="bold",
                va="center", ha="left", color="#2d3748")

    n_eq = int(df["equivalent"].sum())
    ax.set_xlim(-0.085, 0.055)
    ax.set_ylim(-0.9, y + 0.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("selisih AUC (arm A dikurangi arm B), dengan interval kepercayaan 90 persen. "
                  "Ekuivalen bila seluruh interval berada di dalam pita", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.set_title("Fig 16. Ekuivalensi dengan radiomics dalam margin 0.02 AUC, "
                 "dan kontrol negatif yang gagal seperti seharusnya",
                 fontsize=12.5, weight="bold", loc="left", pad=30)
    ax.text(0, 1.035,
            "TOST pada selisih AUC berpasangan, varians DeLong dipool 5 fold; margin "
            f"0.02 ditetapkan atas dasar klinis sebelum analisis (run {RUN_ID}, "
            f"probabilitas {SOURCE_RUN_ID})",
            transform=ax.transAxes, fontsize=8.5, color="#555555")

    # Warna membawa vonis, bentuk membawa rezim -- dua sumbu terpisah, jadi
    # legend-nya juga dipisah. Menggabungkannya akan menyiratkan bahwa belah
    # ketupat selalu berarti tidak ekuivalen, padahal blok bawah memuat belah
    # ketupat biru. Ditaruh di bawah sumbu supaya tidak pernah menimpa interval.
    ax.legend(handles=[
        Patch(facecolor=C_BAND, alpha=0.35, label="pita margin ±0.02 AUC"),
        Line2D([], [], marker="s", linestyle="", color=C_OK, markersize=8,
               label="ekuivalen dalam margin"),
        Line2D([], [], marker="s", linestyle="", color=C_BAD, markersize=8,
               label="tidak ekuivalen"),
        Line2D([], [], marker="o", linestyle="", color="#4a5568", markersize=7,
               label="rezim best (dengan seleksi)"),
        Line2D([], [], marker="D", linestyle="", color="#4a5568", markersize=7,
               label="rezim last (tanpa seleksi)"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3,
        fontsize=8, framealpha=0.95, borderaxespad=0.0)

    fig.text(0.012, -0.05,
             "Blok bawah menjawab pertanyaan sebenarnya: kesetaraan dengan radiomics_only. "
             "Blok atas adalah kontrol negatif — DeLong sudah menyatakan fusion_late jauh "
             "mengungguli cnn_only,\nsehingga TOST wajib menolak ekuivalensi di keenam "
             "barisnya. Lolosnya satu baris di sana berarti ujinya yang salah, bukan datanya.",
             fontsize=8.2, color="#333333", va="top")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SELESAI] {OUT_PNG}  (ekuivalen {n_eq}/12)")


def run(force: bool = False, probs_dir: str = PROBS_DIR, out_csv: str = OUT_CSV,
        kinds: list[str] | None = None, run_id: str = RUN_ID,
        source_run_id: str = SOURCE_RUN_ID, plot: bool = True) -> None:
    if os.path.exists(out_csv) and not force:
        print(f"[LEWAT] {out_csv}")
        return

    kinds = kinds or CKPT_KINDS
    df = compute(probs_dir=probs_dir, kinds=kinds, run_id=run_id,
                 source_run_id=source_run_id,
                 delong_csv=DELONG_CSV if probs_dir == PROBS_DIR else None)
    _check(df, n_expected=len(BACKBONES) * len(kinds) * len(COMPARISONS))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SELESAI] {out_csv}  ({len(df)} baris, margin {MARGIN}, "
          f"commit {_commit_sha()})")

    for comp, label, *_ in COMPARISONS:
        g = df[df["comparison"] == comp]
        print(f"  {label}: ekuivalen {int(g['equivalent'].sum())}/{len(g)}, "
              f"non-inferior {int(g['noninferior'].sum())}/{len(g)}")
        for _, r in g.iterrows():
            print(f"    {r['backbone']:<14} {r['ckpt_kind']:<5} "
                  f"delta {r['delta']:+.4f}  CI90 ({r['ci90_low']:+.4f}, "
                  f"{r['ci90_high']:+.4f})  p_tost {r['p_tost']:.4g}  "
                  f"delong {r['delong_p']:.4g}  "
                  f"{'EKUIVALEN' if r['equivalent'] else 'TIDAK'}")

    # Fig 16 is laid out for the two run02 regimes; a 6-row table would need a
    # different figure, and F-2 reports its numbers as a table.
    if plot:
        _plot(df)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="tulis ulang keluaran yang sudah ada")
    p.add_argument("--self-check", action="store_true", help="jalankan assert saja, tanpa menulis")
    p.add_argument("--probs-dir", default=PROBS_DIR)
    p.add_argument("--out-csv", default=OUT_CSV)
    p.add_argument("--kinds", nargs="+", default=None, choices=CKPT_KINDS)
    p.add_argument("--run-id", default=RUN_ID)
    p.add_argument("--source-run-id", default=SOURCE_RUN_ID)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    run(force=args.force, probs_dir=args.probs_dir, out_csv=args.out_csv,
        kinds=args.kinds, run_id=args.run_id, source_run_id=args.source_run_id,
        plot=not args.no_plot)


if __name__ == "__main__":
    main()
