"""Fig 15 run02: p-value DeLong kedua perbandingan, kedua rezim checkpoint, satu sumbu.

Ini versi tergambar dari klaim inti Track 1 (docs/laporan/LAPORAN_TRACK1_FUSION_XAI.md
sec 6.3.3) beserta batasannya (sec 8.9). Keduanya sekarang hanya berupa tabel angka,
padahal yang membedakan keduanya adalah pola posisi, bukan nilai:

  * `fusion_late` vs `cnn_only` -- enam titik (3 backbone x 2 rezim) menumpuk jauh di
    kiri garis alpha. Rezim checkpoint tidak menggeser apa pun yang penting, karena
    kedua arm memakai checkpoint CNN yang sama sehingga keuntungan seleksi masuk ke
    dua sisi perbandingan dan saling meniadakan. Kemenangan tanpa syarat.
  * `fusion_late` vs `radiomics_only` -- titik rezim `best` semuanya di kanan garis
    (setara), tetapi dua di antaranya menyeberang ke kiri pada rezim `last`
    (ConvNeXt-Tiny 0.0246, DenseNet121 0.0144), yang berarti signifikan LEBIH BURUK.
    Hanya DenseNet201 bertahan di kanan pada kedua rezim, dan itulah alasan tunggal
    ia dipilih sebagai model utama (sec 6.3.2).

Panah antara pasangan best/last digambar supaya arah pergeseran akibat pencabutan
keuntungan seleksi terbaca langsung, bukan disimpulkan dari dua angka berjauhan.

Sumbernya satu berkas, artifacts/results/run02/delong_run02.csv, tanpa hitung ulang
apa pun. Skrip ini murni memplot; nol eksperimen baru.

Keluaran: artifacts/results/run02/fig15_delong_pvalues.png
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.stage_08b_run02_xai import OUT_DIR, RUN_ID, _commit_sha

DELONG_CSV = os.path.join(OUT_DIR, "delong_run02.csv")
OUT_PNG = os.path.join(OUT_DIR, "fig15_delong_pvalues.png")

ALPHA = 0.05
# Urutan bawah-ke-atas pada sumbu y; DenseNet201 paling atas di tiap blok karena
# ia model utama Track 1.
BACKBONES = ["densenet121", "convnext_tiny", "densenet201"]
LABELS = {
    "densenet201": "DenseNet201  (model utama)",
    "convnext_tiny": "ConvNeXt-Tiny",
    "densenet121": "DenseNet121",
}
COMPARISONS = [
    ("fusion_late_vs_radiomics_only", "fusion_late  vs  radiomics_only"),
    ("fusion_late_vs_cnn_only", "fusion_late  vs  cnn_only"),
]
C_BEST = "#2b6cb0"   # rezim best -- dengan keuntungan seleksi checkpoint
C_LAST = "#c05621"   # rezim last -- tanpa seleksi
C_BAD = "#c53030"    # sorotan untuk yang menyeberang jadi signifikan lebih buruk


def _check(df: pd.DataFrame) -> None:
    """Gerbang murah supaya figure tidak pernah terbit dari tabel yang cacat.

    Kegagalan senyap berulang di proyek ini (sec 8.7 laporan) selalu berbentuk
    keluaran yang sah bentuknya tapi kosong atau tak lengkap isinya. Empat assert
    ini menutup bentuk itu untuk figure ini.
    """
    assert len(df) == 12, f"delong_run02.csv harus 12 baris, dapat {len(df)}"
    assert df["delong_p"].between(0, 1, inclusive="right").all(), \
        "ada delong_p di luar (0, 1]"
    assert set(df["ckpt_kind"]) == {"best", "last"}, \
        f"rezim checkpoint tak terduga: {sorted(set(df['ckpt_kind']))}"
    assert set(df["backbone"]) == set(BACKBONES), \
        f"backbone tak terduga: {sorted(set(df['backbone']))}"


def _load() -> pd.DataFrame:
    if not os.path.exists(DELONG_CSV):
        raise FileNotFoundError(
            f"{DELONG_CSV} belum ada. Jalankan: python -m src.stage_08a_run02_probs")
    df = pd.read_csv(DELONG_CSV)
    _check(df)
    return df


def run(force: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if os.path.exists(OUT_PNG) and not force:
        print(f"[LEWAT] {OUT_PNG}")
        return

    df = _load()
    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    yticks, ylabels, y = [], [], 0.0
    n_crossed = 0
    for ci, (comp, comp_label) in enumerate(COMPARISONS):
        if ci:
            y += 0.9  # jarak antar blok perbandingan
        block_start = y
        for bb in BACKBONES:
            rows = df[(df["comparison"] == comp) & (df["backbone"] == bb)]
            p_best = float(rows[rows["ckpt_kind"] == "best"]["delong_p"].iloc[0])
            p_last = float(rows[rows["ckpt_kind"] == "last"]["delong_p"].iloc[0])

            # panah best -> last: arah pergeseran saat keuntungan seleksi dicabut
            ax.annotate("", xy=(p_last, y), xytext=(p_best, y),
                        arrowprops=dict(arrowstyle="-|>", color="#a0aec0",
                                        linewidth=1.1, shrinkA=4, shrinkB=4))
            ax.scatter(p_best, y, s=78, color=C_BEST, zorder=3,
                       edgecolors="white", linewidths=0.8)
            crossed = (p_last < ALPHA) and (p_best >= ALPHA)
            n_crossed += int(crossed)
            ax.scatter(p_last, y, s=90, marker="D", zorder=3,
                       color=C_BAD if crossed else C_LAST,
                       edgecolors="white", linewidths=0.8)
            if crossed:
                ax.annotate(f"p={p_last:.4f}", xy=(p_last, y), xytext=(0, -17),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=C_BAD, weight="bold")
            yticks.append(y)
            ylabels.append(LABELS[bb])
            y += 1.0

        # judul blok di atas bloknya, bukan di tengah, supaya tidak menimpa titik
        ax.text(1.4e-16, y - 0.62, comp_label, fontsize=10,
                weight="bold", va="center", ha="left", color="#2d3748")
        del block_start

    ax.axvline(ALPHA, color=C_BAD, linestyle="--", linewidth=1.3, zorder=1)
    # label di ujung atas garis: satu-satunya ruang yang tidak ditempati anotasi p
    ax.text(ALPHA * 0.82, y - 0.35, "alpha = 0.05", rotation=90, fontsize=8,
            color=C_BAD, ha="right", va="top")

    ax.set_xscale("log")
    ax.set_xlim(1e-16, 3.0)
    ax.set_ylim(-0.95, y - 0.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("p-value DeLong (skala log). Kiri garis = beda signifikan; "
                  "kanan garis = tidak terbedakan", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.set_title("Fig 15. Kemenangan atas cnn_only tidak bersyarat; kesetaraan dengan "
                 "radiomics bersyarat", fontsize=12.5, weight="bold", loc="left", pad=30)
    ax.text(0, 1.035,
            f"DeLong dipool 5 fold, run {RUN_ID}; pangkal panah = checkpoint best "
            "(dengan seleksi), ujung panah = last (tanpa seleksi)",
            transform=ax.transAxes, fontsize=8.5, color="#555555")

    # kiri-bawah: satu-satunya sudut yang kosong pada kedua blok
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=C_BEST, markersize=8,
               label="rezim best (dengan keuntungan seleksi)"),
        Line2D([], [], marker="D", linestyle="", color=C_LAST, markersize=8,
               label="rezim last (tanpa seleksi)"),
        Line2D([], [], marker="D", linestyle="", color=C_BAD, markersize=8,
               label="menyeberang jadi signifikan lebih buruk"),
    ], loc="lower left", fontsize=8, framealpha=0.95)

    fig.text(0.012, 0.0,
             "Blok atas: enam titik di kiri garis pada kedua rezim. Kedua arm berbagi "
             "checkpoint CNN yang sama, jadi keuntungan seleksi masuk ke dua sisi "
             "perbandingan dan meniadakan diri.\n"
             "Blok bawah sensitif justru karena radiomics_only tidak pernah memakai "
             "eval_set, sehingga ia satu-satunya pembanding yang tidak bergerak antar "
             "rezim. Dua backbone menyeberang; DenseNet201 tidak.",
             fontsize=8.2, color="#333333", va="top")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SELESAI] {OUT_PNG}  (run {RUN_ID}, commit {_commit_sha()}, "
          f"menyeberang={n_crossed})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="tulis ulang figure yang sudah ada")
    args = p.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
