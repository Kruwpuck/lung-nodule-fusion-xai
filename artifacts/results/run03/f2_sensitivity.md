# F-2 — sensitivitas hasil fusi terhadap protokol checkpoint CNN

`run_id` 2026-08-22-run03 · `condition` `honest_nested_cv_sensitivity` ·
sumber `run03/fusion/ablation_summary.csv` (75 baris), `run03/fusion/delong_fusion.csv`
(9 baris), `run03/probs/{backbone}.npz`

**Ini analisis sensitivitas, bukan pengganti.** Tabel Track 1 yang terbit tidak
bergerak. Yang dijawab di sini satu pertanyaan yang tabel itu tidak bisa jawab:
seberapa besar hasil fusi bergantung pada cabang CNN yang checkpoint-nya dipilih di
fold yang sama yang kemudian dipakai menilainya.

Checkpoint pembanding: sel run03 unfreeze 100%, pemenang inner-val pada ketiga
backbone (`f1_summary.md`). Fine-tuning **tidak diadopsi** — nol varian kandidat
mengalahkan kontrol; keempat varian dilaporkan di `f1_summary.md` §1.

---

## 1. Kontrol negatif — lolos, dan lebih kuat dari yang gate minta

Gate pra-registrasi §5 butir 4: `radiomics_only` harus tetap dalam ±0,0036 dari
0,9318, yaitu pita [0,9282; 0,9354].

| backbone | pooled `radiomics_only` | simpangan | rerata-fold | putusan |
|---|---|---|---|---|
| convnext_tiny | 0,9336 | +0,0018 | 0,9343 | lolos |
| densenet201 | 0,9336 | +0,0018 | 0,9343 | lolos |
| densenet121 | 0,9336 | +0,0018 | 0,9343 | lolos |

Lebih dari sekadar lolos: nilai pooled-nya **bit-identik dengan run02** (delta
+0,0000 di ketiga backbone). Itu konfirmasi langsung asimetri yang jadi alasan
analisis ini ada — `radiomics_only` adalah XGBoost tanpa `eval_set`, jadi ia tidak
pernah menikmati seleksi checkpoint dan tidak bergerak sedikit pun ketika cabang CNN
diganti. Setiap pergerakan di bawah ini karena itu murni milik sisi CNN.

---

## 2. Lima arm, rerata-fold AUC

| backbone | arm | terbit | jujur | Δ |
|---|---|---|---|---|
| convnext_tiny | cnn_only | 0,9091 | 0,8455 | −0,0635 |
| | fusion_early | 0,9136 | 0,9096 | −0,0040 |
| | fusion_intermediate | 0,9263 | 0,9172 | −0,0091 |
| | fusion_late | 0,9351 | 0,9201 | −0,0150 |
| | radiomics_only | 0,9322 | 0,9343 | +0,0020 |
| densenet201 | cnn_only | 0,9023 | 0,8420 | −0,0603 |
| | fusion_early | 0,9089 | 0,9117 | +0,0029 |
| | fusion_intermediate | 0,9051 | 0,9131 | +0,0079 |
| | fusion_late | 0,9368 | 0,9188 | −0,0180 |
| | radiomics_only | 0,9301 | 0,9343 | +0,0042 |
| densenet121 | cnn_only | 0,8989 | 0,8649 | −0,0340 |
| | fusion_early | 0,9155 | 0,9237 | +0,0083 |
| | fusion_intermediate | 0,9228 | 0,9219 | −0,0009 |
| | fusion_late | 0,9340 | 0,9269 | −0,0070 |
| | radiomics_only | 0,9343 | 0,9343 | +0,0000 |

**Pola yang paling informatif: arm fusi terlatih hampir tidak bergerak.**
`fusion_early` dan `fusion_intermediate` bergeser −0,0091 sampai +0,0083, bahkan
**naik** di empat dari enam sel, meski cabang CNN-nya kehilangan 3,4–6,4 poin.
Keduanya melatih ulang di atas cabang CNN — XGBoost di atas embedding, dan FusionNet
ujung-ke-ujung — sehingga cabang CNN yang lebih lemah sebagian besar dikompensasi
sisi radiomiknya.

`fusion_late` sebaliknya bergerak −0,0070 sampai −0,0180, kira-kira **separuh**
penurunan `cnn_only`-nya. Itu bukan temuan empiris melainkan aritmetika:
`average_fusion(cnn, rad, weight_cnn=0.5)` merata-rata dua probabilitas jadi, sisi
radiomiknya beku, jadi ia mewarisi persis setengah pergerakan sisi CNN.

---

## 3. DeLong tiap arm fusi lawan `radiomics_only`, checkpoint jujur

| backbone | arm | AUC fusi | AUC radiomics | p | putusan |
|---|---|---|---|---|---|
| convnext_tiny | fusion_early | 0,9082 | 0,9336 | 0,0000 | radiomics menang |
| convnext_tiny | fusion_intermediate | 0,9129 | 0,9336 | 0,0003 | radiomics menang |
| convnext_tiny | fusion_late | 0,9195 | 0,9336 | **0,0013** | radiomics menang |
| densenet201 | fusion_early | 0,9104 | 0,9336 | 0,0001 | radiomics menang |
| densenet201 | fusion_intermediate | 0,9095 | 0,9336 | 0,0007 | radiomics menang |
| densenet201 | fusion_late | 0,9171 | 0,9336 | **0,0003** | radiomics menang |
| densenet121 | fusion_early | 0,9219 | 0,9336 | 0,0437 | radiomics menang |
| densenet121 | fusion_intermediate | 0,9190 | 0,9336 | 0,0221 | radiomics menang |
| densenet121 | fusion_late | 0,9279 | 0,9336 | 0,1558 | tidak signifikan |

**`radiomics_only` mengungguli signifikan di 8 dari 9 sel.** Nol arm fusi yang
mengalahkan radiomics di sel mana pun.

---

## 4. Dua pertanyaan yang diminta, dijawab

### 4a. `fusion_late` masih setara `radiomics_only`? **Tidak.**

| backbone | p, checkpoint terbit | p, checkpoint jujur |
|---|---|---|
| convnext_tiny | 0,4555 | **0,0013** |
| densenet201 | 0,4901 | **0,0003** |
| densenet121 | 0,6793 | 0,1558 |

Kolom kiri dari `run02/t0_checkpoint_sensitivity.csv`, kolom `p_late_vs_rad_best`.

Di bawah checkpoint terbit, `fusion_late` tidak terpisah signifikan dari radiomics di
ketiga backbone — itu dasar klaim kesetaraan. Di bawah protokol jujur klaim itu
**runtuh di dua dari tiga backbone** dan berubah jadi kekalahan signifikan. Hanya
densenet121 yang bertahan, dan itu backbone yang denda protokolnya paling kecil
(−0,0340).

### 4b. `fusion_late` masih mengalahkan `cnn_only` signifikan? **Ya, ketiganya.**

| backbone | Δ AUC | p |
|---|---|---|
| convnext_tiny | +0,0728 | <0,0001 |
| densenet201 | +0,0811 | <0,0001 |
| densenet121 | +0,0659 | <0,0001 |

Bertahan, dan keunggulannya **melebar**. Tapi ini bukan bukti independen. Dugaan
"keduanya turun bersama dan keuntungannya saling meniadakan" tidak berlaku justru
karena bentuk `average_fusion`: hanya sisi CNN yang turun, sisi radiomiknya beku,
jadi `fusion_late` turun setengah dari `cnn_only` dan jaraknya otomatis melebar.
Klaim ini bertahan karena aritmetika rata-rata, bukan karena diuji ulang secara
independen.

---

## 5. Arah kesimpulan: menguat, tidak berbalik

Klaim utama Track 1 — radiomics setara atau mengungguli fusi pada n≈1400 — **lebih
kuat** di bawah protokol jujur, bukan lebih lemah. Yang terbit: `fusion_late` setara
radiomics, arm fusi lain kalah. Yang jujur: **seluruh** arm fusi kalah, delapan dari
sembilan secara signifikan.

Yang runtuh adalah satu klaim yang lebih sempit dan lebih menguntungkan fusi —
kesetaraan `fusion_late` — dan ia runtuh persis karena bersandar pada cabang CNN yang
menikmati seleksi fold luar.

---

## 6. Batas yang tidak boleh dilewati saat mengutip ini

**`cnn_only` run03 adalah batas bawah untuk angka jujur, bukan angkanya.** Checkpoint
run03 berbeda dari resep terbit dalam **empat** hal sekaligus (`f1_summary.md` §2b):
seleksi epoch pindah ke inner-val, data latih berkurang 15%, jadwal dua tahap, dan
BatchNorm dibekukan pada densenet. Hanya yang pertama yang ingin diukur.

Konsekuensinya untuk §4a: retrain nested-CV satu tahap tanpa pembekuan BN akan
mendarat **di antara** 0,8360 dan 0,8959, dan p `fusion_late` lawan radiomics bisa
saja kembali tidak signifikan. Analisis ini **menunjukkan klaim kesetaraan rapuh
terhadap protokol**; ia belum membuktikan klaim itu salah di bawah protokol jujur
yang minimal.

Karena itu tabel Track 1 yang terbit tidak diubah atas dasar hasil ini. Mengganti
angka terbit menuntut estimasi, bukan batas bawah.

**Batas lain:**

- Tiga backbone, satu dataset, satu ukuran. Bukan pernyataan umum tentang fusi.
- `fusion_late` tidak punya jalur gradien ke citra; §2 dan §4b menunjukkan
  konsekuensinya bisa diprediksi di atas kertas dari bentuk `average_fusion`.
- Sebaran `radiomics_only` antar backbone (0,9301–0,9343 pada run terbit) berasal
  dari `mutual_info_classif` yang jalan tanpa `random_state`. Tidak diseed di sini
  karena akan mengubah angka `radiomics_only` yang sudah terbit.
