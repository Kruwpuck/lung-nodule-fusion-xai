# F-1 — hasil two-stage fine-tuning, 60 sel

`run_id` 2026-08-22-run03 · sumber `finetune_cnn_only.csv` (60 baris, nol duplikat) ·
60 sel selesai dalam 97 menit · pra-registrasi `handoff/PREREG_run03.md`

Seluruh keempat varian dilaporkan, termasuk yang kalah. AUC di bawah diukur pada
subset irisan labels ∩ radiomics (`outer_auc_merged`), sebanding dengan
`ablation_summary.csv` dan dengan `radiomics_only`.

---

## 1. Tabel penuh: 3 backbone × 4 varian

| backbone | varian | bobot terbuka | modul BN | inner-val (rerata) | outer rerata-fold | outer pooled |
|---|---|---|---|---|---|---|
| convnext_tiny | uf 0% | 0,01% | 0 | 0,7498 | 0,7554 | 0,7551 |
| convnext_tiny | uf 10% | 51,37% | 0 | 0,8509 | 0,8272 | 0,8242 |
| convnext_tiny | uf 20% | 55,61% | 0 | 0,8516 | 0,8346 | 0,8276 |
| convnext_tiny | **uf 100%** | 100% | 0 | **0,8645** | **0,8455** | **0,8467** |
| densenet121 | uf 0% | 0,03% | 121 | 0,7801 | 0,7785 | 0,7742 |
| densenet121 | uf 10% | 31,08% | 121 | 0,8666 | 0,8528 | 0,8502 |
| densenet121 | uf 20% | 38,65% | 121 | 0,8701 | 0,8564 | 0,8473 |
| densenet121 | **uf 100%** | 100% | 121 | **0,8781** | **0,8649** | **0,8620** |
| densenet201 | uf 0% | 0,02% | 201 | 0,7585 | 0,7546 | 0,7519 |
| densenet201 | uf 10% | 38,60% | 201 | 0,8468 | 0,8240 | 0,8195 |
| densenet201 | uf 20% | 47,50% | 201 | 0,8564 | 0,8416 | 0,8403 |
| densenet201 | **uf 100%** | 100% | 201 | **0,8679** | **0,8420** | **0,8360** |

**Pemenang per backbone menurut inner-val — bukan fold luar — adalah sel 100% pada
ketiganya.** Urutan inner-val monoton naik terhadap cakupan unfreeze di ketiga
backbone tanpa kecuali: 0% < 10% < 20% < 100%.

---

## 2. Tiga selisih, dilaporkan terpisah

### 2a. Efek fine-tuning murni — varian kandidat lawan sel 100%

Rerata-fold. Ini besaran yang bebas dari denda protokol, karena keempat sel memakai
protokol yang persis sama.

| backbone | uf 0% | uf 10% | uf 20% |
|---|---|---|---|
| convnext_tiny | −0,0902 | −0,0183 | −0,0110 |
| densenet201 | −0,0874 | −0,0180 | −0,0004 |
| densenet121 | −0,0864 | −0,0121 | −0,0085 |

**Negatif di sembilan dari sembilan sel.** Membekukan sebagian backbone tidak pernah
mengalahkan fine-tuning penuh pada n≈1400, dan kerugiannya mengecil monoton seiring
cakupan unfreeze membesar. Hipotesis regularisasi yang dipra-registrasi di §1
pra-registrasi **tidak didukung**: pada data ini kapasitas yang hilang lebih mahal
daripada overfitting yang dicegah.

### 2b. Denda protokol — sel 100% lawan baseline lama

| backbone | sel 100% rerata-fold | lawan baseline 4b | sel 100% pooled | lawan `cnn_best` 4a | lawan `cnn_last` 4a |
|---|---|---|---|---|---|
| convnext_tiny | 0,8455 | −0,0636 | 0,8467 | −0,0440 | −0,0339 |
| densenet201 | 0,8420 | −0,0603 | 0,8360 | −0,0599 | −0,0528 |
| densenet121 | 0,8649 | −0,0340 | 0,8620 | −0,0345 | −0,0105 |

**Ini batas atas, bukan estimasi bersih.** Sel 100% berbeda dari resep baseline dalam
**empat** hal sekaligus, bukan satu:

1. Epoch dipilih di inner-val, bukan di fold luar — ini yang memang ingin diukur.
2. Data latih berkurang 15% karena inner split dipotong dari fold latih.
3. Jadwalnya dua tahap (20 epoch head-only lalu 50 epoch penuh), bukan satu tahap 50 epoch.
4. BatchNorm dibekukan — pada densenet121/201 saja; convnext_tiny memuat nol modul BN.

Jadi angka −0,034 sampai −0,064 adalah biaya **seluruh paket protokol**, dan komponen
nested CV-nya lebih kecil dari itu. Batas bawah untuk komponen seleksi checkpoint saja
sudah ada dari run02: `cnn_best` lawan `cnn_last` memberi 0,0101 · 0,0071 · 0,0240.
Selisih antara kedua batas itu — kira-kira 1 sampai 4 poin — milik ketiga faktor lain
bersama-sama, dan run ini tidak memisahkannya lebih jauh.

Faktor 4 dapat sedikit dipersempit: convnext_tiny tidak menerima pembekuan BN sama
sekali, tapi dendanya (−0,0636) justru **di antara** kedua densenet (−0,0340 dan
−0,0603). Pembekuan BN karena itu tidak menjelaskan pola dendanya.

### 2c. Efek gabungan — varian terbaik lawan baseline lama

Karena varian terbaik menurut inner-val adalah sel 100% pada ketiga backbone, efek
gabungan **identik dengan §2b**. Nol konfigurasi kandidat yang lolos ke sini.

---

## 3. Putusan gate F-1

Aturan keputusan 1 pra-registrasi: *fine-tuning dipakai kalau `cnn_only` naik dan
tidak menurunkan arm lain.* `cnn_only` **tidak naik** di satu sel kandidat pun.

Gate F-1 butir terakhir berlaku: nol varian menaikkan `cnn_only` di atas sel 100%,
jadi hasilnya dilaporkan apa adanya dan fine-tuning **tidak diadopsi**.

**Ini hasil negatif, dan statusnya `done`, bukan gagal.** Dua pernyataan yang
didukung datanya:

1. Pada n≈1400, membekukan sebagian besar backbone pralatih tidak mengalahkan
   fine-tuning penuh, dan kerugiannya monoton terhadap seberapa banyak yang dibekukan.
2. Angka `cnn_only` yang terbit selama ini berutang 3,4 sampai 6,4 poin AUC pada
   protokol yang dipakai menghasilkannya, dan setidaknya 1,0 sampai 2,4 poin dari
   utang itu adalah seleksi checkpoint pada fold penilaiannya sendiri.

Pernyataan kedua lebih penting daripada yang pertama, dan ia hanya bisa dinyatakan
karena sel kontrol 100% ada. Tanpa kontrol itu, hasil ini akan terbaca sebagai
"fine-tuning menurunkan performa 3 sampai 6 poin", yang salah.

---

## 4. Yang tidak bisa disimpulkan dari sini

- **Bukan pernyataan bahwa fine-tuning bertahap tidak pernah berguna.** Ia diuji pada
  satu dataset, satu ukuran (n≈1400), satu jadwal LR, tiga backbone.
- **Bukan pemisahan bersih denda nested CV.** Lihat §2b: empat faktor bergerak
  bersamaan; yang tersedia batas atas dan batas bawah, bukan titik.
- **Pada convnext_tiny, sel 10% dan 20% nyaris model yang sama** (51,4% lawan 55,6%
  bobot), jadi selisih 0,0073 antar keduanya di backbone itu tidak informatif. Sudah
  diantisipasi di pra-registrasi §2a.
- **Pembekuan BatchNorm tidak teruji terpisah.** convnext_tiny tidak menerimanya sama
  sekali, jadi ketiga backbone bukan tiga replikasi perlakuan yang sama
  (pra-registrasi §2b).
