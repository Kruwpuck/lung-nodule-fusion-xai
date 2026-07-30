# Laporan hasil Track 2: komparasi backbone, stabilitas hyperparameter, dan granularitas label

## 0. Identitas penelitian

- **Judul**: Komparasi backbone, stabilitas hyperparameter, dan pengaruh granularitas label untuk klasifikasi malignansi nodul paru pada LIDC-IDRI
- **Repo**: `lung-nodule-fusion-xai`
- **Tugas**: Track 2 dari pemisahan dua paper (lihat `docs/Review Revisi 1.md` §8)
- **Dataset**: LIDC-IDRI
- **Tanggal laporan**: 30 Juli 2026

---

## 1. Ringkasan eksekutif

Track 2 membandingkan empat backbone (MobileNetV2, EfficientNet-B0, ResNet50, VGG16) lintas sweep 3 optimizer x 3 weight decay x 5 fold, menguji klaim stabilitas dengan uji varians formal (bukan sekadar CV), dan membandingkan empat granularitas label pada subset kasus yang sama.

### Status komponen

| Komponen | Status | Sumber |
|---|---|---|
| Sweep 4 backbone x 3 optimizer x 3 weight_decay x 5 fold | Selesai, 180/207 run, 27 gagal di backbone lain | `artifacts/logs/runs.csv` |
| Uji varians Brown-Forsythe/Levene + ANOVA faktorial | Selesai, dijalankan ulang oleh laporan ini | `artifacts/results/track2_variance_brown_forsythe.csv`, `track2_anova_eta_sq.csv` |
| Evaluasi common-subset 4 arm granularitas | Selesai | `artifacts/results/common_subset_auc.csv` |
| DeLong + Friedman/Nemenyi lintas arm | Selesai | `artifacts/results/delong_arms_common_subset.csv`, `friedman_ranks_common_subset.csv` |
| Sweep learning rate khusus SGD | Kode selesai, belum dieksekusi | `docs/revisi/rev1/TASKBOARD.md` tugas 7 |
| Evaluasi mobilenetv2 di `summary_binary.csv` | **Belum lengkap**, mobilenetv2 ada di `runs.csv` (45 run selesai) tapi hilang dari `summary_binary.csv` | lihat §8 butir 3 |

---

## 2. Latar belakang dan kontribusi

Pemilihan backbone, optimizer, dan granularitas label adalah tiga keputusan desain yang jarang dikuantifikasi kontribusinya secara terpisah pada dataset dan split yang sama. Kontribusi laporan ini: (1) sweep hyperparameter penuh dengan uji varians formal untuk klaim stabilitas, bukan perbandingan titik estimasi CV semata, (2) dekomposisi ANOVA yang memisahkan efek optimizer dari weight decay, (3) perbandingan granularitas label pada subset kasus identik dengan kolaps probabilitas, bukan argmax.

---

## 3. Metodologi

### 3.1 Dataset dan split

Sama dengan Track 1: LIDC-IDRI, split 5-fold berbasis pasien seed 42, patch 2.5D. Empat arm label dilatih: biner, ordinal (1–5), 3-kelas (benign/malignant/indeterminate), 4-kelas (3-kelas plus kelas no-nodule).

### 3.2 Backbone

Empat backbone: MobileNetV2 dan EfficientNet-B0 (ringan, mobile-oriented), ResNet50 dan VGG16 (berat, deep convolutional klasik), dipilih untuk mengontraskan dua filosofi arsitektur dan dua rezim jumlah parameter yang tegas berbeda. Semua dievaluasi pada resolusi native 64 piksel, tanpa upsampling.

### 3.3 Sweep hyperparameter

Tiga optimizer (SGD momentum 0.9, Adam, AdamW), tiga weight decay per optimizer (default PyTorch ±1 orde besaran), lima fold. Grid penuh: 4 backbone x 3 optimizer x 3 weight_decay x 5 fold = 180 run.

### 3.4 Uji stabilitas

CV dilaporkan deskriptif. Untuk klaim "backbone X lebih stabil dari Y" secara formal, dipakai uji Brown-Forsythe (varian Levene berbasis median, lebih tahan skew) pada varians AUC tiap pasangan backbone, dengan koreksi Holm lintas 6 pasangan. ANOVA faktorial atas backbone, optimizer, weight_decay, fold melaporkan eta-squared per faktor.

### 3.5 Perbandingan granularitas label

Tiap arm dikolaps ke probabilitas benign-vs-malignant lewat renormalisasi, bukan argmax, karena renormalisasi mempertahankan skor kontinu yang dibutuhkan AUC. Arm biner pakai probabilitas malignant langsung. Arm ordinal dipetakan lewat probabilitas kumulatif rating di atas 3. Arm 3-kelas direnormalisasi $P(malignant)/(P(benign)+P(malignant))$. Arm 4-kelas dibatasi ke subset nodul dulu (buang kelas no-nodule) baru direnormalisasi sama seperti 3-kelas. Pembatasan ini pertahanan terhadap inflasi AUC dari negatif mudah. Semua arm dievaluasi pada 1391 nodul biner-eligible yang identik per fold. Uji DeLong berpasangan membandingkan tiap pasang arm pada subset ini. Uji Friedman dengan post-hoc Nemenyi memberi peringkat omnibus lintas arm.

### 3.6 Metrik ordinal-native

QWK, MAE, dan akurasi one-off untuk arm ordinal pada skala penuh 1–5.

---

## 4. Dataset

Lihat §3.1. Split dan preprocessing identik dengan Track 1.

---

## 5. Konfigurasi

`configs/config.yaml` blok `tracks.track2`: 4 backbone di atas, `input_size: null` (native). Blok `track2_sweep`: 3 optimizer, weight decay per optimizer.

---

## 6. Hasil

### 6.1 Komparasi backbone

AUC rata-rata per backbone, dipool lintas sweep 3x3 dan 5 fold (45 run per backbone), dari `artifacts/logs/runs.csv`:

| Backbone | AUC rata-rata | CV |
|---|---|---|
| VGG16 | 0.8966 | 0.0400 |
| ResNet50 | 0.8553 | 0.0792 |
| MobileNetV2 | 0.8262 | 0.0615 |
| EfficientNet-B0 | 0.8241 | 0.0865 |

### 6.2 Stabilitas hyperparameter Track 2

Uji Brown-Forsythe berpasangan dengan koreksi Holm lintas 6 pasangan backbone (`track2_variance_brown_forsythe.csv`): **hanya satu** pasangan signifikan, ResNet50 vs VGG16 ($p_{holm}=0.0426$). VGG16 vs EfficientNet-B0, yang sebelumnya diklaim sebagai bukti "VGG16 paling stabil", **tidak lolos koreksi** ($p_{holm}=0.0840$).

#### Batasan pada klaim stabilitas

Klaim umum "VGG16 backbone paling stabil" **tidak didukung** uji ini pada level pasangan individual. Yang bisa dinyatakan secara defensibel hanya perbedaan varians ResNet50 vs VGG16.

ANOVA faktorial (`track2_anova_eta_sq.csv`) melaporkan eta-squared: optimizer 0.4533, backbone 0.2060, fold 0.0684, weight_decay 0.0002, residual 0.1827. Efek optimizer terhadap varians AUC lebih dari 200 kali efek weight decay pada dekomposisi ini, rasio jauh lebih ekstrem dari perbandingan delta AUC mentah (0.1457 vs 0.0218) yang dipakai sebelumnya.

### 6.3 Perbandingan granularitas label

Uji Friedman omnibus lintas 4 arm (`friedman_ranks_common_subset.csv`) signifikan: $\chi^2=43.96$, $p=1.5\times10^{-9}$. Peringkat rata-rata: 4-kelas 1.33 (terbaik), biner 2.40, 3-kelas 2.77, ordinal 3.50 (terburuk). Post-hoc Nemenyi (`nemenyi_arms_common_subset.csv`): semua pasangan signifikan kecuali biner vs 3-kelas ($p=0.69$).

#### Batasan pada klaim granularitas per model

Pada level model individual, uji DeLong berpasangan biner-vs-4-kelas (`delong_arms_common_subset.csv`) hanya signifikan pada **3 dari 6** model legacy (MobileNetV3-Small, VGG16, ViT-Base), tidak signifikan untuk DenseNet121, EfficientNet-B0, ResNet50 ($p=0.055$–$0.17$). Klaim "granularitas label memengaruhi performa" berlaku agregat lintas model dan fold, **tidak** seragam per model individual.

Metrik ordinal-native (`ordinal_native_metrics.csv`): QWK 0.5451, MAE 0.6433, akurasi one-off 0.9155.

---

## 7. Figur

| Figur | Berkas | Kegunaan |
|---|---|---|
| AUC per optimizer | `artifacts/results/figures/track2_auc_by_optimizer.png` | Visualisasi sebaran AUC per optimizer |
| Heatmap AUC | `artifacts/results/figures/track2_auc_heatmap.png` | Peta AUC backbone x optimizer x weight_decay |
| Kurva training Track 2 | `artifacts/results/figures/curves_track2.png` | Kurva loss/AUC per epoch |

---

## 8. Batasan

1. Klaim "VGG16 paling stabil" tidak didukung uji Brown-Forsythe pada level pasangan individual; hanya ResNet50 vs VGG16 signifikan (§6.2).
2. Klaim "granularitas label memengaruhi performa" berlaku agregat, tidak seragam per model (§6.3).
3. **`mobilenetv2` sudah punya 45 run selesai di `runs.csv` tapi hilang dari `summary_binary.csv`.** Ini bukan masalah data eksekusi, tapi `stage_04_evaluate` belum di-*rerun* untuk memasukkannya ke tabel evaluasi. Perlu ditindaklanjuti sebelum tabel §6.1 dianggap lengkap untuk seluruh 4 backbone Track 2 di semua metrik evaluasi (bukan hanya AUC dari `runs.csv`).
4. Sweep learning rate khusus SGD (§Rencana lanjutan butir 1) belum dieksekusi; efek optimizer yang dominan pada §6.2 kemungkinan mencerminkan mismatch learning rate SGD, bukan properti optimizer itu sendiri.
5. Perbandingan granularitas label baru mencakup 6 model legacy dengan checkpoint ordinal dan 4-kelas; belum diperluas ke 4 backbone Track 2.

---

## 9. Rencana lanjutan

1. Jalankan sweep learning rate khusus SGD (`{1e-3, 1e-2, 1e-1}`) untuk menguji apakah defisit SGD adalah mismatch learning rate (kode sudah siap, `stage_03c_sweep.py`).
2. Jalankan ulang `stage_04_evaluate` untuk memasukkan mobilenetv2 ke `summary_binary.csv`.
3. Perluas perbandingan granularitas label ke 4 backbone Track 2.
4. Tambahkan sitasi yang hilang lewat Zotero (`docs/laporan/REFERENSI_DIBUTUHKAN.md`).

---

## 10. Integritas riset

Semua angka pada laporan ini ditelusuri ke baris CSV nyata yang ditarik dari mesin remote pada 30 Juli 2026 dan dianalisis ulang secara lokal (`stage_04b_stability.py` dijalankan langsung terhadap `runs.csv` yang baru ditarik). Klaim yang tidak lolos uji formal (stabilitas VGG16, granularitas per model) dinyatakan eksplisit sebagai tidak didukung, bukan disembunyikan atau dihaluskan.

---

## Lampiran: berkas hasil

| Berkas | Isi |
|---|---|
| `artifacts/logs/runs.csv` | 332 baris, log run-level lengkap sweep Track 2 dan training Track 1 |
| `artifacts/results/track2_stability.csv` | AUC rata-rata, SD, CV per kombinasi backbone/optimizer/weight_decay |
| `artifacts/results/track2_variance_brown_forsythe.csv` | Uji varians berpasangan antar backbone |
| `artifacts/results/track2_anova_eta_sq.csv` | Dekomposisi varians ANOVA faktorial |
| `artifacts/results/common_subset_auc.csv` | 120 baris, AUC per model/fold/arm pada subset biner-eligible |
| `artifacts/results/delong_arms_common_subset.csv` | 36 baris, DeLong berpasangan lintas arm per model |
| `artifacts/results/friedman_ranks_common_subset.csv` | Peringkat Friedman lintas arm |
| `artifacts/results/nemenyi_arms_common_subset.csv` | Post-hoc Nemenyi berpasangan |
| `artifacts/results/ordinal_native_metrics.csv` | QWK, MAE, akurasi one-off arm ordinal |
