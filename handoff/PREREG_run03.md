# Pra-registrasi run03 — fine-tuning bertahap + ensembling backbone

**Ditulis 22 Agustus 2026, sebelum satu pun GPU dijalankan.** Ini yang membedakan
perbaikan metodologis dari pengejaran angka: prediksi, aturan keputusan, dan tabel
pembanding semuanya ditetapkan di muka dan tidak boleh diubah setelah hasil dilihat.

```
RUN_ID           = 2026-08-22-run03
COMMIT_SHA       = fd4e50a          (keadaan repo saat pra-registrasi ini ditulis)
GOAL             = Fine-tuning bertahap + ensembling backbone untuk menaikkan cabang
                   CNN, lalu turunkan ulang metrik hilir yang bergantung padanya
BACKBONE         = convnext_tiny, densenet201, densenet121   (dikunci, GOAL2.md)
FOLD             = 0..4                                       (folds.json tidak disentuh)
INPUT_SIZE       = 96                                         (cfg.tracks.track1.input_size)
RESERVED         = artifacts/results/_baseline_pre_rev2/, artifacts/results/_leaky_pre_nestedcv/,
                   artifacts/splits/folds.json, artifacts/checkpoints/{backbone}/,
                   artifacts/results/preds/, .venv
```

Sumber tugas: `docs/revisi/rev3/implement2.md`. Empat penyimpangan dari brief itu
dicatat di bagian 7, masing-masing dengan bukti yang bisa dicek.

---

## 1. Hipotesis dan prediksi eksplisit

**Prediksi utama.** Fine-tuning bertahap menaikkan `cnn_only` sebesar +1 sampai +3
poin AUC; ensembling backbone menambah +1 sampai +2 poin. Kedua rentang diambil dari
kelaziman literatur, bukan dari data ini.

**Peringatan yang wajib dibaca bersama prediksi itu.** Baseline yang ada sekarang
**sudah 100% unfreeze**: `src/stage_03_train.py` tidak membekukan parameter apa pun.
Jadi ketiga varian yang diuji (0%, 10%, 20%) semuanya berkapasitas latih **lebih
kecil** daripada baseline. Two-stage fine-tuning di sini adalah hipotesis
**regularisasi**, bukan penambahan kapasitas: dugaannya, pada n≈1400 membekukan
sebagian besar backbone pralatih mengurangi overfitting lebih banyak daripada
kehilangan kapasitas yang ditimbulkannya.

Konsekuensinya untuk penafsiran: hasil nol atau negatif **bukan** "fine-tuning
gagal". Pernyataan yang benar adalah "pada n≈1400, membekukan lebih banyak backbone
tidak mengalahkan fine-tuning penuh". Itu hasil yang layak dilaporkan apa adanya.

---

## 2. Konfigurasi yang diuji — ditetapkan di muka, tidak boleh ditambah

Empat sel `unfreeze`, seluruhnya dijalankan di bawah nested CV yang sama:

| sel | cakupan unfreeze tahap 2 | status |
|---|---|---|
| 0% | head saja, backbone beku penuh | kandidat konfigurasi |
| 10% | `ceil(0,10 × n_child)` child terakhir badan backbone | kandidat konfigurasi |
| 20% | `ceil(0,20 × n_child)` child terakhir badan backbone | kandidat konfigurasi |
| 100% | seluruh backbone | **kontrol denda protokol, bukan kandidat** |

**Sel 100% bukan perluasan ruang pencarian.** Ia adalah resep yang sudah dipakai
sekarang, dijalankan ulang di bawah protokol yang jujur. Tanpa sel ini, hasil negatif
tidak dapat ditafsirkan: `cnn_only` turun karena fine-tuning tidak menolong, atau
karena nested CV menghapus keuntungan seleksi checkpoint? Dua penjelasan berlawanan
yang tanpa kontrol tidak terpisahkan. Sel 100% **tidak boleh** dilaporkan sebagai
konfigurasi pemenang temuan fine-tuning; ia hanya dibawa ke F-2 kalau ketiga kandidat
kalah terhadapnya.

Total: 3 backbone × 5 fold × 4 sel = **60 run**. Tidak ada sel kelima. Penambahan
varian di tengah jalan dilarang.

### 2a. Granularitas nyata tiap sel — diukur, bukan diasumsikan

Fraksi dihitung atas **grup lapisan** (child module badan backbone), bukan atas
bobot. Karena bobot tidak tersebar rata antar grup, pangsa parameter yang benar-benar
terbuka jauh berbeda dari angka pada nama sel. Diukur pada 22 Agustus 2026 sebelum
run dimulai:

| backbone | jumlah child | uf 0% | uf 10% | uf 20% | uf 100% |
|---|---|---|---|---|---|
| convnext_tiny | 8 | 0 modul · 0,0% param | 1 modul · **51,4%** param | 2 modul · **55,6%** param | 8 modul · 100% |
| densenet121 | 12 | 0 modul · 0,0% param | 2 modul · 31,1% param | 3 modul · 38,7% param | 12 modul · 100% |
| densenet201 | 12 | 0 modul · 0,0% param | 2 modul · 38,6% param | 3 modul · 47,5% param | 12 modul · 100% |

Dua konsekuensi yang wajib dibawa ke pelaporan:

1. **Nama sel tidak boleh dibaca sebagai pangsa bobot.** "unfreeze 10%" berarti 10%
   grup lapisan terakhir dibulatkan ke atas, yang pada convnext_tiny berarti separuh
   jaringan. `n_trainable`/`n_total` dicatat per baris supaya pangsa sebenarnya
   selalu terlihat di samping angka AUC-nya.
2. **Pada convnext_tiny, sel 10% dan 20% nyaris model yang sama** (51,4% lawan
   55,6% bobot). Selisih AUC antar keduanya di backbone itu karena itu **tidak
   informatif** dan tidak boleh ditafsirkan sebagai efek besaran unfreeze. Kedua sel
   tetap dijalankan dan dilaporkan; batasannya yang dinyatakan.

Granularitas per child module dipertahankan meski labelnya kasar, karena alternatifnya
— membuka sebagian isi satu grup untuk mengejar pangsa bobot tertentu — persis
kegagalan yang brief larang: membuka weight tanpa bias, atau separuh lapisan norm.

### 2b. Pembekuan BatchNorm tidak berlaku sama di ketiga backbone

Diukur pada sel percontohan sebelum run penuh dimulai: **`convnext_tiny` memuat nol
modul BatchNorm.** Arsitekturnya memakai LayerNorm sepanjang jaringan, jadi
`apply_bn_eval` menyentuh 0 modul di sana. Perlakuan "BatchNorm dibekukan" yang jadi
salah satu dari dua mekanisme run ini **hanya benar-benar berlaku pada densenet121
dan densenet201** (121 modul BN pada densenet121).

Konsekuensinya untuk penafsiran, ditetapkan sekarang:

- Sel `convnext_tiny` menguji **staged unfreezing saja**, bukan staged unfreezing
  plus pembekuan BN. Ia bukan replikasi ketiga dari perlakuan yang sama.
- Kolom `n_batchnorm` dicatat di tiap baris hasil supaya perbedaan ini terlihat di
  tabel, bukan harus disimpulkan ulang dari arsitektur.
- Kalau densenet naik dan convnext tidak (atau sebaliknya), penjelasan "efek
  pembekuan BN" **boleh** diajukan; kalau ketiganya bergerak searah, penjelasan itu
  tidak didukung karena satu backbone tidak menerima perlakuannya.

**Seluruh konfigurasi dilaporkan**, termasuk yang kalah. Varian yang kalah adalah
bagian dari temuan.

---

## 3. Protokol — ditetapkan di muka

- **Nested CV.** Inner split patient-level dari fold latih luar, `GroupShuffleSplit`
  dengan `test_size=0.15` dan `random_state=fold`. Angka ini **identik** dengan yang
  sudah dipakai cabang fusi (`src/stage_03b_fusion.py:187`), supaya kedua cabang
  sebanding. Pemilihan epoch memakai inner-val. **Fold luar dinilai tepat sekali, di
  akhir**, dan tidak pernah memengaruhi bobot mana yang disimpan.
- **Tahap 1 (head only):** seluruh backbone dibekukan, head dilatih sampai konvergen.
  Batas 20 epoch, early stopping patience 10 pada inner-val.
- **Tahap 2 (unfreeze bertahap):** `ceil(fraction × n_child)` child terakhir badan
  backbone dibuka, dengan learning rate diskriminatif dibagi 2,6 per grup lapisan
  mundur. Batas 50 epoch, early stopping patience 10 pada inner-val.
- **`head_lr` = 1e-4**, mengikuti `cfg.train.lr`.
- **BatchNorm dibekukan di dalam loop epoch.** `model.train()` mengembalikan BN ke
  mode train setiap epoch, jadi `apply_bn_eval()` dipanggil tepat setelah
  `model.train()` di setiap epoch, bukan sekali sebelum loop. Ini kegagalan senyap
  yang paling mungkin terjadi pada langkah ini dan punya assert sendiri.
- **Unfreeze per child module**, bukan slice `named_parameters()`. Slice per-parameter
  bisa membuka weight tanpa bias-nya.
- **Sel 100% tetap dua tahap dan tetap `apply_bn_eval`**, hanya cakupan unfreeze-nya
  yang berbeda. Dengan begitu selisih terhadap 10%/20% murni soal cakupan, bukan
  soal protokol.
- **`folds.json` tidak disentuh. `input_size` 96 diteruskan eksplisit.**

---

## 4. Tabel pembanding — dipatok sekarang, dari disk

Gate F-1 dibandingkan **hanya** terhadap angka di bawah ini. Semuanya disalin dari
berkas yang ada di disk pada 22 Agustus 2026, bukan dari brief.

### 4a. Pooled AUC seluruh 5 fold digabung

Sumber: `artifacts/results/run02/t0_checkpoint_sensitivity.csv`
(`run_id=2026-08-04-run02`, `commit_sha=5220afb`).

| backbone | `cnn_best` | `cnn_last` | `radiomics_only` |
|---|---|---|---|
| convnext_tiny | 0,8907 | 0,8806 | 0,9336 |
| densenet201 | 0,8959 | 0,8888 | 0,9336 |
| densenet121 | 0,8965 | 0,8725 | 0,9336 |

`cnn_best` adalah checkpoint yang dipilih berdasarkan AUC pada fold yang kemudian
dipakai melaporkan hasil — jadi ia mengandung keuntungan seleksi. `cnn_last` adalah
checkpoint akhir latihan yang tidak dipilih berdasarkan apa pun. Selisih keduanya
(−0,0101 · −0,0071 · −0,0240) adalah ukuran empiris besarnya keuntungan itu, dan
**sebesar efek +1..+3 poin yang run ini cari**. Itulah sebabnya sel kontrol 100% ada.

### 4b. Rerata AUC per fold

Sumber: `artifacts/results/fusion/ablation_summary.csv`, arm inti, n=5 fold tiap sel.

| backbone | `cnn_only` | `radiomics_only` | `fusion_early` | `fusion_intermediate` | `fusion_late` |
|---|---|---|---|---|---|
| convnext_tiny | 0,9091 ± 0,0217 | 0,9322 ± 0,0140 | 0,9136 | 0,9263 | 0,9351 |
| densenet201 | 0,9023 ± 0,0220 | 0,9301 ± 0,0176 | 0,9089 | 0,9051 | 0,9368 |
| densenet121 | 0,8989 ± 0,0216 | 0,9343 ± 0,0163 | 0,9155 | 0,9228 | 0,9340 |

Kedua tabel dilaporkan berdampingan karena keduanya sah dan **berbeda** — pooled AUC
bukan rerata AUC per fold. Angka run03 dibandingkan terhadap tabel yang cara
agregasinya sama, tidak pernah dicampur.

---

## 5. Aturan keputusan — ditetapkan sekarang

1. **Fine-tuning dipakai** kalau `cnn_only` naik **dan** tidak menurunkan arm lain.
2. **Varian pemenang per backbone ditentukan dari inner-val**, bukan dari fold luar.
   Fold luar hanya dilaporkan.
3. **Fusi dinyatakan menang atas radiomics HANYA kalau DeLong p<0,05.** Tidak ada
   pelonggaran, tidak ada "mendekati signifikan".
4. **Kontrol negatif F-2:** `radiomics_only` harus tetap dalam **±0,005** dari nilai
   per-backbone di tabel 4b. Di luar itu, berhenti — artinya ada yang berubah di
   pipeline yang seharusnya tidak. Alasan angka 0,005 ada di bagian 7 butir 3.
5. **Prasyarat F-3:** urutan kasus antar backbone dalam fold yang sama wajib
   diverifikasi identik dan buktinya dicetak sebelum satu pun probabilitas dirata.
6. **Kalau setelah fine-tuning dan ensembling fusi tetap tidak mengalahkan
   `radiomics_only` dengan p<0,05, pencarian performa ditutup permanen.** Statusnya
   `done`, bukan gagal. Ruang pencarian tidak diperluas atas inisiatif agent.

---

## 6. Tiga selisih yang dilaporkan terpisah

Karena sel kontrol ada, tiga besaran yang selama ini tercampur sekarang terpisah.
Ketiganya wajib dilaporkan sendiri-sendiri:

| selisih | menjawab |
|---|---|
| sel 100% lawan baseline lama (tabel 4) | **denda protokol** — berapa banyak angka lama berutang pada seleksi checkpoint di fold penilaiannya sendiri |
| sel 10%/20% lawan sel 100% | **efek fine-tuning murni**, bebas dari denda protokol |
| varian terbaik lawan baseline lama | **efek gabungan** — angka yang masuk naskah |

---

## 7. Penyimpangan dari `implement2.md`, dengan buktinya

**1. Sel kontrol 100% ditambahkan** (brief hanya menetapkan 0/10/20). Alasannya di
bagian 2 dan 4a. Ditetapkan di muka di dokumen ini, jadi bukan penambahan varian di
tengah jalan.

**2. Angka pembanding brief tidak dipakai.** Brief menulis `cnn_only` 0,8927,
baseline 0,9055 / 0,8988 / 0,8959, dan `radiomics_only` 0,9318. Tidak satu pun cocok
dengan disk pada cara agregasi mana pun (bandingkan tabel 4a dan 4b). Nilai 0,8959
yang brief berikan untuk densenet121 adalah nilai densenet201 di disk. Membandingkan
gate terhadap angka yang tidak bisa direproduksi adalah persis insiden densenet121
terulang, jadi tabel 4 yang dipakai.

**3. Toleransi kontrol negatif dilonggarkan dari ±0,0036 menjadi ±0,005.**
`radiomics_only` tidak menyentuh checkpoint CNN sama sekali, tapi tetap bervariasi
0,9301–0,9343 antar backbone di `ablation_summary.csv` — rentang 0,0042, sudah
melebihi toleransi yang brief tetapkan. Sumbernya `mutual_info_classif` di
`_select_fold_features` yang jalan tanpa `random_state`. Menyetel seed di situ
ditolak karena akan mengubah angka `radiomics_only` yang sudah terbit; yang dilakukan
adalah menyetel toleransi ke sedikit di atas sebaran teramati dan mencatat alasannya.

**4. Regenerasi statistik empat-arm Track 2 dibatalkan.** Brief menuntutnya di F-4,
tapi checkpoint legacy `artifacts/checkpoints/densenet121/` tidak disentuh run ini —
checkpoint fine-tuned masuk direktori terpisah. Statistik Friedman/Nemenyi/DeLong
per-arm karena itu tetap sah tanpa dihitung ulang. Sebagai gantinya, **pemisahan
checkpoint dinyatakan eksplisit di kedua naskah dan kedua laporan**: Track 2 memakai
checkpoint legacy, Track 1 memakai run03, berikut alasannya. Pembaca yang
membandingkan angka densenet121 antar-naskah harus tahu keduanya berasal dari
checkpoint berbeda.

---

## 8. Aturan provenansi

Berlaku untuk seluruh run, konsekuensi langsung dari insiden densenet121: prediksi
ditimpa di tempat oleh pelatihan ulang, metrik hilir tidak pernah diturunkan ulang,
dan satu paragraf naskah menggambarkan model yang sudah tidak ada.

- **Prediksi ber-versi, tidak ditimpa.** Keluaran run ini masuk
  `artifacts/results/run03/preds/`; `artifacts/results/preds/*.npz` yang lama tidak
  disentuh. Checkpoint masuk `artifacts/checkpoints/run03/{backbone}_uf{pct}/`.
- **Tiap baris hasil membawa** `run_id`, `commit_sha`, `input_size`,
  `checkpoint_mtime`. Skema `artifacts/results/run03/finetune_cnn_only.csv`:

  ```
  run_id,commit_sha,input_size,checkpoint_mtime,backbone,unfreeze_pct,fold,
  n_trainable,n_total,inner_val_auc,outer_auc_merged,outer_auc_full,
  stage1_epochs_ran,stage2_epochs_ran
  ```

  `checkpoint_mtime` ditulis ISO 8601 sampai detik, contoh `2026-08-22T14:05:33`.
- **Sebelum menurunkan metrik apa pun**, `checkpoint_mtime` diverifikasi cocok dengan
  yang tercatat. Kebasian terdeteksi otomatis, bukan lewat ingatan.
- `outer_auc_merged` dihitung pada subset irisan labels ∩ radiomics (42 nodul dengan
  kunci ambigu dibuang), sebanding dengan `ablation_summary.csv` dan dengan
  `radiomics_only`. `outer_auc_full` dihitung pada seluruh baris biner, sebanding
  dengan `summary_binary.csv`. Keduanya dicatat supaya tidak ada perbandingan yang
  diam-diam memakai subset berbeda.

---

## 9. Larangan

- Dilarang menyentuh `_baseline_pre_rev2/`, `_leaky_pre_nestedcv/`, `folds.json`,
  `.venv`, `configs/config*.yaml` yang sudah ada,
  `artifacts/xai/fixed_display_samples.json`.
- Dilarang menimpa `artifacts/results/preds/*.npz` dan
  `artifacts/checkpoints/{backbone}/` yang sudah ada.
- Dilarang mengubah daftar 3 backbone.
- Dilarang menambah varian unfreeze di luar empat yang ditetapkan bagian 2.
- Dilarang mengembalikan seleksi epoch ke fold luar.
- Dilarang mencari mekanisme fusi baru — run ini soal mutu cabang CNN.
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git push`. Push tetap tugas manusia.

---

## 10. Konsekuensi yang sudah diterima

Run ini membatalkan sebagian hasil Track 1 yang sudah ditulis: tabel ablasi fusi
dihitung ulang, metrik XAI 3 backbone dihitung ulang, penulisan naskah Track 1 mundur
sampai angka baru stabil.

Yang **tidak** mundur dan tetap dikerjakan berbarengan, karena nol ketergantungan
pada checkpoint: penulisan ulang `paper/track2/main.tex:241` beserta §6.3 laporan
Track 2 (koreksi klaim keterpisahan antar-arm), dan penulisan positioning korpus.
Keduanya koreksi kesalahan pelaporan yang berdiri sendiri.
