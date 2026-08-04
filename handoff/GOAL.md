# GOAL — Langkah 3: Arm Fusi Rebalanced

## Slot isian

```
PROJECT_ROOT     = C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai
GOAL             = Jalankan arm fusi rebalanced (branch_norm + gmu) untuk 3 backbone terpilih
                   dan uji apakah ada yang mengalahkan radiomics_only secara signifikan
DONE_CRITERIA    = lihat blok "Kriteria selesai" di bawah, semua 6 gate harus lolos
GPU_HOST         = mesin ini sendiri. RTX 3060 12 GB. Training lokal, bukan SSH ke mesin lain
MAX_ITER         = 15
MAX_SUBAGENT     = 5
MAX_FETCH_PER_SUB= 8
RUN_ID           = 2026-08-04-run01
RESERVED_ENV     = artifacts/results/_baseline_pre_rev2/, artifacts/splits/folds.json, .venv
```

---

## GOAL

Jalankan dua arm fusi rebalanced yang kodenya **sudah ada** (`branch_norm` dan `gmu`)
untuk 3 backbone terpilih, lalu uji dengan DeLong apakah ada yang mengalahkan
`radiomics_only` secara signifikan.

Tidak menulis arsitektur fusi baru. Tidak menambah dependency. Murni konfigurasi,
eksekusi GPU, dan pelaporan.

**Backbone (tetap, tidak boleh diganti agent):**
- `convnext_tiny`
- `densenet201`
- `densenet121`

Dipilih berdasarkan pointing accuracy XAI tertinggi (0.7167 / 0.7000 / 0.7167) dan AUC
standalone, ditetapkan sebelum eksperimen.

---

## Perintah

Tiga langkah, berurutan. Jangan paralel: satu GPU, dua proses training serentak akan
kehabisan memori.

```
.venv\Scripts\python.exe -m src.stage_03b_fusion --config configs/config_l3_plain.yaml
.venv\Scripts\python.exe -m src.stage_03b_fusion --config configs/config_l3_reg.yaml
.venv\Scripts\python.exe -m src.stage_03d_merge_l3
```

Kondisi A (`config_l3_plain.yaml`): `modality_dropout_rate: 0.0`, `aux_loss_weight: 0.0`.
Kondisi B (`config_l3_reg.yaml`): `modality_dropout_rate: 0.2`, `aux_loss_weight: 0.3`.

### Kenapa dua config dan bukan satu

`modality_dropout_rate` dan `aux_loss_weight` dibaca sekali per proses lalu diteruskan
ke setiap arm, jadi menyalakannya mengubah ketiga arm sekaligus. Satu proses tidak bisa
memisahkan kontribusi GMU dari kontribusi regularizer.

### Kenapa `paths.results` dipisah, dan kenapa butuh langkah merge

`src/stage_03b_fusion.py:305` memanggil `summary.to_csv(out_csv)` yang **menimpa**
`ablation_summary.csv`, bukan menambah. Menjalankannya langsung terhadap
`artifacts/results` akan menghapus 175 baris hasil Langkah 1, termasuk seluruh baris
`radiomics_only` yang justru jadi patokan gate G-2.

Karena itu kedua config menulis ke `artifacts/results/l3_plain/` dan
`artifacts/results/l3_reg/`, lalu `stage_03d_merge_l3` menyalin 60 baris arm baru ke
CSV utama sambil menambahkan kolom provenance yang diminta G-6.

`stage_03d_merge_l3` menolak menimpa baris yang sudah ada dan tidak pernah menyentuh
baris `radiomics_only`, `cnn_only`, `fusion_early`, `fusion_late`, atau
`fusion_intermediate` yang lama.

---

## Kriteria selesai (6 gate, semua wajib lolos)

Agent tidak boleh menandai `done` sebelum keenamnya diverifikasi **dengan bukti
tercetak**, bukan asumsi. Perintah pemeriksanya satu:

```
.venv\Scripts\python.exe -m src.stage_03d_merge_l3 --check
```

**G-1. Kelengkapan baris.**
`artifacts/results/fusion/ablation_summary.csv` bertambah **tepat 60 baris** baru:
3 backbone x 5 fold x 2 arm (`branch_norm`, `gmu`) x 2 kondisi regularizer = 60.
Bukan "file ada", hitung barisnya dan cetak angkanya.

Catatan nama: nama arm yang benar-benar ditulis kode adalah
`fusion_intermediate_branch_norm`, `fusion_intermediate_gmu`,
`fusion_intermediate_branch_norm_moddrop_auxloss`, dan
`fusion_intermediate_gmu_moddrop_auxloss`. Pencocokan gate memakai substring, bukan
kesamaan persis. Ambang 60 tidak berubah.

**G-2. Kontrol negatif, `radiomics_only` tidak bergerak.**
Rerata `radiomics_only` harus tetap dalam **plus minus 0.0036** dari **0.9318**
(lantai derau terukur).

Diperiksa pada dua tempat, keduanya wajib lolos:
1. CSV utama, yang seharusnya tidak berubah sama sekali karena merge hanya menambah
   baris arm baru. Nilainya sekarang 0.932384.
2. `radiomics_only` hasil run **baru** di `l3_plain` dan `l3_reg`. Ini kontrol negatif
   yang sebenarnya. Kalau pipeline berubah diam-diam, di sinilah ketahuan.

Kalau salah satu bergerak lebih jauh dari pita: **BERHENTI, tulis STUCK.md**. Itu tanda
ada yang salah di pipeline, bukan hasil eksperimen.

**G-3. `fs_method` konsisten.**
Kolom `fs_method` terisi 100% di semua baris baru, satu nilai unik
(`mutual_info_classif`). Nol kosong.

**G-4. Kondisi benar-benar berbeda.**
Tidak boleh ada dua kondisi dengan AUC **persis identik** pada backbone dan fold yang
sama. AUC identik berarti config tidak benar-benar berubah, arm atau regularizer tidak
terpasang.

**G-5. DeLong dijalankan.**
`delong_fusion.csv` memuat uji `branch_norm` lawan `radiomics_only` dan `gmu` lawan
`radiomics_only` untuk ketiga backbone. Laporkan: berapa yang **signifikan lebih baik**
(p<0.05), berapa seri, berapa lebih buruk.

Sudah terverifikasi bahwa ini berjalan tanpa perubahan kode: kolom `best_single_arm`
bernilai `radiomics` di ketujuh backbone hasil Langkah 1, jadi pembanding DeLong memang
sudah `radiomics_only`.

**G-6. Provenance tercatat.**
Setiap baris baru punya `run_id`, `input_size` yang benar-benar terpakai, `commit_sha`,
dan `condition`. Tidak boleh ada artefak tanpa asal-usul. Kolom ini ditambahkan oleh
`stage_03d_merge_l3`, bukan oleh `stage_03b_fusion`.

---

## Prasyarat

Ketiganya sudah selesai sebelum loop mulai. Jangan dikerjakan ulang.

- [x] Ablasi fusi `densenet121` diulang setelah retrain fold 2/3. Terverifikasi:
      baris `densenet121` di CSV utama identik per-fold dengan `artifacts/results/_tmp_dn121/`,
      masuk lewat commit `c1b14d0`.
- [x] Pohon git bersih. Artefak di-commit di `c1b14d0`, skrip ad-hoc masuk `.gitignore`.
- [x] Sha commit hijau terakhir tercatat di `STATE.json`.

---

## Batasan

- Dilarang menyentuh: `artifacts/results/_baseline_pre_rev2/`, `artifacts/splits/folds.json`, `.venv`
- Dilarang mengubah daftar 3 backbone di atas
- Dilarang mengubah ambang gate (0.0036, 0.9318, 60 baris)
- Training jalan di PC ini. Verifikasi proses masih hidup tiap siklus sebelum melaporkan status
- Maks iterasi: 15
- Satu proses training pada satu waktu

## Larangan keras

- Dilarang mengedit atau menghapus file test
- Dilarang mengubah definisi metrik atau ambang kriteria selesai
- Dilarang mengubah file GOAL ini
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git clean -fdx`, `git push`
- Dilarang menimpa apa pun di `_baseline_pre_rev2/` atau `_leaky_pre_nestedcv/`
- Dilarang mengubah `configs/config.yaml`
- Dilarang menulis arsitektur fusi baru. Arm `branch_norm` dan `gmu` sudah ada, cuma dijalankan

---

## Catatan kalau hasilnya negatif

Kalau setelah keenam gate lolos ternyata **nol dari uji DeLong** mendukung fusi, itu
**hasil sah dan loop selesai dengan status `done`**, bukan gagal. Jangan mencari
konfigurasi lain untuk membalikkannya. "Radiomics mengalahkan fusion" adalah temuan
yang dilaporkan apa adanya.

Angka pembanding dari Langkah 1, tujuh backbone, arm lama: `radiomics_only` 0.929 sampai
0.934, dan nol dari 21 uji DeLong mendukung fusi. `fusion_late` seri, `fusion_early` dan
`fusion_intermediate` kalah signifikan.

Agent dilarang memperluas ruang pencarian atas inisiatif sendiri.

---

## Catatan yang sudah diketahui, jangan diteliti ulang

- `pymrmr` tidak terpasang. Seleksi fitur yang benar-benar berjalan adalah
  `mutual_info_classif`, dan kolom `fs_method` sudah melaporkannya jujur. Bukan bug.
- Nested CV sudah aktif tanpa syarat di `_train_fusion_fold`, bukan di balik flag config.
- Aktivasi gate GMU tersimpan di `model.last_branch_norms` sebagai `img_gate` dan
  `rad_gate`, tetapi belum ditulis ke CSV mana pun. Itu butuh kode baru dan berada di
  luar keenam gate. Catat sebagai pekerjaan lanjutan, jangan kerjakan.
