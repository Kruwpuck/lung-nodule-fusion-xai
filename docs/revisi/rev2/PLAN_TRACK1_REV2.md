# Rencana implementasi Rev2 untuk Track 1 (fusi radiomics-CNN + XAI)

Sumber: `docs/revisi/rev2/compass_artifact_wf-b2cfd7b3-bbdf-5b30-bbe0-ebfc775f5829_text_markdown.md`
Cakupan: Track 1 saja. Track 2 tidak disentuh kecuali disebut eksplisit.
Status: rencana. Belum ada satu baris kode pun yang diubah.

---

## 0. Keputusan yang sudah dikunci

| Pertanyaan | Keputusan |
|---|---|
| Safeguard anti-overfitting | Nested CV. Fold pasien existing tidak diubah, sehingga 215 run Track 2 dan angka paper Track 2 tetap valid |
| Cakupan | Langkah high-yield rev2: 1, 2, 3, 6. OGM-GE, Optuna, multi-view, sweep classifier radiomics tidak dikerjakan |
| Target primer | AUC >= 0.95, ditambah accuracy pada operating point tetap, ditambah analisis *ceiling* sebagai kontribusi tersendiri |
| Backbone untuk eksperimen baru | 3 backbone: ConvNeXt-Tiny, DenseNet201, DenseNet121 |

Catatan tentang batas backbone: pemangkasan ke 3 backbone berlaku untuk **eksperimen baru** (langkah 2 sampai 5). Langkah 1 tetap dijalankan pada ketujuh backbone Track 1, karena langkah 1 adalah pengulangan tabel ablasi yang sudah ada dan hasilnya harus menggantikan `ablation_summary.csv` secara utuh, bukan sebagian.

---

## 1. Apa yang sudah ada, apa yang belum

Audit terhadap kode nyata, bukan terhadap asumsi dokumen rev2.

| Rekomendasi rev2 | Status di repo | Bukti |
|---|---|---|
| 1. Perbaiki bug resolusi, jalankan ulang ablasi | Kode selesai, belum pernah dieksekusi | `docs/revisi/rev1/TASKBOARD.md` tugas 1, status `done-code` |
| 2. Fine-tuning dua tahap dengan BatchNorm dibekukan | Belum ada sama sekali | Nol hasil untuk pencarian `requires_grad`, `norm_eval`, atau `freeze` di seluruh `src/` selain hitungan parameter di `src/evaluation/efficiency.py:13` |
| 3. Fusi yang direbalans | Sudah dikode lengkap, belum dieksekusi | `src/models/fusion_net.py:12` mendefinisikan `FUSION_ARMS = ("concat", "branch_norm", "gmu")`; L2-normalisasi per cabang di baris 165-167, proyeksi 256 ke 32 di baris 116, modality dropout di baris 153-160, *auxiliary loss* per cabang di baris 140-143 |
| 4. OGM-GE | Belum ada | Di luar cakupan yang dipilih |
| 5. Tambah fitur wavelet dan LoG | **Sudah aktif.** Klaim rev2 keliru | `configs/radiomics_params.yaml:2-6` mengaktifkan `LoG` dengan sigma `[1.0, 2.0, 3.0]` dan `Wavelet` |
| 6. Ensembling fold dan backbone | Belum ada | Nol hasil untuk pencarian `ensemble` di `src/` |
| 7. Optuna dan ASHA | Belum ada, `optuna` tidak terdaftar di `requirements.txt` | Di luar cakupan yang dipilih |
| 7b. Held-out test set | Tidak ada. `src/stage_02_split.py` hanya membentuk 5 fold, tanpa *outer holdout* | Diganti nested CV, lihat langkah 4 |
| 8. Input multi-view 3 bidang | Belum ada, `data.n_slices: 3` hanya bidang aksial | Di luar cakupan yang dipilih |

Dua koreksi terhadap dokumen rev2 yang perlu dicatat di manuskrip agar tidak salah kutip:

1. Cabang radiomics **sudah** memakai fitur wavelet dan LoG sejak awal. Rekomendasi nomor 5 rev2 dibangun di atas asumsi yang salah. Yang benar-benar belum diverifikasi adalah apakah `pymrmr` benar-benar terpasang di mesin remote. `src/radiomics/feature_selection.py:58-67` diam-diam jatuh ke `mutual_info_classif` jika impor gagal, dan hanya menulis `logger.warning`. Kalau fallback ini yang aktif selama ini, kata "mRMR" di manuskrip Track 1 adalah klaim yang tidak akurat. Pemeriksaan ini murah dan harus dilakukan pada langkah 0.
2. Rekomendasi ensembling fold rev2 ("rata-ratakan probabilitas 5 model fold") tidak bisa diterapkan apa adanya. Alasannya di langkah 5.

---

## 2. Langkah 0: sinkronkan mesin remote (memblokir semua langkah lain)

Kondisi saat ini, diverifikasi lewat SSH:

- HEAD remote `5693315`, HEAD lokal `121c3fe`. Remote belum punya perbaikan `input_size`, arm fusi baru, maupun kedua manuskrip.
- Remote punya perubahan yang belum di-*commit* pada `artifacts/results/fusion/ablation_summary.csv`, `delong_fusion.csv`, lima berkas `preds/densenet121_fold*.npz`, `xai/xai_densenet121.png`, dan `xai_metrics.csv`.
- Remote punya berkas asing yang belum diperiksa: `fix-resilience.bundle`, `fix-resilience2.bundle`, `run_all_log.txt`.
- Nol proses python berjalan.

Yang harus diputuskan sebelum `git pull` di remote: perubahan lokal remote itu diperiksa dulu atau ditimpa. Menimpa tanpa memeriksa berisiko menghapus hasil percobaan `densenet121` yang sempat dijalankan manual di sana.

Sekalian di langkah ini: jalankan `python -c "import pymrmr"` di remote untuk memastikan seleksi fitur yang selama ini berjalan benar-benar mRMR, bukan fallback `mutual_info_classif`.

Tidak ada langkah lain yang boleh dimulai sebelum ini beres.

---

## 3. Langkah 1: jalankan ulang ablasi fusi dengan perbaikan resolusi

Nol kode baru. Kodenya sudah selesai sejak Rev1 tugas 1.

Perintah: `python -m src.stage_03b_fusion --config configs/config.yaml`

Cakupan: 7 backbone Track 1, 5 arm, 5 fold. Menggantikan `artifacts/results/fusion/ablation_summary.csv` dan `delong_fusion.csv` secara utuh.

Gerbang verifikasi: `cnn_only` untuk `densenet201` harus naik dari 0.6432 mendekati AUC *standalone*-nya 0.8988 di `summary_binary.csv`. Kalau tidak naik, perbaikan `input_size` tidak benar-benar terpasang dan seluruh langkah berikutnya ditunda.

Pertanyaan yang dijawab langkah ini: setelah bug diperbaiki, apakah ada arm fusi yang mengalahkan radiomics-only 0.9313? Kalau ada, seluruh temuan negatif Track 1 harus ditulis ulang.

Biaya perkiraan: setengah hari GPU.

---

## 4. Langkah 2: fine-tuning dua tahap dengan BatchNorm dibekukan

Ini satu-satunya bagian yang butuh kode baru dalam jumlah berarti.

### Berkas baru: `src/training/finetune.py`

Fungsi yang perlu ada:

- `freeze_all(model)` menyetel `requires_grad = False` pada seluruh parameter.
- `unfreeze_top_modules(backbone, fraction)` membuka kembali blok teratas backbone.
- `apply_bn_eval(model)` menyetel setiap modul `BatchNorm2d` ke `.eval()`.
- `build_param_groups(model, head_lr, decay_factor)` menghasilkan `param_groups` untuk optimizer dengan *learning rate* yang menurun ke arah lapisan bawah.

Dua jebakan yang harus ditangani eksplisit, keduanya tidak disebut dokumen rev2:

1. **`model.train()` mengembalikan BatchNorm ke mode train.** Kalau `apply_bn_eval` hanya dipanggil sekali sebelum *loop* epoch, pembekuan statistik BatchNorm akan batal pada epoch pertama. Pemanggilannya harus berada di dalam *loop* epoch, tepat setelah `model.train()`. Ini penyebab kegagalan senyap yang paling mungkin terjadi pada langkah ini, dan harus punya *unit test* sendiri.
2. **Potongan kode rev2 memilih lapisan lewat `named_parameters()[-10%:]`.** Cara ini memotong berdasarkan urutan parameter, bukan batas blok arsitektur, sehingga bisa membuka `weight` sebuah konvolusi tanpa `bias`-nya, atau membuka separuh blok. Rencana ini menyimpang dengan sengaja: pembukaan dilakukan per *child module* teratas dari backbone. Penyimpangan ini dicatat di manuskrip.

### Titik sisip

| Berkas | Baris | Perubahan |
|---|---|---|
| `src/stage_03_train.py` | 198 | `_build_optimizer(optimizer_name, model.parameters(), ...)` menerima `param_groups`, bukan `model.parameters()` |
| `src/stage_03_train.py` | 169 | `build_run_id` menambah sufiks `_ft{fraction:g}` **hanya** saat fine-tuning aktif, sehingga 215 `run_id` yang sudah ada tetap identik byte per byte. Disiplin yang sama seperti Rev1 tugas 7 |
| `src/stage_03b_fusion.py` | 176 | `torch.optim.AdamW(model.parameters(), ...)` menerima `param_groups` |
| `configs/config.yaml` | blok `train` | Tambah sub-blok `finetune` dengan kunci `enabled: false`, `stage1_epochs`, `unfreeze_fraction`, `bn_eval`, `lr_decay_factor: 2.6` |

`enabled: false` sebagai *default* wajib, supaya jalur konfigurasi lama menghasilkan angka yang sama persis.

### Uji yang ditulis lebih dulu

1. Setelah `model.train()` dipanggil, setiap modul `BatchNorm2d` masih `training == False` ketika `bn_eval` aktif.
2. `unfreeze_top_modules(backbone, 0.10)` menghasilkan jumlah parameter yang dapat dilatih lebih besar dari nol dan lebih kecil dari total.
3. `build_param_groups` menghasilkan *learning rate* kelompok backbone sama dengan `head_lr / 2.6`.
4. Dengan `finetune.enabled: false`, `build_run_id` menghasilkan string yang identik dengan sebelum perubahan.

### Eksekusi

3 backbone terpilih, 5 fold, dua konfigurasi: `unfreeze_fraction` 0.10 dan 0.20. Total 30 run.

Gerbang: AUC CNN-only >= 0.92 pada minimal satu backbone. Kalau tidak tercapai, coba `bn_eval: false` sebagai pembanding sebelum melanjutkan ke langkah 3.

Biaya perkiraan: satu hari GPU.

---

## 5. Langkah 3: jalankan arm fusi yang sudah direbalans

Nol kode baru untuk arsitekturnya. Yang dibutuhkan hanya perubahan konfigurasi dan satu keputusan desain sweep.

Konfigurasi target di `configs/config.yaml` blok `track1_fusion`:

```yaml
fusion_arms: ["concat", "branch_norm", "gmu"]
branch_norm_proj_dim: 32
modality_dropout_rate: 0.2
aux_loss_weight: 0.3
```

### Masalah desain yang harus dibereskan dulu

`modality_dropout_rate` dan `aux_loss_weight` dibaca sekali per proses di `src/stage_03b_fusion.py:170-172`, lalu diteruskan ke setiap arm. Artinya menyalakan keduanya akan mengubah **semua** arm sekaligus. Dengan konfigurasi di atas, tidak mungkin memisahkan kontribusi GMU dari kontribusi modality dropout, padahal rev2 memperlakukan keduanya sebagai intervensi terpisah.

Dua pilihan, keduanya sah:

- **Pilihan A, tanpa kode baru:** jalankan `stage_03b_fusion` dua kali dengan dua berkas konfigurasi berbeda, satu dengan regularizer mati dan satu dengan regularizer hidup. Sufiks `run_id` yang sudah ada (`_moddrop{rate}`, `_aux{weight}`, lihat `src/stage_03b_fusion.py:216-220`) sudah cukup memisahkan barisnya di `runs.csv`. Biaya: dua kali waktu GPU.
- **Pilihan B, kode kecil:** jadikan regularizer sumbu sweep sendiri, misalnya `regularizer_grid: [{moddrop: 0.0, aux: 0.0}, {moddrop: 0.2, aux: 0.3}]`, lalu iterasi silang dengan `fusion_arms`. Lebih rapi dan hanya satu kali proses, tetapi menyentuh `_run_backbone_arms`.

Rencana ini memilih **Pilihan A**, karena langkah 3 seharusnya nol kode baru dan Pilihan B menambah risiko regresi pada jalur yang sudah terbukti jalan.

### Eksekusi

3 backbone, 3 arm, 5 fold, 2 kondisi regularizer. Total 90 run fusi.

Gerbang: AUC fusi intermediate melampaui 0.9313, atau minimal satu uji DeLong berpasangan mendukung fusi. Kalau nol dari sekian pasang tetap mendukung fusi setelah bug diperbaiki, fine-tuning dijalankan, dan gating dipakai, maka temuan negatif Track 1 justru menjadi jauh lebih kuat dan layak jadi klaim utama manuskrip.

Observabilitas yang sudah tersedia dan wajib dicatat: `model.last_branch_norms` menyimpan `img_gate` dan `rad_gate` untuk arm GMU (`src/models/fusion_net.py:186-189`). Rata-rata nilai gate per fold adalah bukti langsung apakah gate benar-benar menekan cabang CNN yang lemah. Angka ini masuk manuskrip sebagai tabel tersendiri.

Biaya perkiraan: 1.5 hari GPU.

---

## 6. Langkah 4: nested CV

Ini yang membuat angka akhir jujur, dan ini juga yang akan menurunkan angka.

### Kebocoran yang sedang terjadi

Pada `src/stage_03b_fusion.py:194-207`, *early stopping* dan pemilihan *checkpoint* terbaik dilakukan terhadap `val_loader`, yaitu fold validasi luar itu sendiri. Fold yang sama kemudian dipakai untuk melaporkan AUC. Jadi AUC yang dilaporkan sekarang adalah AUC dari epoch terbaik **pada fold itu**, bukan AUC dari model yang dipilih secara buta. Pola yang sama ada di `src/stage_03_train.py`.

### Perubahan

Berkas baru `src/utils/nested_split.py` dengan satu fungsi:

```
inner_split(train_df, seed, val_fraction=0.2) -> (inner_train_df, inner_val_df)
```

Pemisahan dilakukan per `patient_id`, bukan per nodul, agar konsisten dengan disiplin split yang sudah dipakai di `src/data_loading/lidc_loader.py`.

Di `_train_fusion_fold`, *early stopping* dan penyimpanan `best_pt` berpindah ke `inner_val`. Fold luar hanya di-*score* satu kali di akhir, memakai *checkpoint* yang sudah terpilih.

Kunci konfigurasi baru: `evaluation.nested_cv`, *default* `false`.

### Konsekuensi yang wajib ditulis, bukan disembunyikan

AUC fold luar hampir pasti **turun** dibanding angka sekarang. Itu justru tujuannya. Konsekuensi untuk penulisan:

- Angka lama tidak dihapus. Angka lama dilaporkan sebagai "seleksi model pada fold validasi", angka baru sebagai "nested CV".
- Selisih antara keduanya adalah estimasi besarnya bias seleksi pada protokol ini. Ini kontribusi metodologis yang layak dilaporkan, sejalan dengan rev2 bagian 7.
- Radiomics-only juga harus dijalankan ulang di bawah nested CV. Kalau tidak, perbandingan fusi lawan radiomics jadi tidak adil: satu arm dievaluasi ketat, arm lain longgar.

Biaya perkiraan: setengah hari untuk kode, plus pengulangan langkah 3 di bawah nested CV.

---

## 7. Langkah 5: ensembling

### Koreksi terhadap rev2

Dokumen rev2 menyarankan "rata-ratakan probabilitas 5 model fold". Ini tidak bisa dilakukan pada protokol yang ada. Kelima model fold divalidasi pada subset pasien yang **berbeda-beda**, jadi tidak ada satu pun kasus yang punya prediksi dari kelima model sekaligus. Merata-ratakannya butuh test set bersama, dan test set bersama itu justru yang tidak ada.

Yang sah dikerjakan:

- **Ensembling backbone, per fold luar.** Untuk satu fold luar, ketiga backbone memprediksi kasus validasi yang persis sama. Rata-rata probabilitasnya sah dan bisa diuji DeLong berpasangan terhadap backbone tunggal terbaik. Ini yang dikerjakan.
- **Ensembling inner-fold, di bawah nested CV.** Setelah langkah 4, beberapa model yang dilatih pada *inner split* berbeda bisa dirata-ratakan lalu di-*score* satu kali pada fold luar. Sah, tetapi biayanya berlipat dan hanya dikerjakan kalau waktu tersisa.
- **Ensembling fold lintas fold luar: tidak dikerjakan.** Alasannya ditulis di bagian batasan manuskrip.

### Berkas baru

- `src/evaluation/ensemble.py`: `average_probs(prob_arrays)` dan `rank_average_probs(prob_arrays)`.
- `src/stage_04d_ensemble.py`: membaca `artifacts/results/preds/*.npz`, menghasilkan `artifacts/results/fusion/ensemble_summary.csv` dan `delong_ensemble.csv`.

Prasyarat yang harus dicek dulu: apakah `preds/*.npz` menyimpan urutan kasus yang sama antar backbone untuk fold yang sama. Kalau tidak, dibutuhkan kunci `(patient_id, nodule_idx)` di dalam `.npz`, dan itu berarti perubahan pada tahap yang menulis berkas tersebut.

Biaya perkiraan: setengah hari.

---

## 8. Langkah 6: perbarui manuskrip dan laporan

Berkas yang diperbarui:

- `paper/track1/main.tex`: bagian Results, Discussion, dan Limitations ditulis ulang dengan angka pasca perbaikan. Kalau kesimpulan berbalik dan fusi menang, framing temuan negatif dibuang seluruhnya.
- `docs/laporan/LAPORAN_TRACK1_FUSION_XAI.md`: subsection `#### Batasan pada kolom cnn_only` di bagian 6.1 dihapus setelah langkah 1 selesai, digantikan angka nyata.
- `docs/laporan/REFERENSI_DIBUTUHKAN.md`: tambah referensi yang dikutip rev2, terutama Raghu dkk. (Transfusion, NeurIPS 2019, arXiv:1902.07208), Howard dan Ruder (ULMFiT, ACL 2018, arXiv:1801.06146), Peng dkk. (OGM-GE, CVPR 2022) sebagai *related work* meskipun metodenya tidak dipakai, Baltatzis dkk. (MLMI 2021, arXiv:2108.05386) untuk analisis *ceiling*, dan Demircioglu (Insights into Imaging 2021, DOI 10.1186/s13244-021-01115-1) untuk seleksi fitur di dalam *loop* CV.
- `docs/revisi/rev1/TASKBOARD.md`: tugas 1, 5a, 5b, 5c berpindah dari `done-code` ke `done`.

Jangan mengarang citekey. Referensi di atas belum ada di `paper/refs.bib`, jadi kalimatnya ditulis dulu tanpa `\cite{}` dan kuncinya ditambahkan lewat Zotero.

Bagian baru yang layak ditulis di manuskrip Track 1, karena tidak ada di versi sekarang:

1. Analisis *ceiling*: mengapa angka 94 sampai 98 persen di literatur tidak sebanding dengan protokol ini. Bahan lengkap ada di rev2 bagian 5.
2. Selisih AUC antara seleksi pada fold validasi dan nested CV, sebagai ukuran bias seleksi.
3. Tabel aktivasi gate GMU sebagai bukti kuantitatif dominasi modalitas.

---

## 9. Verifikasi

1. Seluruh `pytest` baru untuk `src/training/finetune.py` hijau. Catatan: 12 kegagalan `pytest` yang sudah ada di `main` sebelum perubahan ini (statsmodels tidak terpasang, hitungan `_NAME_MAP` basi, `KeyError` pada `grade3`) tidak dihitung sebagai regresi, tetapi jumlahnya harus tetap 12, tidak boleh bertambah.
2. Dengan `finetune.enabled: false` dan `nested_cv: false`, satu run ulang menghasilkan `run_id` dan AUC yang identik dengan baris lama di `runs.csv`. Ini gerbang non-regresi terpenting.
3. `densenet201 cnn_only` naik dari 0.6432 mendekati 0.8988.
4. Setiap angka yang masuk manuskrip bisa ditunjuk ke satu baris CSV nyata.
5. BatchNorm benar-benar beku: uji unit yang memanggil `model.train()` lalu memeriksa `module.training` pada setiap `BatchNorm2d`.
6. Audit anti-slop pada seluruh teks baru: nol tanda pisah em, nol kata dari daftar terlarang.

---

## 10. Yang tidak dikerjakan

- OGM-GE gradient modulation. Di luar cakupan yang dipilih. Disebut di *related work* sebagai arah lanjutan.
- Optuna dan ASHA. Di luar cakupan. Konsekuensinya, `unfreeze_fraction` hanya dicoba pada dua nilai, bukan dicari.
- Input multi-view 3 bidang. Butuh pengulangan preprocessing patch dari DICOM, biayanya tidak sebanding dalam anggaran ini.
- Sweep classifier radiomics (RBF-SVM, LightGBM). Di luar cakupan.
- Test-time augmentation. Rev2 sendiri melaporkan bukti bahwa TTA sering menurunkan akurasi.
- Held-out test set terpisah. Diganti nested CV sesuai keputusan yang dikunci.
- Perubahan apa pun pada Track 2. Fold pasien tidak berubah, jadi Track 2 tetap valid tanpa dijalankan ulang.

---

## 11. Urutan eksekusi dan ketergantungan

```
Langkah 0  sinkron remote            (memblokir semua)
   |
Langkah 1  re-run ablasi, 7 backbone (memblokir 3 dan 5)
   |
   +-- Langkah 2  fine-tuning        (kode baru, boleh ditulis paralel saat langkah 1 berjalan)
   |
Langkah 3  arm fusi rebalanced       (butuh 1 dan 2)
   |
Langkah 4  nested CV                 (mengulang 3 di bawah protokol ketat)
   |
Langkah 5  ensembling backbone       (butuh 4)
   |
Langkah 6  manuskrip dan laporan     (butuh semua)
```

Total perkiraan: 4 sampai 5 hari GPU, ditambah sekitar satu hari untuk kode dan uji langkah 2 dan 4.

Satu-satunya bagian yang boleh dikerjakan paralel adalah penulisan kode langkah 2 sambil langkah 1 berjalan di GPU, karena keduanya menyentuh berkas yang berbeda.
