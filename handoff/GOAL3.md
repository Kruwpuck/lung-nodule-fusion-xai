# GOAL run03b — Faithfulness, bukan sekadar lokalisasi

**Belum dijalankan.** Ditulis 19 Agustus 2026 sebagai fase terpisah setelah uji
ekuivalensi (`2026-08-19-run03`) selesai dan naskah stabil. Dipisah dengan sengaja:
beban GPU-nya besar, dan hasilnya bisa **mengubah pembingkaian Limitations**. Kalau
ROAD menunjukkan peta `fusion_late` kurang faithful, batasan 8 dan 9 laporan Track 1
harus ditulis ulang — lebih murah menulis satu bagian sekali daripada menulis seluruh
naskah dua kali.

---

## Masalah yang ditutup

Manuskrip Track 1 melaporkan lima metrik XAI: dice, IoU, dice size-matched, pointing
accuracy, energy pointing. Kelimanya mengukur **satu hal yang sama** — di mana peta
jatuh relatif terhadap mask radiolog. Tidak satu pun menanyakan apakah peta itu
mencerminkan komputasi yang benar-benar dilakukan model.

Peta bisa jatuh tepat di nodul dan tetap tidak faithful: ia bisa menyorot wilayah yang
kalau dihapus tidak mengubah prediksi sama sekali. Reviewer akan menyebut ini, dan
Limitations sudah mengakuinya lebih dulu. Fase ini mengubah pengakuan itu jadi angka.

---

## Slot isian

```
PROJECT_ROOT     = <path repo lung-nodule-fusion-xai>
GOAL             = Kuantifikasi faithfulness peta cnn_only dan fusion_late, plus perluasan jalur CAM ke arm fusi berparameter
DONE_CRITERIA    = 5 gate di bawah, semua wajib lolos dengan bukti tercetak
GPU_HOST         = PC remote ini
RUN_ID           = <YYYY-MM-DD-runNN, tetapkan saat mulai>
SOURCE_RUN_ID    = 2026-08-04-run02
RESERVED_ENV     = artifacts/results/_baseline_pre_rev2/, artifacts/splits/folds.json, .venv
```

---

## Tugas

### F-0. ROAD pada kedua arm yang sudah punya CAM

`grad-cam` 1.5.3 sudah terpasang dan memuat `pytorch_grad_cam.metrics.road`
(`ROADMostRelevantFirst`, `ROADLeastRelevantFirst`, `ROADCombined`). Tidak ada
dependency baru.

Jalankan pada `cnn_only` dan `fusion_late`, ketiga backbone, kedua himpunan sampel
(`fixed6` dan `fold0_60`), memakai jalur CAM yang sudah ada di
`src/xai/gradcam_utils.py:127`. Laporkan per persentase piksel yang dihapus.

### F-1. Kurva deletion dan insertion

Hand-rolled, bukan library — bentuknya sederhana dan `grad-cam` tidak menyediakannya
langsung. Urutkan piksel menurut nilai CAM, hapus atau masukkan bertahap, catat
probabilitas kelas di tiap langkah, laporkan AUC kurvanya.

Catatan teknis yang menentukan implementasinya:

- **Masking terjadi pada tensor 64×64 ternormalisasi**, bukan pada 96×96 hasil resize.
  `BackboneClassifier.forward` (`src/models/backbones.py:254`) melakukan interpolasi
  64→96 di dalam dirinya, dan CAM yang dikembalikan `compute_gradcam` juga sudah
  64×64 dalam `[0, 1]`. Jadi peta dan tensor sudah sejajar; jangan menambah resize.
- **Satu batch, bukan K forward pass.** `forward` adalah
  `classifier(features(_maybe_resize(x)))` tanpa hook maupun state, jadi tumpukan
  `(K, 3, 64, 64)` berisi salinan bertopeng bertahap bisa lewat sekali jalan di bawah
  `torch.no_grad()`.
- **Baseline perlu dipilih dan dicatat.** Patch mentah `patch.npy` adalah HU float32
  (kisaran sekitar −1009 sampai +841), dan tensor CAM adalah min-max reversibel darinya,
  jadi baseline blur, mean-fill, maupun lantai HU semuanya tersedia. Pilih satu,
  tulis alasannya, jangan diam-diam.

### F-2. HiResCAM sebagai pembanding

**Sudah terkabel**, `src/xai/gradcam_utils.py:163,177`, dan sudah dipakai grid metode
CAM di `src/stage_07e_grid_cam_method.py:24`. Ini argumen `method="hirescam"`, bukan
kode CAM baru. Jalankan F-0 dan F-1 pada Layer-CAM dan HiResCAM, lalu bandingkan.

Perhatikan bahwa jaminan faithfulness HiResCAM terbatas pada arsitektur tanpa pooling
setelah konvolusi terakhir (`docs/implementation/fix xai 2.md:128`). Periksa apakah
ketiga backbone Track 1 memenuhi syarat itu sebelum mengutip jaminannya; kalau tidak,
laporkan HiResCAM sebagai pembanding empiris saja.

### F-3. Jalur CAM untuk arm fusi berparameter

Ini yang memperlebar klaim explainability dari dua arm jadi empat, dan tanpa ini
batasan 9 laporan Track 1 tetap berlaku.

`fusion_early` dan `fusion_intermediate` **tidak punya jalur CAM sama sekali** hari ini.
`_get_target_layer` sudah memuat cabang `hasattr(model, "cnn_branch")` untuk `FusionNet`
(`src/xai/gradcam_utils.py:27-82`), tapi `FusionNet.forward` menerima **dua argumen**
dan `pytorch_grad_cam` tidak bisa menjalankannya tanpa wrapper. Wrapper itu pekerjaan
sebenarnya: bekukan vektor radiomik satu sampel, ekspos `forward(x_image)` saja.

`fusion_early` lebih sulit — cabang keputusannya XGBoost di atas embedding CNN, sehingga
tidak ada gradien yang mengalir balik ke citra. Kemungkinan besar arm ini memang tidak
punya peta spasial, dan **kalau begitu kesimpulannya, tulis begitu**; itu temuan, bukan
kegagalan.

### F-4. Perbarui satu bagian, bukan seluruh naskah

Setelah F-0 sampai F-3 punya angka:

- Laporan Track 1: bagian baru §6.5, plus batasan 8 dan 9 §8 ditulis ulang.
- Manuskrip: paragraf faithfulness di Limitations diganti hasil, penanda `\CITE{ROAD}`
  yang sudah terpasang diisi.
- Jangan menyentuh §6.4 maupun tabel TOST. Fase ini tidak mengubah klaim AUC.

---

## Gate

| Gate | Syarat |
|---|---|
| **G-0** | Sampel XAI berasal dari `artifacts/xai/fixed_display_samples.json`, dibaca saja, tidak pernah ditulis |
| **G-1** | Setiap metrik faithfulness punya baseline yang dinyatakan eksplisit di CSV-nya, bukan tersirat di kode |
| **G-2** | `n_disagree` dilaporkan berdampingan dengan tiap metrik, sama seperti `xai_metrics_fusion.csv` sekarang |
| **G-3** | Kalau ROAD dan pointing accuracy berlawanan arah, **keduanya dilaporkan**. Dilarang memilih metrik yang lebih menguntungkan |
| **G-4** | Kolom `run_id` dan `commit_sha` jadi dua kolom pertama tiap CSV baru, mengikuti konvensi run02 |

---

## Batasan

- Sampel XAI **wajib** dari `fixed_display_samples.json`, tidak boleh dipilih ulang
- Backbone tetap: `convnext_tiny`, `densenet201`, `densenet121`
- Dilarang menyentuh `_baseline_pre_rev2/`, `folds.json`, `.venv`
- Dilarang mengubah ambang atau definisi metrik yang sudah ada
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git push`
- Dilarang menyimpulkan keunggulan faithfulness tanpa angka pendukung
- **Dilarang menyembunyikan hasil F-0 kalau tidak menguntungkan**

## Perangkap yang sudah diketahui

1. **Nol yang tersimpan per-sampel.** Seluruh CSV XAI sekarang hanya menyimpan rerata.
   Kurva deletion/insertion butuh sink per-sampel yang **belum ada** — tidak ada yang
   bisa diperluas, harus dibuat.
2. **`stage_08b.run_xai` dijaga keberadaan `xai_metrics_fusion.csv`** (`:224-226`).
   Menjalankannya ulang berarti berurusan dengan berkas itu, dan `rm` masuk daftar
   tolak pada sesi bertipe ini.
3. **Gate G-2/G-3 `stage_08c_run02_gates.py` mengunci jumlah baris.** `XAI_METRICS`
   di `:52` hard-coded empat metrik dan gate-nya menuntut tepat
   3 backbone × 2 sample_set × 4 metrik = 24 baris. Menambah kolom ke
   `xai_metrics_fusion.csv` aman; menambah nama ke `XAI_METRICS` **mengubah jumlah baris
   yang diperiksa gate**. Berkas itu juga masuk daftar tolak sunting pada sesi terakhir.
4. **Target layer yang sebenarnya dipakai bukan yang tertulis.** `_auto_target_layer`
   mencari keluaran 4-D dengan tinggi di `[7, 10]`; pada input 96 piksel, tinggi yang
   tersedia adalah `[1, 3, 6, 12, 24]` (convnext_tiny) dan `[1, 3, 6, 12, 24, 48]`
   (densenet), sehingga **tidak ada yang cocok** dan jalur fallback `_get_target_layer`
   yang berjalan — menghasilkan peta 3×3. Caption di
   `src/stage_07f_xai_comparability.py:89-91` masih menyatakan pita `[7, 10]` dipakai
   dan sudah basi untuk backbone Track 1. Perbaiki caption itu, atau setidaknya jangan
   mengutipnya.

## Yang tidak dikerjakan di fase ini

- Nol pelatihan ulang. Checkpoint yang ada dipakai apa adanya.
- Nol perubahan pada klaim AUC, tabel DeLong, maupun tabel TOST.
- Nol pencarian arsitektur fusi baru.
- `STATE.json` disentuh hanya kalau loop harness memang dijalankan untuk fase ini.
