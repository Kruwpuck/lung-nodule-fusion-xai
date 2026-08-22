# GOAL run03 — Fine-tuning Bertahap + Ensembling Backbone

## Tujuan

Menaikkan performa cabang CNN, yang selama ini jadi hambatan fusi. `cnn_only` 0,8927 tertinggal jauh dari `radiomics_only` 0,9318 — selisih 0,039. Dua mekanisme yang **belum pernah dijalankan** dan paling mungkin menutup jarak itu:

1. **Two-stage fine-tuning** dengan BatchNorm dibekukan (Step 2 rev2, tidak pernah dieksekusi)
2. **Ensembling backbone** per fold luar (Step 5 rev2, tidak pernah dieksekusi)

Bukan mekanisme fusi baru. Pencarian mekanisme fusi sudah ditutup — yang dikejar di sini adalah kualitas masukan ke fusi.

---

## Slot isian

```
PROJECT_ROOT     = <path repo lung-nodule-fusion-xai>
GOAL             = Fine-tuning bertahap + ensembling backbone untuk menaikkan cabang CNN, lalu turunkan ulang seluruh metrik hilir
DONE_CRITERIA    = 7 gate di bawah, semua wajib lolos dengan bukti tercetak
GPU_HOST         = PC remote ini
MAX_ITER         = 20
RUN_ID           = 2026-08-05-run03
RESERVED_ENV     = artifacts/results/_baseline_pre_rev2/, artifacts/splits/folds.json, .venv
```

---

## FASE F-0 — Pra-registrasi (WAJIB, sebelum satu pun GPU jalan)

Tulis ke `handoff/PREREG_run03.md` **sebelum** eksperimen:

- [ ] **Prediksi eksplisit:** fine-tuning diperkirakan menaikkan `cnn_only` +1 s.d. +3 poin AUC (rentang lazim literatur); ensembling +1 s.d. +2 poin.
- [ ] **Aturan keputusan, ditetapkan sekarang:**
  - Fine-tuning dipakai kalau `cnn_only` naik **dan** tidak menurunkan arm lain.
  - Fusi dinyatakan menang atas radiomics HANYA kalau DeLong p<0,05. Tidak ada pelonggaran.
  - Kalau setelah fine-tuning + ensembling fusi tetap tidak menang, **pencarian ditutup permanen** dan hasil dilaporkan apa adanya.
- [ ] **Konfigurasi yang diuji ditetapkan di muka**, maksimal 3 varian unfreeze (0%, 10%, 20%). Tidak boleh ditambah di tengah jalan.
- [ ] **Semua konfigurasi dilaporkan**, bukan hanya yang menang.

> Tanpa F-0, seluruh run ini jadi pencarian tanpa batas. Ini yang membedakan perbaikan metodologis dari pengejaran angka.

---

## FASE F-1 — Two-stage fine-tuning

### Cakupan
3 backbone: `convnext_tiny`, `densenet201`, `densenet121`. Semua 5 fold.

### Protokol
**Tahap 1 — head only:** bekukan seluruh backbone, latih head sampai konvergen.
**Tahap 2 — unfreeze bertahap:** buka N% lapisan terakhir dengan discriminative LR (÷2,6 per grup lapisan mundur).

### Rincian teknis yang WAJIB benar

- [ ] **BatchNorm dibekukan di DALAM loop epoch.** `model.train()` mengembalikan BN ke mode train tiap epoch, jadi `apply_bn_eval()` harus dipanggil **setelah** `model.train()` di setiap epoch, bukan sekali di awal. Ini bug yang sudah teridentifikasi sebelumnya — jangan terulang.
- [ ] **Unfreeze per child module**, bukan slice `named_parameters()[-10%:]`. Slice per-parameter bisa membuka weight tanpa bias.
- [ ] **Nested CV dipertahankan**: inner-val untuk seleksi epoch, fold luar hanya dievaluasi sekali. Jangan kembalikan kebocoran yang sudah ditutup.
- [ ] **`folds.json` tidak disentuh.**
- [ ] Input 96px konsisten; `input_size` diteruskan eksplisit ke semua konstruktor.

### Varian yang diuji (dari F-0)
- unfreeze 0% (baseline head-only)
- unfreeze 10%
- unfreeze 20%

**Gate F-1:**
- [ ] `cnn_only` per backbone tercetak untuk ketiga varian × 5 fold
- [ ] Varian terbaik ditentukan dari **inner-val**, bukan fold luar
- [ ] Bandingkan dengan baseline: convnext_tiny 0,9055 · densenet201 0,8988 · densenet121 0,8959
- [ ] **Kalau nol varian menaikkan `cnn_only`**, laporkan dan lanjut ke F-3 (ensembling) tanpa fine-tuning

---

## FASE F-2 — Turunkan ulang ablasi fusi

Checkpoint baru mengubah embedding dan probabilitas, jadi `fusion_early`, `fusion_intermediate`, dan `fusion_late` **wajib** dihitung ulang.

- [ ] Jalankan ulang ablasi untuk 3 backbone dengan checkpoint hasil F-1
- [ ] **Kontrol negatif:** `radiomics_only` harus tetap dalam ±0,0036 dari 0,9318. Kalau bergerak jauh, berhenti — itu tanda ada yang berubah di pipeline yang seharusnya tidak.
- [ ] DeLong: tiap arm fusi vs `radiomics_only`, dan vs `cnn_only` baru

**Gate F-2:**
- [ ] Kontrol negatif lolos
- [ ] Tabel lengkap 3 backbone × 5 arm × 5 fold
- [ ] DeLong tercetak dengan p-value

---

## FASE F-3 — Ensembling backbone

**Catatan penting:** ensembling **fold** tidak mungkin — tiap fold punya pasien validasi berbeda, nol kasus punya prediksi dari kelima model. Yang sah adalah **ensembling backbone per fold luar**.

- [ ] Rerata probabilitas 3 backbone dalam fold luar yang sama
- [ ] Uji pada `cnn_only` dan pada arm fusi terbaik
- [ ] DeLong: ensemble vs backbone tunggal terbaik, dan vs `radiomics_only`

**Prasyarat yang harus dicek dulu:**
- [ ] Verifikasi `preds/*.npz` punya **urutan kasus identik** antar backbone dalam fold yang sama. Kalau tidak, rerata probabilitasnya salah alamat. Cetak buktinya.

**Gate F-3:**
- [ ] Urutan kasus terverifikasi identik
- [ ] AUC ensemble tercetak per fold
- [ ] DeLong vs pembanding

---

## FASE F-4 — Turunkan ulang metrik hilir

Checkpoint berubah ⇒ semua yang diturunkan darinya basi. Ini konsekuensi yang sudah kamu terima.

- [ ] **XAI**: Layer-CAM + pointing accuracy/IoU/Dice/energy untuk checkpoint baru, memakai `fixed_display_samples.json` yang sama
- [ ] **Statistik**: Friedman, Nemenyi, DeLong per-arm — regenerasi dari prediksi baru
- [ ] **Efisiensi**: params/FLOPs tidak berubah (arsitektur sama), tapi latensi bisa berubah — cek

**Gate F-4:**
- [ ] Seluruh CSV hilir tersimpan dengan provenance lengkap
- [ ] Nol angka di naskah yang masih merujuk checkpoint lama

---

## ATURAN PROVENANSI (baru — dari insiden densenet121)

Insiden terakhir: prediksi densenet121 ditimpa di tempat oleh pelatihan ulang, metrik hilir tidak pernah diturunkan ulang, dan satu paragraf naskah menggambarkan model yang sudah tidak ada. Keadaan perantaranya musnah.

Wajib berlaku di run ini:

- [ ] **Prediksi ber-versi, tidak ditimpa.** Simpan ke `preds/{run_id}/` — jangan menimpa `preds/*.npz` yang lama.
- [ ] **Tiap CSV metrik mencatat hash/timestamp masukannya**, sehingga kebasian terdeteksi otomatis.
- [ ] **Tiap baris hasil membawa** `run_id`, `commit_sha`, `input_size`, `checkpoint_mtime`.
- [ ] Sebelum menurunkan metrik apa pun, **verifikasi checkpoint mtime** cocok dengan yang tercatat.

---

## LARANGAN

- Dilarang menyentuh `_baseline_pre_rev2/`, `folds.json`, `.venv`
- Dilarang menimpa `preds/*.npz` yang sudah ada
- Dilarang mengubah daftar 3 backbone
- Dilarang menambah varian unfreeze di luar yang ditetapkan di F-0
- Dilarang mengembalikan seleksi epoch ke fold luar (kebocoran yang sudah ditutup)
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git push`
- Dilarang mencari mekanisme fusi baru — run ini soal kualitas cabang CNN

---

## KALAU HASILNYA NEGATIF

Kalau setelah fine-tuning dan ensembling `cnn_only` tetap tidak naik berarti, atau fusi tetap tidak mengalahkan `radiomics_only` dengan p<0,05:

- Status tetap `done`, bukan gagal.
- Pencarian performa ditutup permanen.
- Hasilnya dilaporkan: dua mekanisme terakhir yang belum dicoba pun tidak membalikkan kesimpulan.
- Itu justru **memperkuat** klaim "radiomics setara/mengungguli fusi pada n≈1400", karena menunjukkan kesimpulan bertahan setelah upaya perbaikan yang wajar habis.

Agent dilarang memperluas ruang pencarian atas inisiatif sendiri.

---

## KONSEKUENSI YANG SUDAH DITERIMA

Run ini membatalkan sebagian hasil yang sudah ditulis:

- Tabel Track 1 (ablasi fusi) dihitung ulang
- Metrik XAI untuk 3 backbone dihitung ulang
- Statistik Friedman/Nemenyi/DeLong diregenerasi
- Penulisan naskah mundur sampai angka baru stabil

Penulisan ulang §6.3 Track 2 dan `main.tex:241` (koreksi klaim keterpisahan) **tetap dikerjakan** — itu koreksi kesalahan pelaporan yang berdiri sendiri, tidak bergantung pada checkpoint.