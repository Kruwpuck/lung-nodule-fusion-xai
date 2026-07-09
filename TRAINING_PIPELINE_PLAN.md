# LungFuseNet — Training Pipeline Architecture (Planning)

Dokumen ini adalah **rencana arsitektur pipeline training** yang resumable & anti-hang, untuk di-review sebelum diimplementasi di repo. Belum ada kode final — ini blueprint + pattern.

---

## Prinsip Inti

1. **Checkpoint-based & resumable** — tiap stage berat disimpan ke disk. Kalau runtime mati/hang, jalanin ulang → stage yang sudah selesai di-**skip otomatis**, lanjut dari yang belum.
2. **Semua metrik ke CSV** — loss, akurasi, AUC, convergence per epoch disimpen ke CSV. Mau ganti plot/analisis? Baca CSV, **tidak perlu training ulang**.
3. **Preprocessing sekali, dipakai selamanya** — patches & radiomics features di-cache. Training baca dari cache.
4. **Modular `.py` + orchestrator tipis** — logic di file `.py` yang bisa di-test & di-resume; notebook/script cuma manggil stage.
5. **Fair setup terjaga** — preprocessing, split, hyperparameter identik untuk semua model (dari 1 config file).

---

## Kenapa BUKAN 1 ipynb Raksasa

| 1 ipynb full | Modular + orchestrator (rekomendasi) |
|---|---|
| Semua nempel 1 runtime | Tiap stage independen |
| Putus di epoch 40 → hilang semua | Putus → lanjut dari checkpoint |
| Susah di-test per bagian | Tiap modul bisa di-test sendiri |
| Rawan hang (RAM menumpuk) | RAM dibersihkan antar stage |
| Ganti analisis → re-run semua | Ganti analisis → baca CSV saja |

**Kesimpulan:** logic di `.py`, orchestrator (notebook untuk Colab / `run.sh` untuk lokal) cuma jadi "remote" yang manggil stage satu per satu.

---

## Struktur File

```
LungFuseNet/
├── configs/
│   └── config.yaml              # SATU sumber kebenaran: path, hyperparam, seed, daftar model
├── src/
│   ├── stage_00_preprocess.py   # DICOM → patches (RESUMABLE, cache ke disk)
│   ├── stage_01_radiomics.py    # patches → radiomic features CSV (RESUMABLE) [Track 1]
│   ├── stage_02_split.py        # patient-level k-fold split (disimpen ke JSON, fixed)
│   ├── stage_03_train.py        # training 1 model 1 fold (checkpoint + CSV log)
│   ├── stage_04_evaluate.py     # baca checkpoint → metrik final ke CSV
│   ├── stage_05_xai.py          # Grad-CAM + SHAP (baca model tersimpan) [Track 1]
│   ├── utils/
│   │   ├── io.py                # cache helper: cek "sudah ada?" sebelum proses
│   │   ├── logger.py            # tulis metrik ke CSV per epoch
│   │   └── seed.py              # fix semua seed (reproducibility)
│   └── models/
│       └── registry.py          # daftar model (lightweight + heavyweight) [Track 2]
├── artifacts/                    # SEMUA output tersimpan di sini (di-gitignore)
│   ├── patches/                 # hasil preprocessing (cache)
│   ├── features/                # radiomics CSV (cache)
│   ├── splits/                  # fold assignment JSON
│   ├── checkpoints/             # model weights per stage
│   └── logs/                    # CSV metrik & convergence
├── run_local.sh                 # orchestrator versi LOKAL
├── run_colab.ipynb              # orchestrator versi COLAB
└── requirements.txt
```

---

## Konsep Resumable (kunci utama)

Tiap stage cek dulu: **"output-ku sudah ada di disk belum?"** Kalau ada → skip. Kalau belum → proses.

```python
# Pola cache universal (src/utils/io.py)
def cached(path):
    """Return True kalau file/folder output sudah ada → boleh skip."""
    return os.path.exists(path) and os.path.getsize(path) > 0

# Contoh di stage_00_preprocess.py
def run(config):
    out = config['paths']['patches']
    if cached(out) and not config['force_rerun']:
        print(f"[SKIP] Patches sudah ada di {out}")
        return
    # ... proses DICOM → patches ...
    save(patches, out)
    print(f"[DONE] Patches disimpan ke {out}")
```

Efeknya: preprocessing berat (yang paling makan waktu & sering bikin hang) **cuma jalan sekali seumur hidup**. Runtime mati pun, jalanin ulang orchestrator → langsung skip ke training.

---

## Penyimpanan Metrik ke CSV (biar ga training ulang)

Tiap epoch, tulis 1 baris ke CSV. Convergence, loss, semua metrik masuk.

```python
# src/utils/logger.py
import csv, os

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        new = not os.path.exists(path)
        self.f = open(path, 'a', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if new:
            self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)
        self.f.flush()   # PENTING: langsung tulis, biar hang pun data aman
```

**Isi CSV log (`artifacts/logs/{model}_{fold}.csv`):**

| epoch | train_loss | val_loss | train_acc | val_acc | val_auc | lr | time_sec |
|---|---|---|---|---|---|---|---|
| 1 | 0.68 | 0.65 | 0.61 | 0.64 | 0.70 | 1e-3 | 42 |
| 2 | 0.55 | 0.58 | 0.72 | 0.69 | 0.78 | 1e-3 | 41 |

Mau bikin grafik convergence, bandingkan model, ganti style plot? **Cukup baca CSV ini.** Tidak sentuh training sama sekali.

**Summary CSV (`artifacts/logs/summary.csv`)** — 1 baris per model per fold, hasil final:

| model | params_M | flops_G | fold | best_epoch | val_auc | val_acc | sensitivity | specificity | f1 | infer_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| mobilenetv3_small | 2.5 | 0.06 | 0 | 38 | 0.91 | 0.88 | 0.89 | 0.87 | 0.88 | 4.2 |
| vgg16 | 138 | 15.5 | 0 | 22 | 0.90 | 0.87 | 0.86 | 0.88 | 0.87 | 28.1 |

Ini yang jadi bahan tabel & scatter plot "params vs AUC" untuk Track 2, langsung dari CSV.

---

## Model Checkpoint (resume training di tengah)

Selain metrik, simpan juga state training tiap N epoch supaya training bisa lanjut kalau putus:

```python
# Di dalam stage_03_train.py
def save_ckpt(path, model, optimizer, epoch, best_auc):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'best_auc': best_auc,
    }, path)

def maybe_resume(path, model, optimizer):
    if os.path.exists(path):
        ck = torch.load(path)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        print(f"[RESUME] Lanjut dari epoch {ck['epoch']}")
        return ck['epoch'] + 1, ck['best_auc']
    return 0, 0.0   # mulai dari awal
```

Simpan 2 checkpoint: `last.pt` (untuk resume) & `best.pt` (AUC terbaik, untuk evaluasi/XAI).

---

## Config Tunggal (jaga fair setup)

Semua hyperparameter di **satu file**. Ganti di sini = berlaku untuk semua model. Ini yang menjamin perbandingan fair.

```yaml
# configs/config.yaml
seed: 42
force_rerun: false          # set true kalau mau paksa proses ulang

paths:
  raw: "./data/raw"
  patches: "./artifacts/patches"
  features: "./artifacts/features/radiomics.csv"
  splits: "./artifacts/splits/folds.json"
  checkpoints: "./artifacts/checkpoints"
  logs: "./artifacts/logs"

data:
  patch_size: [64, 64]
  hu_window: [-1200, 600]
  augmentation: ["flip", "rotate", "gaussian_noise"]

train:
  epochs: 100
  batch_size: 32
  lr: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  loss: "weighted_ce"
  early_stopping_patience: 15
  n_folds: 5
  checkpoint_every: 5        # simpan last.pt tiap 5 epoch

models:                     # Track 2 — daftar model yang dibandingkan
  lightweight: ["mobilenetv3_small", "efficientnet_b0", "densenet121"]
  heavyweight: ["resnet50", "vgg16", "vit_base"]

track1_fusion:              # Track 1 — fusion + XAI
  enabled: true
  fusion_type: "intermediate"   # early | intermediate | late
  backbone: "mobilenetv3_small"
```

---

## Orchestrator — 2 Versi

### Versi Lokal (`run_local.sh`)

```bash
#!/bin/bash
set -e   # stop kalau ada error

python src/stage_00_preprocess.py --config configs/config.yaml
python src/stage_01_radiomics.py  --config configs/config.yaml
python src/stage_02_split.py      --config configs/config.yaml

# Track 2: loop semua model & fold
for model in mobilenetv3_small efficientnet_b0 densenet121 resnet50 vgg16 vit_base; do
  for fold in 0 1 2 3 4; do
    python src/stage_03_train.py --config configs/config.yaml --model $model --fold $fold
  done
done

python src/stage_04_evaluate.py --config configs/config.yaml
python src/stage_05_xai.py       --config configs/config.yaml
```

Jalankan: `bash run_local.sh`. Putus di tengah? Jalankan lagi — stage & fold yang sudah selesai auto-skip.

### Versi Colab (`run_colab.ipynb`)

Notebook tipis, tiap cell = 1 stage. Plus mount Google Drive supaya `artifacts/` **persisten** (Colab suka reset runtime).

```python
# Cell 1 — Setup & mount Drive (artifacts disimpan di Drive, anti-hilang)
from google.colab import drive
drive.mount('/content/drive')
!git clone <repo_url> && cd LungFuseNet
# arahkan artifacts/ ke Drive
!ln -s /content/drive/MyDrive/LungFuseNet_artifacts ./artifacts

# Cell 2 — Preprocessing (sekali saja, hasil di Drive)
!python src/stage_00_preprocess.py --config configs/config.yaml

# Cell 3 — Radiomics
!python src/stage_01_radiomics.py --config configs/config.yaml

# Cell 4 — Split
!python src/stage_02_split.py --config configs/config.yaml

# Cell 5 — Training (bisa dijalankan per model, biar ga kelamaan 1 sesi)
model = "mobilenetv3_small"   # ganti manual, atau loop
for fold in range(5):
    !python src/stage_03_train.py --config configs/config.yaml --model {model} --fold {fold}

# Cell 6 — Evaluate (baca checkpoint, tulis summary.csv)
!python src/stage_04_evaluate.py --config configs/config.yaml

# Cell 7 — XAI
!python src/stage_05_xai.py --config configs/config.yaml
```

**Kunci anti-hang di Colab:** `artifacts/` di Drive → runtime reset tidak menghapus progress. Training bisa dipecah per model per sesi (jalankan MobileNet hari ini, VGG besok) tanpa kehilangan apa-apa.

---

## Alur Kalau Mau Utak-atik (tanpa training ulang)

```
Mau ganti plot convergence      → baca artifacts/logs/{model}_{fold}.csv
Mau ganti tabel perbandingan    → baca artifacts/logs/summary.csv
Mau XAI ulang di model tertentu → stage_05 baca artifacts/checkpoints/{model}/best.pt
Mau tambah model baru           → cukup train model itu; yang lama tetap di cache
Mau ganti threshold/analisis    → semua dari CSV, nol training
```

---

## Pertanyaan Terbuka (untuk revisi)

1. **Preprocessing 2D atau 3D?** Kalau 3D (patch 64³), file cache lebih besar & butuh lebih banyak storage Drive. Kalau 2.5D, lebih ringan. Perlu diputuskan sebelum `stage_00`.
2. **Colab free atau Pro?** Free suka disconnect ~12 jam & GPU terbatas — makin penting checkpoint per-fold. Pro lebih longgar.
3. **Track 1 & Track 2 pakai backbone sama?** Kalau iya, `stage_03` bisa reuse; kalau beda, perlu config terpisah.
4. **Storage Drive cukup?** Estimasi: patches 3D + checkpoints 6 model × 5 fold bisa beberapa GB. Perlu cek kuota Drive.
5. **Mau MLflow/W&B** untuk tracking, atau cukup CSV? CSV paling simpel & sesuai permintaan (portable, no dependency), tapi W&B kasih dashboard otomatis.

---

## Ringkasan Keputusan yang Sudah Fix

- ✅ Environment: **2 versi** — lokal (`run_local.sh`) + Colab (`run_colab.ipynb`)
- ✅ Struktur: **modular `.py` + orchestrator tipis** (bukan 1 ipynb raksasa)
- ✅ Resumable wajib: **preprocessing/patches + radiomics features** (di-cache, skip otomatis)
- ✅ Metrik & convergence: **semua ke CSV** (flush tiap baris, aman dari hang)
- ✅ Checkpoint model: `last.pt` (resume) + `best.pt` (evaluasi/XAI)
- ✅ Fair setup: **1 config.yaml** jadi sumber kebenaran tunggal
