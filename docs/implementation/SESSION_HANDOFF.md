# SESSION HANDOFF — Transfer Knowledge ke Session Baru

> Baca ini dulu sebelum lanjut kerja. Status per 2026-07-18. Copy asli (plan lengkap) ada di `C:\Users\ihabh\.claude\plans\refactored-finding-dusk.md`.

## Proyek + Akses Remote

- **Repo lokal**: `D:\Kuliah\Semester 7\Paper\lung-nodule-fusion-xai`
- **Remote GPU PC** (training/eval/data jalan di sini):
  - Host: `100.98.9.120`
  - User: `Adaptive Network`
  - Password: `SDNNDN`
  - Project path remote: `C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai`
  - Python: `.venv\Scripts\python.exe` (module run: `python -m src.stage_XX_...`)
- Kredensial cuma buat automation script proyek ini.

## Pola Automation

SSH interaktif gak bisa auth non-interaktif di host ini. Semua eksekusi remote lewat **paramiko SSH password-auth script**:
1. Tulis script python ke scratchpad lokal.
2. Script SSH ke `100.98.9.120`, jalanin command di bawah `.venv\Scripts\python.exe`.
3. Panggil via Bash tool: `python <script>.py`.

Gotcha: print stdout/stderr remote yang ada unicode (em-dash dll) crash `UnicodeEncodeError` di Windows lokal (cp1252). Fix: `.encode("ascii","replace").decode()` sebelum print. Command remote tetap sukses walau print lokal crash.

## Terminologi: Track vs Arm

- **Track 1 — Fusion+XAI**: kode ada (`src/fusion/*.py`, `src/models/fusion_net.py`) tapi gak diwire ke training stage manapun. **Di luar scope** — user skip eksplisit.
- **Track 2 — model comparison** (aktif): 6 backbone (`mobilenetv3_small`, `efficientnet_b0`, `densenet121`, `resnet50`, `vgg16`, `vit_base`) × 5 fold. **Arm** = varian label-target:

| Arm | Task flag | Target col | Kelas | Loss | Status |
|-----|-----------|-----------|-------|------|--------|
| A | `binary` | `label` (drop -1) | 2 (exclude median==3) | CE | trained+eval (legacy checkpoint `checkpoints/{model}/`, no suffix) |
| B | `ordinal` | `median_rating` (1-5, drop NaN) | 1 regresi | SmoothL1 | trained+eval, report done |
| C | `grade3` | `grade3` (drop -1) | 3 (benign/indeterminate/malignant) | CE | **belum train/eval/report** |
| D | `grade4` | `grade4` (semua row) | 4 (+LUNA16 hard-neg) | CE | trained+eval, report done |

## State Sekarang

Data prep selesai + tervalidasi. `labels.csv` kolom: `median_rating`, `label`, `grade3`, `grade4`, `fold`, `rating_std`.

Kode task-aware semua sudah diupload + tervalidasi di remote:
- `src/stage_03_train.py` — `_TASK_CFG`, `_filter_for_task`, `_evaluate`, `_score`. Checkpoint `checkpoints/{model}_{task}/fold{N}_{best,last}.pt`. Log `logs/{model}_{task}_fold{N}.csv`. `--task binary|ordinal|grade3|grade4`. Resumable.
- `src/stage_04_evaluate.py` — output `summary_{task}.csv`, preds `preds/{model}_{task}_fold{N}.npz` key `y_true`/`y_out` (legacy binary pakai `y_prob`).
- `src/models/registry.py` — `_TASK_N_CLASSES = {"binary":2, "ordinal":1, "grade3":3, "grade4":4}`.
- `src/evaluation/metrics.py` — `compute_metrics` (binary), `ordinal_metrics` (MAE/QWK/±1-acc), `grade4_metrics` (macro-AUC OvR/accuracy/macro-F1, dipake grade3 & grade4).
- `src/stage_06_report.py` — binary `run(cfg)` gak disentuh. Tambah `run_ordinal(cfg)`, `run_multiclass(cfg, task)`. Output `figures_{task}/` + `efficiency_table_{task}.csv/.md`.

Report sudah jalan real (live-tested, bukan cuma syntax check) buat ordinal + grade4 — figure lengkap ada di `artifacts/results/figures_ordinal/` dan `figures_grade4/`.

`--task grade3` dispatch tervalidasi fail rapi (exit 0, log error jelas) karena arm C belum ditraining — bukan bug.

## Sisa Kerja (user-side, kode sudah siap)

```bat
:: 1. Training arm C — 6 model x 5 fold, sekuensial (resumable)
for %M in (mobilenetv3_small efficientnet_b0 densenet121 resnet50 vgg16 vit_base) do (
  for %F in (0 1 2 3 4) do .venv\Scripts\python.exe -m src.stage_03_train --model %M --fold %F --task grade3
)

:: 2. Eval arm C -> summary_grade3.csv (30 baris)
.venv\Scripts\python.exe -m src.stage_04_evaluate --task grade3

:: 3. Report arm C -> artifacts/results/figures_grade3/
.venv\Scripts\python.exe -m src.stage_06_report --task grade3
```

Gate cek: `summary_grade3.csv` 30 baris, `auc_macro` ~0.75–0.88. `figures_grade3/` keisi PNG.

## Gotcha / Catatan Historis

- pylidc 0.2.2 `Annotation.bbox(pad=)` tolak tuple-of-tuples, harus `pad=0` (sudah difix, relevan cuma kalau rebuild data).
- Checkpoint arm A legacy: `checkpoints/{model}/` (no suffix). Arm B/C/D: `checkpoints/{model}_{task}/`.
- Preds legacy binary key `y_prob`; task baru key `y_out`.
