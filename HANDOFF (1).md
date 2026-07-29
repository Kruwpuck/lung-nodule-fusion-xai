# Handoff — lung-nodule-fusion-xai preprocessing fix + retrain + report

## Setup

Project: `lung-nodule-fusion-xai` (lung nodule malignancy classification, LIDC-IDRI data via `pylidc`).
Remote GPU PC (Windows): `100.98.9.120`, user `Adaptive Network`, path `C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai`.
Venv: `.venv\Scripts\python.exe` (always run modules as `python -m src.stageXX_name`, never direct file exec — `ModuleNotFoundError: No module named 'src'` otherwise).

Research plan has 2 tracks:
- **Track 2** (implemented, this handoff's scope): 6-model lightweight-vs-heavyweight comparison (mobilenetv3_small, efficientnet_b0, densenet121, resnet50, vgg16, vit_base), 5-fold CV.
- **Track 1** (NOT implemented, out of scope here): Fusion+XAI — radiomics feature selection (ICC→mRMR→LASSO), FusionNet training, SHAP, ablation (CNN-alone vs radiomics-alone vs fused), DeLong significance. `src/fusion/*` and `src/models/fusion_net.py` exist but unused by `stage_03_train.py`.

## Root-cause bug found (critical)

`pylidc.Scan.to_volume()` returns array axis order **(Y, X, Z)** — Z (slice count) is the LAST axis, often only 2–20 slices. Old code assumed `(Z, Y, X)`:

- `src/data_loading/lidc_loader.py` comment claimed `(Z,Y,X)` — wrong.
- `src/training/dataset.py::_extract_2_5d` did `Z, H, W = volume.shape`, treating the tiny slice-count axis as image WIDTH. Result: patches were ~2/3 black padding, width of black bar = nodule size = malignancy proxy.

**Consequence:** old training run (AUC 0.89–0.94 across all 6 models × 5 folds) is a shortcut-learning artifact, not real signal. Grad-CAM confirmed activation on content/black border, not on tissue. All old `artifacts/results/summary.csv` numbers are invalid.

Two more bugs found:
- **Corner-paste bug**: old `_extract_2_5d` pasted crop into top-left corner (`sl[:crop.shape[0], :crop.shape[1]] = crop`) instead of centering — asymmetric black padding.
- **260 nodules had < 3 slices** (some 0) — too few for 2.5D.
- **Isotropic resampling gap**: config says `resampledPixelSpacing: [1,1,1]` (IBSI requirement) but `sitk.GetImageFromArray()` defaults to spacing (1,1,1) regardless of true voxel size unless `SetSpacing()` called explicitly — resampling was silently a no-op.

## Fix approach (chosen by user over cheaper dataset-layer-only patch)

Re-crop from original scan: fixed **physical window 64×64×16 mm**, centered on consensus-mask centroid, resampled to **1mm isotropic**, output always exactly `(16, 64, 64)` voxels in `(Z, Y, X)` order. Out-of-bounds padding uses **-1000 HU (air)**, not zero. Accepted cost: full pipeline re-run (~4h).

## Files changed (all on remote, via SFTP)

1. **`src/data_loading/lidc_loader.py`** — full rewrite of nodule crop logic:
   - `_crop_fixed_window(vol, center, half_vox, fill_value, dtype)` — fixed-size crop with out-of-bounds fill.
   - `_paste_local(dst, src, offset)` — paste mask into same window frame.
   - `_crop_and_resample_nodule(vol, cmask, cbbox, native_spacing, window_mm, target_spacing)` — full pipeline: compute global centroid, crop native window (fill -1000 HU), paste mask, `scipy.ndimage.zoom` resample (order=3 image / order=0 mask), final center-crop/pad to exact target voxel count.
   - `build_nodule_dataset(..., window_mm=(64,64,16), target_spacing=(1,1,1))` — transposes `(Y,X,Z)→(Z,Y,X)` before saving, skips empty-mask nodules, records correct `centroid_y/x/z`, `spacing_y/x/z`.
   - Verified live on LIDC-IDRI-0078/0069 before full run: patches `(16,64,64)`, non-empty masks, realistic HU.

2. **`src/training/dataset.py`** — `_extract_2_5d` simplified (patches now pre-centered/sized, no crop/pad needed):
   ```python
   def _extract_2_5d(self, volume):
       Z, H, W = volume.shape
       cz = Z // 2
       half = self.n_slices // 2
       slices = []
       for offset in range(-half, half + 1):
           z = max(0, min(Z - 1, cz + offset))
           slices.append(volume[z])
       return np.stack(slices, axis=0)
   ```
   Removed dead `for i, dim in enumerate(volume.shape): pass` in `NoduleDataset3D._center_crop_3d`.

3. **`src/radiomics/extraction.py`** — added explicit spacing so resample isn't a no-op:
   ```python
   image_sitk.SetSpacing((1.0, 1.0, 1.0))
   mask_sitk.SetSpacing((1.0, 1.0, 1.0))
   ```

4. **`src/stage_01_radiomics.py`** — fixed silent stale-cache bug (`extract_dataset_features` defaulted to hardcoded cache path ignoring config):
   ```python
   features_df = extract_dataset_features(
       df, params_yaml=cfg["radiomics"]["params_yaml"],
       output_parquet=out, force_rebuild=cfg.get("force_rerun", False),
   )
   ```

5. **`src/stage_04_evaluate.py`** — added per-sample prediction dump for later ROC/calibration/DeLong:
   ```python
   preds_dir = os.path.join(results_dir, "preds")
   os.makedirs(preds_dir, exist_ok=True)
   y_true, y_prob = evaluate(model, val_loader, device)
   np.savez(os.path.join(preds_dir, f"{model_name}_fold{fold}.npz"), y_true=y_true, y_prob=y_prob)
   ```
   Also fixed: `measure_latency()` moves model to CPU in-place (`.to(device)` mutates caller's object) — breaks subsequent GPU inference in same loop. Fix: `model = model.to(device)` right after calling `measure_latency()`.
   Also fixed: checkpoint `torch.load(weights_only=True)` fails on numpy scalar (`best_auc` was numpy.float64) — catch `(TypeError, pickle.UnpicklingError)`, fallback `weights_only=False` (trusted own checkpoints). Same fix applied in `src/training/trainer.py::maybe_resume` and `src/stage_05_xai.py`, plus casting `best_auc` to plain `float()` before saving so future checkpoints don't hit this.

6. **`src/models/backbones.py`** — ViT needs fixed 224×224 input but config uses `patch_xy=64` → `AssertionError: Wrong image height! Expected 224 but got 64!`. Fix: `BackboneClassifier` adds `self._resize_to = 224 if backbone_name == "vit_b_16" else None` + `_maybe_resize()` via `nn.functional.interpolate` before `self.features(...)` in `forward()` and `get_embedding()`.

7. **`src/xai/gradcam_utils.py`** + **`src/stage_05_xai.py`** — `compute_gradcam()` was missing required `backbone_name` arg; `_get_target_layer()`'s mobilenet_v3 branch assumed FusionNet (`model.cnn_branch[-1]`) without checking. Fixed: pass `backbone_internal` name explicitly, `hasattr(model, "cnn_branch")` check with fallback to `model.features[0][-1]` for plain `BackboneClassifier`. Also forced `matplotlib.use("Agg")` before `pyplot` import — headless SSH GUI-backend failure (`PyCapsule_New called with null pointer`) was silently swallowed by a broad except, so `[DONE]` printed despite zero PNGs saved.

8. **`src/stage_06_report.py`** — brand new (~290 lines), not yet run. Generates all figures/tables from `artifacts/logs/*.csv`, `artifacts/results/summary.csv`, `artifacts/results/preds/*.npz`. Reuses existing helpers, doesn't reimplement: `src/evaluation/efficiency.py::build_efficiency_table/plot_params_vs_auc/plot_flops_vs_auc`, `src/evaluation/metrics.py::bootstrap_ci/build_calibration_data`, `src/evaluation/statistical_tests.py::delong_test`. New: `_plot_patch_qc` (before/after comparison, before uses `_old_buggy_extract` replicating old bug), `_plot_convergence` (per-model 5-fold overlay), `_plot_auc_boxplot`, `_pooled_preds` (concat fold npz — valid since folds disjoint), `_plot_roc_curves` (mean±std bands, `np.interp` over `mean_fpr=linspace(0,1,100)`), `_plot_calibration`, `_plot_confusion_matrices` (2×3 grid from summary.csv tp/tn/fp/fn), `_build_delong_matrix` (pairwise p-values → `delong_matrix.csv`), `_build_efficiency_table` (+ `auc_per_M_params`). Sentinel `artifacts/results/figures/done.txt`, forced `Agg` backend.

Newly pip-installed in remote venv: `ptflops` (0.7.5), `tabulate` (0.10.0).

## Execution order (this session)

1. Backup broken artifacts: `artifacts/patches`→`patches_broken`, `artifacts/results`→`results_broken`.
2. Clear stale caches: delete `data/processed/labels.csv` (critical — `load_and_split()` returns this blindly if present), `data/processed/radiomic_features.parquet`, clear `artifacts/features`, `artifacts/splits`, `artifacts/checkpoints`, `artifacts/logs`.
3. Apply code changes above.
4. `pip install ptflops tabulate`.
5. `python -m src.stage_00_preprocess` — **DONE**, QC gate passed (patch_qc.png confirmed visually: before=thin sliver in flat gray, after=centered nodule in real lung tissue).
6. `python -m src.stage_01_radiomics` — **DONE** (survived a local-monitoring disconnect; verified via remote file check: `radiomics.parquet` has 1391 rows, 1134 cols — confirms genuine completion, not data loss).
7. `python -m src.stage_02_split` — **DONE** (`folds.json`, 5 folds).
8. `python -m src.stage_03_train` × 30 (6 models × 5 folds) — **IN PROGRESS** at time of writing, running via remote orchestrator script `_run_all_training.py` (uploaded to remote project root) that runs each combo as a subprocess, logs OK/FAIL + duration to `_training_progress.log` after every combo — resilient to SSH-session disconnects. Estimated ~1.4h total (sum of old run's `time_sec` logs).
9. `python -m src.stage_04_evaluate` — pending (now dumps `preds/*.npz`, GFLOPs via ptflops).
10. `python -m src.stage_05_xai` — pending (Grad-CAM on corrected data).
11. `python -m src.stage_06_report` — pending (8 figures + 2 tables).
12. Cleanup remote temp files: `_run_all_training.py`, `_training_progress.log`, `_qc_check.py`, `_tmp_timing.py` (leftover from earlier session).
13. Commit + push to GitHub (remote `git config user.name "Kruwpuck"` / `user.email "Kruwpuck@users.noreply.github.com"` already set locally on that repo, not global).

## Key technical notes for whoever continues

- **Windows OpenSSH `exec_command`-spawned processes survive the death of the watching SSH channel.** Confirmed twice (stage_00, stage_01) — process finished correctly with correct output files even after local monitoring task got killed by user closing a terminal. Don't panic on "task stopped" notifications — verify via `tasklist | findstr /I python` + checking output file completeness before assuming failure/data loss.
- `stage_03_train.py` has resumable checkpoints (`maybe_resume`) per model+fold, so rerunning the orchestrator script is always safe — it'll skip/resume completed combos.
- Any inline `python -c "...;..."` multi-line script via paramiko can truncate/fail silently — always write a proper `.py` file and run `python script.py`.
- Windows paths in Python scripts need raw strings (`r'...'`) consistently.

## Verification checklist (from plan, not yet fully executed)

After stage_00 (done): patch shape exactly `(16,64,64)`, no zero dims, `mask.sum()>0` all nodules, nodule count ~1391, visual QC passed.

After stage_03/04 (pending): compare new AUC vs old 0.89–0.94 — **a drop is expected and is proof the black-bar shortcut is gone, not a regression**. If new AUC stays ~0.94 with the same pattern, suspect another leak.

After stage_05 (pending): Grad-CAM must activate on the nodule itself, not frame edges.

After stage_06 (pending): 8 figures + `efficiency_table.csv/.md` + `delong_matrix.csv` present; `delong_matrix.csv` has real p-values (not NaN); `gflops` column populated (not empty).

## Explicitly out of scope

Track 1 (Fusion+XAI) implementation — separate future plan, not started.
