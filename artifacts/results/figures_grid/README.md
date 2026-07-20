# XAI Grid Figures — Notes

Fixed sample set: `artifacts/xai/fixed_display_samples.json` (S1-S6, fold 0),
same 6 nodules used across every figure in this directory for controlled
comparison (see `docs/VISUALIZATION_FIGURES_PLAN.md` Bagian 1).

## Grid A (`grid_backbone.png`) — key finding

Interpretability does NOT track model capacity/params:

- **densenet121, resnet50, vgg16** — Layer-CAM hot spot consistently lands
  inside the mask outline on S1-S4 (TP malignant). Best localization.
- **mobilenetv3_small, efficientnet_b0** — hot spot frequently drifts to the
  edge of the nodule or nearby structures, not centered on the mask.
- **vit_base** — activation map is nearly flat/near-zero on S1-S4 (only
  reacts on S6, an edge artifact). Weakest localization of the 6, consistent
  with `pointing_acc ~= 0` measured earlier in Track 2 XAI (`artifacts/results/xai/xai_metrics.csv`).

Takeaway: a lightweight backbone with strong AUC (Track 2 headline result)
is not automatically the one to pick if visual explainability matters —
localization quality is architecture-dependent, not accuracy-dependent, and
should be reported as a separate axis, not assumed to follow from AUC.

## Grid B (`grid_cam_method.png`)

Same 6 samples, densenet121 (best localization from Grid A), 5 CAM methods
(gradcam/gradcampp/hirescam/layercam/eigencam — eigencam substituted for
ablation-cam, not wired in `gradcam_utils.py`'s `cam_classes`). All 5 broadly
agree on hot-spot location for this backbone; layercam/hirescam give the
tightest, least noisy maps.
