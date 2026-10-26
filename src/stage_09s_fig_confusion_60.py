"""Stage 09s: confusion matrices of the twelve backbones on the 60 explained nodules (Track 2 manuscript).

POST HOC, DESCRIPTIVE. No model is run. Label and predicted class of each backbone on each of the 60 fold-0 nodules are read
from the per-sample file written by stage 09e (the prediction is the class the Layer-CAM explains). These 60 nodules were
sampled once (seed 42) for the localization analysis, so the counts describe the explained set, not classifier performance.

Input  (read only): artifacts/results/track2rev/depth_sweep_12_persample.csv
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track2rev/t2_confusion_60.csv            one row per backbone, with provenance columns
    artifacts/results/track2rev/fig_t2_confusion_print.{png,pdf}

Run:  python src/stage_09s_fig_confusion_60.py
"""
from __future__ import annotations

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DIR = os.path.join("artifacts", "results", "track2rev")
NAME = {"mobilenetv3_small": "MobileNetV3-S", "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet121",
        "resnet50": "ResNet50", "vgg16": "VGG16", "vit_base": "ViT-Base", "inceptionv3": "InceptionV3",
        "xception": "Xception", "googlenet": "GoogLeNet", "convnext_tiny": "ConvNeXt-Tiny",
        "inception_resnet_v2": "Inception-ResNet-v2", "densenet201": "DenseNet201"}
ORDER = list(NAME)


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    p = pd.read_csv(os.path.join(DIR, "depth_sweep_12_persample.csv"))
    run_id = str(p["run_id"].iloc[0])
    q = p.drop_duplicates(["backbone", "sample_idx"])
    assert q.groupby("backbone").size().eq(60).all() and q["backbone"].nunique() == 12
    rows, mats = [], {}
    for b in ORDER:
        g = q[q["backbone"] == b]
        tn = int(((g.label == 0) & (g.pred == 0)).sum())
        fp = int(((g.label == 0) & (g.pred == 1)).sum())
        fn = int(((g.label == 1) & (g.pred == 0)).sum())
        tp = int(((g.label == 1) & (g.pred == 1)).sum())
        assert tn + fp + fn + tp == 60
        mats[b] = np.array([[tn, fp], [fn, tp]])
        rows.append({"run_id": run_id, "commit_sha": _sha(), "input_size": 64, "checkpoint_mtime": "not_recorded_in_source",
                     "note": "post_hoc_descriptive; 60 explained nodules, fold 0; source=depth_sweep_12_persample.csv",
                     "backbone": b, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": (tn + tp) / 60,
                     "malignant_recall": tp / (tp + fn), "benign_recall": tn / (tn + fp)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DIR, "t2_confusion_60.csv"), index=False)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 6, figsize=(7.16, 3.0))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.9, bottom=0.115, wspace=0.12, hspace=0.62)
    vmax = max(m.max() for m in mats.values())
    for ax, b in zip(axes.ravel(), ORDER):
        m = mats[b]
        ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8, color="white" if m[i, j] > 0.55 * vmax else "black")
        r = out[out["backbone"] == b].iloc[0]
        ax.set_title(f"{NAME[b]}\nacc. {r.accuracy:.2f}, recall {r.malignant_recall:.2f}", fontsize=6.5, pad=2, linespacing=1.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["B", "M"], fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["B", "M"] if b in (ORDER[0], ORDER[6]) else [], fontsize=7)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("True", fontsize=7.5, labelpad=1)
    fig.text(0.54, 0.012, "Predicted (B: benign, M: malignant)", ha="center", fontsize=7.5)
    fig.savefig(os.path.join(DIR, "fig_t2_confusion_print.png"), dpi=300)
    fig.savefig(os.path.join(DIR, "fig_t2_confusion_print.pdf"))
    print(out[["backbone", "tn", "fp", "fn", "tp", "accuracy", "malignant_recall"]].round(3).to_string(index=False))
    print("git add -f artifacts/results/track2rev/t2_confusion_60.csv artifacts/results/track2rev/fig_t2_confusion_print.* src/stage_09s_fig_confusion_60.py")


if __name__ == "__main__":
    main()
