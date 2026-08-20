"""Stage E-3: the joint analysis the Track 2 results section is written from.

Everything here is read off CSVs that are already on disk. No model is built, no
checkpoint is loaded, nothing touches a GPU.

The manuscript asks one question -- what governs accuracy and interpretability in a
backbone comparison: capacity, optimizer, or architecture family? The answer is measured
on three nested but unequal subsets, and the whole point of this stage is to keep that
inequality visible:

  * capacity      -- seven backbones that share a 96 px input, because a cost axis only
                     means something when the input size is held constant;
  * localization  -- twelve backbones, because here the input resolution is a genuine
                     architectural variable rather than a confound;
  * optimizer     -- four backbones from the hyperparameter sweep, three of which also
                     appear among the twelve.

Outputs, all under artifacts/results/track2rev/:

  joint_master_table.csv / .md   twelve rows, efficiency columns left empty for the five
                                 backbones that do not share the 96 px input
  joint_correlations.csv         Spearman rho with n and p, reported as descriptive
  silent_failure_census.csv      every explanation site that emits an all-zero CAM
  depth_curve_summary.csv        per-backbone curve shape, labelled by a stated rule
  fig_t2_4_pointing_vs_params.png

Run from the repo root:

    .venv/Scripts/python.exe -m src.stage_09g_joint [--self-check]
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

RUN_ID = "2026-08-20-run04"
OUT_DIR = os.path.join("artifacts", "results", "track2rev")
RESULTS_DIR = os.path.join("artifacts", "results")

CAM12_CSV = os.path.join(OUT_DIR, "cam_12.csv")
CAM12_PS_CSV = os.path.join(OUT_DIR, "cam_12_persample.csv")
DEPTH_CSV = os.path.join(OUT_DIR, "depth_sweep_12.csv")
EFF7_CSV = os.path.join(OUT_DIR, "efficiency_7.csv")
AUDIT_CSV = os.path.join(OUT_DIR, "target_layer_audit.csv")
SUMMARY_BINARY_CSV = os.path.join(RESULTS_DIR, "summary_binary.csv")
ANOVA_CSV = os.path.join(RESULTS_DIR, "track2_anova_eta_sq.csv")
STABILITY_CSV = os.path.join(RESULTS_DIR, "track2_stability_by_model.csv")

MASTER_CSV = os.path.join(OUT_DIR, "joint_master_table.csv")
MASTER_MD = os.path.join(OUT_DIR, "joint_master_table.md")
CORR_CSV = os.path.join(OUT_DIR, "joint_correlations.csv")
CENSUS_CSV = os.path.join(OUT_DIR, "silent_failure_census.csv")
CURVE_CSV = os.path.join(OUT_DIR, "depth_curve_summary.csv")
FIG_PNG = os.path.join(OUT_DIR, "fig_t2_4_pointing_vs_params.png")

# The four backbones of the optimizer sweep. mobilenetv2 is in that sweep but not among
# the twelve, so the optimizer subset is not a subset of the localization subset.
OPTIMIZER_MODELS = ["mobilenetv2", "efficientnet_b0", "resnet50", "vgg16"]

# Metrics that are means over the 60 per-sample rows and so get a bootstrap interval.
BOOT_METRICS = {
    "dice": "dice",
    "iou": "iou",
    "dice_size_matched": "dice_size_matched",
    "energy_mean": "energy",
}
N_BOOT = 10000
BOOT_SEED = 20260820
ALPHA = 0.05

EFF_COLS = [
    "params_M", "gflops", "latency_cpu_ms_median", "latency_cpu_ms_iqr",
    "latency_gpu_ms_median", "latency_gpu_ms_iqr", "peak_mem_gpu_mb",
    "auc_mean", "auc_std", "auc_per_M_params",
]

# Curve-shape rule. A pointing-accuracy difference of 0.05 is three of the sixty samples;
# anything smaller is not a shape, it is sampling noise on a 60-sample proportion.
SHAPE_DELTA = 0.05
SHAPE_RULE = (
    "sites ordered shallow to deep by exec_index_norm; score = pointing_acc; "
    f"flat if max-min < {SHAPE_DELTA}; else shallow_peak if the best site is the first, "
    "deep_peak if it is the last, mid_peak otherwise"
)


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _stamp(df: pd.DataFrame, sha: str) -> pd.DataFrame:
    """Put run_id and commit_sha first, replacing any that the frame already carries."""
    df = df.drop(columns=[c for c in ("run_id", "commit_sha") if c in df.columns])
    df.insert(0, "commit_sha", sha)
    df.insert(0, "run_id", RUN_ID)
    return df


# --- interval estimates ---------------------------------------------------------------


def wilson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The normal approximation is used nowhere here on purpose: several backbones point at
    the lesion in zero of sixty samples, and the normal interval collapses to the single
    point zero there, which is not an honest statement about what sixty samples can rule
    out. The Wilson interval stays inside [0, 1] and stays wide at the boundary.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 b: int = N_BOOT, alpha: float = ALPHA) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of a per-sample metric."""
    draws = rng.choice(values, size=(b, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --- master table ---------------------------------------------------------------------


def build_master(sha: str) -> pd.DataFrame:
    """Twelve rows: localization for all, efficiency only where the input size matches."""
    cam = pd.read_csv(CAM12_CSV)
    per_sample = pd.read_csv(CAM12_PS_CSV)
    eff = pd.read_csv(EFF7_CSV).set_index("model")
    audit = pd.read_csv(AUDIT_CSV).set_index("backbone")
    binary = pd.read_csv(SUMMARY_BINARY_CSV)
    params_any = binary.groupby("model")["params_M"].mean()

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for _, r in cam.iterrows():
        b = r["backbone"]
        ps = per_sample[per_sample["backbone"] == b]
        n = int(r["n"])
        in_eff = b in eff.index
        in_opt = b in OPTIMIZER_MODELS
        subsets = ["localization12"]
        if in_eff:
            subsets.append("efficiency7")
        if in_opt:
            subsets.append("optimizer4")

        row = {
            "backbone": b,
            # "+" rather than "|" so the string survives a markdown pipe table unescaped.
            "subsets": "+".join(subsets),
            "in_localization12": 1,
            "in_efficiency7": int(in_eff),
            "in_optimizer4": int(in_opt),
            "cam_input_size": int(audit.loc[b, "input_size"]),
            "n_samples": n,
            "module_path": r["module_path"],
            "module_class": r["module_class"],
            "spatial_h": r["spatial_h"],
            "spatial_w": r["spatial_w"],
            "path_taken": r["path_taken"],
            "n_zero_cam": int(r["n_zero_cam"]),
            "params_M_any_res": float(params_any.get(b, np.nan)),
        }

        # Pointing accuracy is a proportion over n samples, so it gets a Wilson interval.
        k = int(round(float(r["pointing_acc"]) * n))
        lo, hi = wilson(k, n)
        row["pointing_acc"] = float(r["pointing_acc"])
        row["pointing_acc_ci_lo"] = lo
        row["pointing_acc_ci_hi"] = hi
        row["pointing_acc_hits"] = k

        for name, col in BOOT_METRICS.items():
            lo, hi = bootstrap_ci(ps[col].to_numpy(float), rng)
            row[name] = float(r[name])
            row[f"{name}_ci_lo"] = lo
            row[f"{name}_ci_hi"] = hi

        # The published columns come along so the corrected-resolver deltas stay readable
        # next to the corrected numbers rather than in a second file.
        for name in ("dice", "iou", "dice_size_matched", "pointing_acc", "energy_mean"):
            row[f"{name}_published"] = float(r[f"{name}_published"])
            row[f"{name}_delta"] = float(r[f"{name}_delta"])

        for col in EFF_COLS:
            row[col] = float(eff.loc[b, col]) if in_eff else np.nan
        row["efficiency_input_size"] = int(eff.loc[b, "input_size"]) if in_eff else np.nan
        row["ci_method"] = (
            f"pointing_acc: Wilson score {int((1 - ALPHA) * 100)}%; "
            f"dice/iou/dice_size_matched/energy_mean: percentile bootstrap "
            f"{int((1 - ALPHA) * 100)}% over per-sample rows, B={N_BOOT}"
        )
        row["n_bootstrap"] = N_BOOT
        rows.append(row)

    return _stamp(pd.DataFrame(rows), sha)


# --- silent-failure census ------------------------------------------------------------


def build_census(depth: pd.DataFrame, cam: pd.DataFrame, sha: str) -> pd.DataFrame:
    """One row per explanation site that emits an identically-zero CAM on any sample.

    A site whose CAM is identically zero still yields finite dice, IoU and energy numbers,
    because a constant map thresholds to either the whole patch or nothing. That is the
    failure mode: it is silent. The count over all sites in the sweep is what shows it is
    not a GoogLeNet curiosity.
    """
    n_sites = len(depth)
    hit = depth[depth["n_zero_cam"] > 0].copy()
    canonical = cam.set_index("backbone")["module_path"].to_dict()
    out = pd.DataFrame({
        "backbone": hit["backbone"],
        "module_path": hit["module_path"],
        "module_class": hit["module_class"],
        "spatial_h": hit["spatial_h"],
        "spatial_w": hit["spatial_w"],
        "spatial_size": hit["spatial_h"].astype(str) + "x" + hit["spatial_w"].astype(str),
        "n_zero_cam": hit["n_zero_cam"],
        "n": hit["n"],
        "frac_zero": hit["n_zero_cam"] / hit["n"],
        "failure_class": np.where(hit["n_zero_cam"] >= hit["n"], "all_samples_zero",
                                  "some_samples_zero"),
        # These are the numbers the site still emits while the CAM carries no signal.
        "dice_emitted": hit["dice"],
        "pointing_acc_emitted": hit["pointing_acc"],
        "is_band_pick": hit["is_band_pick"],
        "is_canonical_pick": hit["is_canonical_pick"],
        "is_cam12_headline_site": [
            int(canonical.get(bb) == mp) for bb, mp in zip(hit["backbone"], hit["module_path"])
        ],
        "n_sites_in_sweep": n_sites,
    })
    out = out.sort_values(["failure_class", "backbone", "spatial_h"]).reset_index(drop=True)
    return _stamp(out, sha)


# --- curve shape ----------------------------------------------------------------------


def _shape(scores: np.ndarray) -> str:
    """Label a depth curve from the data. The rule is SHAPE_RULE, stated in the header."""
    if scores.max() - scores.min() < SHAPE_DELTA:
        return "flat"
    best = int(np.argmax(scores))
    if best == 0:
        return "shallow_peak"
    if best == len(scores) - 1:
        return "deep_peak"
    return "mid_peak"


def build_curve_summary(depth: pd.DataFrame, sha: str) -> pd.DataFrame:
    """Per backbone: best site, canonical pick, band pick, the gap, and a shape label."""
    rows = []
    for backbone, d in depth.groupby("backbone", sort=False):
        d = d.sort_values("exec_index_norm").reset_index(drop=True)
        scores = d["pointing_acc"].to_numpy(float)
        best = d.loc[int(np.argmax(scores))]
        canon = d[d["is_canonical_pick"] == 1].iloc[0]
        band = d[d["is_band_pick"] == 1].iloc[0]
        rows.append({
            "backbone": backbone,
            "score_metric": "pointing_acc",
            "n_sites": len(d),
            "best_site": best["module_path"],
            "best_site_spatial": f"{best['spatial_h']}x{best['spatial_w']}",
            "best_score": float(best["pointing_acc"]),
            "canonical_site": canon["module_path"],
            "canonical_spatial": f"{canon['spatial_h']}x{canon['spatial_w']}",
            "canonical_score": float(canon["pointing_acc"]),
            "gap_best_minus_canonical": float(best["pointing_acc"] - canon["pointing_acc"]),
            "band_site": band["module_path"],
            "band_spatial": f"{band['spatial_h']}x{band['spatial_w']}",
            "band_score": float(band["pointing_acc"]),
            "gap_best_minus_band": float(best["pointing_acc"] - band["pointing_acc"]),
            "score_range": float(scores.max() - scores.min()),
            "shape": _shape(scores),
            "shape_rule": SHAPE_RULE,
        })
    return _stamp(pd.DataFrame(rows), sha)


# --- correlations ---------------------------------------------------------------------


def build_correlations(master: pd.DataFrame, sha: str) -> pd.DataFrame:
    """Two Spearman correlations, each reported with its n, its p and a verdict.

    Both samples are small. These are descriptive statements about twelve and seven
    points respectively; neither establishes anything on its own.
    """
    rows = []
    eff = master[master["in_efficiency7"] == 1]
    pairs = [
        ("efficiency7 (96 px input)", "params_M", "auc_mean", eff,
         "capacity against classification accuracy, input size held constant"),
        ("localization12", "params_M_any_res", "pointing_acc", master,
         "capacity against localization quality, input size free to vary"),
    ]
    for subset, xcol, ycol, frame, question in pairs:
        x = frame[xcol].to_numpy(float)
        y = frame[ycol].to_numpy(float)
        rho, p = stats.spearmanr(x, y)
        rows.append({
            "subset": subset,
            "question": question,
            "x": xcol,
            "y": ycol,
            "n": int(len(frame)),
            "spearman_rho": float(rho),
            "p_value": float(p),
            "alpha": ALPHA,
            "significant": bool(p < ALPHA),
            "verdict": ("not significant at alpha=%.2f with n=%d; no monotone association "
                        "is demonstrated" % (ALPHA, len(frame))) if p >= ALPHA else
                       ("significant at alpha=%.2f with n=%d; descriptive only, the sample "
                        "is too small to support a general claim" % (ALPHA, len(frame))),
            "interpretation_note": "descriptive, not inferential; n is small",
        })
    return _stamp(pd.DataFrame(rows), sha)


# --- the near-equal-size pair ----------------------------------------------------------


def near_equal_pair(master: pd.DataFrame) -> dict:
    """GoogLeNet against DenseNet121: two similar-sized models from different families.

    The positioning brief proposed this as a quasi-controlled contrast -- near-equal
    parameter counts, opposite CAM quality. This recomputes what the pair shows under the
    corrected resolver instead of restating what it showed before.
    """
    m = master.set_index("backbone")
    a, b = m.loc["googlenet"], m.loc["densenet121"]
    out = {
        "params_gap_M": abs(float(a["params_M"]) - float(b["params_M"])),
        "params_ratio": max(float(a["params_M"]), float(b["params_M"])) /
                        min(float(a["params_M"]), float(b["params_M"])),
    }
    for name, r in (("googlenet", a), ("densenet121", b)):
        out[name] = {
            "params_M": float(r["params_M"]),
            "auc_mean": float(r["auc_mean"]),
            "pointing_acc": float(r["pointing_acc"]),
            "pointing_ci": (float(r["pointing_acc_ci_lo"]), float(r["pointing_acc_ci_hi"])),
            "pointing_published": float(r["pointing_acc_published"]),
            "dice": float(r["dice"]),
            "dice_ci": (float(r["dice_ci_lo"]), float(r["dice_ci_hi"])),
            "site": r["module_path"],
            "spatial": f"{int(r['spatial_h'])}x{int(r['spatial_w'])}",
        }
    out["pointing_gap_now"] = abs(float(a["pointing_acc"]) - float(b["pointing_acc"]))
    out["pointing_gap_published"] = abs(float(a["pointing_acc_published"]) -
                                        float(b["pointing_acc_published"]))
    # Wilson intervals overlapping is the weakest useful statement: sixty samples cannot
    # separate the two, which is the opposite of the contrast the brief expected.
    out["pointing_ci_overlap"] = bool(
        float(a["pointing_acc_ci_lo"]) <= float(b["pointing_acc_ci_hi"]) and
        float(b["pointing_acc_ci_lo"]) <= float(a["pointing_acc_ci_hi"])
    )
    return out


# --- markdown -------------------------------------------------------------------------


def _fmt(v, nd=4):
    if isinstance(v, float) and np.isnan(v):
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown(master: pd.DataFrame, corr: pd.DataFrame, census: pd.DataFrame,
                   curves: pd.DataFrame, pair: dict, path: str) -> None:
    """The markdown twin of the master table, plus the findings that need words."""
    loc_cols = ["backbone", "subsets", "cam_input_size", "module_path", "cam_spatial",
                "n_zero_cam", "pointing_acc", "pointing_acc_ci_lo", "pointing_acc_ci_hi",
                "dice", "dice_ci_lo", "dice_ci_hi", "dice_size_matched", "energy_mean"]
    master = master.copy()
    # The ViT site has no 2-D feature map, so its spatial size is a token grid, not a
    # number; rendering the column as text keeps that row honest instead of showing "nan".
    master["cam_spatial"] = [
        "n/a (token sequence)" if pd.isna(h) else f"{int(h)}x{int(w)}"
        for h, w in zip(master["spatial_h"], master["spatial_w"])
    ]
    eff_cols = ["backbone", "efficiency_input_size", "params_M", "gflops",
                "latency_cpu_ms_median", "peak_mem_gpu_mb", "auc_mean", "auc_std",
                "auc_per_M_params"]

    loc = master[loc_cols].copy()
    effs = master[eff_cols].copy()
    for c in effs.columns[1:]:
        effs[c] = effs[c].map(lambda v: "" if pd.isna(v) else f"{v:g}")

    anova = pd.read_csv(ANOVA_CSV).set_index("term")
    stab = pd.read_csv(STABILITY_CSV)

    n_all_zero = int((census["failure_class"] == "all_samples_zero").sum())
    n_any = len(census)
    n_sites = int(census["n_sites_in_sweep"].iloc[0])

    lines = [
        "# Joint master table -- Track 2 backbone comparison",
        "",
        f"`run_id` {RUN_ID}, `commit_sha` {master['commit_sha'].iloc[0]}. "
        "Every number here is recomputed from the CSVs in this directory.",
        "",
        "## Subset membership",
        "",
        "The three claims rest on three different sets of models, and the sets are not the",
        "same size. Membership is a column in the CSV (`subsets`, plus the three integer",
        "flags) so no reader has to infer it.",
        "",
        f"* **localization12** -- all twelve rows below. CAM input size varies (64, 96, 224 px), "
        "which is deliberate: at this stage input resolution is an architectural variable.",
        f"* **efficiency7** -- the {int(master['in_efficiency7'].sum())} rows measured at a shared "
        "96 px input. Only these have efficiency columns.",
        f"* **optimizer4** -- the optimizer sweep covers {', '.join(OPTIMIZER_MODELS)}. "
        f"{int(master['in_optimizer4'].sum())} of those four appear among the twelve; "
        "**mobilenetv2 is in the optimizer sweep but not in this table at all**, so the "
        "optimizer subset is not nested inside the localization subset.",
        "",
        "## Localization, twelve backbones",
        "",
        loc.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Intervals: `pointing_acc` uses the Wilson score interval at "
        f"{int((1 - ALPHA) * 100)}% on 60 Bernoulli trials. `dice` uses a percentile "
        f"bootstrap at {int((1 - ALPHA) * 100)}% over the 60 per-sample rows, B={N_BOOT}, "
        f"seed {BOOT_SEED}. Bootstrap intervals for `iou`, `dice_size_matched` and "
        "`energy_mean` are in the CSV. The normal-approximation interval is used nowhere: "
        "several backbones score exactly zero and the normal interval degenerates there.",
        "",
        "## Efficiency, seven backbones at 96 px",
        "",
        effs.to_markdown(index=False),
        "",
        "**Footnote on the five empty rows.** mobilenetv3_small, efficientnet_b0, resnet50 "
        "and vgg16 were trained and profiled at a 64 px input; vit_base at 224 px. Cost "
        "figures -- GFLOPs, latency, peak memory -- scale with input size, so pasting their "
        "own measurements into this block would put numbers from three different input "
        "sizes into one column and invite exactly the comparison the column exists to "
        "support. They are left empty rather than dropped, because they still carry "
        "localization results. `params_M_any_res` in the CSV does hold a parameter count "
        "for all twelve; it is separated from the efficiency block because it was measured "
        "at each model's own input size and is used only as the x axis of the twelve-model "
        "scatter, never as a cost figure.",
        "",
        "## Correlations",
        "",
        corr[["subset", "x", "y", "n", "spearman_rho", "p_value", "significant",
              "verdict"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Both are descriptive summaries of a small sample, reported with n attached.",
        "",
        "## Silent-failure census",
        "",
        f"Of {n_sites} candidate explanation sites across the twelve backbones, {n_any} emit "
        f"an identically-zero CAM on at least one sample and {n_all_zero} emit one on every "
        "sample. Sizes and backbones are in `silent_failure_census.csv`. The failure is "
        "silent because a constant-zero CAM still produces finite dice and energy values -- "
        "the `dice_emitted` column in that file -- so an automatic selector that lands on "
        "such a site reports a plausible number rather than an error.",
        "",
        "One of the twelve headline rows is itself affected: the automatic resolver put "
        "vit_base at `encoder_layer_11.ln_1`, where the CAM is identically zero on "
        f"{int(master.set_index('backbone').loc['vit_base', 'n_zero_cam'])} of 60 samples, "
        "and that row still reports a finite dice. Its `n_zero_cam` column in the master "
        "table is the flag.",
        "",
        "## Depth-curve shapes",
        "",
        curves[["backbone", "n_sites", "best_site_spatial", "best_score",
                "canonical_spatial", "canonical_score", "gap_best_minus_canonical",
                "band_score", "shape"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Shape rule, applied to the data and not copied from any earlier report: {SHAPE_RULE}. "
        f"The {SHAPE_DELTA} threshold is three of sixty samples.",
        "",
        "## The near-equal-size pair, as it now stands",
        "",
        f"GoogLeNet ({pair['googlenet']['params_M']:.3f} M parameters) and DenseNet121 "
        f"({pair['densenet121']['params_M']:.3f} M) differ in size by "
        f"{pair['params_gap_M']:.3f} M, a ratio of {pair['params_ratio']:.2f}, and come from "
        "different architecture families. The positioning brief proposed leading with them as "
        "a quasi-controlled contrast: near-equal capacity, opposite CAM quality.",
        "",
        f"Under the corrected resolver that contrast is largely gone. Pointing accuracy is "
        f"{pair['googlenet']['pointing_acc']:.4f} "
        f"[{pair['googlenet']['pointing_ci'][0]:.3f}, {pair['googlenet']['pointing_ci'][1]:.3f}] "
        f"for GoogLeNet against {pair['densenet121']['pointing_acc']:.4f} "
        f"[{pair['densenet121']['pointing_ci'][0]:.3f}, {pair['densenet121']['pointing_ci'][1]:.3f}] "
        f"for DenseNet121, a gap of {pair['pointing_gap_now']:.4f} with "
        f"{'overlapping' if pair['pointing_ci_overlap'] else 'disjoint'} Wilson intervals. "
        f"The published numbers had a gap of {pair['pointing_gap_published']:.4f}. Dice is "
        f"{pair['googlenet']['dice']:.4f} against {pair['densenet121']['dice']:.4f}. "
        f"GoogLeNet's explanation site is now {pair['googlenet']['site']} at "
        f"{pair['googlenet']['spatial']}; DenseNet121's is {pair['densenet121']['site']} at "
        f"{pair['densenet121']['spatial']}.",
        "",
        "Two similarly sized models from different families now behave similarly once both "
        "are explained at a site with spatial extent. That is consistent with the thesis "
        "that the explanation site, not the architecture family, governs localization "
        "quality -- the contrast in the published numbers was a property of where the CAM "
        "was taken, not of the family. It is a finding, not a loss.",
        "",
        "## Optimizer axis, for context",
        "",
        f"From `track2_anova_eta_sq.csv`: optimizer eta^2 = {anova.loc['optimizer', 'eta_sq']:.4f} "
        f"(p = {anova.loc['optimizer', 'p_value']:.3g}), model eta^2 = "
        f"{anova.loc['model', 'eta_sq']:.4f} (p = {anova.loc['model', 'p_value']:.3g}), "
        f"weight decay eta^2 = {anova.loc['weight_decay', 'eta_sq']:.4f} "
        f"(p = {anova.loc['weight_decay', 'p_value']:.3g}). Measured on "
        f"{', '.join(sorted(stab['model']))} only.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --- figure ---------------------------------------------------------------------------


def make_figure(master: pd.DataFrame, path: str) -> None:
    """Pointing accuracy against parameter count, all twelve, log x.

    The scatter helpers in src/evaluation/efficiency.py plot AUC on a single undifferentiated
    series, and this figure needs a different y variable, error bars and two visually
    distinct groups, so it is written fresh rather than bent into shape.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    groups = [
        (master["in_efficiency7"] == 1, "#3b6ea5", "o",
         "shared 96 px input (efficiency subset, n=7)"),
        (master["in_efficiency7"] == 0, "#d7191c", "^",
         "other input size, 64 or 224 px (n=5)"),
    ]
    for sel, colour, marker, label in groups:
        d = master[sel]
        yerr = np.vstack([d["pointing_acc"] - d["pointing_acc_ci_lo"],
                          d["pointing_acc_ci_hi"] - d["pointing_acc"]])
        ax.errorbar(d["params_M_any_res"], d["pointing_acc"], yerr=yerr, fmt=marker,
                    color=colour, markersize=8, linestyle="none", capsize=3,
                    elinewidth=1.0, alpha=0.9, label=label, zorder=3)

    # Hand-placed offsets for the crowded region between 14 and 90 M parameters, where the
    # default up-and-right label would sit on a neighbour's marker.
    nudge = {
        "densenet201": (-9, 2, "right"),
        "xception": (-9, 0, "right"),
        "inceptionv3": (9, 1, "left"),
        "resnet50": (9, -12, "left"),
        "vgg16": (-9, -13, "right"),
        "inception_resnet_v2": (0, 13, "center"),
        "vit_base": (-9, 0, "right"),
    }
    for _, r in master.iterrows():
        dx, dy, ha = nudge.get(r["backbone"], (7, 4, "left"))
        ax.annotate(f"{r['backbone']} ({r['cam_input_size']} px)",
                    (r["params_M_any_res"], r["pointing_acc"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(0.55, 250)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("parameters (millions, log scale; measured at each model's own input size)",
                  fontsize=10)
    ax.set_ylabel("pointing accuracy\n(fraction of 60 samples, Wilson 95% interval)",
                  fontsize=10)
    ax.set_title("Localization quality does not follow model capacity", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.5)
    # The legend goes in reserved space below the axes so it cannot sit on top of a point.
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- self-check -----------------------------------------------------------------------


def self_check(master: pd.DataFrame) -> None:
    """Three asserts, as specified for this stage."""
    # 1. Twelve rows, seven of them carrying efficiency numbers.
    assert len(master) == 12, f"master table has {len(master)} rows, expected 12"
    filled = master[EFF_COLS].notna().all(axis=1).sum()
    assert filled == 7, f"{filled} rows have efficiency columns populated, expected 7"
    empty = master[master["in_efficiency7"] == 0][EFF_COLS].isna().all(axis=1).sum()
    assert empty == 5, f"{empty} rows have efficiency columns fully empty, expected 5"

    # 2. Recompute every localization figure from the per-sample rows and check it against
    #    cam_12.csv. Agreement means the joint analysis is reading the same numbers the
    #    source table published rather than a parallel second derivation of them.
    cam = pd.read_csv(CAM12_CSV).set_index("backbone")
    per_sample = pd.read_csv(CAM12_PS_CSV)
    m = master.set_index("backbone")
    cols = {"dice": "dice", "iou": "iou", "dice_size_matched": "dice_size_matched",
            "pointing_acc": "pointing_hit", "energy_mean": "energy"}
    for backbone in m.index:
        ps = per_sample[per_sample["backbone"] == backbone]
        for metric, col in cols.items():
            want = float(cam.loc[backbone, metric])
            got = float(m.loc[backbone, metric])
            recomputed = float(ps[col].mean())
            assert abs(got - want) < 1e-12, f"{backbone}.{metric}: master {got} vs cam_12 {want}"
            assert abs(recomputed - want) < 1e-9, \
                f"{backbone}.{metric}: per-sample mean {recomputed} vs cam_12 {want}"

    # 3. Every Wilson interval brackets its point estimate and stays inside [0, 1].
    for _, r in master.iterrows():
        lo, hi, p = r["pointing_acc_ci_lo"], r["pointing_acc_ci_hi"], r["pointing_acc"]
        assert 0.0 <= lo <= p <= hi <= 1.0, \
            f"{r['backbone']}: Wilson interval [{lo}, {hi}] does not bracket {p} inside [0,1]"

    print("[SELF-CHECK] all three asserts passed")


# --- driver ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true",
                   help="rebuild the master table in memory and run the asserts only")
    args = p.parse_args()

    sha = _commit_sha()
    os.makedirs(OUT_DIR, exist_ok=True)
    master = build_master(sha)

    if args.self_check:
        self_check(master)
        return

    cam = pd.read_csv(CAM12_CSV)
    depth = pd.read_csv(DEPTH_CSV)

    census = build_census(depth, cam, sha)
    curves = build_curve_summary(depth, sha)
    corr = build_correlations(master, sha)
    pair = near_equal_pair(master)

    master.to_csv(MASTER_CSV, index=False)
    census.to_csv(CENSUS_CSV, index=False)
    curves.to_csv(CURVE_CSV, index=False)
    corr.to_csv(CORR_CSV, index=False)
    write_markdown(master, corr, census, curves, pair, MASTER_MD)
    make_figure(master, FIG_PNG)

    for path in (MASTER_CSV, MASTER_MD, CENSUS_CSV, CURVE_CSV, CORR_CSV, FIG_PNG):
        print(f"[OK] {path}")
    print(f"[INFO] shape distribution: {curves['shape'].value_counts().to_dict()}")
    print(f"[INFO] silent failures: {len(census)} of {len(depth)} sites emit a zero CAM, "
          f"{int((census['failure_class'] == 'all_samples_zero').sum())} on every sample")
    for _, r in corr.iterrows():
        print(f"[INFO] spearman {r['x']} vs {r['y']} (n={r['n']}): rho={r['spearman_rho']:.4f} "
              f"p={r['p_value']:.4f} significant={r['significant']}")
    self_check(master)


if __name__ == "__main__":
    main()
