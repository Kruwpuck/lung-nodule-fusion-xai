"""Stage 09f: efficiency axis for the seven Track 1 backbones (Track 2 rev).

Only the seven ``tracks.track1`` backbones are measured. They all run at
``input_size: 96``; the other five manuscript backbones run at 64 or 224, and a
cost axis is only comparable when every model sees the same resolution.
DenseNet121 is the proof: the legacy records hold it at 0.473 GFLOPs (64 px) and
1.065 GFLOPs (96 px) for one and the same architecture.

The measured input tensor is 64 px, exactly as in training. The 64 -> 96
bilinear upsample happens inside ``BackboneClassifier.forward``, so it is part
of the cost being measured and is deliberately not bypassed.

Latency is reported as a median with an interquartile range over several
independent repeats, on CPU and on GPU separately, because the single-run
numbers already in the repo disagree with each other (summary_binary.csv has
densenet201 at 118.0 ms, RESEARCH_JOURNEY_REPORT.md has 90.5 ms).

Outputs (never touches the stale artifacts/results/efficiency_table.csv):
    artifacts/results/track2rev/efficiency_7.csv
    artifacts/results/track2rev/fig_t2_3_pareto.png

Usage:
    .venv/Scripts/python.exe -m src.stage_09f_efficiency_7
    .venv/Scripts/python.exe -m src.stage_09f_efficiency_7 --self-check
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess

import yaml

RUN_ID = "2026-08-20-run04"
OUT_DIR = os.path.join("artifacts", "results", "track2rev")
OUT_CSV = os.path.join(OUT_DIR, "efficiency_7.csv")
OUT_PNG = os.path.join(OUT_DIR, "fig_t2_3_pareto.png")
SUMMARY_CSV = os.path.join("artifacts", "results", "summary_binary.csv")

# Independent timing repeats, and timed forward passes inside each repeat.
# measure_latency already does 5 warmup passes before the timed block, so each
# repeat is warm; the spread across repeats is what the IQR describes.
N_REPEATS = 7
N_FORWARDS = 50
BATCH_SIZE = 1  # measure_latency times a single-sample forward pass

# Self-check tolerance: a repeat-to-repeat IQR above this fraction of the median
# means the machine was not quiet during the measurement and the median is not
# a reproducible number. 0.25 is loose enough for a Windows CPU under normal
# desktop background load and tight enough to catch a genuinely noisy run.
MAX_IQR_RATIO = 0.25


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _median_iqr(values) -> tuple[float, float]:
    """Median and interquartile range of a sequence of latency measurements."""
    import numpy as np
    a = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return float(np.median(a)), float(q3 - q1)


def _pareto_flags(rows) -> list[bool]:
    """Mark the rows not dominated on (low GFLOPs, high AUC).

    A model is dominated when another model is at least as cheap and at least
    as accurate, and strictly better on one of the two.
    """
    flags = []
    for r in rows:
        dominated = any(
            o["gflops"] <= r["gflops"] and o["auc_mean"] >= r["auc_mean"]
            and (o["gflops"] < r["gflops"] or o["auc_mean"] > r["auc_mean"])
            for o in rows
        )
        flags.append(not dominated)
    return flags


def _auc_per_model():
    """Mean and std of binary AUC across the five folds, per backbone."""
    import pandas as pd
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["task"] == "binary"]
    g = df.groupby("model")["auc"]
    return g.mean(), g.std(ddof=1), g.size()


def _measure_one(name: str, cfg: dict, input_res: tuple, gpu: bool) -> dict:
    """Params, FLOPs, CPU/GPU latency spread and GPU peak memory for one model."""
    import torch
    from src.models.registry import build_model
    from src.evaluation.efficiency import count_params, measure_flops, measure_latency

    model = build_model(name, cfg, task="binary")
    params_M = count_params(model) / 1e6
    gflops, _ = measure_flops(model, input_res=input_res)

    cpu_runs = [measure_latency(model, input_res=input_res, n=N_FORWARDS, device="cpu")
                for _ in range(N_REPEATS)]
    cpu_med, cpu_iqr = _median_iqr(cpu_runs)

    gpu_med = gpu_iqr = peak_mb = float("nan")
    if gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_runs = [measure_latency(model, input_res=input_res, n=N_FORWARDS, device="cuda")
                    for _ in range(N_REPEATS)]
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        gpu_med, gpu_iqr = _median_iqr(gpu_runs)
        model.cpu()
        del model
        torch.cuda.empty_cache()

    return {
        "params_M": round(params_M, 3),
        "gflops": round(gflops, 4),
        "latency_cpu_ms_median": round(cpu_med, 3),
        "latency_cpu_ms_iqr": round(cpu_iqr, 3),
        "latency_gpu_ms_median": round(gpu_med, 3),
        "latency_gpu_ms_iqr": round(gpu_iqr, 3),
        "peak_mem_gpu_mb": round(peak_mb, 2),
    }


def _plot_pareto(df, out_png: str) -> None:
    """AUC against GFLOPs on a log x-axis with the Pareto front drawn in.

    Built on plot_flops_vs_auc, which already supplies the log-scaled labelled
    scatter; this adds the fold-std error bars, the front, and a legend parked
    outside the axes so it can never sit on top of a point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from src.evaluation.efficiency import plot_flops_vs_auc

    plot_df = df.rename(columns={"auc_mean": "auc"}).sort_values("gflops").reset_index(drop=True)
    fig = plot_flops_vs_auc(plot_df)
    fig.set_size_inches(7.2, 4.6)
    ax = fig.axes[0]

    # The helper's fixed label offset collides in the crowded 1.4-1.7 GFLOPs
    # cluster, so re-place the labels on a stagger with leader lines.
    for t in list(ax.texts):
        t.remove()
    ladder = [(-40, 24), (34, -30), (-46, -26), (-14, 58), (-42, -46), (50, 22), (40, -54)]
    for (_, row), (dx, dy) in zip(plot_df.iterrows(), ladder):
        ax.annotate(row["model"], (row["gflops"], row["auc"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=9,
                    ha="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7,
                                    shrinkA=1, shrinkB=6))

    ax.errorbar(plot_df["gflops"], plot_df["auc"], yerr=plot_df["auc_std"],
                fmt="none", ecolor="0.55", elinewidth=1, capsize=3, zorder=2)

    front = plot_df[plot_df["pareto"]].sort_values("gflops")
    ax.plot(front["gflops"], front["auc"], "-o", color="crimson", markersize=7,
            linewidth=1.5, zorder=4, label="Pareto front")
    ax.scatter(plot_df.loc[~plot_df["pareto"], "gflops"],
               plot_df.loc[~plot_df["pareto"], "auc"],
               facecolors="none", edgecolors="0.35", s=55, zorder=3,
               label="dominated")

    ax.set_xlabel("Cost: GFLOPs per forward pass (log scale)")
    ax.set_ylabel("Binary AUC (mean $\\pm$ SD over 5 folds)")
    ax.set_title("Accuracy vs compute, seven Track 1 backbones at 96 px input")
    # A decade-wide log axis would otherwise show a single labelled tick.
    ax.set_xticks([0.5, 0.7, 1.0, 1.4, 2.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.45, 2.15)
    ax.margins(y=0.34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict) -> None:
    import pandas as pd
    import torch

    backbones = cfg["tracks"]["track1"]["backbones"]
    input_size = cfg["tracks"]["track1"]["input_size"]
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    input_res = (n_slices, patch_xy, patch_xy)

    try:
        import ptflops  # noqa: F401
        print("[ENV] ptflops available -> GFLOPs are measured, not fabricated")
    except ImportError:
        raise SystemExit("[FAIL] ptflops is not installed; measure_flops would "
                         "silently return nan. Install it before running.")

    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else "none"
    if not gpu:
        print("[WARN] no CUDA device: GPU latency and peak memory will be nan")
    print(f"[ENV] torch {torch.__version__} cuda {torch.version.cuda} gpu {gpu_name}")

    auc_mean, auc_std, auc_n = _auc_per_model()
    os.makedirs(OUT_DIR, exist_ok=True)
    sha = _commit_sha()

    provenance = {
        "run_id": RUN_ID,
        "commit_sha": sha,
        "input_size": input_size,
        "input_res": f"{n_slices}x{patch_xy}x{patch_xy}",
        "batch_size": BATCH_SIZE,
        "n_repeats": N_REPEATS,
        "n_forwards_per_repeat": N_FORWARDS,
        "cpu_device": platform.processor() or platform.machine(),
        "gpu_device": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "os": f"{platform.system()} {platform.release()}",
    }

    rows = []
    for i, name in enumerate(backbones, 1):
        print(f"[{i}/{len(backbones)}] {name} ...", flush=True)
        m = _measure_one(name, cfg, input_res, gpu)
        rows.append({
            "run_id": RUN_ID,
            "commit_sha": sha,
            "model": name,
            "input_size": input_size,
            **m,
            "auc_mean": round(float(auc_mean[name]), 6),
            "auc_std": round(float(auc_std[name]), 6),
            "n_folds": int(auc_n[name]),
            "auc_per_M_params": round(float(auc_mean[name]) / m["params_M"], 6),
            **{k: v for k, v in provenance.items() if k not in ("run_id", "commit_sha", "input_size")},
        })
        # Rewrite after every model: an interrupted run still leaves the rows
        # already measured on disk. pareto is recomputed over what exists so far
        # and is only final once all seven rows are in.
        for row, flag in zip(rows, _pareto_flags(rows)):
            row["pareto"] = bool(flag)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"    params {m['params_M']}M  {m['gflops']} GFLOPs  "
              f"cpu {m['latency_cpu_ms_median']}+/-{m['latency_cpu_ms_iqr']} ms  "
              f"gpu {m['latency_gpu_ms_median']}+/-{m['latency_gpu_ms_iqr']} ms  "
              f"peak {m['peak_mem_gpu_mb']} MB")

    df = pd.DataFrame(rows)
    _plot_pareto(df, OUT_PNG)
    print(f"[DONE] {len(df)} rows -> {OUT_CSV}")
    print(f"[DONE] {OUT_PNG}")
    print("       Pareto front: " + ", ".join(df.loc[df["pareto"], "model"]))


def self_check() -> int:
    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    fails = []

    if len(df) != 7:
        fails.append(f"expected 7 rows, found {len(df)}")
    bad_size = df[df["input_size"] != 96]
    if not bad_size.empty:
        fails.append(f"input_size != 96 for {list(bad_size['model'])}")

    numeric = ["params_M", "gflops", "latency_cpu_ms_median", "latency_gpu_ms_median"]
    for col in numeric:
        nan_models = list(df.loc[df[col].isna(), "model"])
        if nan_models:
            fails.append(f"nan in {col} for {nan_models} "
                         f"(a nan GFLOPs means ptflops was missing)")

    for dev in ("cpu", "gpu"):
        med = df[f"latency_{dev}_ms_median"]
        ratio = df[f"latency_{dev}_ms_iqr"] / med
        worst = df.loc[ratio.idxmax()]
        print(f"[CHECK] worst {dev} latency IQR/median: {ratio.max():.3f} "
              f"({worst['model']}), threshold {MAX_IQR_RATIO}")
        over = df.loc[ratio > MAX_IQR_RATIO, "model"]
        if len(over):
            fails.append(f"{dev} latency IQR exceeds {MAX_IQR_RATIO:.0%} of the "
                         f"median for {list(over)}: measurement is unstable")

    for f in fails:
        print(f"[FAIL] {f}")
    if fails:
        return 1
    print(f"[OK] {OUT_CSV}: 7 rows at input_size 96, no nan, latency IQR within "
          f"{MAX_IQR_RATIO:.0%} of the median")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--self-check", action="store_true",
                   help="validate an existing efficiency_7.csv and exit")
    args = p.parse_args()
    if args.self_check:
        raise SystemExit(self_check())
    run(yaml.safe_load(open(args.config)))


if __name__ == "__main__":
    main()
