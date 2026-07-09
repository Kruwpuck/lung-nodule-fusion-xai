"""Model efficiency metrics: params, FLOPs, latency, comparison table + plots."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd


def count_params(model) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, input_res: tuple = (3, 224, 224)) -> tuple[float, int]:
    """Return (gflops, params). Falls back to (nan, count_params) if ptflops absent."""
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_res, as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return float(macs) * 2 / 1e9, params  # MACs → FLOPs
    except ImportError:
        return float("nan"), count_params(model)


def measure_latency(model, input_res: tuple = (3, 224, 224),
                    n: int = 50, device: str = "cpu") -> float:
    """Mean forward-pass latency in ms (n runs after 5 warmup)."""
    import torch
    model = model.to(device).eval()
    dummy = torch.randn(1, *input_res).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(n):
            model(dummy)
    return (time.perf_counter() - t0) / n * 1000


def build_efficiency_table(results: dict) -> pd.DataFrame:
    """Build comparison DataFrame from results dict.

    results: {model_name: {params_M, gflops, latency_ms, auc, auc_ci_low, auc_ci_high}}
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "params_M": round(r.get("params_M", float("nan")), 2),
            "gflops": round(r.get("gflops", float("nan")), 2),
            "latency_ms": round(r.get("latency_ms", float("nan")), 1),
            "auc": round(r.get("auc", float("nan")), 4),
            "auc_ci_low": round(r.get("auc_ci_low", float("nan")), 4),
            "auc_ci_high": round(r.get("auc_ci_high", float("nan")), 4),
        })
    return pd.DataFrame(rows).sort_values("params_M").reset_index(drop=True)


def plot_params_vs_auc(df: pd.DataFrame, out_png: Optional[str] = None):
    """Scatter: log(params) vs AUC with model labels."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["params_M"], df["auc"], zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["params_M"], row["auc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Params vs AUC — Lightweight wins upper-left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


def plot_flops_vs_auc(df: pd.DataFrame, out_png: Optional[str] = None):
    """Scatter: log(GFLOPs) vs AUC with model labels."""
    import matplotlib.pyplot as plt
    df_valid = df.dropna(subset=["gflops"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_valid["gflops"], df_valid["auc"], zorder=3)
    for _, row in df_valid.iterrows():
        ax.annotate(row["model"], (row["gflops"], row["auc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("GFLOPs (log scale)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("FLOPs vs AUC — Compute efficiency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig
