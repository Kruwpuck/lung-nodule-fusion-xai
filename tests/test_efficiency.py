"""Tests for efficiency measurement module."""
import pytest
import numpy as np
import pandas as pd

torch = pytest.importorskip("torch", reason="torch not installed")
from src.models.backbones import BackboneClassifier
from src.evaluation.efficiency import (
    count_params, build_efficiency_table, measure_latency
)


def test_count_params_positive():
    model = BackboneClassifier("mobilenet_v3_small", pretrained=False)
    n = count_params(model)
    assert isinstance(n, int) and n > 0


def test_count_params_small_less_than_large():
    small = count_params(BackboneClassifier("mobilenet_v3_small", pretrained=False))
    large = count_params(BackboneClassifier("resnet50", pretrained=False))
    assert small < large


def test_efficiency_table_columns():
    results = {
        "mobilenet_v3_small": {"params_M": 2.5, "gflops": 0.06, "latency_ms": 5.0, "auc": 0.82, "auc_ci_low": 0.79, "auc_ci_high": 0.85},
        "resnet50":           {"params_M": 25.6, "gflops": 4.1,  "latency_ms": 18.0, "auc": 0.83, "auc_ci_low": 0.80, "auc_ci_high": 0.86},
    }
    df = build_efficiency_table(results)
    assert set(["model", "params_M", "gflops", "latency_ms", "auc", "auc_ci_low", "auc_ci_high"]).issubset(df.columns)
    assert df.iloc[0]["params_M"] < df.iloc[1]["params_M"]  # sorted by params


def test_measure_latency_returns_positive():
    model = BackboneClassifier("mobilenet_v3_small", pretrained=False)
    ms = measure_latency(model, input_res=(3, 64, 64), n=5)
    assert ms > 0
