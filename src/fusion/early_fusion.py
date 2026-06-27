"""Early fusion: CNN deep features + radiomic features -> XGBoost classifier."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_cnn_embeddings(
    model,          # BackboneClassifier or FusionNet with get_cnn_embedding
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract CNN penultimate embeddings for all samples. Returns (N, emb_dim)."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            img = batch[0].to(device)
            if hasattr(model, "get_cnn_embedding"):
                emb = model.get_cnn_embedding(img)
            else:
                emb = model.get_embedding(img)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def build_early_fusion_features(
    cnn_embeddings: np.ndarray,
    radiomic_features: np.ndarray,
) -> np.ndarray:
    """Concatenate CNN embeddings and radiomic features into a single feature vector."""
    assert cnn_embeddings.shape[0] == radiomic_features.shape[0], \
        "Mismatch in number of samples"
    return np.concatenate([cnn_embeddings, radiomic_features], axis=1)


def train_early_fusion_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_params: dict | None = None,
    seed: int = 42,
):
    """Train XGBoost on early-fused features. Returns fitted classifier."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost not installed. Run: pip install xgboost") from e

    default_params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": seed,
        "n_jobs": -1,
    }
    if xgb_params:
        default_params.update(xgb_params)

    clf = xgb.XGBClassifier(**default_params)
    clf.fit(X_train, y_train)
    logger.info("XGBoost trained on %d samples, %d features", *X_train.shape)
    return clf
