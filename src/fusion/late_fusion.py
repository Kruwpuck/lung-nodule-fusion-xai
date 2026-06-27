"""Late fusion: probability averaging / stacking from separate CNN and radiomics models."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def average_fusion(
    prob_cnn: np.ndarray,
    prob_radiomics: np.ndarray,
    weight_cnn: float = 0.5,
) -> np.ndarray:
    """Weighted average of malignancy probabilities from two models."""
    w_r = 1.0 - weight_cnn
    return weight_cnn * prob_cnn + w_r * prob_radiomics


def stacking_fusion(
    prob_train_cnn: np.ndarray,
    prob_train_radiomics: np.ndarray,
    y_train: np.ndarray,
    prob_test_cnn: np.ndarray,
    prob_test_radiomics: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, LogisticRegression]:
    """Meta-learner (logistic regression) stacking of two model probability outputs.

    Returns (test_probabilities, fitted_meta_learner).
    """
    X_train_meta = np.stack([prob_train_cnn, prob_train_radiomics], axis=1)
    X_test_meta = np.stack([prob_test_cnn, prob_test_radiomics], axis=1)

    meta = LogisticRegression(random_state=seed, max_iter=500)
    meta.fit(X_train_meta, y_train)
    probs = meta.predict_proba(X_test_meta)[:, 1]
    return probs, meta
