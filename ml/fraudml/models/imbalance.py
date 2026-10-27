"""Random undersampling of legit rows (train only) and the matching probability correction.

If we keep a fraction beta of the legit rows, the model sees fraud 1/beta times more often than
in reality, so its scores are too high. Bayes' rule gives the correction:
    p = beta * p_s / (beta * p_s - p_s + 1)
(equivalently: true odds = odds seen by the model * beta). Ranking is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def undersample(
    X: pd.DataFrame, y: pd.Series, legit_per_fraud: float = 5.0, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Keep every fraud and a random `legit_per_fraud` legit rows per fraud.

    Returns (X_sampled, y_sampled, beta) where beta = share of legit rows kept. Row order of the
    original frame is preserved.
    """
    y_arr = np.asarray(y).astype(bool)
    pos = np.flatnonzero(y_arr)
    neg = np.flatnonzero(~y_arr)
    n_keep = min(len(neg), int(round(legit_per_fraud * len(pos))))
    rng = np.random.default_rng(seed)
    kept_neg = rng.choice(neg, size=n_keep, replace=False)
    idx = np.sort(np.concatenate([pos, kept_neg]))
    beta = n_keep / len(neg) if len(neg) else 1.0
    return X.iloc[idx], y.iloc[idx], beta


def correct_probability(p_sampled, beta: float) -> np.ndarray:
    """Map scores of a model trained on undersampled data back to the true base rate."""
    p = np.asarray(p_sampled, dtype="float64")
    return beta * p / (beta * p - p + 1.0)
