"""XGBoost with library default settings, for comparison only (no tuning in the lean plan).

Trains on the GPU when available (`device="cuda"`), otherwise on CPU; only speed differs.
Categorical columns use XGBoost's native support (`enable_categorical=True`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def gpu_available() -> bool:
    if not xgb.build_info().get("USE_CUDA"):
        return False
    try:
        X = pd.DataFrame({"a": np.arange(20.0)})
        xgb.XGBClassifier(device="cuda", n_estimators=1).fit(X, np.arange(20) % 2)
        return True
    except xgb.core.XGBoostError:
        return False


def make_xgb(seed: int = 42, device: str | None = None) -> xgb.XGBClassifier:
    device = device or ("cuda" if gpu_available() else "cpu")
    return xgb.XGBClassifier(
        tree_method="hist",
        device=device,
        enable_categorical=True,
        random_state=seed,
        n_jobs=-1,
    )


def predict_proba_cpu(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict on CPU (data lives in host memory; avoids a device-mismatch warning)."""
    model.set_params(device="cpu")
    return model.predict_proba(X)[:, 1]
