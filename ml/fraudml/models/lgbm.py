"""LightGBM with fixed default settings.

The baseline comparison is about imbalance strategies, not tuning.

Categorical columns use LightGBM's native categorical support (pandas `category` dtype, fixed
category lists from the FeatureEncoder) and NaN is handled natively.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier

LGBM_DEFAULTS = {"n_estimators": 500, "learning_rate": 0.05}


def make_lgbm(scale_pos_weight: float | None = None, seed: int = 42, **overrides) -> LGBMClassifier:
    params = {**LGBM_DEFAULTS, "random_state": seed, "n_jobs": -1, "verbose": -1, **overrides}
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**params)
