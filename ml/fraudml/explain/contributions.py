"""Per-transaction SHAP contributions from LightGBM's built-in exact TreeSHAP.

`booster.predict(X, pred_contrib=True)` returns one contribution per feature plus the base value,
in the model's raw log-odds: base_value + sum(contributions) = raw score. This is the same
quantity `shap.TreeExplainer` computes, without needing the `shap` package at scoring time.

Masked feature families are summed into one readable group each, so totals stay exact.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

BASE = "base_value"

# Masked (anonymised) feature families -> one group each. Readable features stay individual.
GROUPS: dict[str, tuple[str, ...]] = {
    "masked_counts": tuple(f"C{i}" for i in range(1, 15)),
    "masked_timing": ("D1n", "D4n", "D10n", "D15n"),
    "masked_match": tuple(f"M{i}" for i in range(1, 10)),
    "masked_distance": ("dist1",),
}
GROUP_OF = {feature: group for group, members in GROUPS.items() for feature in members}


def contributions(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """One column per model feature + `base_value`; rows aligned with X."""
    features = booster.feature_name()
    values = booster.predict(X[features], pred_contrib=True)
    return pd.DataFrame(values, columns=[*features, BASE], index=X.index)


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum masked families into their group; readable features and `base_value` are kept."""
    out = {}
    for col in contrib.columns:
        group = GROUP_OF.get(col)
        if group is None:
            out[col] = contrib[col]
        elif group not in out:
            members = [c for c in GROUPS[group] if c in contrib.columns]
            out[group] = contrib[members].sum(axis=1)
    return pd.DataFrame(out, index=contrib.index)
