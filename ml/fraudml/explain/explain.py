"""Explanation objects: top-k risk factors per transaction, JSON-ready.

Each object holds the base value and raw log-odds (base + sum of all contributions), the
calibrated probability when a calibrator is given, and the top-k factors (readable features and
masked groups) ordered by absolute contribution. This is what `predictions.shap_values` stores.
"""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraudml.explain.contributions import BASE, GROUPS, contributions, group_contributions
from fraudml.explain.templates import describe, direction, sentence, strength

TOP_K = 10


def _json_value(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return str(v)


def explain_batch(booster: lgb.Booster, X: pd.DataFrame, k: int = TOP_K, calibrator=None) -> list:
    contrib = group_contributions(contributions(booster, X))
    keys = [c for c in contrib.columns if c != BASE]
    values = contrib[keys].to_numpy()
    base = contrib[BASE].to_numpy()
    raw = base + values.sum(axis=1)
    prob = calibrator.predict(1 / (1 + np.exp(-raw))) if calibrator is not None else None

    out = []
    for i in range(len(X)):
        row = X.iloc[i]
        order = np.argsort(-np.abs(values[i]), kind="stable")[:k]
        factors = []
        for j in order:
            key, c = keys[j], float(values[i, j])
            value = None if key in GROUPS else _json_value(row[key])
            label, _ = describe(key, value)
            factors.append(
                {
                    "key": key,
                    "masked": key in GROUPS,  # anonymised family: UI may list it last
                    "label": label,
                    "value": value,
                    "contribution": c,
                    "direction": direction(c),
                    "strength": strength(c),
                    "text": sentence(key, value, c),
                }
            )
        shown = sum(f["contribution"] for f in factors)
        item = {
            "base_value": float(base[i]),
            "raw_log_odds": float(raw[i]),
            "other_contribution": float(values[i].sum() - shown),
            "factors": factors,
        }
        if prob is not None:
            item["probability"] = float(prob[i])
        out.append(item)
    return out


def explain_transaction(
    booster: lgb.Booster, row: pd.DataFrame, k: int = TOP_K, calibrator=None
) -> dict:
    """Explain one transaction, given as a one-row DataFrame (e.g. `X.iloc[[i]]`).

    A one-row DataFrame keeps the categorical dtypes the model needs (a Series would not).
    """
    if not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise ValueError("pass a one-row DataFrame, e.g. X.iloc[[i]]")
    return explain_batch(booster, row, k, calibrator)[0]
