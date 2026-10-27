"""Training-serving skew check: compare batch features with online features row by row."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.features.encoder import FeatureEncoder
from fraudml.features.history import HistorySource
from fraudml.features.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES_CURATED,
    build_features_batch,
    compute_features_online,
)

RTOL = 1e-9
ATOL = 1e-9


@dataclass
class Mismatch:
    transaction_id: int
    feature: str
    batch: object
    online: object


def values_match(feature: str, a, b) -> bool:
    if feature in CATEGORICAL_FEATURES:
        return str(a) == str(b)
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return bool(np.isclose(a, b, rtol=RTOL, atol=ATOL))


def compare_rows(
    sample: pd.DataFrame,
    batch_features: pd.DataFrame,
    history: HistorySource,
    encoder: FeatureEncoder,
) -> tuple[list[Mismatch], float]:
    """Compute online features for every row of `sample` and compare with `batch_features`.

    Returns the mismatches and the largest numeric absolute difference seen.
    """
    mismatches: list[Mismatch] = []
    max_abs = 0.0
    for idx, row in sample.iterrows():
        online = compute_features_online(row.to_dict(), history, encoder)
        batch = batch_features.loc[idx]
        for f in FEATURES_CURATED:
            a, b = batch[f], online[f]
            if f not in CATEGORICAL_FEATURES:
                fa, fb = float(a), float(b)
                if not (np.isnan(fa) or np.isnan(fb)):
                    max_abs = max(max_abs, abs(fa - fb))
            if not values_match(f, a, b):
                mismatches.append(Mismatch(int(row[config.ID_COL]), f, a, b))
    return mismatches, max_abs


def batch_features_for(df: pd.DataFrame, encoder: FeatureEncoder) -> pd.DataFrame:
    return build_features_batch(df, encoder)
