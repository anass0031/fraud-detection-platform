"""Load model-ready data: features (X), label (y) and evaluation metadata (meta).

meta holds what the evaluation needs but the model must never see: TransactionID, day index,
exact TransactionAmt (for the cost model) and the `uid_fraud_in_train` flag.
The test period is refused unless `allow_test=True` (it is evaluated once, in L05).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraudml import config
from fraudml.features.keys import day_index
from fraudml.features.pipeline import FEATURES_CURATED

META_COLS = ("TransactionID", "day", "amount", config.EVAL_FLAG_COL)

# Every file path read through this module, so scripts can prove the test set was not opened.
READ_LOG: list[str] = []


@dataclass
class ModelData:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def prepare(
    features: pd.DataFrame,
    amounts: pd.DataFrame,
    feature_list: tuple[str, ...] = FEATURES_CURATED,
) -> ModelData:
    """Split a feature frame into X / y / meta and attach the exact amount by TransactionID."""
    bad = set(feature_list) & config.FORBIDDEN_FEATURES
    if bad:
        raise ValueError(f"forbidden columns in feature list: {sorted(bad)}")
    amt = amounts.set_index(config.ID_COL)["TransactionAmt"].astype("float64")
    ids = features[config.ID_COL]
    amount = ids.map(amt)
    if amount.isna().any():
        raise ValueError(f"{int(amount.isna().sum())} rows without TransactionAmt")
    meta = pd.DataFrame(
        {
            "TransactionID": ids.to_numpy(),
            "day": day_index(features[config.TIME_COL]).to_numpy(),
            "amount": amount.to_numpy(),
            config.EVAL_FLAG_COL: features[config.EVAL_FLAG_COL].astype(bool).to_numpy(),
        },
        index=features.index,
    )
    X = features[list(feature_list)].copy()
    y = features[config.TARGET].astype("int8")
    return ModelData(X=X, y=y, meta=meta)


def load_period(
    period: str,
    feature_list: tuple[str, ...] = FEATURES_CURATED,
    allow_test: bool = False,
) -> ModelData:
    if period not in config.MODEL_PERIODS:
        raise ValueError(f"unknown period {period!r}")
    if period == "test" and not allow_test:
        raise PermissionError("the test period is evaluated once, in L05 (pass allow_test=True)")
    feat_path = config.PROCESSED_DIR / f"features_{period}.parquet"
    raw_path = config.PROCESSED_DIR / "train.parquet"  # Kaggle train file = all labelled rows
    READ_LOG.extend([str(feat_path), str(raw_path)])
    features = pd.read_parquet(feat_path)
    # Raw columns not stored in the feature file (e.g. V*/id_* of FEATURES_FULL) are joined
    # from the raw parquet by TransactionID. They are numeric masked columns, used as is.
    extra = [c for c in feature_list if c not in features.columns]
    raw = pd.read_parquet(raw_path, columns=[config.ID_COL, "TransactionAmt", *extra])
    features = join_raw_columns(features, raw, extra)
    return prepare(features, raw[[config.ID_COL, "TransactionAmt"]], feature_list)


def join_raw_columns(features: pd.DataFrame, raw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add raw numeric columns to the feature frame, matched by TransactionID (float32)."""
    if not cols:
        return features
    joined = raw.set_index(config.ID_COL)[cols].astype("float32")
    joined = joined.reindex(features[config.ID_COL].to_numpy())
    joined.index = features.index
    return pd.concat([features, joined], axis=1)
