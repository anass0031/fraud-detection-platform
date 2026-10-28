"""Feature pipeline: the same feature definitions used in two ways.

- build_features_batch(df, encoder): whole DataFrame (training, evaluation, batch scoring).
- compute_features_online(txn, history, encoder): one transaction + its history (the API).

Both paths order transactions by (TransactionDT, TransactionID) and share the row-level code;
only the history lookup differs. tests/test_skew.py and scripts/check_skew.py prove they agree.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.features.behavior import BEHAVIOR_FEATURES, behavior_batch, behavior_online
from fraudml.features.encoder import FeatureEncoder
from fraudml.features.history import HistorySource
from fraudml.features.keys import make_uid
from fraudml.features.static import C_COLS, M_COLS, as_str, row_features

# Raw input columns a transaction must provide (missing values allowed).
RAW_INPUT_COLUMNS = (
    config.ID_COL,
    config.TIME_COL,
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card4",
    "card6",
    "addr1",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_01",
    "id_30",
    "id_31",
    "D1",
    "D4",
    "D10",
    "D15",
    *C_COLS,
    *M_COLS,
)

CATEGORICAL_FEATURES = (
    "ProductCD",
    "card4",
    "card6",
    "p_email",
    "r_email",
    "device_type",
    "browser_family",
    "os_family",
    *M_COLS,
)

FEATURES_CURATED: tuple[str, ...] = (
    # time
    "hour",
    "weekday",
    # amount
    "amt_log",
    "amt_cents",
    "amt_is_round",
    # product / card
    "ProductCD",
    "card4",
    "card6",
    "card1_freq",
    "card2_freq",
    "addr1_freq",
    # e-mail
    "p_email",
    "r_email",
    "email_same_domain",
    "p_email_freq",
    # device
    "device_type",
    "browser_family",
    "os_family",
    "device_info_freq",
    # missingness
    "has_identity",
    "has_uid",
    "addr1_missing",
    "dist1",
    # behaviour (past only)
    *BEHAVIOR_FEATURES,
    # masked (kept; grouped in explanations)
    *C_COLS,
    "D1n",
    "D4n",
    "D10n",
    "D15n",
    *M_COLS,
)

# Extra raw masked columns for the single curated-vs-full comparison (model selection, batch only).
FULL_EXTRA_COLUMNS: tuple[str, ...] = (
    *(f"V{i}" for i in range(1, 340)),
    *(f"id_{i:02d}" for i in range(1, 12)),
)
FEATURES_FULL: tuple[str, ...] = (*FEATURES_CURATED, *FULL_EXTRA_COLUMNS)

assert not (set(FEATURES_FULL) & config.FORBIDDEN_FEATURES), "forbidden column in feature list"


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, then TransactionID (deterministic order for same-second transactions)."""
    return df.sort_values([config.TIME_COL, config.ID_COL], kind="stable")


def raw_features_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Unencoded features for every row of df (index preserved) and the uid of each row."""
    ordered = canonical_order(df)
    uid = make_uid(ordered)
    raw = row_features(ordered, uid)
    beh = behavior_batch(
        ordered[config.TIME_COL],
        ordered["TransactionAmt"],
        as_str(ordered["DeviceInfo"]),
        as_str(ordered["P_emaildomain"], lower=True),
        uid,
    )
    raw = pd.concat([raw, beh], axis=1)
    return raw.loc[df.index], uid.loc[df.index]


def build_features_batch(
    df: pd.DataFrame, encoder: FeatureEncoder, features: tuple[str, ...] = FEATURES_CURATED
) -> pd.DataFrame:
    raw, _ = raw_features_batch(df)
    out = encoder.transform(raw)
    extra = [c for c in features if c not in out.columns]
    if extra:
        out = pd.concat([out, df.loc[out.index, extra].astype("float64")], axis=1)
    return out[list(features)]


def compute_features_online(
    txn: Mapping, history: HistorySource, encoder: FeatureEncoder
) -> pd.Series:
    """Curated features for one transaction. `history` must not contain this transaction."""
    return compute_features_online_frame(txn, history, encoder).iloc[0]


def compute_features_online_frame(
    txn: Mapping, history: HistorySource, encoder: FeatureEncoder
) -> pd.DataFrame:
    """Same as compute_features_online, as a one-row DataFrame.

    The DataFrame keeps the encoder's categorical dtypes, which the model and SHAP need
    (a Series of one row would turn them into plain objects). Used by the scoring API.
    """
    row = pd.DataFrame([{c: txn.get(c, np.nan) for c in RAW_INPUT_COLUMNS}])
    uid = make_uid(row)
    raw = row_features(row, uid)
    u = uid.iloc[0]
    t = int(row[config.TIME_COL].iloc[0])
    hist = history.get(u, before=t) if pd.notna(u) else None
    beh = behavior_online(
        t,
        float(row["TransactionAmt"].iloc[0]),
        as_str(row["DeviceInfo"]).iloc[0],
        as_str(row["P_emaildomain"], lower=True).iloc[0],
        u if pd.notna(u) else None,
        hist,
    )
    for k, v in beh.items():
        raw[k] = v
    return encoder.transform(raw)[list(FEATURES_CURATED)]
