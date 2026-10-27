"""Where the online path gets a customer's past transactions from.

`HistorySource.get(uid, before)` must return that uid's transactions with TransactionDT < before,
ordered by (TransactionDT, TransactionID), with columns:
TransactionDT, TransactionAmt, device, p_email  (device / p_email normalised like the batch path).

InMemoryHistory is used for tests and offline checks. A PostgreSQL implementation is added
with the scoring API.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.features.keys import make_uid
from fraudml.features.static import as_str

HISTORY_COLUMNS = ["TransactionDT", "TransactionAmt", "device", "p_email"]


class HistorySource(Protocol):
    def get(self, uid: str, before: int) -> pd.DataFrame: ...


class InMemoryHistory:
    """History index over a DataFrame of raw transactions (e.g. the processed train.parquet)."""

    def __init__(self, df: pd.DataFrame):
        uid = make_uid(df)
        hist = pd.DataFrame(
            {
                config.UID_COL: uid,
                config.ID_COL: df[config.ID_COL].to_numpy(),
                "TransactionDT": df[config.TIME_COL].astype("int64").to_numpy(),
                "TransactionAmt": df["TransactionAmt"].astype("float64").to_numpy(),
                "device": as_str(df["DeviceInfo"]).to_numpy(),
                "p_email": as_str(df["P_emaildomain"], lower=True).to_numpy(),
            }
        )
        hist = hist[hist[config.UID_COL].notna()]
        hist = hist.sort_values(
            [config.UID_COL, "TransactionDT", config.ID_COL], kind="stable"
        ).reset_index(drop=True)
        self._frame = hist
        self._times = hist["TransactionDT"].to_numpy()
        uids = hist[config.UID_COL].to_numpy()
        starts = np.flatnonzero(np.r_[True, uids[1:] != uids[:-1]])
        ends = np.r_[starts[1:], len(uids)]
        self._ranges = dict(zip(uids[starts], zip(starts, ends, strict=True), strict=True))

    def get(self, uid: str, before: int) -> pd.DataFrame:
        rng = self._ranges.get(uid)
        if rng is None:
            return self._frame.iloc[0:0][HISTORY_COLUMNS]
        a, b = rng
        cut = a + int(np.searchsorted(self._times[a:b], before, side="left"))
        return self._frame.iloc[a:cut][HISTORY_COLUMNS]
