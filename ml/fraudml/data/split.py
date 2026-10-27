"""Chronological train / gap / validation / gap / test split (dates in config.SPLIT_DATES)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudml import config

SECONDS_PER_DAY = 86_400


def period_bounds_seconds(
    split_dates: dict[str, tuple[str, str]] = config.SPLIT_DATES,
    reference: str = config.REFERENCE_DATETIME,
) -> dict[str, tuple[int, int]]:
    """Convert the split dates into TransactionDT bounds [start, end) in seconds."""
    ref = pd.Timestamp(reference)
    return {
        name: (
            int((pd.Timestamp(a) - ref).total_seconds()),
            int((pd.Timestamp(b) - ref).total_seconds()),
        )
        for name, (a, b) in split_dates.items()
    }


def assign_period(transaction_dt: pd.Series) -> pd.Series:
    """Label each transaction with its period name. Raises if a row falls outside every period."""
    t = transaction_dt.to_numpy(dtype="int64")
    out = np.full(len(t), None, dtype=object)
    for name, (a, b) in period_bounds_seconds().items():
        out[(t >= a) & (t < b)] = name
    if (out == None).any():  # noqa: E711 (element-wise comparison on an object array)
        n = int((out == None).sum())  # noqa: E711
        raise ValueError(f"{n} transactions fall outside the configured split dates")
    return pd.Series(out, index=transaction_dt.index, name=config.PERIOD_COL, dtype="object")
