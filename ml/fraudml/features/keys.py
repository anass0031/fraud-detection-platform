"""Keys shared by all feature code: day index, pseudo customer id (uid), normalised D columns.

uid = card1 + addr1 + (day - D1). D1 looks like "days since the card was first seen", so
day - D1 estimates the card's first-seen day and is constant for one customer. The uid is
missing when card1, addr1 or D1 is missing (about 11% of rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudml import config

SECONDS_PER_DAY = 86_400

# "Days since ..." columns that drift over time; day - D is stable for one customer.
D_TO_NORMALISE = ("D1", "D4", "D10", "D15")


def day_index(transaction_dt: pd.Series) -> pd.Series:
    return (transaction_dt.astype("int64") // SECONDS_PER_DAY).rename("day")


def make_uid(df: pd.DataFrame) -> pd.Series:
    """Build the pseudo customer id as a string; NaN when any input is missing.

    Built explicitly from integers (not with astype(str)) so the result does not depend on the
    pandas version or on float formatting, and is identical in the batch and online paths.
    """
    card1 = pd.to_numeric(df["card1"], errors="coerce").astype("float64")
    addr1 = pd.to_numeric(df["addr1"], errors="coerce").astype("float64")
    d1 = pd.to_numeric(df["D1"], errors="coerce").astype("float64")
    first_day = day_index(df[config.TIME_COL]).astype("float64") - d1
    ok = card1.notna() & addr1.notna() & first_day.notna()
    uid = pd.Series(np.nan, index=df.index, dtype="object")
    if ok.any():
        uid[ok] = (
            card1[ok].astype("int64").astype(str)
            + "_"
            + addr1[ok].astype("int64").astype(str)
            + "_"
            + first_day[ok].round().astype("int64").astype(str)
        )
    return uid.rename(config.UID_COL)


def normalise_d(df: pd.DataFrame, cols: tuple[str, ...] = D_TO_NORMALISE) -> pd.DataFrame:
    """Return day - D for each column (e.g. D1n = estimated first-seen day of the card)."""
    day = day_index(df[config.TIME_COL]).astype("float64")
    return pd.DataFrame(
        {f"{c}n": day - pd.to_numeric(df[c], errors="coerce").astype("float64") for c in cols},
        index=df.index,
    )
