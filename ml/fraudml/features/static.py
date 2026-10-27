"""Row-level features: computed from the transaction itself, no history needed.

The output still contains *raw* categorical strings (e.g. e-mail domain, browser family) and raw
columns used for frequency encoding (card1, DeviceInfo, ...). The FeatureEncoder (fitted on the
train period only) turns these into the final categories and frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudml import config
from fraudml.features.keys import SECONDS_PER_DAY, normalise_d

# Reference date 2017-12-01 is a Friday (Monday = 0 -> Friday = 4).
REFERENCE_WEEKDAY = 4

C_COLS = tuple(f"C{i}" for i in range(1, 15))
M_COLS = tuple(f"M{i}" for i in range(1, 10))

# Raw categorical columns handed to the encoder (rare grouping + fixed category list).
RAW_CATEGORICAL = (
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
# Raw columns handed to the encoder for frequency encoding: output name -> source column.
RAW_FREQUENCY = {
    "card1_freq": "card1",
    "card2_freq": "card2",
    "addr1_freq": "addr1",
    "p_email_freq": "p_email",
    "device_info_freq": "DeviceInfo",
}


def as_str(s: pd.Series, lower: bool = False) -> pd.Series:
    """Object-dtype strings with real NaN for missing values (independent of the input dtype)."""
    out = s.astype("object")
    mask = out.notna()
    cleaned = out[mask].astype(str).str.strip()
    if lower:
        cleaned = cleaned.str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="object")
    out[mask] = cleaned
    return out


def _family(value: str, rules: tuple[tuple[str, str], ...]) -> str:
    v = value.lower()
    for needle, family in rules:
        if needle in v:
            return family
    return "other"


BROWSER_RULES = (
    ("edge", "edge"),
    ("samsung", "samsung"),
    ("opera", "opera"),
    ("firefox", "firefox"),
    ("chrome", "chrome"),
    ("google", "google_app"),
    ("safari", "safari"),
    ("android", "android_browser"),
    ("ie ", "ie"),
    ("ie11", "ie"),
)
OS_RULES = (
    ("windows", "windows"),
    ("ios", "ios"),
    ("mac", "mac"),
    ("android", "android"),
    ("linux", "linux"),
)


def map_family(s: pd.Series, rules: tuple[tuple[str, str], ...]) -> pd.Series:
    """Map version strings to a stable family (e.g. 'chrome 63.0 for android' -> 'chrome')."""
    text = as_str(s)
    uniques = text.dropna().unique()
    mapping = {u: _family(u, rules) for u in uniques}
    return text.map(mapping)


def time_features(transaction_dt: pd.Series) -> pd.DataFrame:
    t = transaction_dt.astype("int64")
    day = t // SECONDS_PER_DAY
    return pd.DataFrame(
        {
            "hour": ((t // 3600) % 24).astype("float64"),
            "weekday": ((day + REFERENCE_WEEKDAY) % 7).astype("float64"),
        },
        index=transaction_dt.index,
    )


def amount_features(amount: pd.Series) -> pd.DataFrame:
    amt = amount.astype("float64")
    cents = np.round(amt * 100) % 100
    return pd.DataFrame(
        {
            "amt_log": np.log1p(amt),
            "amt_cents": cents,
            "amt_is_round": (cents == 0).astype("float64"),
        },
        index=amount.index,
    )


def email_features(p_email: pd.Series, r_email: pd.Series) -> pd.DataFrame:
    p, r = as_str(p_email, lower=True), as_str(r_email, lower=True)
    both = p.notna() & r.notna()
    same = pd.Series(np.nan, index=p.index, dtype="float64")
    same[both] = (p[both] == r[both]).astype("float64")
    return pd.DataFrame({"p_email": p, "r_email": r, "email_same_domain": same}, index=p.index)


def row_features(df: pd.DataFrame, uid: pd.Series) -> pd.DataFrame:
    """All row-level features (raw categoricals still unencoded)."""
    parts = [
        time_features(df[config.TIME_COL]),
        amount_features(df["TransactionAmt"]),
        email_features(df["P_emaildomain"], df["R_emaildomain"]),
    ]
    misc = pd.DataFrame(index=df.index)
    misc["ProductCD"] = as_str(df["ProductCD"])
    misc["card4"] = as_str(df["card4"], lower=True)
    misc["card6"] = as_str(df["card6"], lower=True)
    misc["device_type"] = as_str(df["DeviceType"], lower=True)
    misc["browser_family"] = map_family(df["id_31"], BROWSER_RULES)
    misc["os_family"] = map_family(df["id_30"], OS_RULES)
    misc["has_identity"] = df["id_01"].notna().astype("float64")
    misc["has_uid"] = uid.notna().astype("float64")
    misc["addr1_missing"] = df["addr1"].isna().astype("float64")
    misc["dist1"] = pd.to_numeric(df["dist1"], errors="coerce").astype("float64")
    for c in C_COLS:
        misc[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in M_COLS:
        misc[c] = as_str(df[c])
    # Raw sources for frequency encoding (dropped after encoding).
    misc["card1"] = pd.to_numeric(df["card1"], errors="coerce").astype("float64")
    misc["card2"] = pd.to_numeric(df["card2"], errors="coerce").astype("float64")
    misc["addr1"] = pd.to_numeric(df["addr1"], errors="coerce").astype("float64")
    misc["DeviceInfo"] = as_str(df["DeviceInfo"])
    parts += [misc, normalise_d(df)]
    return pd.concat(parts, axis=1)
