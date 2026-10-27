"""Shared synthetic raw transactions for feature tests (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

DAY = 86_400


def make_synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Raw transactions shaped like the processed parquet, with tricky cases built in:
    repeat customers, same-second ties, missing addr1 / D1 (no uid), rare and unseen categories,
    missing identity data, category / float32 dtypes like the real parquet."""
    rng = np.random.default_rng(seed)
    n_cards = 40
    card = rng.integers(0, n_cards, n)
    first_day = rng.integers(0, 30, n_cards)
    t = np.sort(rng.integers(DAY, 120 * DAY, n))
    t[10] = t[9]  # same-second tie (possibly same customer)
    t[11] = t[9]
    card[10] = card[9]
    day = t // DAY
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(3_000_000, 3_000_000 + n),
            "TransactionDT": t,
            "TransactionAmt": np.round(rng.lognormal(4, 1, n), 2),
            "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n, p=[0.6, 0.2, 0.1, 0.07, 0.03]),
            "card1": (1000 + card).astype("int16"),
            "card2": np.where(rng.random(n) < 0.05, np.nan, 100 + card % 7).astype("float32"),
            "card4": rng.choice(["visa", "mastercard", "discover"], n, p=[0.6, 0.38, 0.02]),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": np.where(rng.random(n) < 0.1, np.nan, 200 + card % 5).astype("float32"),
            "dist1": np.where(rng.random(n) < 0.6, np.nan, rng.integers(0, 500, n)).astype(
                "float32"
            ),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "rare.org", None], n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None, None], n),
            "DeviceType": rng.choice(["desktop", "mobile", None, None], n),
            "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-A300H", None], n),
            "id_01": np.where(rng.random(n) < 0.7, np.nan, -5.0).astype("float32"),
            "id_30": rng.choice(["Windows 10", "iOS 11.1", None, None], n),
            "id_31": rng.choice(["chrome 63.0", "mobile safari 11.0", "edge 16.0", None], n),
            "D1": np.where(rng.random(n) < 0.03, np.nan, day - first_day[card]).astype("float32"),
            "D4": rng.integers(0, 50, n).astype("float32"),
            "D10": rng.integers(0, 50, n).astype("float32"),
            "D15": rng.integers(0, 50, n).astype("float32"),
            **{f"C{i}": rng.integers(0, 5, n).astype("float32") for i in range(1, 15)},
            **{f"M{i}": rng.choice(["T", "F", None], n) for i in range(1, 10)},
            "isFraud": (rng.random(n) < 0.05).astype("int8"),
        }
    )
    for c in [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
        *[f"M{i}" for i in range(1, 10)],
    ]:
        df[c] = df[c].astype("category")
    return df


@pytest.fixture
def synthetic():
    return make_synthetic()
