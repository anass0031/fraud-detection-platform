import numpy as np
import pandas as pd

from fraudml.features.keys import day_index, make_uid, normalise_d


def _frame():
    return pd.DataFrame(
        {
            "TransactionDT": [86_400 * 10 + 5, 86_400 * 12, 86_400 * 3, 86_400 * 4],
            "card1": [1000, 1000, 2000, 3000],
            "addr1": [330.0, 330.0, np.nan, 100.0],
            "D1": [5.0, 7.0, 0.0, np.nan],
            "D4": [1.0, np.nan, 2.0, 3.0],
            "D10": [0.0, 0.0, 0.0, 0.0],
            "D15": [2.0, 4.0, 1.0, 1.0],
        }
    )


def test_day_index():
    assert day_index(_frame()["TransactionDT"]).tolist() == [10, 12, 3, 4]


def test_uid_same_customer_and_missing_inputs():
    uid = make_uid(_frame())
    # Rows 0 and 1: same card/addr, first-seen day 10-5 = 12-7 = 5 -> same uid.
    assert uid[0] == uid[1] == "1000_330_5"
    assert pd.isna(uid[2])  # addr1 missing
    assert pd.isna(uid[3])  # D1 missing


def test_uid_independent_of_input_dtypes():
    df = _frame()
    df32 = df.astype({"card1": "int16", "addr1": "float32", "D1": "float32"})
    pd.testing.assert_series_equal(make_uid(df), make_uid(df32))


def test_normalise_d():
    out = normalise_d(_frame())
    assert out["D1n"].tolist()[:3] == [5.0, 5.0, 3.0]
    assert np.isnan(out["D4n"][1])
    assert list(out.columns) == ["D1n", "D4n", "D10n", "D15n"]
