import numpy as np
import pandas as pd

from fraudml.features.static import (
    BROWSER_RULES,
    OS_RULES,
    amount_features,
    as_str,
    email_features,
    map_family,
    row_features,
    time_features,
)


def test_time_features():
    # 2017-12-01 is a Friday; TransactionDT 86400 = 2017-12-02 00:00 (Saturday).
    out = time_features(pd.Series([86_400, 86_400 + 7 * 3600 + 59, 86_400 * 3]))
    assert out["hour"].tolist() == [0, 7, 0]
    assert out["weekday"].tolist() == [5, 5, 0]  # Saturday, Saturday, Monday


def test_amount_features_exact_cents():
    out = amount_features(pd.Series([117.99, 49.0, 30.927, 0.5]))
    assert out["amt_cents"].tolist() == [99, 0, 93, 50]
    assert out["amt_is_round"].tolist() == [0, 1, 0, 0]
    np.testing.assert_allclose(out["amt_log"], np.log1p([117.99, 49.0, 30.927, 0.5]))


def test_email_same_domain_only_when_both_present():
    out = email_features(
        pd.Series(["gmail.com", "Gmail.com", "yahoo.com", None]),
        pd.Series(["gmail.com", "hotmail.com", None, "gmail.com"]),
    )
    assert out["email_same_domain"].tolist()[:2] == [1.0, 0.0]
    assert out["email_same_domain"][2:].isna().all()
    assert out["p_email"][1] == "gmail.com"  # lower-cased


def test_browser_and_os_families():
    browsers = pd.Series(
        [
            "chrome 63.0 for android",
            "mobile safari 11.0",
            "ie 11.0 for desktop",
            "edge 16.0",
            "samsung browser 6.2",
            "Microsoft/Windows",
            None,
        ]
    )
    assert map_family(browsers, BROWSER_RULES).tolist()[:6] == [
        "chrome",
        "safari",
        "ie",
        "edge",
        "samsung",
        "other",
    ]
    assert pd.isna(map_family(browsers, BROWSER_RULES)[6])
    oses = pd.Series(["Windows 10", "iOS 11.2.1", "Mac OS X 10_13_2", "Android 7.0", "func"])
    assert map_family(oses, OS_RULES).tolist() == ["windows", "ios", "mac", "android", "other"]


def test_as_str_keeps_nan_for_category_input():
    s = pd.Series(["a", None, "b"], dtype="category")
    out = as_str(s)
    assert out.tolist()[0] == "a" and pd.isna(out[1])


def test_row_features_shape_and_flags():
    df = pd.DataFrame(
        {
            "TransactionDT": [86_400, 90_000],
            "TransactionAmt": [10.0, 20.5],
            "P_emaildomain": ["gmail.com", None],
            "R_emaildomain": ["gmail.com", None],
            "ProductCD": ["W", "C"],
            "card1": [1000, 2000],
            "card2": [111.0, np.nan],
            "card4": ["visa", "Mastercard"],
            "card6": ["debit", "credit"],
            "addr1": [330.0, np.nan],
            "dist1": [5.0, np.nan],
            "DeviceType": [None, "mobile"],
            "DeviceInfo": [None, "SM-A300H"],
            "id_01": [np.nan, -5.0],
            "id_30": [None, "Android 5.1.1"],
            "id_31": [None, "chrome 62.0 for android"],
            **{f"C{i}": [1.0, 2.0] for i in range(1, 15)},
            **{f"M{i}": ["T", None] for i in range(1, 10)},
            **{c: [1.0, 2.0] for c in ["D1", "D4", "D10", "D15"]},
        }
    )
    uid = pd.Series(["1000_330_1", np.nan])
    out = row_features(df, uid)
    assert out["has_identity"].tolist() == [0, 1]
    assert out["has_uid"].tolist() == [1, 0]
    assert out["addr1_missing"].tolist() == [0, 1]
    assert out["card4"].tolist() == ["visa", "mastercard"]
    assert out["browser_family"][1] == "chrome" and out["os_family"][1] == "android"
    assert {"D1n", "D15n", "C14", "M9", "DeviceInfo"} <= set(out.columns)
