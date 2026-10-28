import numpy as np
import pandas as pd
import pytest

from fraudml.evaluate.cost import CostConfig
from fraudml.models.retrain import (
    WINDOWS,
    align_features,
    fraud_customers,
    split_windows,
    window_metrics,
)


def test_windows_are_consecutive_weeks_without_overlap():
    names = ["train_extra", "gap", "calibration", "comparison"]
    assert list(WINDOWS) == names
    for a, b in zip(names, names[1:], strict=False):
        assert WINDOWS[a][1] == WINDOWS[b][0]
    for start, end in WINDOWS.values():
        assert (pd.Timestamp(end) - pd.Timestamp(start)).days == 7


def test_split_windows_drops_unlabelled_rows_and_respects_boundaries():
    df = pd.DataFrame(
        {
            "transaction_time": pd.to_datetime(
                [
                    "2018-04-28 00:00",  # train_extra
                    "2018-05-04 23:59",  # train_extra
                    "2018-05-05 00:00",  # gap
                    "2018-05-12 10:00",  # calibration, no label -> dropped
                    "2018-05-19 00:00",  # comparison
                    "2018-05-26 00:00",  # after the comparison week
                ]
            ),
            "label": pd.array([0, 1, 0, None, 1, 0], dtype="Int8"),
        }
    )
    parts = split_windows(df)
    assert {k: len(v) for k, v in parts.items()} == {
        "train_extra": 2,
        "gap": 1,
        "calibration": 0,
        "comparison": 1,
    }


def test_align_features_matches_training_dtypes():
    template = pd.DataFrame(
        {
            "card6": pd.Categorical(["credit", "debit"], categories=["credit", "debit", "other"]),
            "amt_log": np.array([1.0, 2.0], dtype="float64"),
            "uid_n_prior": np.array([0, 1], dtype="float32"),
        }
    )
    db_rows = pd.DataFrame(
        {"card6": ["other", None], "amt_log": [3.5, None], "uid_n_prior": [2, 3]}
    )
    out = align_features(db_rows, template)
    assert list(out.columns) == list(template.columns)
    assert out["card6"].dtype == template["card6"].dtype
    assert out["card6"].tolist()[0] == "other" and pd.isna(out["card6"].tolist()[1])
    assert out["amt_log"].dtype == "float64" and np.isnan(out["amt_log"][1])
    assert out["uid_n_prior"].dtype == "float32"


def test_align_features_refuses_unknown_categories():
    template = pd.DataFrame({"card6": pd.Categorical(["credit"], categories=["credit", "other"])})
    with pytest.raises(ValueError, match="outside the training categories"):
        align_features(pd.DataFrame({"card6": ["prepaid"]}), template)


def test_fraud_customers():
    uid = pd.Series(["a", "a", "b", None, "c"])
    assert fraud_customers(uid, [0, 1, 0, 1, 0]) == {"a"}  # missing uid ignored


def test_window_metrics_use_the_cost_model():
    m = window_metrics(
        y=[1, 1, 0, 0],
        prob=[0.9, 0.2, 0.8, 0.1],
        amount=[100.0, 50.0, 200.0, 10.0],
        alert=[True, False, True, False],
        day=[0, 0, 1, 1],
        cfg=CostConfig(review_cost=5, friction_cost=2),
    )
    assert (m["n"], m["n_fraud"], m["alerts"], m["alerts_per_day"]) == (4, 2, 2, 1.0)
    assert (m["precision"], m["recall"]) == (0.5, 0.5)
    assert m["fraud_value_caught"] == pytest.approx(100 / 150)
    assert m["total_cost"] == 62.0  # missed 50 + 2 reviews x 5 + 1 false alarm x 2
    assert m["net_savings"] == 88.0


def test_paired_bootstrap_detects_a_clearly_better_score():
    from fraudml.models.retrain import paired_bootstrap_pr_auc

    rng = np.random.default_rng(0)
    y = rng.random(7_000) < 0.05
    day = np.repeat(np.arange(7), 1_000)
    weak = rng.random(7_000) + y * 0.3
    strong = rng.random(7_000) + y * 1.5
    r = paired_bootstrap_pr_auc(y, weak, strong, day, n_boot=200)
    assert r["diff"] > 0 and r["low"] > 0 and r["share_positive"] == 1.0
    assert r["days"] == 7
    same = paired_bootstrap_pr_auc(y, weak, weak, day, n_boot=50)
    assert same["diff"] == 0 and same["low"] == 0 and same["high"] == 0
